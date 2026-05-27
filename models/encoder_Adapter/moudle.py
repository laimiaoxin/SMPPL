import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
# from models.encoder_Adapter.ops.modules import MSDeformAttn
from timm.models.layers import DropPath
from math import log
from models.spatialmamba import SpatialMambaBlock
_logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_,
                           dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes1 = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index1 = torch.cat((spatial_shapes1.new_zeros(
        (1,)), spatial_shapes1.prod(1).cumsum(0)[:-1]))
    reference_points1 = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points1, spatial_shapes1, level_start_index1]

    spatial_shapes2 = torch.as_tensor(
        [(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index2 = torch.cat((spatial_shapes2.new_zeros(
        (1,)), spatial_shapes2.prod(1).cumsum(0)[:-1]))
    reference_points2 = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points2, spatial_shapes2, level_start_index2]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(
                dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(
                drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + \
                    self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, refine_channels=1, norm_layer=partial(nn.LayerNorm, eps=1e-6),extra_extractor=False,in_channels_m=64,mid_channels=64,need_fusion=False):
        super().__init__()
        if need_fusion:
            self.fusion = SpatialMambaBlock(hidden_dim=dim)
        else:
            self.fusion = None
        self.extractor = EFEM(in_channels_f=dim, in_channels_m=in_channels_m,mid_channels=mid_channels)


    def forward(self, x, c, blocks, H, W, idx):
        """
        参数:
            x: 高级特征 (例如ViT特征), 形状 [B, L, C]
            c: 低级特征 (例如CNN边缘特征), 形状 [B, C, H, W]
            blocks: Transformer块列表
            H, W: 特征图的高度和宽度
        """

        if self.fusion is not None and idx >= 2:
            # reshape vit_feat → [B, C, H, W]
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

            x = self.fusion(x, c)  # [B, C, H, W]

            # 3. 重塑回序列形式以通过Transformer块
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        for blk in blocks:
            x = blk(x, H, W)

        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        c = self.extractor(x, c)  # [B, C, H, W]

        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        return x, c  # 返回处理后的序列和特征图

class EdgeAwareModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.stem = EdgeStem(in_channels=3, out_channels=64)
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.out_indices = out_indices

        self.edge = LaplaceConv2d(in_channels=3, out_channels=1)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)  # 将c4上采样8倍到c1的尺寸
        self.fusion = nn.Sequential(
            nn.Conv2d(inplanes * 5 + 1, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.down_c1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1),  # H/2 -> H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1),  # H/4 -> H/8
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1),  # H/8 -> H/16
            nn.ReLU(inplace=True)
        )

        self.down_edge = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        B, C, H, W = x.shape
        x1 = self.edge(x)  # 边缘图 [B, 1, H, W]
        c1 = self.stem(x)  # [B, 64, H/2, W/2]
        c2 = self.conv2(c1)  # [B, 128, H/4, W/4]
        c3 = self.conv3(c2)  # [B, 256, H/8, W/8]
        c4 = self.conv4(c3)  # [B, 256, H/16, W/16]

        c1 = self.down_c1(c1)              # [B, 64, H/16, W/16]
        x1 = self.down_edge(x1)

        edge = torch.cat([x1, c1, c4], dim=1)
        edge = self.fusion(edge)  # [B, embed_dim, H, W]

        return edge



class EdgeStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 融合卷积 - 保持通道数为out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            # 第一次下采样：3x3卷积，stride=2，保持通道数不变
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二次下采样：同上，总下采样率仍为4（2×2）
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)

        fused = torch.cat([f1, f2, f3], dim=1)  # [B, 3*out_channels, H, W]
        out = self.fusion(fused)  # [B, out_channels, H, W]
        out = self.downsample(out)      # [B, out_channels, H/4, W/4]
        return out

class LaplaceConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(LaplaceConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

        # Generate Laplace kernel
        laplace_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)  ##8领域
        laplace_kernel = laplace_kernel.unsqueeze(0).unsqueeze(0)
        laplace_kernel = laplace_kernel.repeat((out_channels, in_channels, 1, 1))
        self.conv.weight = nn.Parameter(laplace_kernel)
        self.conv.bias.data.fill_(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.relu(self.bn(x1))

        return x1


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x




def laplacian_convolution(input_feature_map):
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    laplacian_kernel = laplacian_kernel.repeat(input_feature_map.size(1), 1, 1, 1).to(input_feature_map.device)
    laplacian_conv = F.conv2d(input_feature_map, laplacian_kernel, stride=1, padding=1, groups=input_feature_map.size(1))
    return laplacian_conv

def laplacian_and_add(input_feature_map):
    laplacian_output = laplacian_convolution(input_feature_map)
    laplacian_output_relu = F.relu(laplacian_output)
    result = input_feature_map + laplacian_output_relu
    return result


# Conv-BN-ReLU 基础块
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AxialAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 输入通道数

        # 1. 水平轴向注意力（捕捉H方向的通道+空间关联）
        self.axial_h = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim),  # 深度卷积降维
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        # 2. 垂直轴向注意力（捕捉W方向的通道+空间关联）
        self.axial_w = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim),  # 深度卷积降维
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.axial_s = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # 深度卷积降维
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        # 3. 交互融合
        self.interact_fuse = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, sem_feat):

        sem_h = self.axial_h(sem_feat)        # [B, mid_dim, H, W]

        sem_w = self.axial_w(sem_feat)        # [B, mid_dim, H, W]

        sem_s = self.axial_s(sem_feat)        # [B, mid_dim, H, W]

        # 步骤3：融合轴向特征
        sem_fused = torch.cat([sem_feat, sem_h, sem_w, sem_s], dim=1)  # [B, 4*mid_dim, H, W]
        sem_fused = self.interact_fuse(sem_fused)  # 升维回原通道数 → [B, C, H, W]

        return sem_fused
# EFEM 模块
class EFEM(nn.Module):
    def __init__(self, in_channels_f, in_channels_m, mid_channels=64):
        super(EFEM, self).__init__()

        # Kᵢ (边缘分支处理)
        self.cbr_k = CBR(in_channels_m, mid_channels, kernel_size=3, padding=1)

        # vᵢ₋₁ (对齐 fᵢ)
        self.cbr_v = CBR(in_channels_f, mid_channels, kernel_size=1, padding=0)
        self.axial_align = AxialAttention(dim=mid_channels)
        

        # Attention 生成
        self.conv_attn = nn.Sequential(
            CBR(mid_channels * 2, mid_channels, kernel_size=1, padding=0),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        

    def forward(self, f_i, m_prev):
        # --- Laplacian 残差增强 ---
        k_i = laplacian_and_add(m_prev)
        k_i = self.cbr_k(k_i)

        # --- 上采样并对齐 fᵢ ---
        f_i_up = F.interpolate(f_i, size=m_prev.shape[2:], mode='bilinear', align_corners=False)
        v_i = self.cbr_v(f_i_up)
        v_i = self.axial_align(v_i)

        # --- Attention map ---
        concat = torch.cat([k_i, v_i], dim=1)
        l_i = self.conv_attn(concat)

        # --- 融合输出 ---
        m_i = l_i * k_i

        return m_i
