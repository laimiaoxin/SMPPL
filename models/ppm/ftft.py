import math
import fvcore.nn.weight_init as weight_init
from einops import rearrange

import torch
import torchvision
from torch import nn, Tensor
import torch.nn.functional as F

from .module import ConvLayer2d, UpSample2d
from .common import MLP, get_norm, get_act


class CSM(nn.Module):
    def __init__(
            self,
            hidden_dim,
            factor=2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            ConvLayer2d(hidden_dim, hidden_dim // factor, 1, bias=False, norm='LN', act_func='leaky'),
            ConvLayer2d(hidden_dim // factor, hidden_dim, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.mean(x, dim=[2, 3], keepdim=True)
        x_w = self.mlp(x_avg)
        x_w = self.sigmoid(x_w)

        return x + x * x_w

class PrototypePromptEncoder(nn.Module):
    def __init__(
            self,
            feat_dim=256,
            num_protos=5,
            num_classes=1,
            num_heads=8,
    ):
        super().__init__()
        self.prototype_refinement = PrototypeRefinement(
            feat_dim=feat_dim,
            num_heads=num_heads,
            num_classes=num_classes
        )
        # 初始化PromptGenerator
        self.prompt_generator = PromptGenerator(
            feat_dim=feat_dim,
            num_protos=num_protos,
        )
        self.num_classes = num_classes
        self.num_protos = num_protos
        self.num_heads = num_heads
        self.level_image = nn.Embedding(1, feat_dim)
        self.level_mask = nn.Embedding(1, num_classes)

    def forward(
            self,
            out_embed: torch.Tensor,
            mask_embed: torch.Tensor,
            intra_prototypes: torch.Tensor,
            intra_embed: torch.Tensor,
            masks: torch.Tensor
    ):

        # masks = torch.softmax(masks, dim=1)
        out_embed = out_embed + self.level_image.weight.reshape(1, -1, 1, 1)
        mask_embed = mask_embed + self.level_mask.weight.reshape(1, -1, 1, 1)

        intra_prototypes = self.prototype_refinement(
            image_embed=out_embed,
            mask_embed=mask_embed,
            intra_prototypes=intra_prototypes,
            intra_embed=intra_embed,
            masks=masks
        )

        dense_prompts, sparse_prompts = self.prompt_generator(
            out_embed=out_embed,
            intra_prototypes=intra_prototypes,
            intra_embed=intra_embed,
            masks=masks
        )

        return out_embed, mask_embed, intra_prototypes, dense_prompts, sparse_prompts


class PrototypeRefinement(nn.Module):
    def __init__(
            self,
            feat_dim=256,
            num_heads=8,
            num_classes=1,
    ):
        super().__init__()

        self.in_proj = nn.Linear(feat_dim, feat_dim)

        self.class_attn = ClassAttention(feat_dim, num_heads, num_classes)
        self.intra = nn.Parameter(torch.ones(feat_dim) * 1e-3)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image_embed: torch.Tensor, mask_embed: torch.Tensor, intra_prototypes: torch.Tensor,
                intra_embed: torch.Tensor,masks: torch.Tensor):
        intra_prototypes = intra_prototypes + intra_embed
        prototypes = self.in_proj(intra_prototypes)
        prototypes = self.class_attn(prototypes,  # [b, 5, c]
                                     image_embed,  # [b, c, h, w]
                                     mask_embed,  # [b, 1, h, w]
                                     masks  # [b, 1, h, w]
                                     )
        return prototypes

class ClassAttention(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            num_classes: int,

    ):
        super().__init__()

        self.q_norm = get_norm('LN', embedding_dim, d=False)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.x_proj = nn.Sequential(
            ConvLayer2d(embedding_dim + num_classes, embedding_dim, 1, act_func='gelu'),
            ConvLayer2d(embedding_dim, embedding_dim, 1, norm='LN')
        )
        self.x_conv = ConvLayer2d(embedding_dim, embedding_dim, 3, groups=embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.num_heads = num_heads
        self.num_classes = num_classes
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, query_prototypes: Tensor, image_embed: Tensor, mask_embed: Tensor, masks: Tensor) -> Tensor:

        # Input
        query = self.q_proj(self.q_norm(query_prototypes))
        x = self.x_proj(torch.cat((image_embed, mask_embed), dim=1))
        x = self.x_conv(x).flatten(2).permute(0, 2, 1)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # Multi-head Attn
        query = self._separate_heads(query, self.num_heads)  # [b, n, q, c]
        key = self._separate_heads(key, self.num_heads)  # [b, n, d, c]
        value = self._separate_heads(value, self.num_heads)  # [b, n, d, c]
        B, N, Q, C = query.shape

        # Query Attn
        att_query = query @ key.permute(0, 1, 3, 2)  # [b, n, q, d]
        masks = masks.flatten(2).unsqueeze(1).expand(-1, self.num_heads, Q, -1)
        att_query = att_query * masks
        # Output
        att_weight = torch.softmax(att_query, dim=-1)
        out = att_weight @ value
        out = self._recombine_heads(out)  # [b, q, c]
        out = self.out_proj(out)

        return out

class PromptGenerator(nn.Module):
    def __init__(self, feat_dim=256, num_protos=5):
        super().__init__()
        self.dense_prompt_generator = DensePromptGenerator(
            feat_dim=feat_dim, num_protos=num_protos)
        self.sparse_prompt_generator = SparsePromptGenerator(
            feat_dim=feat_dim, num_protos=num_protos)

    def forward(self, out_embed, intra_prototypes, intra_embed, masks=None):
        # 原型融合增强
        intra_prototypes = intra_prototypes + intra_embed
        dense_prompts = self.dense_prompt_generator(out_embed, intra_prototypes, masks)
        sparse_prompts = self.sparse_prompt_generator(out_embed, intra_prototypes, masks)
        return dense_prompts, sparse_prompts


class DensePromptGenerator(nn.Module):
    def __init__(
            self,
            feat_dim=256,
            num_protos=5,  # 农田子原型数量
    ):
        super().__init__()

        self.proj = nn.Linear(feat_dim, feat_dim)
        self.act_func = get_act('gelu')
        self.linear = nn.Linear(feat_dim, feat_dim)
        self.norm = get_norm('LN', feat_dim, d=False)
        self.proto_gate = nn.Parameter(torch.ones(num_protos))

        self.dcn = DeformAttn(feat_dim)

        self.out = nn.Sequential(
            ConvLayer2d(feat_dim, feat_dim // 2, 1),
            get_act('gelu'),
            ConvLayer2d(feat_dim // 2, feat_dim, 1)
        )

        self.alpha = nn.Parameter(torch.ones(feat_dim) * 1e-3)
        self.scale = feat_dim ** -0.5
        self.num_protos = num_protos
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image_embed, intra_prototypes, masks=None):
        """
        Args:
            image_embed: [B, C, H, W]
            intra_prototypes: [B, num_protos, C]  # 农田子原型集合
            masks: [B, 1, H, W]  可选
        """
        intra_prototypes = intra_prototypes * torch.sigmoid(self.proto_gate).view(1, -1, 1)
        proto_sim = torch.einsum('bqc,bkc->bqk', intra_prototypes, intra_prototypes)  # [B, Q, Q]
        proto_sim = proto_sim * self.scale
        proto_attn = torch.softmax(proto_sim, dim=-1)

        enhanced_proto = torch.einsum('bqk,bkc->bqc', proto_attn, intra_prototypes)
        prototype_tokens = self.proj(enhanced_proto)
        prototype_tokens = self.act_func(prototype_tokens)
        prototype_tokens = self.linear(prototype_tokens)
        prototype_tokens = self.norm(prototype_tokens)

        if masks is not None:
            prompt_embed = intra_prototypes.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
            image_embed = image_embed * (1 + prompt_embed * masks)

        dense_embed = self.dcn(image_embed)

        attn_embed = torch.einsum("bchw,bqc->bqhw", dense_embed, prototype_tokens)
        attn_embed = torch.softmax(attn_embed * self.scale, dim=1)
        attn_embed = torch.einsum("bqhw,bqc->bchw", attn_embed, prototype_tokens)

        attn_embed = attn_embed + dense_embed * self.alpha.reshape(1, -1, 1, 1)
        dense_prompts = self.out(attn_embed)

        return dense_prompts

class DeformAttn(nn.Module):
    def __init__(
            self,
            dim
    ):
        super().__init__()

        self.in_proj = ConvLayer2d(dim, dim, 1)
        self.act_func = get_act('gelu')
        self.deform_conv0 = DeformConv(dim, 3, groups=dim, dilation=1)
        self.deform_conv1 = DeformConv(dim, 3, groups=dim, dilation=3)
        self.conv = ConvLayer2d(dim, dim, 1)

        self.norm = get_norm('LN', dim)
        self.out_proj = nn.Sequential(
            ConvLayer2d(dim, dim, 1),
            ConvLayer2d(dim, dim, 3, groups=dim, act_func='gelu'),
            ConvLayer2d(dim, dim, 1)
        )

    def forward(self, image_embed):
        x = self.in_proj(image_embed)
        x = self.act_func(x)
        x = self.deform_attn(x) + image_embed

        x = self.norm(x)
        x = self.out_proj(x) + x

        return x

    def deform_attn(self, x):
        attn = self.deform_conv0(x)
        attn = self.deform_conv1(attn)
        attn = self.conv(attn)

        return x * attn

class DeformConv(nn.Module):
    def __init__(
            self,
            d_modal,
            kernel_size=3,
            groups=1,
            dilation=1
    ):
        super().__init__()

        padding = kernel_size // 2 * dilation
        offset_channels = 3 * kernel_size * kernel_size
        self.dcn_offset = nn.Conv2d(in_channels=d_modal,
                                    out_channels=offset_channels,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    dilation=dilation)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=d_modal,
                                                        out_channels=d_modal,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        dilation=dilation)

        self._reset_parameters()

    def _reset_parameters(self):
        weight_init.c2_msra_fill(self.deform_conv)

        nn.init.constant_(self.dcn_offset.weight, 0)
        nn.init.constant_(self.dcn_offset.bias, 0)

    def forward(self, x):
        offsets = self.dcn_offset(x)
        offset_x, offset_y, mask = torch.chunk(offsets, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()

        out = self.deform_conv(x, offset, mask)
        return out


class SparsePromptGenerator(nn.Module):
    def __init__(self, feat_dim=256, num_protos=5):
        super().__init__()
        self.dconv = ConvLayer2d(feat_dim, feat_dim, 3, dilation=2)
        self.prototype_adapter = PrototypeAdapter(feat_dim, num_protos)

        self.out = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            get_act('gelu'),
            nn.Linear(feat_dim // 2, feat_dim)
        )

        self.alpha = nn.Parameter(torch.ones(feat_dim) * 1e-3)
        self.scale = feat_dim ** -0.5
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image_embed, intra_prototypes, masks=None):
        """
        image_embed: [B, C, H, W]
        intra_prototypes: [B, num_protos, C]
        masks: [B, 1, H, W] 可选
        """
        B, C, H, W = image_embed.shape

        # Step 1: 局部增强特征
        image_embed = self.dconv(image_embed)

        # Step 2: 原型自适应更新（soft dynamic gating）
        sparse_embed = self.prototype_adapter(image_embed, intra_prototypes)  # [B, P, C]

        # Step 3: 稀疏 Attention 聚合
        img_flat = image_embed.flatten(2)  # [B, C, HW]
        attn_score = torch.einsum("bqc,bcd->bqd", sparse_embed, img_flat) * self.scale
        attn_weight = torch.softmax(attn_score, dim=-1)

        if masks is not None:
            attn_weight = attn_weight * masks.flatten(2)

        attn_out = torch.einsum("bqd,bcd->bqc", attn_weight, img_flat)

        # Step 4: 残差融合 + MLP 输出
        fused = attn_out + sparse_embed * self.alpha.reshape(1, 1, -1)
        sparse_prompts = self.out(fused)

        return sparse_prompts

class PrototypeAdapter(nn.Module):
    def __init__(self, feat_dim=256, num_protos=5):
        super().__init__()
        self.num_protos = num_protos
        self.scale = feat_dim ** -0.5

        self.in_proj = nn.Linear(feat_dim, feat_dim)
        self.act_func = get_act('gelu')
        self.linear = nn.Linear(feat_dim, feat_dim)
        self.norm = get_norm('LN', feat_dim, d=False)

        # 每个原型的动态门控参数（可学习）
        self.proto_gate = nn.Parameter(torch.ones(num_protos))
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image_embed, intra_prototypes):
        """
        image_embed: [B, C, H, W]
        intra_prototypes: [B, P, C]
        """
        B, C, H, W = image_embed.shape
        P = self.num_protos

        feat_flat = rearrange(image_embed, 'b c h w -> b (h w) c')
        feat_norm = F.normalize(feat_flat, dim=-1)
        proto_norm = F.normalize(intra_prototypes, dim=-1)

        # 相似度 [B, N, P]
        sim = torch.einsum('bnc,bpc->bnp', feat_norm, proto_norm)
        proto_act = sim.mean(dim=1) * self.scale  # [B, P]

        # === Soft Dynamic Gating ===
        gate = torch.sigmoid(self.proto_gate).view(1, P)
        sparse_weight = F.softmax(proto_act * gate, dim=-1)   # [B, P]
        sparse_weight = sparse_weight ** 2                    # 增强稀疏性
        sparse_weight = sparse_weight / sparse_weight.sum(dim=-1, keepdim=True)

        # === 加权原型融合 ===
        act_proto = intra_prototypes * sparse_weight.unsqueeze(-1)  # [B, P, C]
        adapt_proto = self.in_proj(act_proto)
        adapt_proto = self.act_func(adapt_proto)
        adapt_proto = self.linear(adapt_proto)
        adapt_proto = self.norm(adapt_proto)
        return adapt_proto

class GlobalPrototypes(nn.Module):
    def __init__(self, feat_dim=256, num_protos=5):
        super().__init__()

        self.num_protos = num_protos
        self.feat_dim = feat_dim

        self.intra_prototypes = nn.Embedding(num_protos, feat_dim)
        self.intra_embed = nn.Embedding(num_protos, feat_dim)

        # 初始化
        nn.init.trunc_normal_(self.intra_prototypes.weight, std=0.02)
        nn.init.trunc_normal_(self.intra_embed.weight, std=0.02)

    def forward(self):

        return self.intra_prototypes.weight, self.intra_embed.weight