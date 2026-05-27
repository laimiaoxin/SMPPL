# -----------------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# -----------------------------------------------------------------------------------
# VMamba: Visual State Space Model
# Copyright (c) 2024 MzeroMiko
# -----------------------------------------------------------------------------------
# Spatial-Mamba: Effective Visual State Space Models via Structure-Aware State Fusion
# Modified by Chaodong Xiao
# -----------------------------------------------------------------------------------

import math
import copy
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


from .utils import selective_scan_state_flop_jit, selective_scan_fn, Stem, DownSampling

from .dwconv_layer import DepthwiseFunction



class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StateFusion(nn.Module):
    def __init__(self, dim):
        super(StateFusion, self).__init__()

        self.dim = dim
        self.kernel_3   = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):

        if self.training:
            h1 = F.conv2d(self.padding(h, (1,1,1,1)), self.kernel_3,   padding=0, dilation=1, groups=self.dim)
            h2 = F.conv2d(self.padding(h, (3,3,3,3)), self.kernel_3_1, padding=0, dilation=3, groups=self.dim)
            h3 = F.conv2d(self.padding(h, (5,5,5,5)), self.kernel_3_2, padding=0, dilation=5, groups=self.dim)
            out = self.alpha[0]*h1 + self.alpha[1]*h2 + self.alpha[2]*h3
            return out

        else:
            if not hasattr(self, "_merge_weight"):
                self._merge_weight = torch.zeros((self.dim, 1, 11, 11), device=h.device)
                self._merge_weight[:, :, 4:7, 4:7] = self.alpha[0]*self.kernel_3

                self._merge_weight[:, :, 2:3, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,0:1,0:1]
                self._merge_weight[:, :, 2:3, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,0:1,1:2]
                self._merge_weight[:, :, 2:3, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[1]*self.kernel_3_1[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,1:2,2:3]
                self._merge_weight[:, :, 8:9, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,2:3,0:1]
                self._merge_weight[:, :, 8:9, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,2:3,1:2]
                self._merge_weight[:, :, 8:9, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,2:3,2:3]

                self._merge_weight[:, :, 0:1, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,0:1,0:1]
                self._merge_weight[:, :, 0:1, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,0:1,1:2]
                self._merge_weight[:, :, 0:1, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[2]*self.kernel_3_2[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,1:2,2:3]
                self._merge_weight[:, :, 10:11, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,2:3,0:1]
                self._merge_weight[:, :, 10:11, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,2:3,1:2]
                self._merge_weight[:, :, 10:11, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,2:3,2:3]

            out = DepthwiseFunction.apply(h, self._merge_weight, None, 11//2, 11//2, False)

            return out

class StructureAwareSSM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 共享输入投影层（语义+结构特征）
        self.in_proj_semantic = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_structural = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 语义特征专用卷积层
        self.conv2d_semantic = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # 结构特征专用卷积层
        self.conv2d_structural = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2, **factory_kwargs,
        )

        self.act = nn.SiLU()

        # 语义特征SSM参数
        self.x_proj_semantic = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight_semantic = nn.Parameter(self.x_proj_semantic.weight)
        del self.x_proj_semantic

        self.dt_projs_semantic = self.dt_init(
            self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs
        )
        self.dt_projs_weight_semantic = nn.Parameter(self.dt_projs_semantic.weight)
        self.dt_projs_bias_semantic = nn.Parameter(self.dt_projs_semantic.bias)
        del self.dt_projs_semantic

        self.A_logs_semantic = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds_semantic = self.D_init(self.d_inner, dt_init)

        # 结构特征SSM参数
        self.x_proj_structural = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                                           **factory_kwargs)
        self.x_proj_weight_structural = nn.Parameter(self.x_proj_structural.weight)
        del self.x_proj_structural

        self.dt_projs_structural = self.dt_init(
            self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs
        )
        self.dt_projs_weight_structural = nn.Parameter(self.dt_projs_structural.weight)
        self.dt_projs_bias_structural = nn.Parameter(self.dt_projs_structural.bias)
        del self.dt_projs_structural

        self.A_logs_structural = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds_structural = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn
        self.state_fusion = StateFusion(self.d_inner)

        # 输出融合层
        # self.out_norm = nn.LayerNorm(self.d_inner*2)
        self.out_norm_semantic = nn.LayerNorm(self.d_inner)
        self.out_norm_structural = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)  # 融合语义+结构输出
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D)
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm_semantic(self, x: torch.Tensor):
        """语义特征SSM处理，返回输出和最终隐藏状态"""
        B, C, H, W = x.shape
        L = H * W
        xs = x.view(B, -1, L)

        x_dbl = torch.matmul(self.x_proj_weight_semantic.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight_semantic.view(1, C, -1), dts)

        As = -torch.exp(self.A_logs_semantic)
        Ds = self.Ds_semantic
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias_semantic


        # 保留最后一个隐藏状态h_last
        h, h_last = self.selective_scan(
            xs, dts,
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=True,  # 新增：返回最后状态
        )

        h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
        h = self.state_fusion(h)
        h = rearrange(h, "b d h w -> b d (h w)")

        y = h * Cs
        y = y + xs * Ds.view(-1, 1)
        return y, h_last  # 返回输出和最终隐藏状态

    def ssm_structural(self, s: torch.Tensor, init_hidden=None):
        """结构特征SSM处理，支持初始隐藏状态输入"""
        B, C, H, W = s.shape
        L = H * W
        ss = s.view(B, -1, L)

        x_dbl = torch.matmul(self.x_proj_weight_structural.view(1, -1, C), ss)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight_structural.view(1, C, -1), dts)

        As = -torch.exp(self.A_logs_structural)
        Ds = self.Ds_structural
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias_structural

        # 使用语义SSM的最终状态作为初始隐藏状态
        h = self.selective_scan(
            ss, dts,
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
            # init_hidden=init_hidden
        )

        h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
        # h = rearrange(h, "b d (h w) -> b d h w", h=H, w=W)
        h = self.state_fusion(h)
        h = rearrange(h, "b d h w -> b d (h w)")

        y = h * Cs
        y = y + ss * Ds.view(-1, 1)
        return y

    def forward(self, x: torch.Tensor, s: torch.Tensor, **kwargs):
        """
        x: 语义特征 (B, H, W, C)
        s: 结构特征 (B, H, W, C)
        """
        B, H, W, C = x.shape

        # 语义特征处理
        xz = self.in_proj_semantic(x)
        x, z = xz.chunk(2, dim=-1)
        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.act(self.conv2d_semantic(x))
        y_semantic, h_last = self.ssm_semantic(x)  # 获取语义SSM输出和最终状态

        # 结构特征处理（使用语义SSM的最终状态作为初始状态）
        sz = self.in_proj_structural(s)
        s, z_struct = sz.chunk(2, dim=-1)
        s = rearrange(s, 'b h w d -> b d h w').contiguous()
        s = self.act(self.conv2d_structural(s))
        y_structural = self.ssm_structural(s, init_hidden=h_last)  # 传入初始隐藏状态

        # 融合两层输出
        y_semantic = rearrange(y_semantic, 'b d (h w)-> b h w d', h=H, w=W)
        y_structural = rearrange(y_structural, 'b d (h w)-> b h w d', h=H, w=W)


        y_semantic = self.out_norm_semantic(y_semantic)
        y_structural = self.out_norm_structural(y_structural)
        y_semantic = y_semantic * F.silu(z)
        y_structural = y_structural * F.silu(z)
        y_fused = y_structural + y_semantic
        y_fused = self.out_proj(y_fused)

        if self.dropout is not None:
            y_fused = self.dropout(y_fused)
        return y_fused

class StructureAwareSSM_1(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            ** kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 共享输入投影层（语义+结构特征）
        self.in_proj_semantic = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_structural = nn.Linear(self.d_model, self.d_inner * 2, bias=bias,** factory_kwargs)

        # 语义特征专用卷积层
        self.conv2d_semantic = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # 结构特征专用卷积层
        self.conv2d_structural = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,** factory_kwargs,
        )

        self.act = nn.SiLU()

        # --------------------------
        # 核心修改：共享C权重的投影层
        # --------------------------
        # 语义SSM的x_proj：仅生成dt和B（不再生成C）
        self.x_proj_semantic = nn.Linear(self.d_inner, (self.dt_rank + self.d_state), bias=False, **factory_kwargs)
        self.x_proj_weight_semantic = nn.Parameter(self.x_proj_semantic.weight)
        del self.x_proj_semantic

        # 结构SSM的x_proj：仅生成dt和B（不再生成C）
        self.x_proj_structural = nn.Linear(self.d_inner, (self.dt_rank + self.d_state), bias=False,** factory_kwargs)
        self.x_proj_weight_structural = nn.Parameter(self.x_proj_structural.weight)
        del self.x_proj_structural

        # 共享的C投影层：为语义和结构SSM提供C权重
        self.C_proj_shared = nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs)
        self.C_proj_weight_shared = nn.Parameter(self.C_proj_shared.weight)
        del self.C_proj_shared

        # 语义特征SSM其他参数（保持独立）
        self.dt_projs_semantic = self.dt_init(
            self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,** factory_kwargs
        )
        self.dt_projs_weight_semantic = nn.Parameter(self.dt_projs_semantic.weight)
        self.dt_projs_bias_semantic = nn.Parameter(self.dt_projs_semantic.bias)
        del self.dt_projs_semantic

        self.A_logs_semantic = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds_semantic = self.D_init(self.d_inner, dt_init)

        # 结构特征SSM其他参数（保持独立）
        self.dt_projs_structural = self.dt_init(
            self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs
        )
        self.dt_projs_weight_structural = nn.Parameter(self.dt_projs_structural.weight)
        self.dt_projs_bias_structural = nn.Parameter(self.dt_projs_structural.bias)
        del self.dt_projs_structural

        self.A_logs_structural = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds_structural = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn
        self.state_fusion = StateFusion(self.d_inner)

        # 输出层
        self.out_norm = nn.LayerNorm(self.d_inner*2)

        self.out_proj = nn.Linear(self.d_inner*2, self.d_model, bias=bias,** factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    # 以下dt_init、A_log_init、D_init方法与原代码一致，省略...
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                bias=True, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init == "random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init == "simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init == "zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init == "random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D)
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm_semantic(self, x: torch.Tensor):
        """语义特征SSM处理，使用共享C权重"""
        B, C, H, W = x.shape
        L = H * W
        xs = x.view(B, -1, L)

        # 1. 生成dt和B（语义SSM专用）
        x_dbl = torch.matmul(self.x_proj_weight_semantic.view(1, -1, C), xs)  # 维度：(1, dt_rank + d_state, L)
        dts, Bs = torch.split(x_dbl, [self.dt_rank, self.d_state], dim=1)  # 拆分dt和B
        dts = torch.matmul(self.dt_projs_weight_semantic.view(1, C, -1), dts)

        # 2. 生成共享C权重（语义和结构SSM共用）
        Cs = torch.matmul(self.C_proj_weight_shared.view(1, -1, C), xs)  # 维度：(1, d_state, L)

        As = -torch.exp(self.A_logs_semantic)
        Ds = self.Ds_semantic
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias_semantic

        # 保留最后一个隐藏状态h_last
        h, h_last = self.selective_scan(
            xs, dts,
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=True,
        )

        h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
        h = self.state_fusion(h)
        h = rearrange(h, "b d h w -> b d (h w)")

        y = h * Cs  # 使用共享C
        y = y + xs * Ds.view(-1, 1)
        return y, h_last

    def ssm_structural(self, s: torch.Tensor, init_hidden=None):
        """结构特征SSM处理，使用共享C权重"""
        B, C, H, W = s.shape
        L = H * W
        ss = s.view(B, -1, L)

        # 1. 生成dt和B（结构SSM专用）
        x_dbl = torch.matmul(self.x_proj_weight_structural.view(1, -1, C), ss)  # 维度：(1, dt_rank + d_state, L)
        dts, Bs = torch.split(x_dbl, [self.dt_rank, self.d_state], dim=1)  # 拆分dt和B
        dts = torch.matmul(self.dt_projs_weight_structural.view(1, C, -1), dts)

        # 2. 生成共享C权重（与语义SSM共用同一投影层）
        Cs = torch.matmul(self.C_proj_weight_shared.view(1, -1, C), ss)  # 维度：(1, d_state, L)

        As = -torch.exp(self.A_logs_structural)
        Ds = self.Ds_structural
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias_structural

        # 使用语义SSM的最终状态作为初始隐藏状态
        h = self.selective_scan(
            ss, dts,
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
        h = self.state_fusion(h)
        h = rearrange(h, "b d h w -> b d (h w)")

        y = h * Cs  # 使用共享C
        y = y + ss * Ds.view(-1, 1)
        return y

    # forward方法与原代码一致（仅使用共享C，无需修改）
    def forward(self, x: torch.Tensor, s: torch.Tensor, **kwargs):
        """
        x: 语义特征 (B, H, W, C)
        s: 结构特征 (B, H, W, C)
        """
        B, H, W, C = x.shape

        # 语义特征处理
        xz = self.in_proj_semantic(x)
        x, z = xz.chunk(2, dim=-1)
        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.act(self.conv2d_semantic(x))
        y_semantic, h_last = self.ssm_semantic(x)  # 获取语义SSM输出和最终状态

        # 结构特征处理（使用语义SSM的最终状态作为初始状态）
        sz = self.in_proj_structural(s)
        s, z_struct = sz.chunk(2, dim=-1)
        s = rearrange(s, 'b h w d -> b d h w').contiguous()
        s = self.act(self.conv2d_structural(s))
        y_structural = self.ssm_structural(s, init_hidden=h_last)  # 传入初始隐藏状态

        # 融合两层输出
        y_semantic = rearrange(y_semantic, 'b d (h w)-> b h w d', h=H, w=W)
        y_structural = rearrange(y_structural, 'b d (h w)-> b h w d', h=H, w=W)

        # 特征融合（拼接后投影）
        y_fused = torch.cat([y_semantic, y_structural], dim=-1)
        z_fused = torch.cat([z, z_struct], dim=-1)
        y_fused = self.out_norm(y_fused)
        y_fused = y_fused * F.silu(z_fused)  # 使用语义分支的门控
        y_fused = self.out_proj(y_fused)
        if self.dropout is not None:
            y_fused = self.dropout(y_fused)
        return y_fused
class SpatialMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 1,
        dt_init: str = "random",
        **kwargs,
    ):
        super().__init__()

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = StructureAwareSSM_1(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, dt_init=dt_init, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.y_down = Downsampler(in_channels=64, out_channels=hidden_dim)

        self.ln_2 = norm_layer(hidden_dim)
    def forward(self, x: torch.Tensor, s: torch.Tensor):
        shortcut = x
        semantic_x = self.ln_1(x.permute(0, 2, 3, 1).contiguous())
        structural_x = self.y_down(s)
        structural_x = self.ln_2(structural_x.permute(0, 2, 3, 1).contiguous())

        y = self.self_attention(semantic_x, structural_x)

        return shortcut + self.drop_path(y.permute(0, 3, 1, 2))


class Downsampler(nn.Module):
    def __init__(self, in_channels=64, out_channels=256, kernel_size=3):
        super().__init__()
        # 第一次卷积：1→128通道，128×128→64×64
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels // 2,  # 中间通道：256//2=128
                kernel_size=kernel_size,
                stride=2,  # 步长=2，缩小1/2
                padding=1,  # 确保尺寸缩小1/2
                bias=False  # 后续有BN，可省略bias
            ),
            nn.BatchNorm2d(out_channels // 2),  # 稳定特征分布
            nn.GELU()  # 非线性激活，增强表达能力
        )

        # 第二次卷积：128→256通道，64×64→32×32
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels // 2,
                out_channels=out_channels,  # 目标通道：256
                kernel_size=kernel_size,
                stride=2,  # 步长=2，再缩小1/2
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x: 输入特征，如[B,1,128,128]
        x = self.conv1(x)  # 输出：[B,128,64,64]
        x = self.conv2(x)  # 输出：[B,256,32,32]
        return x



