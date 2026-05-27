import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt
# from models.encoder_Adapter.ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
from functools import partial
from .moudle import (InteractionBlock, EdgeStem,
                              deform_inputs)
from .SSCM import HierarchicalVSSBlock
from models.sammodel import ImageEncoderViT
from models.spatialmamba import SpatialMambaBlock

_logger = logging.getLogger(__name__)



class ImageEncoderViTAdapter(ImageEncoderViT):
    def __init__(self, *args, interaction_indexes=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), pretrain_size=512, drop_rate=0., **kwargs):
        super().__init__(*args, **kwargs)

        dim = self.embed_dim
        self.level_embed = nn.Parameter(torch.zeros(3, dim))
        self.interaction_indexes = interaction_indexes
        self.norm_layer = norm_layer

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.stem = EdgeStem(in_channels=3, out_channels=64)


        self.interactions = nn.Sequential(*[
            InteractionBlock(
                dim=dim,
                refine_channels=dim,  # RefineLayer的refine_channels参数
                norm_layer=self.norm_layer,
                extra_extractor=(i == len(self.interaction_indexes)-1),
                in_channels_m=64,
                mid_channels=64,
                need_fusion=False #(i >= 2)
            )
            for i in range(len(self.interaction_indexes))
        ])
        self.fusion = SpatialMambaBlock(hidden_dim=256)

        self.interactions.apply(self._init_weights)
        normal_(self.level_embed)
        


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                

    def forward(self, x):
        # deform inputs for attention
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        c = x
        c = self.stem(c)

        # Patch embedding
        x = self.patch_embed(x)
       
        B, H, W, _= x.shape

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # Interactions
        for i, block in enumerate(self.interactions):
            idx = self.interaction_indexes[i]
            x, c = block(x, c, self.blocks[idx[0]:idx[-1]+1],
                        H, W, i)
            
        x = self.neck(x.permute(0, 3, 1, 2))
        x = self.fusion(x, c)

        return x, c
