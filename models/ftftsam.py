import matplotlib.pyplot as plt
import torchvision
import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.sammodel import ImageEncoderViT, PromptEncoder, MaskDecoder, TwoWayTransformer
from models.encoder_Adapter import ImageEncoderViTAdapter
from models.ppm import PrototypePromptEncoder, GlobalPrototypes, CSM, PromptGenerator
from models.ppm import HierMaskDecoder
from models.ppm import TwoWayTransformer as PPMTwoWayTransformer
from models.ppm import ConvLayer2d, UpSample2d
from .GroupCBAMEnhancer import GroupCBAMEnhancer, MLFusion

logger = logging.getLogger(__name__)
from model.detr import SetCriterion  # DETR相关模块
from model.matcher import HungarianMatcher  # DETR的匈牙利匹配器
from model.detr_Decoder import DETRDecoder
from .iou_loss import BoundaryDoULoss, IOU
from typing import Any, Optional, Tuple



# boundary loss
class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''

    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss


def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

def prototype_alignment_loss(
    intra_prototypes, image_embed, masks,
    lambda_g=0.4, lambda_l=0.4, lambda_o=0.2
):
    """
    Args:
        intra_prototypes: [B, P, C]  # 子原型集合
        image_embed: [B, C, H, W]    # 图像特征
        masks: [B, 1, H, W]          # 掩码
    Returns:
        L_proto: scalar
    """

    B, P, C = intra_prototypes.shape
    _, _, H, W = image_embed.shape

    # === Step 1: 掩码区域特征均值 ===
    masked_feat = image_embed * masks
    feat_mean = masked_feat.sum(dim=(2, 3)) / (masks.sum(dim=(2, 3)) + 1e-6)  # [B, C]

    # === Step 2: 原型均值 ===
    proto_mean = intra_prototypes.mean(dim=1)  # [B, C]

    # === Step 3: 全局一致性损失 ===
    global_sim = F.cosine_similarity(proto_mean, feat_mean, dim=-1)  # [B]
    L_global = 1 - global_sim.mean()

    # === Step 4: 局部覆盖损失 ===
    feat_flat = F.normalize(image_embed.flatten(2).permute(0, 2, 1), dim=-1)  # [B, HW, C]
    mask_flat = masks.flatten(2).permute(0, 2, 1)                             # [B, HW, 1]
    feat_flat = feat_flat * (mask_flat > 0.5).float()

    proto_norm = F.normalize(intra_prototypes, dim=-1)  # [B, P, C]
    sim = torch.einsum('bpc,bnc->bpn', proto_norm, feat_flat)  # [B, P, HW]
    max_sim = sim.max(dim=-1)[0]                            # [B, P]
    L_local = 1 - max_sim.mean()

    # === Step 5: 正交约束 ===
    # 防止梯度爆炸：detach() 一般不加，但可以用 normalize 防稳定
    P_norm = F.normalize(intra_prototypes, dim=-1)
    ortho_loss = []
    for b in range(B):
        proto_corr = P_norm[b] @ P_norm[b].T       # [P, P]
        I = torch.eye(P, device=proto_corr.device)
        ortho_loss.append(((proto_corr - I) ** 2).sum())
    L_ortho = torch.stack(ortho_loss).mean() / (P * P)

    # === Step 6: 融合 ===
    L_proto = lambda_g * L_global + lambda_l * L_local + lambda_o * L_ortho
    return L_proto



class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.

    removed forward_with_coords which is 这个方法可以用于在对图像做处理时，对非归一化的点坐标进行位置编码，以便在后续的模型中使用
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

@register('ftftsam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None, bd_weight=0.3, feat_dim=256, num_protos=5,
                 num_classes=1, num_multimask_output=3, num_heads=8):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViTAdapter(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
            interaction_indexes=encoder_mode['interaction_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])


        # self.mask_decoder = MaskDecoder(
        #     num_multimask_outputs=3,
        #     transformer=TwoWayTransformer(
        #         depth=2,
        #         embedding_dim=self.prompt_embed_dim,
        #         mlp_dim=2048,
        #         num_heads=8,
        #     ),
        #     transformer_dim=self.prompt_embed_dim,
        #     iou_head_depth=3,
        #     iou_head_hidden_dim=256,
        # )
        self.mask_embed = nn.Embedding(1, feat_dim)
        self.csm = CSM(feat_dim)

        self.global_prototypes = GlobalPrototypes(
            feat_dim=feat_dim,
            num_protos=num_protos,
        )
        # self.prompt_generator = PromptGenerator(
        #     feat_dim=feat_dim,
        #     num_protos=num_protos,
        # )

        # 初始化PrototypePromptEncoder
        self.prototype_prompt_encoder = PrototypePromptEncoder(
            feat_dim=feat_dim,
            num_protos=num_protos,
            num_classes=num_classes,
            num_heads=num_heads,
        )
        self.mask_decoder = HierMaskDecoder(
            transformer=PPMTwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            embed_dim=self.prompt_embed_dim,
            num_multimask_outputs=num_multimask_output,
        )
        # self.Hiermask_decoder = HierMaskDecoder1(
        #     transformer=PPMTwoWayTransformer(
        #         depth=2,
        #         embedding_dim=self.prompt_embed_dim,
        #         mlp_dim=2048,
        #         num_heads=8,
        #     ),
        #     embed_dim=self.prompt_embed_dim,
        #     num_multimask_outputs=num_multimask_output,
        # )

        # self.boundary_proj = nn.Conv2d(64, 1, kernel_size=1)
        self.edge_pred_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)  # 输出logits，用于BCE损失
        )


        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()
            self.bd_weight = bd_weight
            self.criterionDOU = BoundaryDoULoss()

    def set_input(self, input, gt_mask, bd_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)
        self.bd_mask = bd_mask.to(self.device)


    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    # def get_dense_pe(self) -> torch.Tensor:
    #     """
    #     返回 positional encoding，形状为 [B, C, H, W]，与 image_embeddings 对齐
    #     """
    #     pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)  # [1, C, H, W]
    #     B = self.input.shape[0]
    #     return pe.expand(B, -1, -1, -1)  # [B, C, H, W]

    def forward(self):
        bs = 4

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        # self.features, boundary_features = self.image_encoder(self.input)
        self.features, self.pred_bd = self.image_encoder(self.input)

        intra_prototypes, intra_embed = self.global_prototypes()
        B = self.features.shape[0]
        # 扩展批量维度：[8,256] → [B,8,256]
        intra_prototypes = intra_prototypes.unsqueeze(0).expand(B, -1, -1)
        intra_embed = intra_embed.unsqueeze(0).expand(B, -1, -1)

        low_res_masks, mask_embed = self.mask_decoder(
            image_embeddings=self.features,
            dense_prompt_embeddings=dense_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            up_embeds=self.pred_bd,
            mask_embeds=None,
            ps_masks=None,
            multimask_output=False,
        )

        # low_res_masks, iou_predictions = self.mask_decoder(
        #     image_embeddings=self.features,
        #     image_pe=self.get_dense_pe(),
        #     sparse_prompt_embeddings=sparse_embeddings,
        #     dense_prompt_embeddings=dense_embeddings,
        #     multimask_output=False,
        # )
        out_embed = self.features + self.mask_embed.weight.reshape(1, -1, 1, 1)
        out_embed = self.csm(out_embed)
        ps_masks = F.interpolate(low_res_masks, (self.features.shape[2], self.features.shape[3]),
                                 mode="bilinear", align_corners=False)
        # dense_prompts, sparse_prompts = self.prompt_generator(
        #     out_embed=out_embed,
        #     intra_prototypes=intra_prototypes,
        #     intra_embed=intra_embed,
        #     masks=ps_masks
        # )
        # low_res_masks, mask_embed = self.Hiermask_decoder(
        #     image_embeddings=out_embed,  # [b, c, h, w]
        #     dense_prompt_embeddings=dense_prompts,  # [b, c, h, w]
        #     sparse_prompt_embeddings=sparse_prompts,  # [b, q, c]
        #     up_embeds=None,
        #     mask_embeds=None,
        #     ps_masks=ps_masks,
        #     multimask_output=False,
        # )
        out_embed, mask_embed, intra_prototypes, dense_prompts, sparse_prompts = self.prototype_prompt_encoder(
            out_embed=out_embed,
            mask_embed=mask_embed,
            intra_prototypes=intra_prototypes,
            intra_embed=intra_embed,
            masks=ps_masks
        )
        low_res_masks, mask_embed = self.mask_decoder(
            image_embeddings=out_embed,  # [b, c, h, w]
            dense_prompt_embeddings=dense_prompts,  # [b, c, h, w]
            sparse_prompt_embeddings=sparse_prompts,  # [b, q, c]
            up_embeds=self.pred_bd,
            mask_embeds=mask_embed,
            ps_masks=ps_masks,
            multimask_output=False,
        )
        self.pred_proto = intra_prototypes
        self.pred_embed = out_embed
        #



        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks
        # boundary = self.boundary_proj(self.pred_bd)
        boundary = self.edge_pred_head(self.pred_bd)
        
        self.boundary = self.postprocess_edge(boundary, self.inp_size, self.inp_size)

    def infer(self, input):
        bs = 4

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        # self.features, boundary_features = self.image_encoder(input)
        self.features, self.pred_bd = self.image_encoder(input)

        intra_prototypes, intra_embed = self.global_prototypes()
        B = self.features.shape[0]
        # 扩展批量维度：[8,256] → [B,8,256]
        intra_prototypes = intra_prototypes.unsqueeze(0).expand(B, -1, -1)
        intra_embed = intra_embed.unsqueeze(0).expand(B, -1, -1)



        low_res_masks, mask_embed = self.mask_decoder(
            image_embeddings=self.features,
            dense_prompt_embeddings=dense_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            up_embeds=self.pred_bd,
            mask_embeds=None,
            ps_masks=None,
            multimask_output=False,
        )

        # low_res_masks, iou_predictions = self.mask_decoder(
        #     image_embeddings=self.features,
        #     image_pe=self.get_dense_pe(),
        #     sparse_prompt_embeddings=sparse_embeddings,
        #     dense_prompt_embeddings=dense_embeddings,
        #     multimask_output=False,
        # )
        out_embed = self.features + self.mask_embed.weight.reshape(1, -1, 1, 1)
        out_embed = self.csm(out_embed)
        ps_masks = F.interpolate(low_res_masks, (self.features.shape[2], self.features.shape[3]),
                                 mode="bilinear", align_corners=False)
        # dense_prompts, sparse_prompts = self.prompt_generator(
        #     out_embed=out_embed,
        #     intra_prototypes=intra_prototypes,
        #     intra_embed=intra_embed,
        #     masks=ps_masks
        # )
        # low_res_masks, mask_embed = self.mask_decoder(
        #     image_embeddings=out_embed,  # [b, c, h, w]
        #     dense_prompt_embeddings=dense_prompts,  # [b, c, h, w]
        #     sparse_prompt_embeddings=sparse_prompts,  # [b, q, c]
        #     up_embeds=None,
        #     mask_embeds=None,
        #     ps_masks=ps_masks,
        #     multimask_output=False,
        # )
        out_embed, mask_embed, intra_prototypes, dense_prompts, sparse_prompts = self.prototype_prompt_encoder(
            out_embed=out_embed,
            mask_embed=mask_embed,
            intra_prototypes=intra_prototypes,
            intra_embed=intra_embed,
            masks=ps_masks
        )
        low_res_masks, mask_embed = self.mask_decoder(
            image_embeddings=out_embed,  # [b, c, h, w]
            dense_prompt_embeddings=dense_prompts,  # [b, c, h, w]
            sparse_prompt_embeddings=sparse_prompts,  # [b, q, c]
            up_embeds=self.pred_bd,
            mask_embeds=mask_embed,
            ps_masks=ps_masks,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        # boundary = self.boundary_proj(self.pred_bd)
        boundary = self.edge_pred_head(self.pred_bd)
        self.boundary = self.postprocess_edge(boundary, self.inp_size, self.inp_size)
        return masks

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def postprocess_edge(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)
            self.loss_G += self.bd_weight * self.criterionDOU(self.boundary, self.bd_mask)
            # self.mask = F.interpolate(self.gt_mask, (self.features.shape[2], self.features.shape[3]),
            #                          mode="bilinear", align_corners=False)
            # L_proto = prototype_alignment_loss(
            #     intra_prototypes=self.pred_proto,  # [B,P,C]
            #     image_embed=self.pred_embed,  # [B,C,H,W]
            #     masks=self.mask  # [B,1,H,W]
            # )
            # self.loss_G += L_proto  # λ_proto，可调
            # self.L_proto = L_proto.item()
        

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
