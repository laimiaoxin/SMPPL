import torch
import torch.nn as nn
import torch.nn.functional as F

class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)
class BoundaryDoULoss(nn.Module):
    def __init__(self):
        super(BoundaryDoULoss, self).__init__()

    def _adaptive_size(self, score, target):
        # 确保输入在同一设备上
        device = score.device
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).to(device)

        # 批量边界检测（避免循环）
        Y = F.conv2d(target.unsqueeze(1).float(),
                     kernel.unsqueeze(0).unsqueeze(0),
                     padding=1).squeeze(1)
        Y = Y * target
        Y[Y == 5] = 0

        # 计算边界像素比例
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1
        alpha = min(alpha, 0.8)  # 截断上限

        # 计算改进的Dice损失
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        # 假设inputs是模型输出的logits，先进行sigmoid而非softmax
        inputs = torch.sigmoid(inputs)  # [4, 1, 512, 512]

        # 确保target是二进制掩码（0或1）
        target = target.float()

        # 检查形状匹配
        assert inputs.size() == target.size(), f'predict {inputs.size()} & target {target.size()} shape do not match'

        # 计算损失（直接处理单通道）
        loss = self._adaptive_size(inputs.squeeze(1), target.squeeze(1))
        return loss
