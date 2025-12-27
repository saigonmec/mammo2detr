import torch
import torch.nn as nn


class FocalBBoxLoss(nn.Module):
    """Focal loss for bounding box regression - focus on hard examples"""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred, target):
        # Compute smooth L1 loss
        loss = self.smooth_l1(pred, target)

        # Compute focal weight
        pt = torch.exp(-loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Apply focal weight
        loss = focal_weight * loss
        return loss.mean()


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss - better than L1 for bbox
    Returns values in range [0, 2] where 0 is perfect overlap
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: [N, 4] predicted bbox in format [x1, y1, x2, y2] normalized [0,1]
            target: [N, 4] target bbox in format [x1, y1, x2, y2] normalized [0,1]
        Returns:
            loss: scalar tensor, GIoU loss value (non-negative)
        """
        # Clamp predictions to valid range [0, 1]
        pred = torch.clamp(pred, 0.0, 1.0)

        # Ensure x2 > x1 and y2 > y1 for both pred and target
        pred_x1 = torch.min(pred[:, 0], pred[:, 2])
        pred_y1 = torch.min(pred[:, 1], pred[:, 3])
        pred_x2 = torch.max(pred[:, 0], pred[:, 2])
        pred_y2 = torch.max(pred[:, 1], pred[:, 3])

        target_x1 = torch.min(target[:, 0], target[:, 2])
        target_y1 = torch.min(target[:, 1], target[:, 3])
        target_x2 = torch.max(target[:, 0], target[:, 2])
        target_y2 = torch.max(target[:, 1], target[:, 3])

        # Compute intersection
        x1_i = torch.max(pred_x1, target_x1)
        y1_i = torch.max(pred_y1, target_y1)
        x2_i = torch.min(pred_x2, target_x2)
        y2_i = torch.min(pred_y2, target_y2)

        inter_w = torch.clamp(x2_i - x1_i, min=0)
        inter_h = torch.clamp(y2_i - y1_i, min=0)
        inter_area = inter_w * inter_h

        # Compute union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area + self.eps

        # Compute IoU
        iou = inter_area / union_area

        # Compute enclosing box
        x1_c = torch.min(pred_x1, target_x1)
        y1_c = torch.min(pred_y1, target_y1)
        x2_c = torch.max(pred_x2, target_x2)
        y2_c = torch.max(pred_y2, target_y2)

        c_area = (x2_c - x1_c) * (y2_c - y1_c) + self.eps

        # Compute GIoU
        giou = iou - (c_area - union_area) / c_area

        # Loss = 1 - GIoU, clamped to [0, 2]
        loss = 1 - giou
        loss = torch.clamp(loss, min=0.0, max=2.0)

        return loss.mean()


class SmoothL1BBoxLoss(nn.Module):
    """Smooth L1 Loss for bbox regression (always non-negative)"""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred: [N, 4] predicted bbox
            target: [N, 4] target bbox
        """
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff**2 / self.beta,
            diff - 0.5 * self.beta,
        )
        return loss.sum(dim=1).mean()


class HybridBBoxLoss(nn.Module):
    """
    Hybrid loss: GIoU + SmoothL1
    Combines geometry-aware GIoU with coordinate regression
    """

    def __init__(self, giou_weight=1.0, l1_weight=0.5):
        super().__init__()
        self.giou_loss = GIoULoss()
        self.l1_loss = SmoothL1BBoxLoss()
        self.giou_weight = giou_weight
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        giou = self.giou_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        total_loss = self.giou_weight * giou + self.l1_weight * l1
        return total_loss
