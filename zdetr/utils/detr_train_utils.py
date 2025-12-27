import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from zdetr.utils.detr_common_utils import box_iou
from zdetr.utils.bbox_loss import GIoULoss


class HungarianMatcher(nn.Module):
    """Hungarian matching for DETR"""

    def __init__(self, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_bboxes, target_bboxes, target_mask):
        B, N = pred_bboxes.shape[:2]
        indices = []

        for i in range(B):
            valid_mask = target_mask[i] > 0.5
            num_valid = valid_mask.sum().item()

            if num_valid == 0:
                indices.append(([], []))
                continue

            pred = pred_bboxes[i]
            tgt = target_bboxes[i][valid_mask]

            try:
                cost_bbox = torch.cdist(pred, tgt, p=1)
                cost_giou = -box_iou(pred, tgt)
                C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
                pred_idx, tgt_idx = linear_sum_assignment(C.cpu().numpy())

                if len(pred_idx) > 0:
                    assert max(pred_idx) < N and max(tgt_idx) < num_valid

                indices.append((pred_idx.tolist(), tgt_idx.tolist()))
            except Exception as e:
                print(f"⚠️ Hungarian error: {e}")
                indices.append(([], []))

        return indices


class DETRCriterion(nn.Module):
    """DETR loss with auxiliary losses"""

    def __init__(
        self,
        cls_criterion,
        lambda_bbox=5.0,
        lambda_giou=2.0,
        lambda_obj=1.0,
        aux_loss_weight=0.5,
    ):
        super().__init__()
        self.cls_criterion = cls_criterion
        self.matcher = HungarianMatcher(cost_bbox=lambda_bbox, cost_giou=lambda_giou)
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou
        self.lambda_obj = lambda_obj
        self.aux_loss_weight = aux_loss_weight
        self.bbox_l1_loss = nn.L1Loss(reduction="none")
        self.giou_loss = GIoULoss()

    def forward(self, outputs, targets):
        cls_loss = self.cls_criterion(outputs["cls_logits"], targets["label"])

        indices = self.matcher(
            outputs["pred_bboxes"], targets["bboxes"], targets["bbox_mask"]
        )

        bbox_loss, giou_loss, obj_loss = self._compute_detection_losses(
            outputs["pred_bboxes"],
            outputs["obj_scores"],
            targets["bboxes"],
            targets["bbox_mask"],
            indices,
        )

        total_loss = (
            cls_loss
            + self.lambda_bbox * bbox_loss
            + self.lambda_giou * giou_loss
            + self.lambda_obj * obj_loss
        )

        aux_loss = torch.tensor(0.0, device=total_loss.device)
        if "aux_outputs" in outputs and len(outputs["aux_outputs"]) > 0:
            for aux_out in outputs["aux_outputs"]:
                aux_bbox, aux_giou, aux_obj = self._compute_detection_losses(
                    aux_out["pred_bboxes"],
                    aux_out["obj_scores"],
                    targets["bboxes"],
                    targets["bbox_mask"],
                    indices,
                )
                aux_loss += (
                    self.lambda_bbox * aux_bbox
                    + self.lambda_giou * aux_giou
                    + self.lambda_obj * aux_obj
                )

            aux_loss = aux_loss * self.aux_loss_weight / len(outputs["aux_outputs"])
            total_loss += aux_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "bbox_loss": bbox_loss,
            "giou_loss": giou_loss,
            "obj_loss": obj_loss,
            "aux_loss": aux_loss,
        }

    def _compute_detection_losses(
        self, pred_bboxes, pred_obj, target_bboxes, target_mask, indices
    ):
        B, N = pred_bboxes.shape[:2]
        device = pred_bboxes.device

        bbox_loss = torch.tensor(0.0, device=device)
        giou_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        num_boxes = 0

        for i, (pred_idx, tgt_idx) in enumerate(indices):
            obj_target = torch.zeros(N, 1, device=device)

            if len(pred_idx) == 0:
                obj_loss += nn.functional.binary_cross_entropy_with_logits(
                    pred_obj[i], obj_target, reduction="sum"
                )
                continue

            valid_mask = target_mask[i] > 0.5
            if valid_mask.sum() == 0:
                obj_loss += nn.functional.binary_cross_entropy_with_logits(
                    pred_obj[i], obj_target, reduction="sum"
                )
                continue

            valid_targets = target_bboxes[i][valid_mask]

            try:
                matched_pred = pred_bboxes[i, pred_idx]
                matched_tgt = valid_targets[tgt_idx]

                if (matched_pred < 0).any() or (matched_pred > 1).any():
                    raise ValueError("Invalid pred bbox")
                if (matched_tgt < 0).any() or (matched_tgt > 1).any():
                    raise ValueError("Invalid target bbox")

                bbox_loss += self.bbox_l1_loss(matched_pred, matched_tgt).sum()
                giou_loss += self.giou_loss(matched_pred, matched_tgt)
                obj_target[pred_idx] = 1.0
                num_boxes += len(pred_idx)
            except Exception as e:
                print(f"⚠️ Loss error: {e}")

            obj_loss += nn.functional.binary_cross_entropy_with_logits(
                pred_obj[i], obj_target, reduction="sum"
            )

        if num_boxes > 0:
            bbox_loss = bbox_loss / num_boxes
            giou_loss = giou_loss / num_boxes
        obj_loss = obj_loss / (B * N)

        return bbox_loss, giou_loss, obj_loss
