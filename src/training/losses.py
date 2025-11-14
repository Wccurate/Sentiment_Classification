from __future__ import annotations

import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_one_hot = F.one_hot(target, num_classes=logits.size(-1)).float()
        focal = (1 - probs) ** self.gamma
        loss = -(focal * target_one_hot * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

