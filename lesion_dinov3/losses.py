import torch
import torch.nn.functional as F


def weighted_bce_with_logits(logits, target, pos_weight=20.0):
    weights = torch.ones_like(target)
    weights = weights + target * (pos_weight - 1.0)
    return F.binary_cross_entropy_with_logits(logits, target, weight=weights)


def focal_bce_with_logits(logits, target, gamma=2.0, pos_weight=20.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    focal = (1.0 - pt).pow(gamma)
    weights = torch.ones_like(target) + target * (pos_weight - 1.0)
    return (focal * bce * weights).mean()
