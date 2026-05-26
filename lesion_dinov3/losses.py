import torch
import torch.nn.functional as F
import torch.nn as nn


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

def tversky_loss(logits, target, alpha=0.5, beta=0.5, smooth=1e-6):
    prob = torch.sigmoid(logits)
    prob_flat = prob.view(-1)
    target_flat = target.view(-1)

    true_pos = (prob_flat * target_flat).sum()
    false_neg = ((1 - prob_flat) * target_flat).sum()
    false_pos = (prob_flat * (1 - target_flat)).sum()

    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return 1 - tversky_index
