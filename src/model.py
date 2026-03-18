"""
Shared model components and training utilities.

Imported by pretrain_unet.py, finetune_trainer.py, and visualize_unet.py
to avoid code duplication.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """4-level encoder-decoder U-Net for binary segmentation.

    Input:  (B, in_channels, H, W)
    Output: (B, 1, H, W)  — raw logits, apply sigmoid for probabilities.
    """
    def __init__(self, in_channels=8, base_c=64):
        super().__init__()
        self.d1 = DoubleConv(in_channels, base_c)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base_c, base_c * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base_c * 2, base_c * 4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base_c * 4, base_c * 8)
        self.p4 = nn.MaxPool2d(2)

        self.b = DoubleConv(base_c * 8, base_c * 16)

        self.u4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, 2, stride=2)
        self.c4 = DoubleConv(base_c * 16, base_c * 8)
        self.u3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, 2, stride=2)
        self.c3 = DoubleConv(base_c * 8, base_c * 4)
        self.u2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, stride=2)
        self.c2 = DoubleConv(base_c * 4, base_c * 2)
        self.u1 = nn.ConvTranspose2d(base_c * 2, base_c, 2, stride=2)
        self.c1 = DoubleConv(base_c * 2, base_c)

        self.out = nn.Conv2d(base_c, 1, 1)

    def forward(self, x):
        d1 = self.d1(x);  p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        d4 = self.d4(p3); p4 = self.p4(d4)
        b  = self.b(p4)
        u4 = self.u4(b);  c4 = self.c4(torch.cat([u4, d4], dim=1))
        u3 = self.u3(c4); c3 = self.c3(torch.cat([u3, d3], dim=1))
        u2 = self.u2(c3); c2 = self.c2(torch.cat([u2, d2], dim=1))
        u1 = self.u1(c2); c1 = self.c1(torch.cat([u1, d1], dim=1))
        return self.out(c1)


def bce_dice_loss(logits, targets, smooth=1.0):
    """BCE + Dice loss (50/50) for binary segmentation.

    logits:  (B, 1, H, W) — raw model output
    targets: (B, 1, H, W) — float in {0, 1}
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    num = (2.0 * (probs * targets).sum(dim=(2, 3)) + smooth)
    den = (probs.pow(2).sum(dim=(2, 3)) + targets.pow(2).sum(dim=(2, 3)) + smooth)
    dice = 1.0 - (num / den)
    return 0.5 * bce + 0.5 * dice.mean()


@torch.no_grad()
def iou_f1_from_logits(logits, targets, thr=0.5, eps=1e-7):
    """Mean-per-image IoU and F1 from raw logits. Returns (iou, f1) as Python floats."""
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))
    iou = (inter + eps) / (union + eps)
    tp = inter
    fp = (preds * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - preds) * targets).sum(dim=(2, 3))
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return iou.mean().item(), f1.mean().item()
