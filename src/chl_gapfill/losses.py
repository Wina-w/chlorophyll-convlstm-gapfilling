"""Loss functions for chlorophyll cloud gap-filling.

The core objective is a **masked** reconstruction loss: error is measured only on
the pixels that were hidden (the synthetic cloud "hole"), because those are the
pixels the model actually has to infer. Scoring the whole image would be
dominated by pixels the model can copy straight from the input.

All tensors are (B, 1, H, W); masks are 1 on hole pixels, 0 elsewhere.
"""

import torch
from torch import nn


def masked_mse(pred, target, hole_mask, eps=1e-8):
    """Mean squared error over hole pixels only."""
    diff2 = (pred - target) ** 2 * hole_mask
    return diff2.sum() / (hole_mask.sum() + eps)


def masked_l1(pred, target, hole_mask, eps=1e-8):
    diff = (pred - target).abs() * hole_mask
    return diff.sum() / (hole_mask.sum() + eps)


def gradient_loss(pred, target, hole_mask=None, eps=1e-8):
    """Encourage matching spatial gradients (sharper, less blurry fills).

    Computed on finite-difference gradients; if a hole mask is given, only
    gradient elements touching a hole pixel are counted.
    """

    def grads(x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    pgx, pgy = grads(pred)
    tgx, tgy = grads(target)
    if hole_mask is not None:
        mx = torch.maximum(hole_mask[:, :, :, 1:], hole_mask[:, :, :, :-1])
        my = torch.maximum(hole_mask[:, :, 1:, :], hole_mask[:, :, :-1, :])
        lx = ((pgx - tgx).abs() * mx).sum() / (mx.sum() + eps)
        ly = ((pgy - tgy).abs() * my).sum() / (my.sum() + eps)
    else:
        lx = (pgx - tgx).abs().mean()
        ly = (pgy - tgy).abs().mean()
    return lx + ly


class GapFillLoss(nn.Module):
    """Masked reconstruction loss with a light valid-region + gradient term.

    total = w_hole * MSE(hole) + w_valid * MSE(observed) + w_grad * grad(hole)

    The small valid-region term keeps the model consistent with pixels it can
    see (a standard inpainting regulariser); the hole term is what we report.
    """

    def __init__(self, w_hole=1.0, w_valid=0.1, w_grad=0.05):
        super().__init__()
        self.w_hole, self.w_valid, self.w_grad = w_hole, w_valid, w_grad

    def forward(self, pred, target, hole_mask, valid_mask=None):
        loss = self.w_hole * masked_mse(pred, target, hole_mask)
        if self.w_valid and valid_mask is not None:
            loss = loss + self.w_valid * masked_mse(pred, target, valid_mask)
        if self.w_grad:
            loss = loss + self.w_grad * gradient_loss(pred, target, hole_mask)
        return loss
