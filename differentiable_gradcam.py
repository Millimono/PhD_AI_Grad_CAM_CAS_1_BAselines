import torch
import torch.nn.functional as F
import torch.nn as nn

class DifferentiableGradCAM(nn.Module):
    def forward(self, feature_maps, gradients):
        # gradients: (B, C, H, W)
        # feature_maps: (B, C, H, W)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        return cam
