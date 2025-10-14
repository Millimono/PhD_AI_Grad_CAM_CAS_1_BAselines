import torch
import torch.nn.functional as F
import torch.nn as nn

class DifferentiableGradCAM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_maps, gradients):
        """
        Calcule la carte Grad-CAM différentiable.
        """
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # α_k = moyenne spatiale
        cam_pre_relu = (weights * feature_maps).sum(dim=1, keepdim=True)  # S_uv
        cam = F.relu(cam_pre_relu)  # CAM = ReLU(S)
        return cam, cam_pre_relu, weights

    def gradcam_loss(self, cam, target_mask):
        """
        L_cam = ||CAM - M||^2
        """
        return F.mse_loss(cam, target_mask)
    
    
# Example usage:
if __name__ == "__main__":
    # Simulated feature maps and gradients
    feature_maps = torch.randn(1, 512, 14, 14, requires_grad=True)  # (B, C, H, W)
    gradients = torch.randn(1, 512, 14, 14)  # (B, C, H, W)
    target_mask = torch.randn(1, 1, 14, 14)  # (B, 1, H, W)

    gradcam = DifferentiableGradCAM()
    cam, cam_pre_relu, weights = gradcam(feature_maps, gradients)
    loss = gradcam.gradcam_loss(cam, target_mask)

    print("CAM shape:", cam.shape)
    print("Loss:", loss.item())