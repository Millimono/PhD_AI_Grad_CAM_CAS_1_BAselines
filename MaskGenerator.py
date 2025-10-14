import torch

class MaskGenerator:
    def __init__(self, device='cuda'):
        self.device = device

    def generate(self, shape, mask_type="center", cam=None, latent_mask=None, **kwargs):
        """
        Génère un masque selon le type demandé.

        Args:
            shape : tuple (B, C, H, W)
            mask_type : str, "center", "circle", "border", "diffuse", "latent"
            cam : torch.Tensor, nécessaire pour le masque diffuse
            latent_mask : torch.Tensor, nécessaire pour le masque latent
            kwargs : paramètres supplémentaires (sigma, radius, etc.)
        """
        if mask_type == "center":
            return self.center_mask(shape, **kwargs)
        elif mask_type == "circle":
            return self.circle_mask(shape, **kwargs)
        elif mask_type == "border":
            return self.border_mask(shape, **kwargs)
        elif mask_type == "diffuse":
            if cam is None:
                raise ValueError("Pour 'diffuse', il faut passer la CAM avec cam=<tensor>")
            return self.diffuse_mask(cam, **kwargs)
        elif mask_type == "latent":
            if latent_mask is None:
                raise ValueError("Pour 'latent', il faut passer le masque latent avec latent_mask=<tensor>")
            return self.latent_mask(latent_mask, shape, **kwargs)
        else:
            raise ValueError(f"Masque inconnu: {mask_type}")

    # ---- Masques prédéfinis ----
    def center_mask(self, shape, sigma=0.5):
        B, C, H, W = shape
        y = torch.linspace(-1, 1, steps=H, device=self.device).view(H, 1).expand(H, W)
        x = torch.linspace(-1, 1, steps=W, device=self.device).view(1, W).expand(H, W)
        grid = torch.stack([x, y], dim=0)
        dist_squared = grid[0]**2 + grid[1]**2
        mask = torch.exp(-dist_squared / (2*sigma**2))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        return mask

    def circle_mask(self, shape, radius=0.5):
        B, C, H, W = shape
        cy, cx = H/2, W/2
        y = torch.arange(H, device=self.device).view(H,1).expand(H,W)
        x = torch.arange(W, device=self.device).view(1,W).expand(H,W)
        mask = ((x - cx)**2 + (y - cy)**2 <= (radius * min(H,W))**2).float()
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B,1,H,W)
        return mask

    def border_mask(self, shape, sigma=0.5):
        # 1 - center mask
        return 1.0 - self.center_mask(shape, sigma=sigma)

    def diffuse_mask(self, cam, epsilon=1e-5):
        """
        Normalise une CAM pour obtenir un masque diffus.
        cam: (B, 1, H, W)
        """
        mask = cam / (cam.sum(dim=(2,3), keepdim=True) + epsilon)
        return mask

    def latent_mask(self, latent_mask, shape):
        """
        Expands un masque latent appris pour correspondre au batch.
        latent_mask: (1,1,H,W) ou (B,1,H,W)
        shape: tuple (B,C,H,W)
        """
        B, C, H, W = shape
        if latent_mask.shape[0] == 1:
            mask = latent_mask.expand(B, 1, H, W)
        else:
            mask = latent_mask
        return mask
