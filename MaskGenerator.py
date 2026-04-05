import torch
import torch.nn.functional as F   # ← manquait dans ta version

class MaskGenerator:
    def __init__(self, device='cuda'):
        self.device = device

    def generate(self, shape, mask_type="center", cam=None, latent_mask=None,image=None,  **kwargs):
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
        
        elif mask_type == "ellipse":
            return self.ellipse_mask(shape, **kwargs)
        elif mask_type == "tissue":
            if image is None:
                raise ValueError("Pour 'tissue', il faut passer image=<tensor>")
            return self.tissue_mask(image, shape, **kwargs)

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


     # ------------------------------------------------------------------ #
    #  Nouveaux masques anatomiques — mammographie                        #
    # ------------------------------------------------------------------ #
 
    def ellipse_mask(self, shape, side="left", a_ratio=0.45, b_ratio=0.55, sigma_blend=0.08):
        """
        Masque semi-elliptique gaussien, motivé par la morphologie du sein
        en mammographie (vue MLO ou CC).
 
        Formule (coordonnées normalisées dans [-1, 1]) :
 
            M_ellipse(i,j) = exp( - (x_rel/a)^2 - (y_rel/b)^2 )
 
        avec (x_rel, y_rel) le décalage par rapport au centroïde de l'ellipse,
        ancrée sur le bord latéral gauche ou droit de l'image.
 
        Args:
            shape     : (B, C, H, W)
            side      : "left" | "right" — côté du sein dans l'image
            a_ratio   : demi-axe horizontal  (fraction de W, ∈ ]0,1[)
            b_ratio   : demi-axe vertical    (fraction de H, ∈ ]0,1[)
            sigma_blend : lissage gaussien des bords de l'ellipse
        """
        B, C, H, W = shape
 
        # Grille normalisée [-1, 1]
        y_lin = torch.linspace(-1, 1, steps=H, device=self.device)
        x_lin = torch.linspace(-1, 1, steps=W, device=self.device)
        yy, xx = torch.meshgrid(y_lin, x_lin, indexing="ij")   # (H, W)
 
        # Centroïde de l'ellipse : ancré sur le bord latéral, centré verticalement
        cx = -1.0 + a_ratio if side == "left" else 1.0 - a_ratio
        cy = 0.0
 
        # Distance elliptique normalisée
        dx = (xx - cx) / a_ratio   # (H, W)
        dy = (yy - cy) / b_ratio
 
        dist2 = dx**2 + dy**2      # = 1 sur le contour de l'ellipse
 
        # Gaussienne centrée sur l'ellipse : douce à l'intérieur, décroît à l'extérieur
        mask = torch.exp(-dist2 / (2 * sigma_blend**2 + 1e-6))
 
        # Normalisation dans [0, 1]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
 
        return mask.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
 
    def tissue_mask(self, image, shape, smooth_sigma=5.0, dilate_iter=3):
        """
        Masque image-driven par seuillage d'Otsu adaptatif.
 
        Extrait la région du tissu mammaire directement depuis l'intensité
        de l'image, sans annotation :
 
            τ_Otsu = argmax_τ [ w0(τ)·w1(τ)·(μ0(τ) - μ1(τ))² ]
 
        Puis applique un lissage gaussien pour obtenir un masque continu.
 
        Args:
            image        : (B, C, H, W) — images d'entrée (valeurs dans [0,1])
            shape        : (B, C, H, W) — même shape, pour cohérence API
            smooth_sigma : écart-type du lissage gaussien final (en pixels)
            dilate_iter  : nombre d'itérations de dilatation morphologique
        """
        B, C, H, W = shape
 
        # 1. Convertir en niveaux de gris (moyenne des canaux)
        gray = image.mean(dim=1, keepdim=True)          # (B, 1, H, W)
 
        masks = []
        for b in range(B):
            img_b = gray[b, 0]                          # (H, W)
 
            # 2. Seuillage d'Otsu (implémentation différentiable-friendly)
            tau = self._otsu_threshold(img_b)
 
            # 3. Masque binaire : tissu = pixels au-dessus du seuil
            binary = (img_b > tau).float()              # (H, W)
 
            # 4. Dilatation morphologique (max-pooling 3×3 itéré)
            m = binary.unsqueeze(0).unsqueeze(0)        # (1,1,H,W)
            for _ in range(dilate_iter):
                m = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
            binary_dilated = m.squeeze(0)               # (1,H,W)
 
            # 5. Lissage gaussien pour masque continu
            k_size = int(6 * smooth_sigma) | 1          # impair
            gauss_kernel = self._gaussian_kernel_2d(k_size, smooth_sigma)
            gauss_kernel = gauss_kernel.to(self.device).unsqueeze(0).unsqueeze(0)
            pad = k_size // 2
            smoothed = F.conv2d(binary_dilated.unsqueeze(0), gauss_kernel,
                                padding=pad)            # (1,1,H,W)
            smoothed = smoothed.squeeze(0)              # (1,H,W)
 
            # 6. Normalisation dans [0, 1]
            mn, mx = smoothed.min(), smoothed.max()
            smoothed = (smoothed - mn) / (mx - mn + 1e-8)
 
            masks.append(smoothed)
 
        #mask = torch.stack(masks, dim=0)                # (B, 1, H, W)
        #return mask
    
        # APRÈS
        mask = torch.stack(masks, dim=0)
        mask = F.interpolate(mask, size=(shape[2], shape[3]),
                            mode='bilinear', align_corners=False)
        return mask
 
    # ------------------------------------------------------------------ #
    #  Helpers privés                                                      #
    # ------------------------------------------------------------------ #
 
    @staticmethod
    def _otsu_threshold(img: torch.Tensor, n_bins: int = 256) -> float:
        """
        Seuil d'Otsu calculé sur un tenseur 2-D (H, W) dans [0, 1].
 
        Maximise la variance inter-classes :
            σ²_B(τ) = w0·w1·(μ0 - μ1)²
        """
        flat = img.reshape(-1)
        hist = torch.histc(flat, bins=n_bins, min=0.0, max=1.0)   # (n_bins,)
        hist = hist / hist.sum()                                    # probabilités
 
        bins = torch.linspace(0.0, 1.0, steps=n_bins, device=img.device)
 
        best_tau, best_var = 0.0, -1.0
        for t in range(1, n_bins):
            w0 = hist[:t].sum()
            w1 = hist[t:].sum()
            if w0 < 1e-8 or w1 < 1e-8:
                continue
            mu0 = (bins[:t] * hist[:t]).sum() / w0
            mu1 = (bins[t:] * hist[t:]).sum() / w1
            var = (w0 * w1 * (mu0 - mu1)**2).item()
            if var > best_var:
                best_var = var
                best_tau = bins[t].item()
        return best_tau
 
    @staticmethod
    def _gaussian_kernel_2d(k_size: int, sigma: float) -> torch.Tensor:
        """Noyau gaussien 2-D de taille k_size × k_size."""
        ax = torch.arange(k_size).float() - k_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.sum()