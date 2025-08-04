import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

import numpy as np
import cv2  # si tu fais des heatmaps avec OpenCV


class Trainer:
    def __init__(self, model, gradcam_module, optimizer, dataloader, criterion, gradcam_loss_weight=1.0, use_cam_loss=True, fixed_images=None, fixed_labels=None):
        self.use_cam_loss = use_cam_loss
        self.model = model
        self.grad_cam = gradcam_module
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.criterion = criterion
        self.gradcam_loss_weight = gradcam_loss_weight
        self.alpha = 1.0  # poids de la perte de reconstruction
        self.lambda_cam = 0.1  # poids de la CAM loss
        self.current_epoch = 0  # Initialiser l'époque

        self.cam_history = []  # stocke les anciennes CAMs
        self.adaptive_cam_weight = 0.0  # poids d’adaptation au début
        self.max_adaptive_weight = 0.9  # max de confiance dans CAM moyenne

        self.fixed_images = fixed_images
        self.fixed_labels = fixed_labels

        #self.set_fixed_images(self.dataloader, num_images=5)  # Pour visualiser un batch fixe


    def set_fixed_images(self, dataloader, num_images=5):
        for images, labels in dataloader:
            if images.size(0) >= num_images:
                self.fixed_images = images[:num_images].cuda()
                self.fixed_labels = labels[:num_images].cuda()
            else:
                print(f"⚠️ Batch trop petit ({images.size(0)}), impossible de fixer {num_images} images.")
                self.fixed_images = images.cuda()
                self.fixed_labels = labels.cuda()
            break



    def gradcam_loss(self, cam, labels):
        # Ex: for now, dummy loss: sum of CAMs (you can make it class-aware)
        return cam.mean()



    def train_epoch(self):
        self.model.train()
        total_loss = 0

        epoch_cam_accumulator = []  # Pour stocker les CAMs de cette époque

        for batch_idx, (images, labels) in enumerate(self.dataloader):
            images, labels = images.cuda(), labels.cuda()

            self.optimizer.zero_grad()

            # outputs, features = self.model(images)
            # loss_class = self.criterion(outputs, labels)

            # grads = torch.autograd.grad(loss_class, features, retain_graph=True, create_graph=True)[0]

#------------------------------------------------------------------------------>
            outputs, features = self.model(images)  # outputs: (B, C)

            # Sélectionne le logit correspondant à la classe cible pour chaque échantillon
            target_logits = outputs.gather(1, labels.view(-1, 1)).squeeze()  # (B,)

            # Somme des logits cibles pour un backward unique
            target_score = target_logits.sum()

            # Calcul du gradient par rapport aux features
            grads = torch.autograd.grad(target_score, features, retain_graph=True, create_graph=True)[0]
           
            loss_class = self.criterion(outputs, labels)

# ------------------------------------------------------------------------------->            
            
            
            cam = self.grad_cam(features, grads)  # (B, 1, H, W)
            

            # # 👉 Pour visualiser uniquement le premier batch
            # if batch_idx == 0:
            #     self.visualize_gradcam(images, cam, labels, self.current_epoch, batch_idx)

            # 🔁 Accumule les CAMs
            epoch_cam_accumulator.append(cam.detach())

            # 🎯 Génère une CAM target adaptative
            center_mask = self.generate_center_mask(cam.shape, images.device)

            if len(self.cam_history) > 0:
                adaptive_target = self.update_adaptive_target(cam.device, cam.shape)
                w = self.adaptive_cam_weight
                target_mask = (1 - w) * center_mask + w * adaptive_target
            else:
                target_mask = center_mask
            if self.use_cam_loss:
                    loss_cam = F.mse_loss(cam, target_mask)
                    total_loss_batch = loss_class + self.gradcam_loss_weight * loss_cam
            else:
                total_loss_batch = loss_class
            
            # 🔥 Ajoute la perte de reconstruction si nécessaire
            total_loss_batch.backward()
            self.optimizer.step()

            total_loss += total_loss_batch.item()

        # 🔁 Enregistre toutes les CAMs de cette époque
        epoch_cams = torch.cat(epoch_cam_accumulator, dim=0)  # (N, 1, H, W)
        self.cam_history.append(epoch_cams)

        # 🔄 Met à jour le poids adaptatif (progressif)
        self.adaptive_cam_weight = min(self.max_adaptive_weight, self.current_epoch / 10.0)

        self.current_epoch += 1

        # Visualiser les CAMs sur batch fixe après l'époque
        if self.fixed_images is None or self.fixed_images.nelement() == 0:
            print("⚠️ self.fixed_images est vide ou non initialisée.")
        else:
            print(f"✅ self.fixed_images: {self.fixed_images.shape}")
            # self.model.eval()
            # with torch.no_grad():
            outputs, features = self.model(self.fixed_images)
            target_logits = outputs.gather(1, self.fixed_labels.view(-1, 1)).squeeze()
            grads = torch.autograd.grad(target_logits.sum(), features, retain_graph=True, create_graph=True)[0]

            cams = self.grad_cam(features, grads)
            self.visualize_gradcam(self.fixed_images, cams, self.fixed_labels, self.current_epoch, batch_idx=0, max_images=len(self.fixed_images))


        return total_loss / len(self.dataloader)

    def update_adaptive_target(self, device, shape):
        """
        Calcule la moyenne des CAMs passées (sur tous les batches).
        Retourne une CAM moyenne adaptée à la forme du batch courant.
        """
        all_cams = torch.cat(self.cam_history, dim=0)  # (N, 1, H, W)
        mean_cam = all_cams.mean(dim=0, keepdim=True)  # (1, 1, H, W)
        mean_cam = mean_cam.expand(shape[0], 1, shape[2], shape[3])  # (B, 1, H, W)
        return mean_cam.to(device)


    def generate_center_mask(self, shape, device, sigma=0.5):
        """
        Génère un masque gaussien centré normalisé entre 0 et 1.
        shape: (batch_size, 1, H, W)
        """
        B, C, H, W = shape
        y = torch.linspace(-1, 1, steps=H).view(H, 1).expand(H, W)
        x = torch.linspace(-1, 1, steps=W).view(1, W).expand(H, W)
        grid = torch.stack([x, y], dim=0).to(dtype=torch.float32, device=device)  # (2, H, W)

        dist_squared = grid[0]**2 + grid[1]**2
        mask = torch.exp(-dist_squared / (2 * sigma**2))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)  # (B, 1, H, W)

        return mask


    @torch.no_grad()
    def visualize_gradcam(self, images, cams, labels, epoch, batch_idx, max_images=4):
        cams = cams.squeeze(1)  # (B, H, W)
        cams = (cams - cams.min(dim=1, keepdim=True)[0]) / (cams.max(dim=1, keepdim=True)[0] + 1e-5)
        cams = F.interpolate(cams.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)

        images = F.interpolate(images, size=(224, 224))  # (B, 3, 224, 224)
        images = images.cpu().permute(0, 2, 3, 1).numpy()  # (B, 224, 224, 3)

        folder_path = self.create_gradcam_folder()

        for i in range(min(max_images, images.shape[0])):
            img = images[i]
            img = (img - img.min()) / (img.max() + 1e-5)

            cam = cams[i].cpu().numpy()
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

            overlay = heatmap * 0.5 + img * 0.5
            overlay = np.clip(overlay, 0, 1)

            # 🔥 Affichage côte à côte
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img)
            axes[0].set_title("Image originale")
            axes[0].axis("off")

            axes[1].imshow(overlay)
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")



            mode = "cam_supervised" if self.use_cam_loss else "baseline"
            filename = os.path.join(folder_path, f"{mode}_epoch_{epoch}_batch_{batch_idx}_img_{i}.png")
            #filename = os.path.join(folder_path, f"epoch_{epoch}_batch_{batch_idx}_img_{i}.png")

            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            print(f"Saved side-by-side Grad-CAM visualization for image {i} at {filename}")



    def create_gradcam_folder(self):
        mode = "cam_supervised" if self.use_cam_loss else "baseline"
        #folder_path = './logs/cams/'
        folder_path = f'./logs/cams/{mode}/'
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path


    @torch.no_grad()
    def evaluate_accuracy(self):
        self.model.eval()
        correct = 0
        total = 0
        for images, labels in self.dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs, _ = self.model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        return correct / total

    @torch.no_grad()
    def evaluate_predictions(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in self.dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs, _ = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        return all_preds, all_labels, all_probs



    def test_model(self, dataloader, output_folder="test_visualizations", max_images=4):
        self.model.eval()
        os.makedirs(output_folder, exist_ok=True)

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.cuda(), labels.cuda()
            images.requires_grad = True  # pour que le backward fonctionne

            # Passe avant avec gradient pour Grad-CAM
            outputs, features = self.model(images)

            # Prédictions
            preds = outputs.argmax(dim=1)

            # Grad-CAM : on s'intéresse à la classe cible (vraie classe ici)
            target_logits = outputs.gather(1, labels.view(-1, 1)).squeeze()
            target_score = target_logits.sum()

            grads = torch.autograd.grad(target_score, features, retain_graph=True, create_graph=True)[0]

            # Calcul CAM
            cams = self.grad_cam(features, grads)  # (B, 1, H, W)
            cams = F.interpolate(cams, size=(224, 224), mode='bilinear', align_corners=False)
            cams = (cams - cams.min()) / (cams.max() + 1e-8)

            # Redimensionner images pour overlay
            images_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            images_np = images_resized.detach().cpu().permute(0, 2, 3, 1).numpy()

            for i in range(min(max_images, images.shape[0])):
                image = images_np[i]
                image = (image - image.min()) / (image.max() + 1e-8)

                cam = cams[i, 0].detach().cpu().numpy()
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

                overlay = 0.5 * image + 0.5 * heatmap
                overlay = np.clip(overlay, 0, 1)

                true_label = labels[i].item()
                predicted_label = preds[i].item()

                mode = "cam_supervised" if self.use_cam_loss else "baseline"
                filename = f"{mode}_epoch{self.current_epoch}_batch{batch_idx}_img{i}_true{true_label}_pred{predicted_label}.png"

                #filename = f"epoch{self.current_epoch}_batch{batch_idx}_img{i}_true{true_label}_pred{predicted_label}.png"
                path = os.path.join(output_folder, filename)

                plt.imsave(path, overlay)

            #break  # ⚠️ Enlève ce break si tu veux tester tous les batches
    
    @torch.no_grad()
    def evaluate_accuracy_val_data(self, val_dataloader=None):
        if val_dataloader is None:
            print("⚠️ Aucun DataLoader de validation passé. Évaluation ignorée.")
            return None

        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()

        for images, labels in val_dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs, _ = self.model(images)
            loss = criterion(outputs, labels)  # Calcul du loss

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)  # pondéré par la taille du batch

        accuracy = correct / total
        avg_loss = total_loss / total

        return accuracy, avg_loss

    @torch.no_grad()
    def evaluate_predictions_val(self, val_dataloader=None):
        if val_dataloader is None:
            print("⚠️ Aucun DataLoader de validation passé. Évaluation ignorée.")
            return None, None, None

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in val_dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs, _ = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        return all_preds, all_labels, all_probs

    def save_full_model(model, filepath="model_full.pt"):
        """
        Sauvegarde le modèle complet (architecture + poids)
        """
        torch.save(model, filepath)
        print(f"✅ Modèle sauvegardé dans {filepath}")

    @staticmethod
    def load_full_model(filepath, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Charge un modèle complet sauvegardé (structure + poids)
        """
        model = torch.load(filepath, map_location=device)
        model.eval()
        print(f"✅ Modèle chargé depuis {filepath}")
        return model
