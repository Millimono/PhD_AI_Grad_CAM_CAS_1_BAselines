import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

import json
import numpy as np
import cv2  


class Trainer:
    def __init__(self, model, gradcam_module, optimizer, dataloader, criterion, 
                 gradcam_loss_weight=1.0, use_cam_loss=True,
                   fixed_images=None, fixed_labels=None,mask_type="center", mask_generator=None,
                   use_adaptive_supervision=False, total_epochs=20):
        
        self.use_cam_loss = use_cam_loss
        self.model = model
        self.grad_cam = gradcam_module
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.criterion = criterion
        self.gradcam_loss_weight = gradcam_loss_weight
        self.mask_generator = mask_generator   # <-- Important

        self.mask_type = mask_type  # 


        self.alpha = 1.0  # poids de la perte de reconstruction
        self.lambda_cam = 0.1  # poids de la CAM loss
        self.current_epoch = 0  # Initialiser l'époque

        self.cam_history = []  # stocke les anciennes CAMs
        self.adaptive_cam_weight = 0.0  # poids d’adaptation au début
        self.max_adaptive_weight = 0.9  # max de confiance dans CAM moyenne

        self.fixed_images = fixed_images
        self.fixed_labels = fixed_labels
        self.use_adaptive_supervision = use_adaptive_supervision  # ou False si supervision fixe
        self.total_epochs = total_epochs  # passé depuis main()

        #self.set_fixed_images(self.dataloader, num_images=5)  # Pour visualiser un batch fixe


    # def set_fixed_images(self, dataloader, num_images=5):
    #     for images, labels in dataloader:
    #         if images.size(0) >= num_images:
    #             self.fixed_images = images[:num_images].cuda()
    #             self.fixed_labels = labels[:num_images].cuda()
    #         else:
    #             print(f"⚠️ Batch trop petit ({images.size(0)}), impossible de fixer {num_images} images.")
    #             self.fixed_images = images.cuda()
    #             self.fixed_labels = labels.cuda()
    #         break



    # def gradcam_loss(self, cam, labels):
    #     return cam.mean()

    # def gradcam_loss(self, features, grads, target_mask):
    #     """
    #     Implémente la perte L_cam = ||CAM - M||^2 avec dépendance complète via alpha_k(A).
    #     features : activations A^k
    #     grads : gradients dY/dA^k
    #     target_mask : masque heuristique M (B, 1, H, W)
    #     """

    #     # Étape 1 : calcul des poids alpha_k = moyenne spatiale des gradients
    #     B, C, H, W = grads.shape
    #     alpha = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

    #     # Étape 2 : somme pondérée (pré-ReLU)
    #     S = (alpha * features).sum(dim=1, keepdim=True)  # (B, 1, H, W)

    #     # Étape 3 : ReLU (seulement si S>0)
    #     cam = F.relu(S)

    #     # Étape 4 : perte MSE entre CAM et masque M
    #     loss_cam = F.mse_loss(cam, target_mask)

    #     return loss_cam


    def train_epoch(self):
        self.model.train()
        total_loss = 0

        epoch_cam_accumulator = []  # Pour stocker les CAMs de cette époque

        for batch_idx, (images, labels) in enumerate(self.dataloader):
            images, labels = images.cuda(), labels.cuda()

            self.optimizer.zero_grad()
#------------------------------------------------------------------------------>
            outputs, features = self.model(images)  # outputs: (B, C)

            # Sélectionne le logit correspondant à la classe cible pour chaque échantillon
            target_logits = outputs.gather(1, labels.view(-1, 1)).squeeze()  # (B,)

            # Somme des logits cibles pour un backward unique
            target_score = target_logits.sum()

            # Calcul du gradient par rapport aux features
            grads = torch.autograd.grad(target_score, features, retain_graph=True, create_graph=True)[0]
            
            # grads = torch.autograd.grad(target_logits.sum(), features,
            #                 create_graph=True, retain_graph=True)[0]

            loss_class = self.criterion(outputs, labels)

# ------------------------------------------------------------------------------->            
            
            
            # cam = self.grad_cam(features, grads)  # (B, 1, H, W)
            # cam, S, alpha = self.grad_cam(features, grads) # (B, 1, H, W)
            
            cam, cam_pre_relu, weights = self.grad_cam(features, grads)


            # # 👉 Pour visualiser uniquement le premier batch
            # if batch_idx == 0:
            #     self.visualize_gradcam(images, cam, labels, self.current_epoch, batch_idx)

            # 🔁 Accumule les CAMs
            epoch_cam_accumulator.append(cam.detach())

            # 🎯 Génère une CAM target adaptative
            # center_mask = self.generate_center_mask(cam.shape, images.device)

            extra_args = {}
            if self.mask_type == "diffuse":
                extra_args["cam"] = cam
            elif self.mask_type == "latent":
                extra_args["latent_mask"] = self.latent_mask

            mask_a_priori = self.mask_generator.generate(cam.shape, self.mask_type, **extra_args)


            # mask_a_priori = self.mask_generator.generate(cam.shape, self.mask_type)

            if len(self.cam_history) > 0:
                adaptive_target = self.update_adaptive_target(cam.device, cam.shape)
                w = self.adaptive_cam_weight
                target_mask = (1 - w) * mask_a_priori + w * adaptive_target
            else:
                target_mask = mask_a_priori
            
            if self.use_cam_loss:
                    loss_cam = self.grad_cam.gradcam_loss(cam, target_mask)
                    # loss_cam = F.mse_loss(cam, target_mask)

                    if self.use_adaptive_supervision:
                        # Supervision adaptative : alpha_t croît avec l'époque
                        alpha_min = 0.1
                        alpha_max = self.gradcam_loss_weight
                        progress = min(1.0, self.current_epoch / self.total_epochs)
                        alpha_t = alpha_min + (alpha_max - alpha_min) * progress
                        total_loss_batch = loss_class + alpha_t * loss_cam

                    else :    
                        total_loss_batch = loss_class + self.gradcam_loss_weight * loss_cam


            else:
                total_loss_batch = loss_class
            
            # Sauvegarde périodique pour analyse scientifique
            if self.current_epoch % 5 == 0 and batch_idx == 0:  # uniquement 1er batch pour pas saturer le disque
                dataset_name = getattr(self.dataloader.dataset, 'name', 'dataset_unknown')
                mode = "cam_supervised" if self.use_cam_loss else "baseline"

                self.save_cam_analysis(
                            cam=cam.detach(),
                            cam_pre_relu=cam_pre_relu.detach(),
                            weights=weights.detach(),
                            save_dir=f"./logs/gradcam_analysis/{dataset_name}/{mode}/{self.mask_type}",
                            epoch=self.current_epoch,
                            layer_name="layer4"
                        )
            
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

            cams,_,_ = self.grad_cam(features, grads)
            preds = outputs.argmax(dim=1)

            self.visualize_gradcam(self.fixed_images, cams, self.fixed_labels, self.current_epoch, batch_idx=0, max_images=len(self.fixed_images),preds=preds)


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


    # def generate_center_mask(self, shape, device, sigma=0.5):
    #     """
    #     Génère un masque gaussien centré normalisé entre 0 et 1.
    #     shape: (batch_size, 1, H, W)
    #     """
    #     B, C, H, W = shape
    #     y = torch.linspace(-1, 1, steps=H).view(H, 1).expand(H, W)
    #     x = torch.linspace(-1, 1, steps=W).view(1, W).expand(H, W)
    #     grid = torch.stack([x, y], dim=0).to(dtype=torch.float32, device=device)  # (2, H, W)

    #     dist_squared = grid[0]**2 + grid[1]**2
    #     mask = torch.exp(-dist_squared / (2 * sigma**2))
    #     mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)  # (B, 1, H, W)

    #     return mask


    def compute_gradcam(self, images, labels=None, target_class=None):
        outputs, features = self.model(images)
        if labels is not None:
            target_logits = outputs.gather(1, labels.view(-1, 1)).squeeze()
            target_score = target_logits.sum()
        elif target_class is not None:
            target_score = outputs[:, target_class].sum()
        else:
            target_score = outputs.max(dim=1).values.sum()
        
        grads = torch.autograd.grad(target_score, features, retain_graph=True, create_graph=True)[0]
        cam,_,_ = self.grad_cam(features, grads)
        return outputs, features, cam

    @torch.no_grad()
    def visualize_gradcam(self, images, cams, labels, epoch, batch_idx, max_images=4, preds=None):

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

            # idx_to_class = {0: "covid", 1: "nocovid"}
            idx_to_class = self.dataloader.dataset.class_to_idx
            idx_to_class = {v: k for k, v in idx_to_class.items()}


             # Ajout nom classes
            true_cls = idx_to_class[labels[i].item()]
            pred_cls = idx_to_class[preds[i].item()] if preds is not None else "N/A"    




            mode = "cam_supervised" if self.use_cam_loss else "baseline"
            filename = os.path.join(folder_path, 
                    f"{mode}_epoch_{epoch}_batch_{batch_idx}_img_{i}_true_{true_cls}_pred_{pred_cls}.png"
)
            #filename = os.path.join(folder_path, f"epoch_{epoch}_batch_{batch_idx}_img_{i}.png")

            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            print(f"Saved side-by-side Grad-CAM visualization for image {i} at {filename}")



    # def create_gradcam_folder(self):
    #     mode = "cam_supervised" if self.use_cam_loss else "baseline"
    #     #folder_path = './logs/cams/'
    #     folder_path = f'./logs/cams/{mode}/'
        
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #     return folder_path

    def create_gradcam_folder(self):
        """
        Crée un dossier pour stocker les visualisations GradCAM en fonction :
        - du dataset
        - de la supervision CAM
        - du type de masque
        Exemple : ./logs/supervision_fixe/cams/cifar10/cam_supervised/center/
        """
        # Nom du dataset si disponible
        dataset_name = getattr(self.dataloader.dataset, 'name', 'dataset_unknown')
        
        mode = "cam_supervised" if self.use_cam_loss else "baseline"
        folder_path = os.path.join('./logs/supervision_fixe/cams', dataset_name, mode, self.mask_type)

        os.makedirs(folder_path, exist_ok=True)
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
            cams,_,_ = self.grad_cam(features, grads)  # (B, 1, H, W)
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

    
    def save_full_model(self, model, filepath="model_full.pt"):
        """
        Sauvegarde le modèle complet (architecture + poids)
        """

        # Récupération du nom du dataset si disponible
        dataset_name = getattr(self.dataloader.dataset, 'name', 'dataset_unknown')

        # Déterminer le mode de supervision
        mode = "cam_supervised" if self.use_cam_loss else "baseline"

        # Construire le chemin complet du dossier
        model_dir = os.path.join('./logs/supervision_fixe/models', dataset_name, mode, self.mask_type)
        os.makedirs(model_dir, exist_ok=True)

        # Chemin complet du fichier de sauvegarde
        full_path = os.path.join(model_dir, filepath)

        # Sauvegarde du modèle complet
        torch.save(model, full_path)
        print(f"✅ Modèle complet sauvegardé dans : {full_path}")
    
    
    def save_full_and_state_model(self, model, filepath="model_full"):
        """
        Sauvegarde à la fois le modèle complet (.pt) et uniquement les poids (.pth),
        dans un dossier structuré selon le dataset, le mode de supervision et le type de masque.
        """
        # Récupération du nom du dataset si disponible
        dataset_name = getattr(self.dataloader.dataset, 'name', 'dataset_unknown')

        # Déterminer le mode de supervision
        mode = "cam_supervised" if self.use_cam_loss else "baseline"

        # Construire le dossier cible
        model_dir = os.path.join('./logs/supervision_fixe/models', dataset_name, mode, self.mask_type)
        os.makedirs(model_dir, exist_ok=True)

        # Enlever toute extension si elle existe (pour éviter model_full.pt.pt)
        base_name = os.path.splitext(filepath)[0]

        # Construire les chemins
        full_model_path = os.path.join(model_dir, base_name + ".pt")
        state_dict_path = os.path.join(model_dir, base_name + "_weights.pth")

        # Sauvegarde du modèle complet
        torch.save(model, full_model_path)

        # Sauvegarde uniquement des poids
        torch.save(model.state_dict(), state_dict_path)

        print(f"✅ Modèle complet sauvegardé dans : {full_model_path}")
        print(f"✅ Poids du modèle sauvegardés dans : {state_dict_path}")


    def save_list_as_json(self, data_list, filename="metrics_history.json"):
        """
        Sauvegarde une liste de dictionnaires (par ex. historique des métriques)
        dans un fichier JSON dans le même dossier que les modèles.
        
        Args:
            data_list (list): liste de dictionnaires à sauvegarder
            filename (str): nom du fichier JSON à créer (par défaut "metrics_history.json")
        """
        # Récupération du nom du dataset si disponible
        dataset_name = getattr(self.dataloader.dataset, 'name', 'dataset_unknown')

        # Déterminer le mode de supervision
        mode = "cam_supervised" if self.use_cam_loss else "baseline"

        # Construire le dossier cible
        model_dir = os.path.join('./logs/supervision_fixe/metrics', dataset_name, mode, self.mask_type)
        os.makedirs(model_dir, exist_ok=True)
        

        # Construit le chemin complet
        file_path = os.path.join(model_dir, filename)

        # Sauvegarde proprement la liste au format JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)

        print(f"✅ Données sauvegardées dans {file_path}")




    @staticmethod
    def load_full_model(filepath, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Charge un modèle complet sauvegardé (structure + poids)
        """
        # model = torch.load(filepath, map_location=device)
        model = torch.load(filepath, map_location=device, weights_only=False)

        model.eval()
        print(f"✅ Modèle chargé depuis {filepath}")
        return model


    def save_cam_analysis(self,cam, cam_pre_relu, weights, save_dir, epoch, layer_name="last_conv"):
        """
        Sauvegarde les CAM, pré-ReLU et les poids Grad-CAM pour analyse scientifique.
        """
        os.makedirs(save_dir, exist_ok=True)

        # ✅ Conversion CPU et numpy
        cam_np = cam.detach().cpu().numpy()
        cam_pre_relu_np = cam_pre_relu.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()

        # ✅ Sauvegarde des tenseurs bruts pour analyse ultérieure (ex: via NumPy)
        np.save(os.path.join(save_dir, f"cam_epoch{epoch}_{layer_name}.npy"), cam_np)
        np.save(os.path.join(save_dir, f"cam_pre_relu_epoch{epoch}_{layer_name}.npy"), cam_pre_relu_np)
        np.save(os.path.join(save_dir, f"weights_epoch{epoch}_{layer_name}.npy"), weights_np)

        # ✅ Figure visuelle rapide
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cam_pre_relu_np[0, 0], cmap='bwr')
        plt.title("CAM avant ReLU")
        #CAM avant ReLU (zones négatives incluses)
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(cam_np[0, 0], cmap='hot')
        plt.title("CAM après ReLU ")
        # CAM après ReLU (zones actives)
        plt.colorbar()

        plt.subplot(1, 3, 3)
        mean_weights = weights_np[0].mean(axis=(1, 2))  # moyenne par canal
        plt.plot(mean_weights)
        plt.title("Poids moyens par canal (αₖ)")
        plt.xlabel("Canal")
        plt.ylabel("Importance moyenne")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"cam_analysis_epoch{epoch}_{layer_name}.png"))
        plt.close()

        print(f"📊 Analyse CAM sauvegardée dans {save_dir}")
