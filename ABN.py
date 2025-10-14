import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# 1. Backbone CNN (ex: ResNet-18 simplifié)
# ----------------------------------------
from torchvision.models import resnet18

class ABNModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ABNModel, self).__init__()
        # Backbone
        backbone = resnet18(pretrained=pretrained)
        modules = list(backbone.children())[:-2]  # retirer FC et AvgPool
        self.features = nn.Sequential(*modules)
        self.num_features = 512  # dépend du backbone

        # Branch attention (Attention Branch)
        self.att_conv = nn.Conv2d(self.num_features, 1, kernel_size=1)
        self.att_sigmoid = nn.Sigmoid()

        # Branch perception / classification
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        feats = self.features(x)  # (B, C, H, W)

        # ----------- Attention Branch -----------
        att_map = self.att_conv(feats)       # (B, 1, H, W)
        att_map = self.att_sigmoid(att_map)  # [0,1]

        # Appliquer attention
        feats_att = feats * att_map          # broadcasting (B,C,H,W)

        # ----------- Perception Branch -----------
        pooled = self.avgpool(feats_att).view(x.size(0), -1)
        logits = self.fc(pooled)

        return logits, att_map
    
    def abn_loss(logits, labels, att_map, lambda_att=0.05):
        """
        logits : sorties de classification
        labels : labels ground-truth
        att_map: carte d'attention générée par la branche attention
        lambda_att : poids de régularisation attention
        """
        # Loss classification standard
        loss_class = F.cross_entropy(logits, labels)

        # Perte attention (régularisation simple L2 pour stabiliser l'attention)
        loss_att = torch.mean(att_map**2)  # comme dans l'article : encourage cohérence / lissage

        # Loss totale
        loss_total = loss_class + lambda_att * loss_att
        return loss_total

