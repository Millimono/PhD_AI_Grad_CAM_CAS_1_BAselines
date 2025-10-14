import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):

    def __init__(self, backbone_name='resnet18', pretrained=False):
        super().__init__()

        # Charger dynamiquement le backbone
        if hasattr(models, backbone_name):
            backbone = getattr(models, backbone_name)(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone '{backbone_name}' non reconnu par torchvision.models")

        # Obtenir le nombre de canaux du dernier bloc conv (cela dépend du modèle)
        if 'resnet' in backbone_name:
            self.out_channels = backbone.fc.in_features
            self.features = nn.Sequential(*list(backbone.children())[:-2])  # jusqu'à conv5
        
        elif 'vgg' in backbone_name:
            self.out_channels = 512  # VGG16 et VGG19 sortent 512 canaux à la fin du dernier conv
            self.features = backbone.features

        elif 'densenet' in backbone_name:
            self.out_channels = backbone.classifier.in_features
            self.features = nn.Sequential(
                backbone.features,
                nn.ReLU(inplace=True)  # pour suivre l'activation après le dernier batchnorm
            )

        elif 'efficientnet' in backbone_name:
            self.out_channels = backbone.classifier[1].in_features
            self.features = backbone.features

        else:
            raise NotImplementedError(f"Backbone '{backbone_name}' non encore pris en charge.")


        
    def forward(self, x):
        return self.features(x)

class ClassifierHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

class FullModel(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet18', pretrained=False):

        super().__init__()

        self.feature_extractor = FeatureExtractor(backbone_name=backbone_name, pretrained=pretrained)
        self.classifier = ClassifierHead(self.feature_extractor.out_channels, num_classes)


    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

# Example usage:
if __name__ == "__main__":
    model = FullModel(num_classes=10, backbone_name='resnet18', pretrained=True)
    print(model)

    # Test with a random input
    x = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    logits, features = model(x)
    print("Logits shape:", logits.shape)
    print("Features shape:", features.shape)


    # Fine-tuning avec ResNet50 préentraîné
    model = FullModel(num_classes=2, backbone_name='resnet50', pretrained=True)
    # Fine-tuning avec VGG16
    model = FullModel(num_classes=2, backbone_name='vgg16', pretrained=True)
    # Fine-tuning avec DenseNet121
    model = FullModel(num_classes=2, backbone_name='densenet121', pretrained=True)
    # Expérience avec EfficientNet
    model = FullModel(num_classes=2, backbone_name='efficientnet_b0', pretrained=True)
