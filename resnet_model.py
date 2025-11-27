import torch.nn as nn
from torchvision import models

class ResNetEmotion(nn.Module):
    """
    ResNet18 backbone with a new 7-class output layer for emotions.
    Uses pretrained ImageNet weights by default.
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
