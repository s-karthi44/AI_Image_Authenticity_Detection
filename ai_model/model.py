import torch
import torch.nn as nn
from torchvision import models

class AIDetectorModel(nn.Module):
    def __init__(self, base='resnet50', num_classes=2):
        super(AIDetectorModel, self).__init__()
        self.base_name = base
        if base == 'resnet50':
            # Use weights=models.ResNet50_Weights.DEFAULT if torchvision version supports it, 
            # otherwise pretrained=True. For compatibility, we'll use pretrained=True or check.
            # Using new weights API if possible is better, but safe bet:
            self.backbone = models.resnet50(pretrained=True)
            num_features = self.backbone.fc.in_features
            # Replace the final fully connected layer
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif base == 'efficientnet_b7':
            self.backbone = models.efficientnet_b7(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            raise NotImplementedError(f"Base model {base} not implemented")

    def forward(self, x):
        return self.backbone(x)
