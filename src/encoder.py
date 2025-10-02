import torch
import torch.nn as nn
from torchvision import models


class EncoderResNet50(nn.Module):
    """
    ResNet-50 encoder for Show, Attend and Tell.
    Input:  (B, 3, 224, 224)
    Output: (B, L=49, D=2048)  -> annotation vectors
    """

    def __init__(self, pretrained=True, trainable=False):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # keep layers up to layer4 (exclude avgpool & fc)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.out_dim = 2048

        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images):
        """
        images: (B, 3, 224, 224)
        returns: (B, L=49, D=2048)
        """
        x = self.backbone(images)                  # (B, 2048, 7, 7)
        b, d, h, w = x.shape
        return x.view(b, d, h * w).permute(0, 2, 1)  # (B, 49, 2048)
