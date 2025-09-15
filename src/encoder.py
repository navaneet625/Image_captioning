import os
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms

class EncoderCNN(nn.Module):
    """
    Minimal ResNet-based encoder that returns spatial features.
    For standard 224x224 inputs, resnet50 layer4 output is (B, 2048, 7, 7).
    We'll flatten spatial dims -> (B, L, D) where L=7*7=49.
    """

    def __init__(self, pretrained=True, trainable=False):
        super().__init__()
        # load a resnet and cut off classification head
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT  # ✅ use weights instead of deprecated pretrained
        else:
            weights = None

        resnet = models.resnet50(weights=weights)

        # keep everything until layer4 (inclusive)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.out_dim = 2048

        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images):
        """
        images: (B, 3, H, W)
        returns: (B, L, D) where L=H'*W' (spatial locations)
        """
        x = self.backbone(images)           # (B, D, Hf, Wf)
        b, d, hf, wf = x.shape
        x = x.view(b, d, hf * wf).permute(0, 2, 1)  # (B, L, D)
        return x

    def extract_and_save(self, images_dir, out_dir, transform=None, device="cpu"):
        """
        Utility: iterate images in images_dir, compute features and save as .pt per image.
        Not used by train/infer automatically, but convenient.
        """
        os.makedirs(out_dir, exist_ok=True)
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        device = torch.device(device)
        self.to(device).eval()
        for fn in os.listdir(images_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = Image.open(os.path.join(images_dir, fn)).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = self.forward(img_t).squeeze(0).cpu()  # (L, D)
            torch.save(feat, os.path.join(out_dir, fn + ".pt"))
            