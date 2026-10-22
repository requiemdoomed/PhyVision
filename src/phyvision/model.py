"""PhyNet: dual-branch physique classifier.

Branch A  ResNet-18 (single-channel stem) reads the 128x128 normalised
          silhouette and learns spatial shape cues (V-taper, limb mass, belly).
Branch B  An MLP reads the 100-d shape profile (width at 48 body heights +
          global statistics), giving the network explicit proportion data.
Fusion    Concatenated 512 + 64 features -> BatchNorm -> Dropout -> 128 -> 4 classes.
Framing   A second small head on the fused features estimates how much of the
          body is in the photo (1.0 = head to feet, ~0.6 = head to mid-thigh).
          Learning this tells the classifier how to read a cropped photo.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18

from .config import CLASSES, PROFILE_DIM


class PhyNet(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES), profile_dim: int = PROFILE_DIM):
        super().__init__()
        cnn = resnet18(weights=None)
        cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.profile = nn.Sequential(
            nn.Linear(profile_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(nn.BatchNorm1d(512 + 64), nn.Dropout(0.4))
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.framing = nn.Sequential(nn.Linear(512 + 64, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, sil: torch.Tensor, prof: torch.Tensor):
        """Returns (class logits [N, 4], visible body fraction [N] in 0..1)."""
        z = self.fuse(torch.cat([self.cnn(sil), self.profile(prof)], dim=1))
        return self.head(z), torch.sigmoid(self.framing(z)).squeeze(1)
