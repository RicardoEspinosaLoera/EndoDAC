import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ConvBlock, LayerNorm

class MAB(nn.Module):
    """
    Multi-scale Attention Block used before Patch Embedding in HADepth.
    Produces multi-scale CNN features and fuses them into a single feature map.

    Input:  B x 3 x H x W
    Output: B x C x H x W   (default C = embed_dim)
    """

    def __init__(self, in_channels=3, embed_dim=64):
        super(MAB, self).__init__()

        self.embed_dim = embed_dim
        mid = embed_dim // 3

        # ----- Multi-scale convolutions -----
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 7, padding=3),
            nn.ReLU(inplace=True),
        )

        # ----- Optional attention-like weighting -----
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim, 1),
            nn.Sigmoid()
        )

        # ----- Final projection -----
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.norm = LayerNorm(embed_dim, data_format='channels_first')

    def forward(self, x):
        # Multi-scale branches
        f3 = self.branch3(x)
        f5 = self.branch5(x)
        f7 = self.branch7(x)

        # Concatenate: B x (3*mid) x H x W
        feats = torch.cat([f3, f5, f7], dim=1)  # → B x embed_dim x H x W

        # Channel attention
        att = self.channel_att(feats)
        feats = feats * att

        # Final projection + normalization
        feats = self.proj(feats)
        feats = self.norm(feats)

        return feats
