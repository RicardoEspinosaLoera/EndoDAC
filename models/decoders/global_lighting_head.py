import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLightingHead(nn.Module):
    """One affine pair (contrast c, brightness b) per image.

    The *global* illumination-calibration control of the CVIU ablation
    (--illum_calib global). It reads the pose-encoder bottleneck like
    IntrinsicsHead, global-average-pools it and predicts two scalars bounded
    exactly like LightingDecoder (c in [1-alpha, 1+alpha], b in [-beta, beta]).
    Outputs use LightingDecoder's keys, so the trainer does not care which of
    the two it holds; the maps are (B,1,1,1) and the trainer's bilinear
    upsampling broadcasts them to full resolution.
    """

    def __init__(self, num_ch_enc, scales=range(4), alpha=0.10, beta=0.05):
        super(GlobalLightingHead, self).__init__()
        self.scales = list(scales)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(num_ch_enc[-1], 256, 1)
        self.head = nn.Conv2d(256, 2, 1)
        # start at the identity calibration (c=1, b=0); tanh'(0)=1 so gradients flow
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_features):
        x = self.head(F.relu(self.squeeze(self.pool(input_features[-1]))))  # (B,2,1,1)
        c = 1.0 + self.alpha * torch.tanh(x[:, 0:1])
        b = self.beta * torch.tanh(x[:, 1:2])
        outputs = {}
        for s in self.scales:
            outputs[("lighting", s)] = x
            outputs[("contrast", s)] = c
            outputs[("brightness", s)] = b
        return outputs
