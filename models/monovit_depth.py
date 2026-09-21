"""MonoViT depth network: an MPViT encoder + the HR (nested, fSE-attention) decoder.

Trained from scratch inside this repo with --depth_backbone monovit, so MonoViT enters the CVIU
tables as a run of the grid rather than as an external checkpoint. Like models/resnet_depth.py it
wraps encoder+decoder behind the endodac interface (image -> {("disp", s)}), which is all the
trainer and cviu_revision.py's predict stage need.

Two encoder sizes are selectable with --mpvit_variant. MonoViT is published on MPViT-**small**
(22.6M), which is the default here; the submission's MonoIIT row was trained on MPViT-**xsmall**
(10.3M), so reproducing it needs `xsmall`. They are different networks with different channel
widths, not a checkpoint detail: loading one into the other silently drops most of the encoder.

Input convention: [0, 1] images, as everywhere else in the repo (monodepth2's dataloader and
MonoViT's own trainer both feed the backbone unnormalised colour).
"""
import torch.nn as nn

from models.monovit.hr_decoder import DepthDecoderT
from models.monovit.mpvit import mpvit_small, mpvit_xsmall

# stem + the four MHCA stages, at H/2 H/4 H/8 H/16 H/32; verified by running each encoder
MPVIT_VARIANTS = {
    "small": (mpvit_small, [64, 128, 216, 288, 288]),
    "xsmall": (mpvit_xsmall, [64, 128, 192, 256, 256]),
}


class MonoViTDepth(nn.Module):
    NUM_CH_ENC = MPVIT_VARIANTS["small"][1]

    def __init__(self, scales=range(4), pretrained_weights=None, variant="small"):
        super(MonoViTDepth, self).__init__()
        if variant not in MPVIT_VARIANTS:
            raise ValueError("unknown --mpvit_variant {}: expected one of {}".format(
                variant, sorted(MPVIT_VARIANTS)))
        build, num_ch_enc = MPVIT_VARIANTS[variant]
        self.variant = variant
        self.encoder = build(pretrained=pretrained_weights)
        self.encoder.num_ch_enc = list(num_ch_enc)
        self.decoder = DepthDecoderT(ch_enc=list(num_ch_enc), scales=list(scales))

    def forward(self, x):
        return self.decoder(self.encoder(x))
