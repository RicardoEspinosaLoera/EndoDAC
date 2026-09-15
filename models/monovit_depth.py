"""MonoViT depth network: MPViT-small encoder + the HR (nested, fSE-attention) decoder.

Trained from scratch inside this repo with --depth_backbone monovit, so MonoViT enters the CVIU
tables as a run of the grid rather than as an external checkpoint. Like models/resnet_depth.py it
wraps encoder+decoder behind the endodac interface (image -> {("disp", s)}), which is all the
trainer and cviu_revision.py's predict stage need.

Input convention: [0, 1] images, as everywhere else in the repo (monodepth2's dataloader and
MonoViT's own trainer both feed the backbone unnormalised colour).
"""
import torch.nn as nn

from models.monovit.hr_decoder import DepthDecoderT
from models.monovit.mpvit import mpvit_small


class MonoViTDepth(nn.Module):
    # stem + the four MHCA stages of mpvit_small, at H/2 H/4 H/8 H/16 H/32
    NUM_CH_ENC = [64, 128, 216, 288, 288]

    def __init__(self, scales=range(4), pretrained_weights=None):
        super(MonoViTDepth, self).__init__()
        self.encoder = mpvit_small(pretrained=pretrained_weights)
        self.encoder.num_ch_enc = list(self.NUM_CH_ENC)
        self.decoder = DepthDecoderT(ch_enc=list(self.NUM_CH_ENC), scales=list(scales))

    def forward(self, x):
        return self.decoder(self.encoder(x))
