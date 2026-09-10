"""ResNet-18 depth network: the weak-backbone control of the CVIU ablation.

It is the monodepth2 encoder/decoder pair the repo already ships for
AF-SfMLearner checkpoints, wrapped so it has the endodac interface
(image -> {("disp", s)}) and can be swapped in with --depth_backbone resnet18.
"""
import torch.nn as nn

import models.encoders as encoders
import models.decoders as decoders


class ResnetDepth(nn.Module):
    def __init__(self, num_layers=18, pretrained=True, scales=range(4)):
        super(ResnetDepth, self).__init__()
        self.encoder = encoders.ResnetEncoder(num_layers, pretrained)
        self.decoder = decoders.DepthDecoder(self.encoder.num_ch_enc, list(scales))

    def forward(self, x):
        return self.decoder(self.encoder(x))
