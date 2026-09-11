"""Zero-shot DA3-Base single-view depth without the depth_anything_3 package.

Reproduces DepthAnything3Net's depth output for one view with the ported encoder
(da3_vit.DA3Encoder) and DA3's DualDPT main branch: pre-norm, projections + uv pos-embed,
resize layers, fusion, output_conv1, upsampling to the input size, uv pos-embed, output_conv2,
exp. The auxiliary ray branch and the camera decoder are not needed for depth and are skipped.
Source: ByteDance-Seed/Depth-Anything-3 (Apache License 2.0), model/dualdpt.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .da3_vit import DA3Encoder, DA3_HEAD_PREFIX, add_uv_pos_embed, load_matching, read_safetensors
from .endodac import DPTHead


class DA3BaseMono(nn.Module):
    def __init__(self, weights_path=None, process_res=504, patch_size=14):
        super(DA3BaseMono, self).__init__()
        self.process_res = process_res
        self.patch_size = patch_size
        self.encoder = DA3Encoder(weights_path=weights_path)
        self.neck = DPTHead(self.encoder.out_dim, 128, False, [96, 192, 384, 768],
                            pre_norm=True, pos_embed=True)
        self.out = nn.Module()
        self.out.output_conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.out.output_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0))
        if weights_path is not None:
            head = read_safetensors(weights_path, DA3_HEAD_PREFIX)
            load_matching(self.neck, head, "DA3 neck")
            convs = {k[len("scratch."):]: v for k, v in head.items()
                     if k.startswith("scratch.output_conv1.") or k.startswith("scratch.output_conv2.")}
            _, missing, _ = load_matching(self.out, convs, "DA3 output convs")
            if missing:
                raise RuntimeError("DA3 output convs not found in {}: {}".format(weights_path, missing))

    def predict_at(self, x):
        """x: (B,3,H,W) in [0,1], H and W multiples of 14 -> relative depth (B,1,H,W)."""
        B, _, H, W = x.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        feats = self.encoder.get_intermediate_layers(x, 4, return_class_token=True)
        fused = self.out.output_conv1(self.neck.fuse(feats, ph, pw)[0])
        fused = F.interpolate(fused, size=(H, W), mode="bilinear", align_corners=True)
        logits = self.out.output_conv2(add_uv_pos_embed(fused, W, H))
        return torch.exp(logits[:, 0:1])

    def forward(self, img):
        """img: (B,3,h,w) in [0,1] at any size. Processed with its longest side at process_res
        (DA3's default 504), rounded to multiples of 14; depth returned at that size."""
        h, w = img.shape[-2:]
        s = float(self.process_res) / max(h, w)
        H = max(self.patch_size, int(round(h * s / self.patch_size)) * self.patch_size)
        W = max(self.patch_size, int(round(w * s / self.patch_size)) * self.patch_size)
        x = F.interpolate(img, size=(H, W), mode="bilinear", align_corners=False)
        return self.predict_at(x)
