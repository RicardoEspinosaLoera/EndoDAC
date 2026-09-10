"""Depth Anything 3 encoder for endodac (--backbone_weights da3).

DA3-Base is a DINOv2 ViT-B/14 whose blocks from `qknorm_start`/`rope_start` on use
QK-normalisation, rotary position embeddings and alternating (multi-view) attention.
Its weights therefore cannot be copied onto the repo's vanilla ViT without changing
what the network computes. This wrapper keeps DA3's own module, loaded through the
`depth_anything_3` package (Apache-2.0), and exposes the interface endodac expects:

  .blocks                                     the transformer blocks, for DV-LoRA insertion
                                              into blocks[i].mlp.fc1 / fc2
  .get_intermediate_layers(x, 4, return_class_token=True)
                                              4 x (patch tokens (B,N,C), class token (B,C))

EndoDAC's Conv-neck (ResBottleneckBlock after selected blocks) is re-created as forward
hooks on those blocks, with the same `residual_` parameter prefix so the trainable-
parameter policy in layers.py applies unchanged. With a single view the alternating
attention reduces to ordinary self-attention. Inputs follow the repo's convention
(images in [0, 1] without ImageNet normalisation), the same as the Depth Anything v1 path.
"""
import torch
import torch.nn as nn

from models.backbones.layers.utils import ResBottleneckBlock


class DA3Encoder(nn.Module):
    def __init__(self, model_id="depth-anything/da3-base", residual_block_indexes=(),
                 out_layers=(2, 5, 8, 11), patch_size=14):
        super(DA3Encoder, self).__init__()
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise ImportError("--backbone_weights da3 needs the depth_anything_3 package: "
                              "git clone https://github.com/ByteDance-Seed/Depth-Anything-3 && "
                              "cd Depth-Anything-3 && pip install -e .") from e
        da3 = DepthAnything3.from_pretrained(model_id)
        self.vit = da3.backbone.pretrained      # DinoVisionTransformer (DA3 flavour), weights loaded
        del da3                                 # DualDPT head and camera modules are not used
        for attr in ("cat_token",):             # plain patch tokens; the class token is read separately
            if hasattr(self.vit, attr):
                setattr(self.vit, attr, False)
        self.out_layers = list(out_layers)
        self.patch_size = patch_size
        self.embed_dim = int(getattr(self.vit, "embed_dim", 768))
        self.blocks = self._flat_blocks(self.vit)
        self._grid = None
        self._token_half = None
        self.residual_ = nn.ModuleDict()
        for i in residual_block_indexes:
            self.residual_[str(i)] = ResBottleneckBlock(in_channels=self.embed_dim, out_channels=self.embed_dim,
                                                        bottleneck_channels=self.embed_dim // 8)
            self.blocks[i].register_forward_hook(self._make_hook(str(i)))
        print("DA3 encoder {}: {} blocks, dim {}, Conv-neck after {}".format(
            model_id, len(self.blocks), self.embed_dim, list(residual_block_indexes)))

    @staticmethod
    def _flat_blocks(vit):
        """Transformer blocks in order, whether or not the ViT chunks them."""
        blocks = []
        for m in vit.blocks:
            if hasattr(m, "mlp"):
                blocks.append(m)
            else:
                blocks += [b for b in m if hasattr(b, "mlp")]
        if not blocks:
            raise RuntimeError("could not find transformer blocks with an .mlp in the DA3 ViT")
        return blocks

    def _make_hook(self, key):
        def hook(module, inputs, out):
            x = out[0] if isinstance(out, tuple) else out
            ph, pw = self._grid
            n = ph * pw
            if x.dim() != 3 or x.shape[1] < n:
                return None
            B, C = x.shape[0], x.shape[2]
            patches = x[:, -n:, :].reshape(B, ph, pw, C).permute(0, 3, 1, 2)
            patches = self.residual_[key](patches).permute(0, 2, 3, 1).reshape(B, n, C)
            x = torch.cat([x[:, :-n, :], x[:, -n:, :] + patches], 1)
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        return hook

    def _split_token(self, f):
        """If the class token is still concatenated to each patch token, keep the token half."""
        C = self.embed_dim
        if f.shape[-1] <= C:
            return f
        if self._token_half is None:
            second_const = (f[:, :, C:] - f[:, :1, C:]).abs().max().item() == 0
            first_const = (f[:, :, :C] - f[:, :1, :C]).abs().max().item() == 0
            self._token_half = "first" if second_const or not first_const else "second"
        return f[..., :C] if self._token_half == "first" else f[..., C:]

    def get_intermediate_layers(self, x, n=4, return_class_token=True, **kwargs):
        B, _, H, W = x.shape
        self._grid = (H // self.patch_size, W // self.patch_size)
        num_patches = self._grid[0] * self._grid[1]
        try:
            feats = self.vit.get_intermediate_layers(x.unsqueeze(1), self.out_layers)  # (B, S=1, 3, H, W)
        except (RuntimeError, ValueError, IndexError):
            feats = self.vit.get_intermediate_layers(x, self.out_layers)
        outs = []
        for f in feats:
            if f.dim() == 4:                       # (B, S, T, C) -> (B, T, C)
                f = f.reshape(B, -1, f.shape[-1])
            f = self._split_token(f)
            if f.shape[1] > num_patches:           # class / register tokens still in the sequence
                cls, f = f[:, 0], f[:, -num_patches:]
            else:
                cls = f.mean(1)
            outs.append((f, cls) if return_class_token else f)
        return tuple(outs)
