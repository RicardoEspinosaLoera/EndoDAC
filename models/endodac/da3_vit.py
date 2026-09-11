"""Depth Anything 3 (DA3-Base) encoder, ported to this repo (Python >= 3.8, torch >= 1.10,
no extra packages).

Source: ByteDance-Seed/Depth-Anything-3, commit 3d835ec (Apache License 2.0),
src/depth_anything_3/model/dinov2/{vision_transformer.py, layers/*.py} and
src/depth_anything_3/model/utils/head_utils.py.
Copyright (c) Meta Platforms, Inc. and affiliates; Copyright (c) 2025 ByteDance Ltd. and/or
its affiliates. Licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0).

Changes with respect to the original, none of which alters the computation for one view:
  * single-view forward (S = 1). DA3 alternates "local" (per view) and "global" (all views)
    attention from block `alt_start` on; with one view both attend over the same tokens and
    differ only in their RoPE positions, which are kept exactly (global blocks use the
    constant "no-diff" positions). Multi-view reference-view selection and user camera
    conditioning are dropped: the reference camera token is injected as in DA3.
  * no einops / DA3 logger / xformers; Python 3.8 syntax; the position cache is keyed by
    device as well.
  * EndoDAC's Conv-neck (ResBottleneckBlock) can be added after selected blocks, applied to
    the patch tokens exactly as in models/backbones/layers/block.py. Its parameters are named
    `residual_.<block>.*`, so the trainable-parameter policy of layers.py applies unchanged.
  * a pure-Python .safetensors reader, so the official checkpoint loads without the
    `safetensors` package.
The port was checked against the original code with shared weights (max abs difference at
float32 round-off level).

Checkpoint: https://huggingface.co/depth-anything/DA3-BASE/resolve/main/model.safetensors
(542 MB, Apache-2.0). Encoder tensors are stored under "model.backbone.pretrained.", the
DualDPT head under "model.head.".
"""
import json
import math
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.layers.utils import ResBottleneckBlock

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DA3_ENCODER_PREFIX = "model.backbone.pretrained."
DA3_HEAD_PREFIX = "model.head."

# --------------------------------------------------------------------------------------
# checkpoint reader
# --------------------------------------------------------------------------------------

_ST_DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64,
              "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def read_safetensors(path, prefix=""):
    """Tensors of a .safetensors file whose name starts with `prefix`, prefix stripped."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
        base = 8 + n
        for name, info in header.items():
            if name == "__metadata__" or not name.startswith(prefix):
                continue
            start, end = info["data_offsets"]
            f.seek(base + start)
            buf = f.read(end - start)
            if info["dtype"] == "BF16":
                t = torch.from_numpy(np.frombuffer(buf, dtype=np.int16).copy()).view(torch.bfloat16).float()
            else:
                t = torch.from_numpy(np.frombuffer(buf, dtype=_ST_DTYPES[info["dtype"]]).copy())
            out[name[len(prefix):]] = t.reshape(info["shape"])
    return out


def load_matching(module, state, what):
    """load_state_dict restricted to keys present in `module` with the same shape."""
    own = module.state_dict()
    ok = {k: v for k, v in state.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    skipped = sorted(k for k in state if k not in ok)
    module.load_state_dict(ok, strict=False)
    missing = sorted(k for k in own if k not in ok)
    print("{}: loaded {} tensors, {} left at init, {} checkpoint tensors unused".format(
        what, len(ok), len(missing), len(skipped)))
    return ok, missing, skipped


# --------------------------------------------------------------------------------------
# head_utils (uv positional embedding used by DA3's DPT heads)
# --------------------------------------------------------------------------------------

def make_sincos_pos_embed(embed_dim, pos, omega_0=100):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0 ** omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def position_grid_to_embed(pos_grid, embed_dim, omega_0=100):
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    return torch.cat([emb_x, emb_y], dim=-1).view(H, W, embed_dim)


def create_uv_grid(width, height, aspect_ratio=None, dtype=None, device=None):
    """(height, width, 2) grid of normalised UV coordinates (DA3 head_utils.create_uv_grid)."""
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def add_uv_pos_embed(x, W, H, ratio=0.1):
    """DualDPT._add_pos_embed: x (B,C,h,w) + ratio * sincos(uv grid); W/H is the image aspect."""
    pw, ph = x.shape[-1], x.shape[-2]
    pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
    pe = position_grid_to_embed(pe, x.shape[1]) * ratio
    pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
    return x + pe


# --------------------------------------------------------------------------------------
# transformer layers
# --------------------------------------------------------------------------------------

class PositionGetter(object):
    """(y, x) patch coordinates, cached per grid size and device."""

    def __init__(self):
        self.position_cache = {}

    def __call__(self, batch_size, height, width, device):
        key = (height, width, str(device))
        if key not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            self.position_cache[key] = torch.cartesian_prod(y_coords, x_coords)
        cached = self.position_cache[key]
        return cached.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, frequency=100.0, scaling_factor=1.0):
        super(RotaryPositionEmbedding2D, self).__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache = {}

    def _compute_frequency_components(self, dim, seq_len, device, dtype):
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[cache_key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x):
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens, positions):
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2
        feature_dim = tokens.size(-1) // 2
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = self._apply_1d_rope(vertical, positions[..., 0], cos_comp, sin_comp)
        horizontal = self._apply_1d_rope(horizontal, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical, horizontal), dim=-1)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=14, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "input {}x{} is not a multiple of the patch size {}".format(H, W, self.patch_size)
        x = self.proj(x)
        return self.norm(x.flatten(2).transpose(1, 2))


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super(LayerScale, self).__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.drop = nn.Dropout(0.0)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, qk_norm=False, rope=None):
        super(Attention, self).__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(0.0)
        self.rope = rope

    def forward(self, x, pos=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        if hasattr(F, "scaled_dot_product_attention"):   # torch >= 2.0
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            x = attn.softmax(dim=-1) @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, proj_bias=True, ffn_bias=True,
                 init_values=1.0, qk_norm=False, rope=None, ln_eps=1e-6):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=ln_eps)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
                              qk_norm=qk_norm, rope=rope)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=ln_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, pos=None):
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos))
        return x + self.ls2(self.mlp(self.norm2(x)))


# --------------------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------------------

class DA3Encoder(nn.Module):
    """DA3-Base DINOv2 encoder with endodac's interface.

    get_intermediate_layers(x, 4, return_class_token=True) returns, for DA3's output layers
    (5, 7, 9, 11), (patch tokens (B, N, 1536), camera token (B, 1536)). As in DA3, each
    output concatenates the last "local" block state (not normalised) with the normalised
    current state; DA3's head applies a LayerNorm over the 1536 channels before its
    projections (DPTHead(pre_norm=True) does the same).

    Inputs are images in [0, 1]; ImageNet normalisation is applied here, as in DA3's own
    preprocessing (the repo's DA v1 path feeds [0, 1] unnormalised, EndoDAC's convention).
    """

    def __init__(self, embed_dim=768, depth=12, num_heads=12, patch_size=14, img_size=518,
                 alt_start=4, qknorm_start=4, rope_start=4, rope_freq=100, cat_token=True,
                 out_layers=(5, 7, 9, 11), residual_block_indexes=(), weights_path=None,
                 normalize_input=True):
        super(DA3Encoder, self).__init__()
        self.embed_dim = self.num_features = embed_dim
        self.out_dim = 2 * embed_dim if cat_token else embed_dim
        self.patch_size = patch_size
        self.alt_start, self.qknorm_start, self.rope_start = alt_start, qknorm_start, rope_start
        self.cat_token = cat_token
        self.out_layers = list(out_layers)
        self.interpolate_offset = 0.1
        self.normalize_input = normalize_input
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        if rope_start != -1 and rope_freq > 0:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
            self.position_getter = PositionGetter()
        else:
            self.rope = None
            self.position_getter = None
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads,
                  qk_norm=(i >= qknorm_start) if qknorm_start != -1 else False,
                  rope=self.rope if (rope_start != -1 and i >= rope_start) else None)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # EndoDAC Conv-neck, same construction as models/backbones/layers/block.py
        self.residual_ = nn.ModuleDict({
            str(i): ResBottleneckBlock(in_channels=embed_dim, out_channels=embed_dim,
                                       bottleneck_channels=embed_dim // 8, act_layer=nn.GELU,
                                       conv_kernels=3, conv_paddings=1)
            for i in residual_block_indexes})

        if weights_path is not None:
            self.load_da3(weights_path)

    def load_da3(self, path):
        state = read_safetensors(path, DA3_ENCODER_PREFIX)
        if not state:
            raise ValueError("no '{}*' tensors in {}".format(DA3_ENCODER_PREFIX, path))
        _, missing, skipped = load_matching(self, state, "DA3 encoder <- " + path)
        missing = [k for k in missing if not k.startswith("residual_.")]
        if missing or skipped:
            raise RuntimeError("DA3 checkpoint does not match the encoder: missing {} unused {}".format(
                missing[:8], skipped[:8]))

    # ---- original DinoVisionTransformer pieces --------------------------------------
    def interpolate_pos_encoding(self, x, H, W):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and H == W:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        h0 = H // self.patch_size
        w0 = W // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        sx = float(h0 + self.interpolate_offset) / M
        sy = float(w0 + self.interpolate_offset) / M
        patch_pos_embed = F.interpolate(patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                                        mode="bicubic", scale_factor=(sx, sy))
        assert (h0, w0) == tuple(patch_pos_embed.shape[-2:])
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def _prepare_rope(self, B, ph, pw, device):
        if self.rope is None:
            return None, None
        pos = self.position_getter(B, ph, pw, device)
        pos_nodiff = torch.zeros_like(pos)
        pos = pos + 1
        special = torch.zeros(B, 1, 2, device=device, dtype=pos.dtype)
        return torch.cat([special, pos], dim=1), torch.cat([special, pos_nodiff + 1], dim=1)

    def _is_global(self, i):
        return self.alt_start != -1 and i >= self.alt_start and i % 2 == 1

    def _conv_neck(self, i, tokens, ph, pw):
        B, N, C = tokens.shape
        patches = tokens[:, 1:].reshape(B, ph, pw, C).permute(0, 3, 1, 2)
        patches = self.residual_[str(i)](patches).permute(0, 2, 3, 1).reshape(B, ph * pw, C)
        return torch.cat([tokens[:, :1], tokens[:, 1:] + patches], dim=1)

    def get_intermediate_layers(self, x, n=4, return_class_token=True, **kwargs):
        if self.normalize_input:
            x = (x - self.img_mean) / self.img_std
        B, _, H, W = x.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        tokens = self.patch_embed(x)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        tokens = tokens + self.interpolate_pos_encoding(tokens, H, W)
        pos, pos_nodiff = self._prepare_rope(B, ph, pw, x.device)

        raw, local_x = [], tokens
        for i, blk in enumerate(self.blocks):
            is_global = self._is_global(i)
            if self.rope is None or i < self.rope_start:
                p = None
            else:
                p = pos_nodiff if is_global else pos
            if self.alt_start != -1 and i == self.alt_start:   # reference-view camera token
                tokens = torch.cat([self.camera_token[:, :1].expand(B, -1, -1), tokens[:, 1:]], dim=1)
            tokens = blk(tokens, pos=p)
            if str(i) in self.residual_:
                tokens = self._conv_neck(i, tokens, ph, pw)
            if not is_global:
                local_x = tokens
            if i in self.out_layers:
                raw.append(torch.cat([local_x, tokens], dim=-1) if self.cat_token else tokens)

        C = self.embed_dim
        outputs = []
        for out in raw:
            cam_token = out[:, 0]
            if out.shape[-1] == 2 * C:
                out = torch.cat([out[..., :C], self.norm(out[..., C:])], dim=-1)
            else:
                out = self.norm(out)
            patches = out[:, 1:]
            outputs.append((patches, cam_token) if return_class_token else patches)
        return tuple(outputs)

    def forward(self, x):
        return self.get_intermediate_layers(x, len(self.out_layers), return_class_token=True)
