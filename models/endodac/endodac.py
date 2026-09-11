import os
import torch
import torch.nn as nn
import models.backbones as backbones
from models.backbones.mylora import Linear as LoraLinear
from models.backbones.mylora import DVLinear as DVLinear
from models.backbones.flora import Linear as FLinear
from models.backbones.dora import Linear as DLinear
from .layers import HeadDepth
from .layers import mark_only_part_as_trainable,_make_scratch, _make_fusion_block
from .da3_vit import add_uv_pos_embed

def _copy_linear(old, new):
    """Copy weight/bias of a pretrained nn.Linear into its LoRA replacement (same shapes)."""
    with torch.no_grad():
        if getattr(old, "weight", None) is not None and new.weight.shape == old.weight.shape:
            new.weight.copy_(old.weight)
        if getattr(old, "bias", None) is not None and getattr(new, "bias", None) is not None \
                and new.bias.shape == old.bias.shape:
            new.bias.copy_(old.bias)


class DPTHead(nn.Module):
    """DPT neck + EndoDAC multi-scale disparity heads.

    pre_norm / pos_embed reproduce DA3's DualDPT main branch (LayerNorm over the tokens before
    the projections, uv sin-cos embedding added after them), so a DA3 encoder can use DA3's
    trained neck unchanged. Both default to off, which is the Depth Anything v1 / EndoDAC head.
    """
    def __init__(self, in_channels, features=128, use_bn=False, out_channels=[96, 192, 384, 768], use_clstoken=False,
                 pre_norm=False, pos_embed=False):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        self.norm = nn.LayerNorm(in_channels) if pre_norm else None
        self.pos_embed = pos_embed

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.conv_depth_1 = HeadDepth(features)
        self.conv_depth_2 = HeadDepth(features)
        self.conv_depth_3 = HeadDepth(features)
        self.conv_depth_4 = HeadDepth(features)
        
        self.sigmoid = nn.Sigmoid()
    def fuse(self, out_features, patch_h, patch_w):
        """Token features -> fused DPT pyramid (path_1 finest ... path_4 coarsest)."""
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            if self.norm is not None:
                x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            if self.pos_embed:
                x = add_uv_pos_embed(x, patch_w, patch_h)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        return path_1, path_2, path_3, path_4

    def forward(self, out_features, patch_h, patch_w):
        path_1, path_2, path_3, path_4 = self.fuse(out_features, patch_h, patch_w)

        outputs = {}
        outputs[("disp", 3)] = self.sigmoid(self.conv_depth_4(path_4))
        outputs[("disp", 2)] = self.sigmoid(self.conv_depth_3(path_3))
        outputs[("disp", 1)] = self.sigmoid(self.conv_depth_2(path_2))
        outputs[("disp", 0)] = self.sigmoid(self.conv_depth_1(path_1))

        return outputs
    
class endodac(nn.Module):
    """Applies low-rank adaptation to a ViT model's image encoder.

    Args:
        backbone_size: size of pretrained Dinov2 choice from: "small", "base", "large", "giant"
        r: rank of LoRA
        image_shape: input image shape, h,w need to be multiplier of 14, default:(224,280)
        lora_layer: which layer we apply LoRA.
    """

    def __init__(self, 
                 backbone_size = "base", 
                 r=4, 
                 image_shape=(224,280), 
                 lora_type="lora",
                 pretrained_path=None,
                 residual_block_indexes=[],
                 include_cls_token=True,
                 use_cls_token=False,
                 use_bn=False,
                 backbone_weights="da1",
                 da3_weights=None,
                 train_depth_head=False):
        """backbone_weights: "da1" Depth Anything v1 encoder + DPT head (EndoDAC / MonoIIF);
        "da3" DA3-Base encoder + DA3's own DPT neck (models/endodac/da3_vit.py; weights from
        da3_weights, default <pretrained_path>/da3_base.safetensors); "none" random ViT-B with the
        DA v1 head, whole head trained. pretrained_path=None skips every pretrained load (used
        when a trained checkpoint is loaded afterwards)."""
        super(endodac, self).__init__()

        assert r > 0
        self.r = r
        self.backbone_size = backbone_size
        self.backbone = {
            "small": backbones.vits.vit_small(residual_block_indexes=residual_block_indexes,
                                              include_cls_token=include_cls_token),
            "base": backbones.vits.vit_base(residual_block_indexes=residual_block_indexes,
                                            include_cls_token=include_cls_token),
        }
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
        }
        self.depth_head_features = {
            "small": 64,
            "base": 128,
        }
        self.depth_head_out_channels = {
            "small": [48, 96, 192, 384],
            "base": [96, 192, 384, 768],
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        self.depth_head_feature = self.depth_head_features[self.backbone_size]
        self.depth_head_out_channel = self.depth_head_out_channels[self.backbone_size]
        if backbone_weights not in ("da1", "da3", "none"):
            raise ValueError("backbone_weights must be da1, da3 or none, got {}".format(backbone_weights))
        self.backbone_weights = backbone_weights
        head_in_channels, head_kwargs = self.embedding_dim, {}
        if backbone_weights == "da3":
            # DA3-Base's DINOv2 (QK-norm, 2D RoPE, camera token from block 4), ported in da3_vit.py.
            # Its features are 1536-d ([local | global] states) at DA3's layers 5/7/9/11, consumed by
            # DA3's own neck (pre-norm + uv pos-embed), loaded below.
            from .da3_vit import DA3Encoder
            if da3_weights is None and pretrained_path is not None:
                da3_weights = os.path.join(pretrained_path, "da3_base.safetensors")
            if pretrained_path is not None and not os.path.exists(da3_weights):
                raise FileNotFoundError(
                    "DA3-Base weights not found: {} (wget -O {} "
                    "https://huggingface.co/depth-anything/DA3-BASE/resolve/main/model.safetensors)".format(
                        da3_weights, da3_weights))
            encoder = DA3Encoder(residual_block_indexes=residual_block_indexes,
                                 weights_path=da3_weights if pretrained_path is not None else None)
            head_in_channels, head_kwargs = encoder.out_dim, {"pre_norm": True, "pos_embed": True}
        else:
            encoder = self.backbone[self.backbone_size]

        self.image_shape = image_shape

        if lora_type != "none":
            for t_layer_i, blk in enumerate(encoder.blocks):
                old_fc1, old_fc2 = blk.mlp.fc1, blk.mlp.fc2
                mlp_in_features = blk.mlp.fc1.in_features
                mlp_hidden_features = blk.mlp.fc1.out_features
                mlp_out_features = blk.mlp.fc2.out_features
                if lora_type == "dvlora":
                    blk.mlp.fc1 = DVLinear(mlp_in_features, mlp_hidden_features, r=self.r, lora_alpha=self.r)
                    blk.mlp.fc2 = DVLinear(mlp_hidden_features, mlp_out_features, r=self.r, lora_alpha=self.r)
                elif lora_type == "lora":
                    blk.mlp.fc1 = LoraLinear(mlp_in_features, mlp_hidden_features, r=self.r)
                    blk.mlp.fc2 = LoraLinear(mlp_hidden_features, mlp_out_features, r=self.r)
                elif lora_type == "flora":
                    blk.mlp.fc1 = FLinear(mlp_in_features, mlp_hidden_features, r=self.r)
                    blk.mlp.fc2 = FLinear(mlp_hidden_features, mlp_out_features, r=self.r)
                elif lora_type == "dora":
                    blk.mlp.fc1 = DLinear(mlp_in_features, mlp_hidden_features, r=self.r)
                    blk.mlp.fc2 = DLinear(mlp_hidden_features, mlp_out_features, r=self.r)
                else:
                    raise ValueError(f"unknown lora_type '{lora_type}'; expected dvlora, lora, flora, dora or none")
                # keep the base weights an already-loaded encoder (DA3) carries; for the DA v1
                # path they are overwritten by load_state_dict below anyway
                _copy_linear(old_fc1, blk.mlp.fc1)
                _copy_linear(old_fc2, blk.mlp.fc2)

        self.encoder = encoder
        self.depth_head = DPTHead(head_in_channels, self.depth_head_feature, use_bn, out_channels=self.depth_head_out_channel,
                                  use_clstoken=use_cls_token, **head_kwargs)

        if pretrained_path is not None and backbone_weights == "da3":
            # DA3's neck (pre-norm, projections, resize layers, fusion blocks); EndoDAC's
            # conv_depth heads have no DA3 counterpart and start from init, as with DA v1
            from .da3_vit import read_safetensors, load_matching, DA3_HEAD_PREFIX
            load_matching(self.depth_head, read_safetensors(da3_weights, DA3_HEAD_PREFIX),
                          "DA3 neck <- " + da3_weights)
        elif pretrained_path is not None:
            pretrained_path = os.path.join(pretrained_path, "depth_anything_{}.pth".format(self.backbone_arch))
            if os.path.exists(pretrained_path):
                pretrained_dict = torch.load(pretrained_path, map_location="cpu")
                if backbone_weights == "none":
                    # only the DPT head is taken from Depth Anything v1; the encoder stays random
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith("depth_head.")}
                self.load_state_dict(pretrained_dict, strict=False)
                print("load pretrained weight from {} ({} keys)\n".format(pretrained_path, len(pretrained_dict)))
            elif backbone_weights == "da1":
                raise FileNotFoundError("Depth Anything v1 weights not found: {}".format(pretrained_path))
            else:
                print("no {}: DPT head starts from random init".format(pretrained_path))

        # EndoDAC freezes the DPT neck (only the conv_depth heads train). That is right when the
        # neck was trained on the same encoder's features (da1, and da3 with DA3's neck); with a
        # random encoder the DA v1 neck no longer matches, so the whole head trains.
        self.train_depth_head = bool(train_depth_head) or backbone_weights == "none"
        mark_only_part_as_trainable(self.encoder)
        mark_only_part_as_trainable(self.depth_head)
        if self.train_depth_head:
            for p in self.depth_head.parameters():
                p.requires_grad = True
    def forward(self, pixel_values):
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=self.image_shape, mode="bilinear", align_corners=True)
        h, w = pixel_values.shape[-2:]
        
        features = self.encoder.get_intermediate_layers(pixel_values, 4, return_class_token=True)
        patch_h, patch_w = h // 14, w // 14

        disp = self.depth_head(features, patch_h, patch_w)

        return disp