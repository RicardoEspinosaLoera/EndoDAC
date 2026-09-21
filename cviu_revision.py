#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cviu_revision.py: every new experiment of the MonoIIF CVIU revision, in one script.

Stages (each is idempotent: finished work is skipped unless --force is given):

  train         launch the ablation grid (E/R/C runs x seeds) as train_end_to_end.py subprocesses
  predict       per-frame depth metrics for every trained run and external method -> per_frame.csv
  illum-fit     model-free test of the affine illumination model on ground-truth geometry (SCARED)
  illum-params  statistics of the learned (c, b) calibration maps of a trained run
  illum-sens    response of the calibration to synthetic gain/bias, and depth robustness to it
  illum-grad    how much gradient each loss term gives to the illumination calibration
  da3           zero-shot Depth Anything 3 rows
  stats         sequence-level aggregation, bootstrap CIs, paired tests, LaTeX tables
  report        markdown report and figures
  all           predict, illum-fit, illum-params, illum-sens, da3, stats, report (not train)
  full          train, then everything in `all` on the finished runs (failed runs are listed in the report)

Examples:
  python cviu_revision.py full --gpus 0 1 2 3          # the whole revision in one go (run it in tmux)
  python cviu_revision.py train --gpus 0 1 2 3 --dry_run
  python cviu_revision.py train --gpus 0 --only E3 E8 --extra_flags "--num_epochs 1"
  python cviu_revision.py predict --datasets scared
  python cviu_revision.py stats && python cviu_revision.py report

The rationale of every experiment is in CVIU_REVISION_PLAN.md. All settings live in
cviu_config.yaml (missing keys fall back to DEFAULT_CONFIG below).
"""
from __future__ import absolute_import, division, print_function

import argparse
import collections
import copy
import csv
import glob
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

try:  # stats/report run without torch; everything else needs it
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(ROOT, "splits", "endovis")
METRICS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
LOWER_IS_BETTER = {"abs_rel": True, "sq_rel": True, "rmse": True, "rmse_log": True,
                   "a1": False, "a2": False, "a3": False}
MIN_DEPTH = 1e-3
MAX_DEPTH = {"scared": 150.0, "hamlyn": 150.0, "c3vd": 100.0}
H, W = 256, 320            # training / evaluation resolution of the repo
DEPTH_RANGE = (0.1, 150.0)  # --min_depth / --max_depth defaults of options.py

# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "data": {"scared": "/mnt/data-hdd2/Beilei/Dataset/SCARED",
             "hamlyn": "/mnt/data-hdd2/Beilei/Dataset/Hamlyn",
             "c3vd": "/mnt/data-hdd2/Beilei/Dataset/C3VD"},
    "log_dir": "./logs",
    "out_dir": "./results/cviu",
    "pretrained_path": "./pretrained_model",
    # ImageNet MPViT-small checkpoint for the MonoViT rows, if it is not already in
    # <pretrained_path>/mpvit_small.pth (the official download link is dead, see MPVIT_URL)
    "mpvit_weights": None,
    "python": sys.executable,
    "num_workers": 4,
    "train": {
        "common_flags": "--num_epochs 20 --batch_size 8 --learn_intrinsics True --wandb_mode offline",
        "seeds": [314],
        "multi_seeds": [314, 1, 2],
        # The M-grid is the published method and carries its claim. E4 (calibration only, lambda1=0)
        # is its ablation arm. E5/E7 stay single-seed: at lambda1=0.1 they are sensitivity points,
        # not the method.
        "multi_seed_runs": ["E3", "E4", "E7", "E8", "C0", "C1", "R1", "R2", "D3",
                            "M-local", "M-none", "M-global", "lam-da-025", "lam-da-100", "lam-da-200",
                            "E8-DVLoRA", "E8-IIF", "C2-sup", "C1-lora",
                            "da-bas-0", "da-bas-1", "da-bas-2",
                            "da-lam-010", "da-lam-025", "da-lam-050",
                            "MonoII", "MonoViT", "MonoViT-II",
                            "MonoII-none", "MonoII-glob",
                            "A-MonoII-none", "A-MonoII-glob", "A-MonoII",
                            "lam-res-000", "lam-res-010", "lam-res-025", "lam-res-100", "lam-res-200",
                  "bas-res-0", "bas-res-1", "bas-res-2", "bas-res-3",
                  "lam-bas-000", "lam-bas-010", "lam-bas-025", "lam-bas-100", "lam-bas-200",
                  "MonoII-ssim", "bas-res-2-ssim",
                  "lam-ssim-010", "lam-ssim-025", "lam-ssim-100", "lam-ssim-200",
                            "bas-res-0", "bas-res-1", "bas-res-2", "bas-res-3",
                            "lam-bas-000", "lam-bas-010", "lam-bas-025", "lam-bas-100", "lam-bas-200",
                            "MonoII-ssim", "bas-res-2-ssim",
                            "lam-ssim-010", "lam-ssim-025", "lam-ssim-100", "lam-ssim-200"],
        "checkpoint": "best",
        "runs": {},
        "skip_runs": [],
    },
    "proposed": "E8",
    "datasets": ["scared", "hamlyn", "c3vd"],
    "save_pred": True,
    "methods": {},
    "da3": {"models": ["da3-base", "depth-anything/da3mono-large"], "affine_rows": True},
    "illum": {"runs": ["E8"], "seed": 314, "fit_depth_run": "E3", "sequences": ["sequence1", "sequence2"],
              "patch_sizes": [64, 32, 16, 8], "alpha": 0.10, "beta": 0.05, "ridge": 0.01,
              # the perturbations must stay inside what the decoder can express (c in 1+-alpha,
              # b in +-beta), otherwise the ideal response is unreachable and the fitted slope
              # is flattened by clipping rather than by the module ignoring illumination
              "sens_stride": 5, "gains": [0.95, 0.975, 1.0, 1.025, 1.05],
              "biases": [-0.03, -0.015, 0.0, 0.015, 0.03]},
    "stats": {"n_boot": 10000, "alpha": 0.05, "paired_metrics": ["abs_rel", "sq_rel", "rmse", "a1"],
              "table_methods": None},
}

# The training grid of CVIU_REVISION_PLAN.md section 3. Flags come after common_flags.
GRID = {
    # E-grid: component build-up and leave-one-out
    "E1": {"group": "E", "desc": "DA frozen + DPT output heads, standard loss",
           "flags": "--lora_type none --residual_block_indexes --illum_calib none --illumination_invariant 0 --photometric standard"},
    "E2": {"group": "E", "desc": "E1 + Conv-neck residual blocks",
           "flags": "--lora_type none --illum_calib none --illumination_invariant 0 --photometric standard"},
    "E3": {"group": "E", "desc": "E2 + DV-LoRA (EndoDAC baseline)",
           "flags": "--illum_calib none --illumination_invariant 0 --photometric standard"},
    "E4": {"group": "E", "desc": "E3 + local calibration",
           "flags": "--illumination_invariant 0 --photometric standard"},
    "E5": {"group": "E", "desc": "E3 + IIF loss",
           "flags": "--illum_calib none --photometric standard"},
    "E6": {"group": "E", "desc": "E3 + highlight-aware loss",
           "flags": "--illum_calib none --illumination_invariant 0"},
    "E7": {"group": "E", "desc": "E3 + calibration + IIF",
           "flags": "--photometric standard"},
    "E8": {"group": "E", "desc": "MonoIIF (full)", "flags": ""},
    "E8-IIF": {"group": "E", "desc": "full minus IIF", "flags": "--illumination_invariant 0"},
    "C0": {"group": "EC", "desc": "full minus calibration (no lighting model)", "flags": "--illum_calib none"},
    "E8-DVLoRA": {"group": "E", "desc": "full with plain LoRA", "flags": "--lora_type lora"},
    # R-grid: same losses on a weak backbone
    "R1": {"group": "R", "desc": "ResNet-18 + standard loss (= Monodepth2 under this recipe)",
           "flags": "--depth_backbone resnet18 --illum_calib none --illumination_invariant 0 --photometric standard"},
    "R2": {"group": "R", "desc": "ResNet-18 + calibration + IIF + highlight",
           "flags": "--depth_backbone resnet18"},
    # B-grid: the competing architectures, trained here from scratch under the SHARED recipe
    # (plan section 12) -- same pose net, same splits, same 20 epochs, same learned intrinsics,
    # each method's own published loss. Their difference against E3/M-local is then the depth
    # network, not the training protocol. MonoII and MonoViT-II carry the method's two components
    # (local affine calibration + II loss at the published lambda1 = 0.5, monodepth2 photometric
    # loss); R2 is NOT MonoII: it uses lambda1 = 0.1 and HADepth's highlight term.
    "MonoII": {"group": "B", "desc": "MonoII: ResNet-18 + local calibration + II (lambda1=0.5)",
               "flags": "--depth_backbone resnet18 --photometric standard --illumination_invariant 0.5"},
    # AM-grid: the R2 question (none / global / local affine calibration) as a full factorial with
    # the colour-augmentation defect of section 9, on the cheapest backbone. ResNet-18 trains in
    # ~4-6 h instead of ~10.5, so 3 calibrations x 2 augmentations x 3 seeds costs about what six
    # foundation-backbone runs cost. Everything else is the published recipe: II at lambda1 = 0.5,
    # monodepth2 photometric loss. MonoII is the {local, broken} cell.
    "MonoII-none": {"group": "AM", "desc": "ResNet-18, no calibration, II (0.5)",
                    "flags": "--depth_backbone resnet18 --illum_calib none --photometric standard "
                             "--illumination_invariant 0.5"},
    "MonoII-glob": {"group": "AM", "desc": "ResNet-18, global affine calibration, II (0.5)",
                    "flags": "--depth_backbone resnet18 --illum_calib global --photometric standard "
                             "--illumination_invariant 0.5"},
    "A-MonoII-none": {"group": "AM", "desc": "ResNet-18, no calibration, II (0.5), consistent jitter",
                      "flags": "--depth_backbone resnet18 --illum_calib none --photometric standard "
                               "--illumination_invariant 0.5 --color_aug_consistent True"},
    "A-MonoII-glob": {"group": "AM", "desc": "ResNet-18, global calibration, II (0.5), consistent jitter",
                      "flags": "--depth_backbone resnet18 --illum_calib global --photometric standard "
                               "--illumination_invariant 0.5 --color_aug_consistent True"},
    "A-MonoII": {"group": "AM", "desc": "MonoII (local calibration), consistent jitter",
                 "flags": "--depth_backbone resnet18 --photometric standard "
                          "--illumination_invariant 0.5 --color_aug_consistent True"},
    # bas-res grid: the calibration as ONE family parameterised by capacity, on ResNet-18.
    # degree 0 is the global model, the dense LightingDecoder is the limit, and MonoII /
    # MonoII-none supply the two ends of the curve. Everything else is the published recipe.
    "bas-res-0": {"group": "bas-res", "desc": "ResNet-18 + basis calibration, degree 0 (k=1, constant = global)",
                    "flags": "--depth_backbone resnet18 --photometric standard "
                             "--illumination_invariant 0.5 --illum_calib basis "
                             "--illum_basis_degree 0"},
    "bas-res-1": {"group": "bas-res", "desc": "ResNet-18 + basis calibration, degree 1 (k=3, linear gradient)",
                    "flags": "--depth_backbone resnet18 --photometric standard "
                             "--illumination_invariant 0.5 --illum_calib basis "
                             "--illum_basis_degree 1"},
    "bas-res-2": {"group": "bas-res", "desc": "ResNet-18 + basis calibration, degree 2 (k=6, quadratic, vignetting)",
                    "flags": "--depth_backbone resnet18 --photometric standard "
                             "--illumination_invariant 0.5 --illum_calib basis "
                             "--illum_basis_degree 2"},
    "bas-res-3": {"group": "bas-res", "desc": "ResNet-18 + basis calibration, degree 3 (k=10, cubic)",
                    "flags": "--depth_backbone resnet18 --photometric standard "
                             "--illumination_invariant 0.5 --illum_calib basis "
                             "--illum_basis_degree 3"},
    # lam-bas grid: lambda1 swept on the BASIS calibration (degree 2), the other arm of the
    # cross through (degree 2, lambda1 = 0.5) = bas-res-2. lam-res-* sweeps the same weight
    # on the dense map instead, which audits the paper's Table 2; this one finds the weight
    # that suits the constrained field.
    "lam-bas-000": {"group": "lam-bas", "desc": "ResNet-18 + basis calibration (degree 2), II lambda1=0",
                     "flags": "--depth_backbone resnet18 --photometric standard "
                              "--illum_calib basis --illum_basis_degree 2 "
                              "--illumination_invariant 0"},
    "lam-bas-010": {"group": "lam-bas", "desc": "ResNet-18 + basis calibration (degree 2), II lambda1=0.1",
                     "flags": "--depth_backbone resnet18 --photometric standard "
                              "--illum_calib basis --illum_basis_degree 2 "
                              "--illumination_invariant 0.1"},
    "lam-bas-025": {"group": "lam-bas", "desc": "ResNet-18 + basis calibration (degree 2), II lambda1=0.25",
                     "flags": "--depth_backbone resnet18 --photometric standard "
                              "--illum_calib basis --illum_basis_degree 2 "
                              "--illumination_invariant 0.25"},
    "lam-bas-100": {"group": "lam-bas", "desc": "ResNet-18 + basis calibration (degree 2), II lambda1=1.0",
                     "flags": "--depth_backbone resnet18 --photometric standard "
                              "--illum_calib basis --illum_basis_degree 2 "
                              "--illumination_invariant 1.0"},
    "lam-bas-200": {"group": "lam-bas", "desc": "ResNet-18 + basis calibration (degree 2), II lambda1=2.0",
                     "flags": "--depth_backbone resnet18 --photometric standard "
                              "--illum_calib basis --illum_basis_degree 2 "
                              "--illumination_invariant 2.0"},
    # II-comparator check: the paper's Eq. (14)-(15) compare the descriptor images with SSIM,
    # but --iif_loss defaults to l2 (0.25*||u_p-u_t||^2, from the 2026-09-09 review) and every
    # run of this study used it. Same two cells at the published lambda1 = 0.5 with the
    # published comparator: if they tie their l2 twins, the comparator is not what hurts.
    "MonoII-ssim": {"group": "ssim", "desc": "MonoII with the paper's SSIM_II comparator",
                    "flags": "--depth_backbone resnet18 --photometric standard "
                             "--illumination_invariant 0.5 --iif_loss ssim"},
    "bas-res-2-ssim": {"group": "ssim", "desc": "basis degree 2 + II (0.5) with the SSIM_II comparator",
                       "flags": "--depth_backbone resnet18 --photometric standard --illum_calib basis "
                                "--illum_basis_degree 2 --illumination_invariant 0.5 --iif_loss ssim"},
    # lam-ssim grid: the same lambda1 sweep as lam-bas-*, with the paper's SSIM_II comparator
    # (Eq. 14-15) instead of the l2 that every other run used. Same structure throughout:
    # ResNet-18, basis calibration degree 2, monodepth2 photometric loss. lambda1 = 0 needs no
    # run of its own -- the trainer skips the II branch entirely at weight 0, so lam-bas-000 is
    # the shared origin of both sweeps -- and 0.5 is bas-res-2-ssim.
    "lam-ssim-010": {"group": "lam-ssim", "desc": "basis degree 2 + II (SSIM comparator), lambda1=0.1",
                      "flags": "--depth_backbone resnet18 --photometric standard "
                               "--illum_calib basis --illum_basis_degree 2 --iif_loss ssim "
                               "--illumination_invariant 0.1"},
    "lam-ssim-025": {"group": "lam-ssim", "desc": "basis degree 2 + II (SSIM comparator), lambda1=0.25",
                      "flags": "--depth_backbone resnet18 --photometric standard "
                               "--illum_calib basis --illum_basis_degree 2 --iif_loss ssim "
                               "--illumination_invariant 0.25"},
    "lam-ssim-100": {"group": "lam-ssim", "desc": "basis degree 2 + II (SSIM comparator), lambda1=1.0",
                      "flags": "--depth_backbone resnet18 --photometric standard "
                               "--illum_calib basis --illum_basis_degree 2 --iif_loss ssim "
                               "--illumination_invariant 1.0"},
    "lam-ssim-200": {"group": "lam-ssim", "desc": "basis degree 2 + II (SSIM comparator), lambda1=2.0",
                      "flags": "--depth_backbone resnet18 --photometric standard "
                               "--illum_calib basis --illum_basis_degree 2 --iif_loss ssim "
                               "--illumination_invariant 2.0"},
    # da-grid: the clean recipe on the paper's own backbone -- Depth Anything v1 + DV-LoRA with
    # monodepth2's photometric loss, i.e. WITHOUT HADepth's highlight term, which is not part of
    # the method. Two axes crossing at (basis degree 1, lambda1 = 0):
    #   capacity  : E3 (no calibration) - da-bas-0 (global) - da-bas-1 - da-bas-2 - E4 (dense)
    #   II weight : da-bas-1 (0) - da-lam-010 - da-lam-025 - da-lam-050
    # E3 and E4 already exist and are reused; E4 gains its two missing seeds.
    "da-bas-0": {"group": "da", "desc": "DA v1 + basis calibration degree 0 (global), no II loss",
                 "flags": "--photometric standard --illumination_invariant 0 "
                          "--illum_calib basis --illum_basis_degree 0"},
    "da-bas-1": {"group": "da", "desc": "DA v1 + basis calibration degree 1, no II loss",
                 "flags": "--photometric standard --illumination_invariant 0 "
                          "--illum_calib basis --illum_basis_degree 1"},
    "da-bas-2": {"group": "da", "desc": "DA v1 + basis calibration degree 2, no II loss",
                 "flags": "--photometric standard --illumination_invariant 0 "
                          "--illum_calib basis --illum_basis_degree 2"},
    "da-lam-010": {"group": "da", "desc": "DA v1 + basis degree 1 + II loss, lambda1=0.1",
                   "flags": "--photometric standard --illumination_invariant 0.1 "
                            "--illum_calib basis --illum_basis_degree 1"},
    "da-lam-025": {"group": "da", "desc": "DA v1 + basis degree 1 + II loss, lambda1=0.25",
                   "flags": "--photometric standard --illumination_invariant 0.25 "
                            "--illum_calib basis --illum_basis_degree 1"},
    "da-lam-050": {"group": "da", "desc": "DA v1 + basis degree 1 + II loss, lambda1=0.5",
                   "flags": "--photometric standard --illumination_invariant 0.5 "
                            "--illum_calib basis --illum_basis_degree 1"},
    # lam-res grid: sweep of lambda1, the WEIGHT OF THE II LOSS (--illumination_invariant,
    # eq. 18 of the paper), on the ResNet-18 backbone. Not the learning rate, i.e. an audit of the paper's own
    # Table 2, which swept lambda1 there with ONE seed and frame-level means and then
    # applied the resulting 0.5 to all three variants including MonoIIF. Structure fixed to
    # MonoII (local calibration, monodepth2 photometric loss); lambda1 = 0.5 is MonoII itself.
    "lam-res-000": {"group": "lam-res", "desc": "ResNet-18 + local calibration, II lambda1=0",
             "flags": "--depth_backbone resnet18 --photometric standard --illumination_invariant 0"},
    "lam-res-010": {"group": "lam-res", "desc": "ResNet-18 + local calibration, II lambda1=0.1",
             "flags": "--depth_backbone resnet18 --photometric standard --illumination_invariant 0.1"},
    "lam-res-025": {"group": "lam-res", "desc": "ResNet-18 + local calibration, II lambda1=0.25",
             "flags": "--depth_backbone resnet18 --photometric standard --illumination_invariant 0.25"},
    "lam-res-100": {"group": "lam-res", "desc": "ResNet-18 + local calibration, II lambda1=1.0",
             "flags": "--depth_backbone resnet18 --photometric standard --illumination_invariant 1.0"},
    "lam-res-200": {"group": "lam-res", "desc": "ResNet-18 + local calibration, II lambda1=2.0",
             "flags": "--depth_backbone resnet18 --photometric standard --illumination_invariant 2.0"},
    "MonoViT": {"group": "B", "desc": "MonoViT: MPViT-small + HR decoder, standard loss",
                "flags": "--depth_backbone monovit --illum_calib none --illumination_invariant 0 "
                         "--photometric standard"},
    "MonoViT-II": {"group": "B", "desc": "MonoViT + local calibration + II (lambda1=0.5)",
                   "flags": "--depth_backbone monovit --photometric standard --illumination_invariant 0.5"},
    # The submission's MonoIIT row was trained on MPViT-**xsmall**, not the MPViT-small MonoViT is
    # published with, so these two reproduce it. They mirror R2/R1 on ResNet-18 and E8/E3 on Depth
    # Anything: MonoIIT carries the method's two components at the repo defaults (local calibration,
    # II at lambda1 = 0.1, HADepth's photometric term) and MonoViT-xs is the same network with
    # neither, so the pair isolates the components and the column isolates the architecture.
    # They are NOT MonoViT as published -- that is `MonoViT`/`MonoViT-II` above, still blocked on
    # the MPViT-small weights -- and the paper must say which encoder size it used.
    "MonoViT-xs": {"group": "B", "desc": "MPViT-xsmall + HR decoder, standard loss (plain twin of MonoIIT)",
                   "flags": "--depth_backbone monovit --mpvit_variant xsmall --illum_calib none "
                            "--illumination_invariant 0 --photometric standard"},
    "MonoIIT": {"group": "B", "desc": "MonoIIT: MPViT-xsmall + local calibration + II, repo defaults",
                "flags": "--depth_backbone monovit --mpvit_variant xsmall"},
    # bas-res-2-ssim on the third backbone: whether the best C3VD point of the ResNet family
    # (0.3282, basis degree 2 + II at 0.5 with the paper's SSIM_II comparator) is a property of
    # that cell or of ResNet-18. Its twins are MonoViT-II (same weight, dense map, l2 comparator)
    # and MonoViT (neither component).
    "bas-vit-2-ssim": {"group": "vit-ssim", "desc": "MonoViT + basis degree 2 + II (0.5), SSIM_II comparator",
                       "flags": "--depth_backbone monovit --photometric standard --illum_calib basis "
                                "--illum_basis_degree 2 --illumination_invariant 0.5 --iif_loss ssim"},
    # C-grid: illumination model
    "C1": {"group": "C", "desc": "global affine calibration", "flags": "--illum_calib global"},
    # diagnostic: does the calibration behave once it is supervised with its least-squares fit?
    "C2-sup": {"group": "C", "desc": "MonoIIF with the calibration supervised by the LS fit",
               "flags": "--calib_supervision 0.05"},
    # M-grid: MonoIIF EXACTLY as published (CVIU submission, sections 2.2.5 and 3.3): EndoDAC's
    # adapted backbone + the per-pixel affine calibration + the II loss at lambda1 = 0.5, with
    # monodepth2's photometric loss. The highlight-aware term of E8/C0/C1 is HADepth's and is not
    # part of the method, and every E/C/D/R run above used the repo default lambda1 = 0.1, which is
    # BELOW the whole range the paper's Table 2 sweeps ({0.25 ... 10}, optimum 0.5). No run of that
    # grid is therefore the published method; these are.
    "M-local": {"group": "M", "desc": "MonoIIF as published: local calibration + II (lambda1=0.5)",
                "flags": "--photometric standard --illumination_invariant 0.5"},
    "M-none": {"group": "M", "desc": "II (lambda1=0.5) only, no calibration",
               "flags": "--illum_calib none --photometric standard --illumination_invariant 0.5"},
    "M-global": {"group": "M", "desc": "global calibration + II (lambda1=0.5)",
                 "flags": "--illum_calib global --photometric standard --illumination_invariant 0.5"},
    # L-grid: lambda1 sweep for the II loss on the FOUNDATION backbone. The paper's Table 2 sweeps
    # lambda1 on the ResNet variant, single seed, frame-level; MonoIIF has no sweep at all. Together
    # with E4 (lambda1=0), E7 (0.1) and M-local (0.5) these give six points, 3 seeds each, all with
    # the method's structure (local calibration, monodepth2 photometric loss).
    "lam-da-025": {"group": "lam-da", "desc": "local calibration + II (lambda1=0.25)",
             "flags": "--photometric standard --illumination_invariant 0.25"},
    "lam-da-100": {"group": "lam-da", "desc": "local calibration + II (lambda1=1.0)",
             "flags": "--photometric standard --illumination_invariant 1.0"},
    "lam-da-200": {"group": "lam-da", "desc": "local calibration + II (lambda1=2.0)",
             "flags": "--photometric standard --illumination_invariant 2.0"},
    # the two settings that won their own comparison, combined
    "C1-lora": {"group": "C", "desc": "global calibration + plain LoRA",
                "flags": "--illum_calib global --lora_type lora"},
    # A-grid: the whole grid above was trained with a colour jitter re-sampled per frame, so the
    # frames of one item carry different illumination and the pose/lighting heads read noise
    # (datasets/mono_dataset.py::_sample_color_aug). These three repeat the calibration
    # comparison with the same jitter for every frame: if the learned maps start tracking
    # illumination (illum-sens slope -> 1) the module was starved of signal, not ill-posed.
    "A-C0": {"group": "A", "desc": "no calibration, consistent colour jitter",
             "flags": "--illum_calib none --color_aug_consistent True"},
    "A-C1": {"group": "A", "desc": "global calibration, consistent colour jitter",
             "flags": "--illum_calib global --color_aug_consistent True"},
    "A-E8": {"group": "A", "desc": "MonoIIF (local calibration), consistent colour jitter",
             "flags": "--color_aug_consistent True"},
    # D-grid: Depth Anything 3 encoder (its own DinoV2 with QK-norm/RoPE) under the same recipe
    "D3": {"group": "D", "desc": "MonoIIF with DA3-Base encoder", "flags": "--backbone_weights da3"},
    "D3-EndoDAC": {"group": "D", "desc": "EndoDAC recipe with DA3-Base encoder",
                   "flags": "--backbone_weights da3 --illum_calib none --illumination_invariant 0 --photometric standard"},
    "N0": {"group": "D", "desc": "MonoIIF with a randomly initialised encoder (no foundation weights)",
           "flags": "--backbone_weights none"},
}
ABLATION_ORDER = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E8-IIF", "C0", "C1", "C2-sup",
                  "E8-DVLoRA", "C1-lora", "M-none", "M-global", "M-local",
                  "lam-da-025", "lam-da-100", "lam-da-200", "R1", "R2", "MonoII", "MonoViT", "MonoViT-II",
                  "MonoII-none", "MonoII-glob", "A-MonoII-none", "A-MonoII-glob", "A-MonoII",
                  "lam-res-000", "lam-res-010", "lam-res-025", "lam-res-100", "lam-res-200",
                  "N0", "D3-EndoDAC", "D3", "A-C0", "A-C1", "A-E8"]
# R2 as a 3 x 2 factorial on the ResNet-18 backbone: calibration x colour augmentation. Reading
# down a column gives the calibration comparison the reviewer asked for; reading across a row
# gives the effect of the augmentation defect on that calibration model.
# lambda1 swept with the paper's SSIM_II comparator; lam-bas-000 is the shared origin because at
# weight 0 the II branch is skipped and the comparator does not apply
LAMBDA_SSIM_ORDER = [("lam-bas-000", "$\lambda_1 = 0$ (sin II)"), ("lam-ssim-010", "$\lambda_1 = 0.1$"),
                     ("lam-ssim-025", "$\lambda_1 = 0.25$"), ("bas-res-2-ssim", "$\lambda_1 = 0.5$"),
                     ("lam-ssim-100", "$\lambda_1 = 1.0$"), ("lam-ssim-200", "$\lambda_1 = 2.0$")]
# Paired contrasts that are not against cfg["proposed"], so paired.csv cannot express them: each
# row names its own reference. diff = method - reference per sequence, so for a lower-is-better
# metric a NEGATIVE diff means the method is better and `wins` counts the sequences where it is.
# Rows whose method or reference has no results are written as "--".
CONTRASTS = [
    ("contrast_capacity.tex",
     "capacity of the calibration field, lambda1 = 0.5 fixed; negative = the calibration helps",
     [("degree 0 (global)", "bas-res-0", "MonoII-none"),
      ("degree 1", "bas-res-1", "MonoII-none"),
      ("degree 2", "bas-res-2", "MonoII-none"),
      ("degree 3", "bas-res-3", "MonoII-none"),
      ("dense map (submission)", "MonoII", "MonoII-none")]),
    ("contrast_degree1.tex",
     "the degree-1 field against each alternative; positive = degree 1 is better",
     [("global affine (Ozyoruk et al.)", "bas-res-0", "bas-res-1"),
      ("dense map (submission)", "MonoII", "bas-res-1"),
      ("no calibration", "MonoII-none", "bas-res-1")]),
    ("contrast_components.tex",
     "calibration and II loss against the plain ResNet-18; negative = the cell is better",
     [("calibration only", "lam-bas-000", "R1"),
      ("II loss only", "MonoII-none", "R1"),
      ("both", "bas-res-2", "R1"),
      ("calibration only vs both", "lam-bas-000", "bas-res-2")]),
    ("contrast_jitter.tex",
     "consistent colour jitter against the jitter as shipped; negative = the fix helps",
     [("no calibration", "A-MonoII-none", "MonoII-none"),
      ("global affine", "A-MonoII-glob", "MonoII-glob"),
      ("local, dense map", "A-MonoII", "MonoII")]),
    ("contrast_da_capacity.tex",
     "Depth Anything, clean recipe (no highlight term), lambda1 = 0; negative = the calibration helps",
     [("basis degree 0 (global)", "da-bas-0", "E3"),
      ("basis degree 1", "da-bas-1", "E3"),
      ("basis degree 2", "da-bas-2", "E3"),
      ("dense map", "E4", "E3")]),
    ("contrast_da_lambda.tex",
     "Depth Anything, II weight at basis degree 1; negative = that weight helps",
     [("lambda1 = 0.1", "da-lam-010", "da-bas-1"),
      ("lambda1 = 0.25", "da-lam-025", "da-bas-1"),
      ("lambda1 = 0.5", "da-lam-050", "da-bas-1")]),
    ("contrast_comparator.tex",
     "II comparator: the paper's SSIM against the l2 that was trained; negative = SSIM is better",
     [("dense map", "MonoII-ssim", "MonoII"),
      ("basis, degree 2", "bas-res-2-ssim", "bas-res-2")]),
]


# the II comparator: l2 (what was trained) against SSIM (what the paper writes), same cells
COMPARATOR_ORDER = [("MonoII", "dense map, II l2"), ("MonoII-ssim", "dense map, II SSIM"),
                    ("bas-res-2", "basis degree 2, II l2"), ("bas-res-2-ssim", "basis degree 2, II SSIM")]
# lambda1 swept with the basis calibration held at degree 2
LAMBDA_BAS_ORDER = [("lam-bas-000", "$\lambda_1 = 0$"), ("lam-bas-010", "$\lambda_1 = 0.1$"),
                    ("lam-bas-025", "$\lambda_1 = 0.25$"), ("bas-res-2", "$\lambda_1 = 0.5$"),
                    ("lam-bas-100", "$\lambda_1 = 1.0$"), ("lam-bas-200", "$\lambda_1 = 2.0$")]
# capacity of the calibration field, from 0 free parameters to the dense map: the curve that
# turns the none/global/local ternary of R2 into one family
BASIS_ORDER = [("MonoII-none", "no calibration"), ("bas-res-0", "degree 0 ($k$=1, global)"),
               ("bas-res-1", "degree 1 ($k$=3)"), ("bas-res-2", "degree 2 ($k$=6)"),
               ("bas-res-3", "degree 3 ($k$=10)"), ("MonoII", "dense map (LightingDecoder)")]
# lambda1 sweep on ResNet-18: the audit of the paper's Table 2, three seeds instead of one
LAMBDA_RES_ORDER = [("lam-res-000", "$\lambda_1 = 0$"), ("lam-res-010", "$\lambda_1 = 0.1$"),
                    ("lam-res-025", "$\lambda_1 = 0.25$"), ("MonoII", "$\lambda_1 = 0.5$ (paper)"),
                    ("lam-res-100", "$\lambda_1 = 1.0$"), ("lam-res-200", "$\lambda_1 = 2.0$")]
MONOII_CALIB_ORDER = [("MonoII-none", "none, jitter as shipped"),
                      ("MonoII-glob", "global affine, jitter as shipped"),
                      ("MonoII", "local affine (MonoII), jitter as shipped"),
                      ("A-MonoII-none", "none, consistent jitter"),
                      ("A-MonoII-glob", "global affine, consistent jitter"),
                      ("A-MonoII", "local affine (MonoII), consistent jitter")]
# R3, the reviewer's question about what the backbone contributes: the same two components
# (local affine calibration + II loss at lambda1 = 0.5, monodepth2 photometric loss) on three
# depth networks of very different capacity, each also trained plain. Every row is 3 seeds and
# the shared recipe, so a row pair differs only in the components and a column pair only in the
# architecture.
BACKBONE_ORDER = [("R1", "ResNet-18 (Monodepth2), plain"),
                  ("MonoII", "ResNet-18 + calibration + II (MonoII)"),
                  ("MonoViT", "MPViT-small + HR decoder (MonoViT), plain"),
                  ("MonoViT-II", "MPViT-small + calibration + II"),
                  ("E3", "Depth Anything v1 + DV-LoRA (EndoDAC), plain"),
                  ("M-local", "Depth Anything v1 + calibration + II (MonoIIF)")]
# lambda1 sweep of the II loss, in increasing order: the table the paper's Table 2 lacks for the
# foundation backbone. Every point has the method's structure and differs only in lambda1.
LAMBDA_ORDER = [("E4", "$\lambda_1 = 0$"), ("E7", "$\lambda_1 = 0.1$"),
                ("lam-da-025", "$\lambda_1 = 0.25$"), ("M-local", "$\lambda_1 = 0.5$ (paper)"),
                ("lam-da-100", "$\lambda_1 = 1.0$"), ("lam-da-200", "$\lambda_1 = 2.0$")]
CALIB_ORDER = [("C0", "none"), ("C1", "global affine"), ("E8", "local affine (MonoIIF)"),
               ("A-C0", "none, consistent jitter"), ("A-C1", "global affine, consistent jitter"),
               ("A-E8", "local affine, consistent jitter")]


def deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            text = f.read()
        try:
            import yaml
            user = yaml.safe_load(text) or {}
        except ImportError:
            user = json.loads(text)
        deep_update(cfg, user)
    elif path:
        print("[config] {} not found, using DEFAULT_CONFIG".format(path))
    return cfg


def all_runs(cfg):
    runs = copy.deepcopy(GRID)
    for name, spec in (cfg["train"].get("runs") or {}).items():
        runs.setdefault(name, {"group": "X", "desc": name, "flags": ""}).update(spec)
    for name in cfg["train"].get("skip_runs") or []:
        runs.pop(name, None)
    return runs


def run_seeds(cfg, run):
    t = cfg["train"]
    return list(t["multi_seeds"] if run in t["multi_seed_runs"] else t["seeds"])


def run_name(run, seed):
    return "cviu_{}_s{}".format(run, seed)


def weights_folder(cfg, run, seed):
    d = os.path.join(cfg["log_dir"], run_name(run, seed), "models")
    if cfg["train"]["checkpoint"] == "last":
        p = os.path.join(d, "weights_last")
        return p if os.path.isdir(p) else None
    best, best_ep = None, -1
    for p in glob.glob(os.path.join(d, "weights_*")):
        tail = os.path.basename(p).split("_")[-1]
        if tail.isdigit() and int(tail) > best_ep:
            best, best_ep = p, int(tail)
    return best


def git_hash():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def out_path(cfg, *parts):
    p = os.path.join(cfg["out_dir"], *parts)
    ensure_dir(os.path.dirname(p))
    return p


def readlines(path):
    with open(path) as f:
        return f.read().splitlines()


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(path, rows, fieldnames):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def device_of(args):
    if torch is None:
        raise RuntimeError("this stage needs torch")
    return torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")


# --------------------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------------------

def preflight(cfg):
    """Everything a training subprocess needs, checked before the grid is launched."""
    problems = []
    dp = cfg["data"]["scared"]
    if not os.path.isdir(dp):
        problems.append("data.scared is not a directory: {}".format(dp))
    else:
        folder, frame = readlines(os.path.join(SPLITS, "train_files.txt"))[0].split()[:2]
        img = os.path.join(dp, folder, "data", frame + ".jpg")  # SCAREDRAWDataset.get_image_path
        if not os.path.exists(img):
            problems.append("first training image not found: {} (dataset layout or data.scared)".format(img))
    pw = os.path.join(cfg["pretrained_path"], "depth_anything_vitb14.pth")
    if not os.path.exists(pw):
        problems.append("Depth Anything weights not found: {} (README: pretrained_model/)".format(pw))
    gt = os.path.join(SPLITS, "gt_depths.npz")
    if not os.path.exists(gt):
        problems.append("{} missing: python export_gt_depth.py --data_path {} --split endovis --useage eval".format(gt, dp))
    py = cfg["python"]
    if shutil.which(py) is None and not os.path.exists(py):
        problems.append("python interpreter not found: {}".format(py))
    else:
        missing = []
        for mod in ("torch", "kornia", "wandb", "tensorboardX", "skimage", "cv2", "matplotlib",
                    "yaml", "scipy", "fvcore", "PIL"):
            if subprocess.call([py, "-c", "import " + mod], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
                missing.append(mod)
        if missing:
            pip_names = {"skimage": "scikit-image", "cv2": "opencv-python-headless", "yaml": "pyyaml", "PIL": "pillow"}
            problems.append("'{}' cannot import {}: {} -m pip install {}".format(
                py, missing, py, " ".join(pip_names.get(m, m) for m in missing)))
        elif subprocess.call([py, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                             cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            problems.append("'{}' has torch but torch.cuda.is_available() is False (CPU-only wheel or the "
                            "driver is not visible): {} -c \"import torch; print(torch.__version__, torch.version.cuda)\"".format(py, py))
    return problems


DA3_URL = "https://huggingface.co/depth-anything/DA3-BASE/resolve/main/model.safetensors"


def da3_weights_path(cfg):
    """DA3-Base checkpoint used by the D-grid and the zero-shot rows (ported code, no package)."""
    return os.path.join(cfg["pretrained_path"], "da3_base.safetensors")


def _valid_safetensors(path):
    try:
        import struct
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n).decode("utf-8"))
        return any(k.startswith("model.backbone.pretrained.") for k in header)
    except Exception:
        return False


def ensure_da3_weights(cfg, download=True):
    """True if the DA3-Base checkpoint is available, downloading it (542 MB) when missing."""
    path = da3_weights_path(cfg)
    if os.path.exists(path):
        if _valid_safetensors(path):
            return True
        print("[da3] {} is not a valid DA3 checkpoint (interrupted download?), fetching it again".format(path))
    if not download:
        return False
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    part = path + ".part"
    print("[da3] downloading the DA3-Base weights (542 MB) -> {}".format(path))
    try:
        import urllib.request
        urllib.request.urlretrieve(DA3_URL, part)
        if not _valid_safetensors(part):
            raise ValueError("downloaded file is not a DA3 safetensors checkpoint")
        os.replace(part, path)
        return True
    except Exception as e:
        if os.path.exists(part):
            os.remove(part)
        print("[da3] download failed ({}); get it by hand: wget -O {} {}".format(e, path, DA3_URL))
        return False


# The link the MPViT and MonoViT READMEs both give. Checked 2026-09-15: Dropbox retired the
# /s/<id>/ links and it now answers with an HTML page, so the download below will normally fail
# and the file has to come from a copy that already exists somewhere.
MPVIT_URL = {
    "small": "https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth",
    "xsmall": "https://dl.dropbox.com/s/vvpq2m474g8tvyq/mpvit_xsmall.pth",
}
# <name> is filled with mpvit_small / mpvit_xsmall: MonoViT is published on small, the
# submission's MonoIIT row was trained on xsmall, and the two encoders are not interchangeable.
MPVIT_FALLBACKS = [
    "/workspace/endo-manydepth/manydepth/mpvit/{name}.pth",  # hard-coded in models/monovit/mpvit.py before this change
    "./ckpt/{name}.pth",                                     # where MonoViT's README puts it
    "~/{name}.pth",
]


def mpvit_weights_path(cfg, variant="small"):
    """ImageNet MPViT checkpoint for this encoder size."""
    return os.path.join(cfg["pretrained_path"], "mpvit_{}.pth".format(variant))


def ensure_mpvit_weights(cfg, download=True, variant="small"):
    """True if the MPViT checkpoint of `variant` is available, fetching it when missing.

    Looked for in <pretrained_path>, then in cfg["mpvit_weights"] and the known local copies, then
    downloaded. Without it the MonoViT rows would train from random weights, which is a different
    experiment and would understate the baseline, so the B-grid runs are skipped rather than
    mislabelled.
    """
    path = mpvit_weights_path(cfg, variant)

    def valid(p):
        if torch is None:  # stats/report environments have no torch; size is the only check left
            return os.path.getsize(p) > (10 << 20)
        try:
            ck = torch.load(p, map_location="cpu")
            return isinstance(ck, dict) and ("model" in ck or "state_dict" in ck)
        except Exception:
            return False

    if os.path.exists(path):
        if valid(path):
            return True
        print("[mpvit] {} is not a valid MPViT checkpoint (interrupted download?), fetching it again".format(path))
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    cands = [cfg.get("mpvit_weights")] if cfg.get("mpvit_weights") else []
    cands += [c.format(name="mpvit_{}".format(variant)) for c in MPVIT_FALLBACKS]
    for cand in cands:
        cand = os.path.expanduser(cand)
        if os.path.exists(cand) and valid(cand):
            print("[mpvit] using the copy already on this machine: {}".format(cand))
            shutil.copy2(cand, path)
            return True
    if not download:
        return False
    part = path + ".part"
    print("[mpvit] downloading the ImageNet MPViT-{} weights -> {}".format(variant, path))
    try:
        import urllib.request
        urllib.request.urlretrieve(MPVIT_URL[variant], part)
        if not valid(part):
            raise ValueError("downloaded file is not an MPViT checkpoint")
        os.replace(part, path)
        return True
    except Exception as e:
        if os.path.exists(part):
            os.remove(part)
        print("[mpvit] download failed ({}).\n"
              "        The official link ({}) is dead: Dropbox retired the /s/ links, so it answers\n"
              "        with an HTML page. Copy an existing mpvit_{}.pth to {} (look for one in\n"
              "        {}), or set mpvit_weights: <path> in the config."
              .format(e, MPVIT_URL[variant], variant, path,
                      ", ".join(c.format(name="mpvit_{}".format(variant)) for c in MPVIT_FALLBACKS)))
        return False


def _last_flag_value(cmd, flag):
    vals = [cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i] == flag]
    return vals[-1] if vals else None


def trained_epochs(cfg, name):
    """num_epochs a run was launched with (its models/opt.json), or None."""
    try:
        with open(os.path.join(cfg["log_dir"], name, "models", "opt.json")) as f:
            return int(json.load(f)["num_epochs"])
    except Exception:
        return None


def clear_done_markers(cfg, name):
    for marker in ("cviu_done.txt", "train_complete.txt"):
        p = os.path.join(cfg["log_dir"], name, marker)
        if os.path.exists(p):
            os.remove(p)


def gpu_status():
    """{gpu index: (MiB used, MiB total)} from nvidia-smi; {} when it is not available."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                       "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return {}
    status = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            status[parts[0]] = (int(parts[1]), int(parts[2]))
    return status


def running_cviu_runs():
    """model names of train_end_to_end.py processes of this grid that are still alive."""
    try:
        out = subprocess.check_output(["ps", "-eo", "args"], stderr=subprocess.DEVNULL).decode(errors="replace")
    except Exception:
        return set()
    names = set()
    for line in out.splitlines():
        if "train_end_to_end.py" in line and "--model_name" in line:
            toks = line.split()
            i = toks.index("--model_name") if "--model_name" in toks else -1
            if 0 <= i < len(toks) - 1 and toks[i + 1].startswith("cviu_"):
                names.add(toks[i + 1])
    return names


def log_shows_completion(log_path):
    """True if the last launch recorded in a train log ran all its epochs without an error.

    Covers runs whose launcher died while the training process went on: nobody wrote their
    cviu_done.txt, but their output kept going to the log (one "Evaluating" line per epoch).
    """
    try:
        with open(log_path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return False
    starts = [i for i, l in enumerate(lines) if l.startswith("# ") and "train_end_to_end.py" in l]
    if not starts:
        return False
    toks = lines[starts[-1]].split()
    epochs = [int(toks[i + 1]) for i in range(len(toks) - 1) if toks[i] == "--num_epochs" and toks[i + 1].isdigit()]
    section = lines[starts[-1]:]
    n_eval = sum(1 for l in section if l.strip() == "Evaluating")
    return bool(epochs) and n_eval >= epochs[-1] and not any(l.startswith("Traceback") for l in section)


def run_is_done(cfg, name, alive=()):
    run_dir = os.path.join(cfg["log_dir"], name)
    if os.path.exists(os.path.join(run_dir, "cviu_done.txt")) or os.path.exists(os.path.join(run_dir, "train_complete.txt")):
        return True
    if name in alive or not os.path.isdir(os.path.join(run_dir, "models", "weights_last")):
        return False
    if log_shows_completion(os.path.join(cfg["out_dir"], "train_logs", name + ".log")):
        with open(os.path.join(run_dir, "cviu_done.txt"), "w") as f:
            f.write("recovered from its train log (the launcher was gone when it finished)\n")
        return True
    return False


def tail(path, n=25):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def stage_train(cfg, args):
    runs = all_runs(cfg)
    selected = [r for r in runs if not args.only or r in args.only]
    problems = preflight(cfg)
    for p in problems:
        print("[train] PREFLIGHT: " + p)
    if problems and not args.dry_run and not args.skip_preflight:
        print("[train] fix the above (or pass --skip_preflight) before launching")
        sys.exit(1)
    # DA3-encoder rows need the DA3-Base checkpoint (ported code, same Python env as the rest); it is
    # downloaded when missing. If that fails they are left out so the rest of the grid still runs.
    da3_runs = [r for r in selected if "--backbone_weights da3" in runs[r]["flags"]]
    if da3_runs and not ensure_da3_weights(cfg, download=not args.dry_run):
        print("[train] {} need the DA3-Base weights: wget -O {} {}".format(
            da3_runs, da3_weights_path(cfg), DA3_URL))
        if not args.dry_run and not args.skip_preflight:
            print("[train] skipping {} for now".format(da3_runs))
            selected = [r for r in selected if r not in da3_runs]
    # MonoViT rows need the ImageNet MPViT weights of their encoder size; without them the encoder
    # would start from random init and the baseline would be understated.
    for mv_variant in ("small", "xsmall"):
        mv_runs = [r for r in selected
                   if "--depth_backbone monovit" in runs[r]["flags"]
                   and (_last_flag_value(runs[r]["flags"], "--mpvit_variant") or "small") == mv_variant]
        if mv_runs and not ensure_mpvit_weights(cfg, download=not args.dry_run, variant=mv_variant):
            print("[train] {} need the ImageNet MPViT-{} weights at {} (see the message above)".format(
                mv_runs, mv_variant, mpvit_weights_path(cfg, mv_variant)))
            if not args.dry_run and not args.skip_preflight:
                print("[train] skipping {} for now".format(mv_runs))
                selected = [r for r in selected if r not in mv_runs]
    alive = running_cviu_runs()   # e.g. jobs left running after their launcher died
    if alive:
        print("[train] still training from an earlier launch: {}".format(sorted(alive)))
    jobs = []
    for run, spec in runs.items():
        if run not in selected:
            continue
        for seed in run_seeds(cfg, run):
            name = run_name(run, seed)
            cmd = [cfg["python"], os.path.join(ROOT, "train_end_to_end.py"),
                   "--data_path", cfg["data"]["scared"], "--log_dir", cfg["log_dir"],
                   "--pretrained_path", cfg["pretrained_path"],
                   "--model_name", name, "--seed", str(seed)]
            cmd += shlex.split(cfg["train"]["common_flags"]) + shlex.split(spec["flags"])
            cmd += shlex.split(args.extra_flags or "")
            if name in alive:
                print("[train] {} is still running, skipping (it would write into the same folder)".format(name))
                continue
            if run_is_done(cfg, name, alive) and not args.force:
                planned, trained = _last_flag_value(cmd, "--num_epochs"), trained_epochs(cfg, name)
                if planned is not None and trained is not None and trained < int(planned):
                    # e.g. a 1-epoch smoke test: it must not stand in for the real run
                    print("[train] {} was trained for {} epoch(s), the grid asks for {}: retraining".format(
                        name, trained, planned))
                    clear_done_markers(cfg, name)
                else:
                    print("[train] {} done, skipping".format(name))
                    continue
            jobs.append({"run": run, "seed": seed, "name": name, "desc": spec["desc"], "cmd": cmd})

    manifest = {"git": git_hash(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": jobs}
    with open(out_path(cfg, "train_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[train] git {}: {} job(s)".format(manifest["git"], len(jobs)))
    for j in jobs:
        print("  {:<22} {}".format(j["name"], " ".join(shlex.quote(c) for c in j["cmd"])))
    if args.dry_run or not jobs:
        return 0

    gpus = [str(g) for g in (args.gpus or ["0"])]
    status = gpu_status()
    busy = [g for g in gpus if g in status and status[g][0] > max(1024, 0.05 * status[g][1])]
    if busy:
        print("[train] GPUs already in use (MiB used/total): {}".format(
            ", ".join("{}: {}/{}".format(g, *status[g]) for g in busy)))
        if not args.use_busy_gpus:
            gpus = [g for g in gpus if g not in busy]
            print("[train] leaving them out (--use_busy_gpus to override); using {}".format(gpus))
    if not gpus:
        print("[train] no free GPU among the requested ones; see nvidia-smi")
        sys.exit(1)
    q = queue.Queue()
    failed = []
    for j in jobs:
        q.put(j)
    log_dir = ensure_dir(os.path.join(cfg["out_dir"], "train_logs"))

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            log = os.path.join(log_dir, j["name"] + ".log")
            # The job list was built when this launcher started; another launcher (or a longer
            # queue of our own) may have begun this run since. Two processes with the same
            # --model_name write into the same models/ folder and ruin each other's checkpoints.
            now_alive = running_cviu_runs()
            if j["name"] in now_alive:
                print("[train] gpu {} -> {} started elsewhere since the job list was built, "
                      "skipping (it would write into the same folder)".format(gpu, j["name"]))
                continue
            if run_is_done(cfg, j["name"], now_alive) and not args.force:
                print("[train] gpu {} -> {} finished elsewhere, skipping".format(gpu, j["name"]))
                continue
            print("[train] gpu {} -> {} (log: {})".format(gpu, j["name"], log))
            t0 = time.time()
            with open(log, "a") as lf:
                lf.write("# {}\n# {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(j["cmd"])))
                lf.flush()
                rc = subprocess.call(j["cmd"], cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
            secs = time.time() - t0
            if rc == 0:
                with open(os.path.join(cfg["log_dir"], j["name"], "cviu_done.txt"), "w") as f:
                    f.write("git {}\nhours {:.2f}\ncmd {}\n".format(manifest["git"], secs / 3600.0, " ".join(j["cmd"])))
                print("[train] {} finished in {:.1f} h".format(j["name"], secs / 3600.0))
            else:
                failed.append(j["name"])
                log_tail = tail(log)
                print("[train] {} FAILED (rc={}) after {:.0f} s, see {}\n----- log tail -----\n{}--------------------".format(
                    j["name"], rc, secs, log, log_tail))
                if "out of memory" in log_tail or "valid cuDNN algorithm" in log_tail:
                    print("[train] {} ran out of GPU memory on GPU {}: check `nvidia-smi` for other processes "
                          "on it".format(j["name"], gpu))
                if secs < 180:
                    # died before training started: a setup problem of this run (its other seeds would
                    # fail the same way) or, once several runs do it, of the whole grid
                    with lock:
                        fast_failed_runs.add(j["run"])
                        drain_all = len(fast_failed_runs) >= 3
                        kept, dropped = [], []
                        while True:
                            try:
                                j2 = q.get_nowait()
                                q.task_done()
                            except queue.Empty:
                                break
                            (dropped if drain_all or j2["run"] == j["run"] else kept).append(j2)
                        for j2 in kept:
                            q.put(j2)
                    if drain_all:
                        print("[train] {} different runs failed within 3 minutes {}: aborting the remaining {} "
                              "job(s); fix the error and relaunch (finished runs are skipped)".format(
                                  len(fast_failed_runs), sorted(fast_failed_runs), len(dropped)))
                    elif dropped:
                        print("[train] {} failed within 3 minutes: dropping its other seeds {}; the rest of the grid "
                              "goes on".format(j["name"], [d["name"] for d in dropped]))
                    failed.extend(d["name"] + " (not started)" for d in dropped)
            q.task_done()

    lock = threading.Lock()
    fast_failed_runs = set()
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(out_path(cfg, "train_failures.json"), "w") as f:
        json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "failed": failed}, f, indent=2)
    if failed:
        print("[train] {} job(s) failed: {}".format(len(failed), failed))
    return len(failed)


# --------------------------------------------------------------------------------------
# datasets and per-frame evaluation
# --------------------------------------------------------------------------------------

def compute_errors(gt, pred):
    """monodepth2 metrics (identical to utils.utils.compute_errors)."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate_prediction(pred, kind, align, gt, max_depth):
    """pred: (h,w) disparity or depth at any resolution. Returns (metrics dict, ratio)."""
    import cv2
    gh, gw = gt.shape[:2]
    pred = cv2.resize(pred.astype(np.float32), (gw, gh))
    mask = np.logical_and(gt > MIN_DEPTH, gt < max_depth)
    g = gt[mask]
    if g.size == 0:
        # no usable ground truth in this frame (some C3VD frames are entirely invalid):
        # scoring it would produce NaN metrics and poison its sequence mean
        return None, float("nan")
    if kind == "disp":
        disp = np.clip(pred, 1e-6, None)[mask]
    else:
        disp = 1.0 / np.clip(pred, 1e-6, None)[mask]
    if align == "affine":  # disparity shift + scale (mytest_da.py), for relative-depth models
        gd = 1.0 / g
        t_gt, t_p = np.median(gd), np.median(disp)
        s_gt, s_p = np.mean(np.abs(gd - t_gt)), np.mean(np.abs(disp - t_p)) + 1e-12
        p = 1.0 / np.clip((disp - t_p) * (s_gt / s_p) + t_gt, 1e-6, None)
        ratio = float("nan")
    else:
        p = 1.0 / disp
        ratio = float(np.median(g) / np.median(p))
        p = p * ratio
    p = np.clip(p, MIN_DEPTH, max_depth)
    m = compute_errors(g, p)
    return dict(zip(METRICS, [float(x) for x in m])), ratio


def make_dataset(cfg, name):
    """Returns (dataset, meta) with meta(i, batch) -> (sequence, frame, gt_depth)."""
    if name == "scared":
        from datasets.scared_dataset import SCAREDRAWDataset
        lines = readlines(os.path.join(SPLITS, "test_files.txt"))
        ds = SCAREDRAWDataset(cfg["data"]["scared"], lines, H, W, [0], 4, is_train=False)
        gts = np.load(os.path.join(SPLITS, "gt_depths.npz"), fix_imports=True, encoding="latin1")["data"]

        def meta(i, data):
            folder, frame = lines[i].split()[:2]
            return folder, frame, gts[i]
    elif name == "hamlyn":
        from datasets.hamlyn_dataset import HamlynDataset
        ds = HamlynDataset(cfg["data"]["hamlyn"], H, W, [0], 4, is_train=False)

        def meta(i, data):
            return ("rectified{:02d}".format(int(data["sequence"][0])), str(int(data["index"][0])),
                    data["depth_gt"][0].numpy())
    elif name == "c3vd":
        from datasets.c3vd_dataset import C3VDDataset
        ds = C3VDDataset(cfg["data"]["c3vd"], H, W, [0], 4, is_train=False, split="test")

        def meta(i, data):
            return str(data["sequence"][0]), str(i), data["depth_gt"][0].numpy()
    else:
        raise ValueError("unknown dataset " + name)
    return ds, meta


def iterate_dataset(cfg, name):
    from torch.utils.data import DataLoader
    ds, meta = make_dataset(cfg, name)
    loader = DataLoader(ds, 1, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    for i, data in enumerate(loader):
        seq, frame, gt = meta(i, data)
        gt = np.asarray(gt, dtype=np.float32)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        yield i, data[("color", 0, 0)], seq, frame, gt


def _pretrained_or_none(cfg):
    p = cfg["pretrained_path"]
    return p if os.path.exists(os.path.join(p, "depth_anything_vitb14.pth")) else None


def load_depth_model_from_run(cfg, run, seed, device):
    """Depth network of a grid run (endodac or resnet18), from its saved opt.json + weights."""
    wf = weights_folder(cfg, run, seed)
    if wf is None:
        raise FileNotFoundError("no weights for {} seed {}".format(run, seed))
    with open(os.path.join(os.path.dirname(wf), "opt.json")) as f:
        opt = json.load(f)
    sd = torch.load(os.path.join(wf, "depth_model.pth"), map_location="cpu")
    backbone = opt.get("depth_backbone", "endodac")
    if backbone == "resnet18":
        from models.resnet_depth import ResnetDepth
        model = ResnetDepth(opt["num_layers"], False, opt["scales"])
    elif backbone == "monovit":
        from models.monovit_depth import MonoViTDepth
        # pretrained_weights=None: the checkpoint holds every weight, no need to reload MPViT
        model = MonoViTDepth(scales=opt["scales"], pretrained_weights=None,
                             variant=opt.get("mpvit_variant", "small"))
    else:
        import models.endodac as endodac
        # pretrained_path=None: the checkpoint holds every weight, no need to reload DA v1
        model = endodac.endodac(backbone_size="base", r=opt["lora_rank"], lora_type=opt["lora_type"],
                                image_shape=(224, 280), pretrained_path=None,
                                residual_block_indexes=opt["residual_block_indexes"],
                                include_cls_token=opt.get("include_cls_token", True),
                                backbone_weights=opt.get("backbone_weights", "da1"))
    md = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in md}, strict=False)
    return model.to(device).eval(), opt, wf


def build_predictor(cfg, name, spec, device):
    """Returns (predict(color)->np array, kind, align) or ('npy', arrays, kind, align)."""
    if spec["type"] == "npy":
        arrays = {ds: np.load(p) for ds, p in spec["files"].items()}
        return ("npy", arrays, spec.get("kind", "disp"), spec.get("align", "median"))
    if spec["type"] == "run":
        model, _, _ = load_depth_model_from_run(cfg, spec["run"], spec["seed"], device)
    else:
        from evaluate_depth_all import DepthModelFactory
        ns = argparse.Namespace(load_weights_folder=spec["weights"], model_type=spec["type"],
                                lora_rank=spec.get("lora_rank", 4), lora_type=spec.get("lora_type", "dvlora"),
                                pretrained_path=_pretrained_or_none(cfg),
                                residual_block_indexes=spec.get("residual_block_indexes", [2, 5, 8, 11]),
                                include_cls_token=spec.get("include_cls_token", True),
                                num_layers=spec.get("num_layers", 18),
                                residual_block_kind=spec.get("residual_block_kind", "lka"))
        model, _ = DepthModelFactory.load_model(spec["type"], ns)
    from utils.layers import disp_to_depth

    def predict(color):
        with torch.no_grad():
            out = model(color.to(device))
        disp = out[("disp", 0)] if isinstance(out, dict) else out
        scaled, _ = disp_to_depth(disp, *DEPTH_RANGE)
        return scaled[0, 0].cpu().numpy()
    return ("model", predict, "disp", "median")


def predict_methods(cfg, args):
    """All (method name, seed, spec) triples the predict stage should cover."""
    items = []
    alive = running_cviu_runs()
    for run in all_runs(cfg):
        for seed in run_seeds(cfg, run):
            if weights_folder(cfg, run, seed) is None:
                continue
            if not run_is_done(cfg, run_name(run, seed), alive):
                # interrupted or still training: its checkpoint must not enter the tables
                print("[predict] {} has not finished training, skipped".format(run_name(run, seed)))
                continue
            items.append((run, seed, {"type": "run", "run": run, "seed": seed}))
    for name, spec in (cfg.get("methods") or {}).items():
        items.append((name, 0, spec))
    if args.only:
        items = [it for it in items if it[0] in args.only]
    return items


def stage_predict(cfg, args):
    device = device_of(args)
    per_frame = out_path(cfg, "per_frame.csv")
    done = {(r["method"], r["seed"], r["dataset"]) for r in read_csv(per_frame)}
    fields = ["method", "seed", "dataset", "sequence", "frame"] + METRICS + ["ratio", "infer_ms", "git"]
    datasets = args.datasets or cfg["datasets"]
    g = git_hash()
    for method, seed, spec in predict_methods(cfg, args):
        todo = [d for d in datasets if (method, str(seed), d) not in done or args.force]
        if spec["type"] == "npy":
            todo = [d for d in todo if d in spec["files"]]
        if not todo:
            continue
        print("[predict] {} seed {} on {}".format(method, seed, todo))
        try:
            pred = build_predictor(cfg, method, spec, device)
        except Exception as e:
            print("[predict] cannot load {}: {}".format(method, e))
            continue
        for ds in todo:
            rows, disps, n_skipped = [], [], 0
            for i, color, seq, frame, gt in iterate_dataset(cfg, ds):
                t0 = time.time()
                if pred[0] == "npy":
                    p = pred[1][ds][i]
                    p = p[0] if p.ndim == 3 else p
                else:
                    p = pred[1](color)
                ms = (time.time() - t0) * 1000.0
                if cfg["save_pred"] and pred[0] != "npy":
                    disps.append(p.astype(np.float16))
                m, ratio = evaluate_prediction(p, pred[2], pred[3], gt, MAX_DEPTH[ds])
                if m is None:
                    n_skipped += 1
                    continue
                row = {"method": method, "seed": seed, "dataset": ds, "sequence": seq, "frame": frame,
                       "ratio": ratio, "infer_ms": ms, "git": g}
                row.update(m)
                rows.append(row)
            if args.force:  # drop stale rows of this (method, seed, dataset) before appending
                keep = [r for r in read_csv(per_frame)
                        if not (r["method"] == method and r["seed"] == str(seed) and r["dataset"] == ds)]
                write_csv(per_frame, keep, fields)
            append_rows(per_frame, rows, fields)
            if disps:
                np.save(out_path(cfg, "pred", "{}_s{}_{}.npy".format(method, seed, ds)), np.stack(disps))
            mean = {k: float(np.mean([r[k] for r in rows])) for k in METRICS} if rows else {k: float("nan") for k in METRICS}
            print("[predict] {:<20} {:<7} frames={:<4} abs_rel={:.4f} rmse={:.3f} a1={:.4f}{}".format(
                method, ds, len(rows), mean["abs_rel"], mean["rmse"], mean["a1"],
                "" if not n_skipped else "  ({} frames without ground truth skipped)".format(n_skipped)))


# --------------------------------------------------------------------------------------
# illumination model: shared torch helpers
# --------------------------------------------------------------------------------------

def gaussian_blur(x, k=11):
    """Depthwise Gaussian blur with LightingDecoder's kernel/sigma heuristic."""
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2 = torch.outer(g, g)[None, None].expand(x.shape[1], 1, k, k).contiguous()
    pad = k // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), k2, groups=x.shape[1])


def fit_affine(w, t, v, P, ridge):
    """Weighted least-squares affine fit t ~ c*w + b, per PxP patch (P=None: per image).

    Thin wrapper over utils.layers.fit_patch_affine, so this analysis and the training-time
    supervision (--calib_supervision) solve exactly the same problem.
    """
    from utils.layers import fit_patch_affine
    return fit_patch_affine(w, t, v, patch=P, ridge=ridge)


def masked_l1(x, t, e):
    return ((x - t).abs() * e).sum() / (e.sum() * x.shape[1] + 1e-6)


def checkerboard(like):
    B, _, Hh, Ww = like.shape
    yy, xx = torch.meshgrid(torch.arange(Hh, device=like.device), torch.arange(Ww, device=like.device), indexing="ij")
    par = ((yy + xx) % 2).to(like.dtype)[None, None]
    return par, 1.0 - par


def affine_residuals(w, t, v, patch_sizes, alpha, beta, ridge, ssim=None):
    """Cross-validated residuals of the illumination models on one batch of warped pairs.

    Fits on one checkerboard half of the valid pixels and evaluates on the other (both ways),
    so extra degrees of freedom cannot win by construction. Returns a dict of scalars.
    """
    par0, par1 = checkerboard(v)
    halves = [(v * par0, v * par1), (v * par1, v * par0)]
    res = {"r_none": masked_l1(w, t, v).item(), "valid_frac": v.mean().item()}
    if ssim is not None:
        res["ssim_none"] = (ssim(w, t).mean(1, True) * v).sum().item() / (v.sum().item() + 1e-6)
    for name, P in [("global", None)] + [("local{}".format(P), P) for P in patch_sizes]:
        r, rb = 0.0, 0.0
        for fit_v, ev_v in halves:
            c, b = fit_affine(w, t, fit_v, P, ridge)
            r += masked_l1(c * w + b, t, ev_v).item()
            if P is not None:  # what the LightingDecoder is allowed to express
                cb = gaussian_blur(c.clamp(1 - alpha, 1 + alpha))
                bb = gaussian_blur(b.clamp(-beta, beta))
                rb += masked_l1(cb * w + bb, t, ev_v).item()
        res["r_" + name] = r / 2
        if P is not None:
            res["r_bounded{}".format(P)] = rb / 2
        # in-sample parameters (all valid pixels) for plausibility statistics
        c, b = fit_affine(w, t, v, P, ridge)
        if P is None:
            res["c_global"], res["b_global"] = c.mean().item(), b.mean().item()
        cv, bv = c[v > 0], b[v > 0]
        if cv.numel() > 0:
            res["c_{}_p1".format(name)] = torch.quantile(cv, 0.01).item()
            res["c_{}_p99".format(name)] = torch.quantile(cv, 0.99).item()
            res["b_{}_p1".format(name)] = torch.quantile(bv, 0.01).item()
            res["b_{}_p99".format(name)] = torch.quantile(bv, 0.99).item()
            res["out_of_bounds_{}".format(name)] = (((cv - 1).abs() > alpha) | (bv.abs() > beta)).float().mean().item()
        if ssim is not None and name in ("global", "local16"):
            res["ssim_" + name] = (ssim(c * w + b, t).mean(1, True) * v).sum().item() / (v.sum().item() + 1e-6)
    return res


class Warper(object):
    """GT/predicted-geometry warping of a source frame into the target view (repo layers)."""

    def __init__(self, device):
        from utils.layers import BackprojectDepth, Project3D
        self.device = device
        self._bp, self._pr = {}, {}
        self.BackprojectDepth, self.Project3D = BackprojectDepth, Project3D

    def _mods(self, B):
        if B not in self._bp:
            self._bp[B] = self.BackprojectDepth(B, H, W).to(self.device)
            self._pr[B] = self.Project3D(B, H, W).to(self.device)
        return self._bp[B], self._pr[B]

    def warp(self, source, depth_t, K, inv_K, T, depth_s=None, tol=0.05):
        """Returns warped source (B,3,H,W), validity (B,1,H,W) and the z-buffer test rate."""
        B = source.shape[0]
        bp, pr = self._mods(B)
        cam_points = bp(depth_t, inv_K)
        pix = pr(cam_points, K, T)
        warped = F.grid_sample(source, pix, padding_mode="border", align_corners=True)
        valid = ((pix.abs() <= 1).all(-1).unsqueeze(1) & (depth_t > 0)).to(source.dtype)
        occl = None
        if depth_s is not None:  # depth-consistency occlusion check with the source GT depth
            z = torch.matmul(T, cam_points)[:, 2].view(B, 1, H, W)
            d_s = F.grid_sample(depth_s, pix, mode="nearest", padding_mode="border", align_corners=True)
            ok = ((d_s > 0) & ((z - d_s).abs() < tol * d_s.clamp_min(1e-6))).to(source.dtype)
            occl = (ok * valid).sum() / (valid.sum() + 1e-6)
            valid = valid * ok
        return warped, valid, occl


def photometric_map(ssim, pred, target):
    return 0.85 * ssim(pred, target).mean(1, True) + 0.15 * (pred - target).abs().mean(1, True)


def highlight_mask(image):
    """HADepth highlight test without kornia: S = (max-min)/max, V = max; 1 = highlight."""
    v, _ = image.max(1, keepdim=True)
    mn, _ = image.min(1, keepdim=True)
    s = (v - mn) / (v + 1e-6)
    return ((s < 0.1) & (v > 0.9)).to(image.dtype)


def pearson(a, b):
    """Per-image Pearson correlation of two (B,1,H,W) maps -> (B,)"""
    a = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    b = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


def sequence_pairs(cfg, seq):
    """Consecutive-frame pairs of a SCARED pose sequence: (lines, pair indices, gt T per line)."""
    lines = readlines(os.path.join(SPLITS, "test_files_{}.txt".format(seq)))
    pairs = []
    for i in range(len(lines) - 1):
        f0, f1 = lines[i].split(), lines[i + 1].split()
        if f0[0] == f1[0] and int(f1[1]) == int(f0[1]) + 1:
            pairs.append(i)
    gt_T = None
    p = os.path.join(SPLITS, "curve", "gt_poses_{}.npz".format(seq))
    if os.path.exists(p):
        gt_T = np.load(p)["data"]
    return lines, pairs, gt_T


def scared_file(data_path, folder, *parts):
    """Path of a SCARED file under <folder>/data/<parts>, whichever layout the copy uses.

    The repo is inconsistent: the image loader expects <data>/<folder>/data/..., while the
    depth and pose loaders expect <data>/<train|test>/<folder>/data/... . Try both.
    """
    split = "train" if int(folder[7]) < 8 else "test"
    candidates = [os.path.join(data_path, split, folder, "data", *parts),
                  os.path.join(data_path, folder, "data", *parts)]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise IOError("SCARED file not found, tried:\n  " + "\n  ".join(candidates))


def scared_depth(data_path, folder, frame_index):
    """Ground-truth depth of one SCARED frame, read as export_gt_depth.py does."""
    import cv2
    path = scared_file(data_path, folder, "scene_points",
                       "scene_points{:06d}.tiff".format(frame_index - 1))
    depth = cv2.imread(path, 3)
    if depth is None:
        raise IOError("cannot read {}".format(path))
    return depth[:, :, 0][0:1024, :].astype(np.float32)


def gt_relative_pose(data_path, folder, frame):
    """T = P(frame+1) @ pinv(P(frame)) from SCARED frame_data json (export_gt_pose.py)."""
    poses = []
    for idx in (frame - 1, frame):
        with open(scared_file(data_path, folder, "frame_data", "frame_data{:06d}.json".format(idx))) as f:
            poses.append(np.array(json.load(f)["camera-pose"]))
    return (poses[1] @ np.linalg.pinv(poses[0])).astype(np.float32)


# --------------------------------------------------------------------------------------
# stage: illum-fit
# --------------------------------------------------------------------------------------

def gt_depth_index():
    """(folder, frame) -> row of the exported splits/endovis/gt_depths.npz."""
    gts = np.load(os.path.join(SPLITS, "gt_depths.npz"), fix_imports=True, encoding="latin1")["data"]
    index = {}
    for k, line in enumerate(readlines(os.path.join(SPLITS, "test_files.txt"))):
        folder, frame = line.split()[:2]
        index[(folder, int(frame))] = k
    return gts, index


def model_depth_source(cfg, ds, lines, device, run, seed):
    """Depth from a trained run, rescaled to the units of the exported ground truth.

    Used when this copy of SCARED ships only the images (no per-frame scene_points): the warp
    needs depth in the same units as the ground-truth poses, so the run's scale-ambiguous depth
    is multiplied by the median ground-truth/prediction ratio over the frames of this sequence
    that do appear in gt_depths.npz. Returns (depth_of, scale).
    """
    import cv2
    from utils.layers import disp_to_depth
    model, _, _ = load_depth_model_from_run(cfg, run, seed, device)

    def depth_of(x):
        with torch.no_grad():
            out = model(x.to(device))
        disp = F.interpolate(out[("disp", 0)], size=x.shape[-2:], mode="bilinear", align_corners=True)
        return disp_to_depth(disp, *DEPTH_RANGE)[1]

    gts, index = gt_depth_index()
    ratios = []
    for i, line in enumerate(lines):
        folder, frame = line.split()[:2]
        key = (folder, int(frame))
        if key not in index:
            continue
        gt = np.asarray(gts[index[key]], dtype=np.float32)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        d = depth_of(ds[i][("color", 0, 0)][None])[0, 0].cpu().numpy()
        d = cv2.resize(d, (gt.shape[1], gt.shape[0]))
        m = np.logical_and(gt > MIN_DEPTH, gt < MAX_DEPTH["scared"])
        if m.sum():
            ratios.append(float(np.median(gt[m]) / np.median(d[m])))
    if not ratios:
        raise IOError("no frame of this sequence appears in gt_depths.npz, cannot scale the model depth")
    scale = float(np.median(ratios))
    print("[illum-fit] depth from run {} seed {}: scale {:.2f} estimated on {} frames with exported "
          "ground truth (relative spread {:.1%})".format(run, seed, scale, len(ratios), float(np.std(ratios)) / scale))
    return depth_of, scale


def stage_illum_fit(cfg, args):
    import cv2
    from datasets.scared_dataset import SCAREDRAWDataset
    from utils.layers import SSIM
    device = device_of(args)
    ic = cfg["illum"]
    warper = Warper(device)
    ssim = SSIM().to(device)
    for seq in ic["sequences"]:
        path = out_path(cfg, "illum", "fit_{}.csv".format(seq))
        if os.path.exists(path) and not args.force:
            print("[illum-fit] {} exists, skipping".format(path))
            continue
        lines, pairs, gt_T = sequence_pairs(cfg, seq)
        ds = SCAREDRAWDataset(cfg["data"]["scared"], lines, H, W, [0, 1], 4, is_train=False)
        rows, convention = [], None

        # per-frame ground-truth depth when this SCARED copy has it, otherwise the depth of a
        # trained run scaled to the exported ground truth
        folder0, frame0 = lines[0].split()[:2]
        try:
            scared_depth(cfg["data"]["scared"], folder0, int(frame0))
            depth_of, depth_scale, depth_source = None, 1.0, "gt"
        except (IOError, OSError) as e:
            run = ic.get("fit_depth_run", "E3")
            print("[illum-fit] {}: no per-frame ground-truth depth on disk ({})".format(
                seq, str(e).splitlines()[0]))
            depth_of, depth_scale = model_depth_source(cfg, ds, lines, device, run, ic["seed"])
            depth_source = "model:{}_s{}".format(run, ic["seed"])

        def load_pair(i):
            folder, frame, side = lines[i].split()
            frame = int(frame)
            item = ds[i]
            I_t = item[("color", 0, 0)][None].to(device)
            I_s = item[("color", 1, 0)][None].to(device)
            K, inv_K = item[("K", 0)][None].to(device), item[("inv_K", 0)][None].to(device)
            if depth_of is None:
                d_t = cv2.resize(scared_depth(cfg["data"]["scared"], folder, frame), (W, H), interpolation=cv2.INTER_NEAREST)
                d_s = cv2.resize(scared_depth(cfg["data"]["scared"], folder, frame + 1), (W, H), interpolation=cv2.INTER_NEAREST)
                d_t = torch.from_numpy(d_t)[None, None].to(device)
                d_s = torch.from_numpy(d_s)[None, None].to(device)
                d_t = d_t * ((d_t > MIN_DEPTH) & (d_t < MAX_DEPTH["scared"])).to(d_t.dtype)
            else:
                d_t = depth_of(I_t) * depth_scale
                d_s = depth_of(I_s) * depth_scale
            T = gt_T[i] if gt_T is not None else gt_relative_pose(cfg["data"]["scared"], folder, frame)
            T = torch.from_numpy(np.asarray(T, dtype=np.float32))[None].to(device)
            return folder, frame, I_t, I_s, K, inv_K, d_t, d_s, T

        # the pose json convention is checked empirically on the first pairs: the direction
        # whose GT warp has the lower photometric residual is used for the whole sequence
        scores = {"direct": 0.0, "inverse": 0.0}
        probe, errors = [], []
        for i in pairs:
            try:
                probe.append(load_pair(i))
            except (IOError, OSError) as e:
                errors.append(str(e))
                continue
            if len(probe) >= 20:
                break
        if not probe:
            raise IOError("no usable frame of {}: ground-truth depth or pose files are missing.\n{}".format(
                seq, errors[0] if errors else ""))
        for _, _, I_t, I_s, K, inv_K, d_t, d_s, T in probe:
            for name, Tx in (("direct", T), ("inverse", torch.inverse(T))):
                wpd, v, _ = warper.warp(I_s, d_t, K, inv_K, Tx, d_s)
                scores[name] += masked_l1(wpd, I_t, v).item()
        convention = min(scores, key=scores.get)
        print("[illum-fit] {}: pose convention '{}' (residuals {})".format(seq, convention, scores))

        t0 = time.time()
        n_missing = 0
        for n, i in enumerate(pairs):
            try:
                folder, frame, I_t, I_s, K, inv_K, d_t, d_s, T = load_pair(i)
            except (IOError, OSError):   # frame without ground-truth depth / pose on disk
                n_missing += 1
                continue
            if convention == "inverse":
                T = torch.inverse(T)
            wpd, v, occl = warper.warp(I_s, d_t, K, inv_K, T, d_s)
            with torch.no_grad():
                res = affine_residuals(wpd, I_t, v, ic["patch_sizes"], ic["alpha"], ic["beta"], ic["ridge"], ssim)
            res.update({"sequence": folder, "frame": frame, "r_identity": masked_l1(I_s, I_t, v).item(),
                        "zbuffer_pass": float(occl), "convention": convention, "depth_source": depth_source})
            rows.append(res)
            if n % 100 == 0:
                print("[illum-fit] {} {}/{}  none={:.4f} global={:.4f} local16={:.4f} bounded16={:.4f} ({:.0f}s)".format(
                    seq, n, len(pairs), res["r_none"], res["r_global"], res["r_local16"], res["r_bounded16"], time.time() - t0))
        if not rows:
            raise IOError("no pair of {} could be evaluated ({} skipped for missing files)".format(seq, n_missing))
        if n_missing:
            print("[illum-fit] {}: {} of {} pairs skipped, their files are not on disk".format(
                seq, n_missing, len(pairs)))
        fixed = ("sequence", "frame", "convention", "depth_source", "valid_frac", "zbuffer_pass", "r_identity")
        fields = list(fixed) + sorted(k for k in rows[0] if k not in fixed)
        write_csv(path, rows, fields)
        print("[illum-fit] wrote {} ({} pairs)".format(path, len(rows)))


# --------------------------------------------------------------------------------------
# stage: illum-params / illum-sens (learned calibration)
# --------------------------------------------------------------------------------------

def load_run_models(cfg, run, seed, device):
    """Depth, pose, intrinsics and lighting networks of a grid run."""
    import models.decoders as decoders
    from models.encoders import ResnetEncoder
    depth, opt, wf = load_depth_model_from_run(cfg, run, seed, device)
    models = {"depth": depth}
    pe = ResnetEncoder(opt["num_layers"], False, num_input_images=2)
    pe.load_state_dict(torch.load(os.path.join(wf, "pose_encoder.pth"), map_location="cpu"))
    models["pose_encoder"] = pe.to(device).eval()
    pd = decoders.PoseDecoder(pe.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
    pd.load_state_dict(torch.load(os.path.join(wf, "pose.pth"), map_location="cpu"))
    models["pose"] = pd.to(device).eval()
    if opt.get("learn_intrinsics", True) and os.path.exists(os.path.join(wf, "intrinsics_head.pth")):
        ih = decoders.IntrinsicsHead(pe.num_ch_enc)
        ih.load_state_dict(torch.load(os.path.join(wf, "intrinsics_head.pth"), map_location="cpu"))
        models["intrinsics"] = ih.to(device).eval()
    calib = opt.get("illum_calib", "local")
    lp = os.path.join(wf, "lighting.pth")
    if calib != "none" and os.path.exists(lp):
        if calib == "global":
            lt = decoders.GlobalLightingHead(pe.num_ch_enc, opt["scales"])
        elif opt.get("illum_calib") == "basis":
            lt = decoders.BasisLightingHead(pe.num_ch_enc, opt["scales"],
                                            degree=opt.get("illum_basis_degree", 2),
                                            grid=(opt["height"], opt["width"]))
        else:
            lt = decoders.LightingDecoder(pe.num_ch_enc, opt["scales"])
        lt.load_state_dict(torch.load(lp, map_location="cpu"))
        models["lighting"] = lt.to(device).eval()
    models["_opt"] = opt
    return models


def run_pair(models, warper, ssim, target, source, K_in, inv_K_in):
    """One forward pass of the full pipeline on (target, source) batches. No grad."""
    from utils.layers import disp_to_depth, transformation_from_parameters
    with torch.no_grad():
        disp = models["depth"](target)[("disp", 0)]
        disp = F.interpolate(disp, [H, W], mode="bilinear", align_corners=True)
        _, depth = disp_to_depth(disp, *DEPTH_RANGE)
        feats = models["pose_encoder"](torch.cat([source, target], 1))
        axisangle, translation, inter = models["pose"]([feats])
        if "intrinsics" in models:
            K = models["intrinsics"](inter, W, H)
            inv_K = torch.inverse(K)
        else:
            K, inv_K = K_in, inv_K_in
        T = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
        warped, valid, _ = warper.warp(source, depth, K, inv_K, T)
        if "lighting" in models:
            lo = models["lighting"](feats)
            c = F.interpolate(lo[("contrast", 0)], [H, W], mode="bilinear", align_corners=False)
            b = F.interpolate(lo[("brightness", 0)], [H, W], mode="bilinear", align_corners=False)
        else:
            c, b = torch.ones_like(valid), torch.zeros_like(valid)
        automask = (photometric_map(ssim, warped, target) < photometric_map(ssim, source, target)).to(target.dtype) * valid
    return {"depth": depth, "T": T, "K": K, "warped": warped, "valid": valid, "c": c, "b": b, "automask": automask}


def stage_illum_params(cfg, args):
    from torch.utils.data import DataLoader
    from datasets.scared_dataset import SCAREDRAWDataset
    from utils.layers import SSIM
    device = device_of(args)
    ic = cfg["illum"]
    warper, ssim = Warper(device), SSIM().to(device)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    radial = torch.sqrt(yy ** 2 + xx ** 2)[None, None].to(device)
    runs = args.only or ic["runs"]
    for run in runs:
        for seed in ([ic["seed"]] if not args.all_seeds else run_seeds(cfg, run)):
            try:
                models = load_run_models(cfg, run, seed, device)
            except Exception as e:
                print("[illum-params] cannot load {} s{}: {}".format(run, seed, e))
                continue
            lt = models.get("lighting")
            alpha = getattr(lt, "alpha", ic["alpha"])
            beta = getattr(lt, "beta", ic["beta"])
            hist = {"c": np.zeros(200), "b": np.zeros(200)}
            c_edges, b_edges = np.linspace(1 - 2 * alpha, 1 + 2 * alpha, 201), np.linspace(-2 * beta, 2 * beta, 201)
            for seq in ic["sequences"]:
                path = out_path(cfg, "illum", "params_{}_s{}_{}.csv".format(run, seed, seq))
                if os.path.exists(path) and not args.force:
                    print("[illum-params] {} exists, skipping".format(path))
                    continue
                lines, _, gt_T = sequence_pairs(cfg, seq)
                # frames whose both neighbours exist in the split
                idx = [i for i in range(1, len(lines) - 1)
                       if lines[i - 1].split()[0] == lines[i].split()[0] == lines[i + 1].split()[0]
                       and int(lines[i + 1].split()[1]) - int(lines[i - 1].split()[1]) == 2]
                sub = [lines[i] for i in idx]
                ds = SCAREDRAWDataset(cfg["data"]["scared"], sub, H, W, [0, -1, 1], 4, is_train=False)
                loader = DataLoader(ds, 8, shuffle=False, num_workers=cfg["num_workers"], drop_last=False)
                rows, prev_c_fwd, prev_frame = [], None, None
                for bi, data in enumerate(loader):
                    target = data[("color", 0, 0)].to(device)
                    K_in, inv_K_in = data[("K", 0)].to(device), data[("inv_K", 0)].to(device)
                    outs = {f: run_pair(models, warper, ssim, target, data[("color", f, 0)].to(device), K_in, inv_K_in) for f in (-1, 1)}
                    hl = highlight_mask(target)
                    B = target.shape[0]
                    for f in (-1, 1):
                        o = outs[f]
                        c, b, am = o["c"], o["b"], o["automask"]
                        refined = c * o["warped"] + b
                        # least-squares oracle (patch 16) on the same pixels, for reference
                        c_or, b_or = fit_affine(o["warped"], target, am, 16, ic["ridge"])
                        gc = (c[:, :, :, 1:] - c[:, :, :, :-1]).abs().mean((1, 2, 3)) + (c[:, :, 1:] - c[:, :, :-1]).abs().mean((1, 2, 3))
                        gb = (b[:, :, :, 1:] - b[:, :, :, :-1]).abs().mean((1, 2, 3)) + (b[:, :, 1:] - b[:, :, :-1]).abs().mean((1, 2, 3))
                        stats = {
                            "c_mean": c.mean((1, 2, 3)), "c_std": c.flatten(1).std(1), "b_mean": b.mean((1, 2, 3)), "b_std": b.flatten(1).std(1),
                            "c_p1": torch.quantile(c.flatten(1), 0.01, dim=1), "c_p99": torch.quantile(c.flatten(1), 0.99, dim=1),
                            "b_p1": torch.quantile(b.flatten(1), 0.01, dim=1), "b_p99": torch.quantile(b.flatten(1), 0.99, dim=1),
                            "sat_c": ((c - 1).abs() > 0.95 * alpha).float().mean((1, 2, 3)), "sat_b": (b.abs() > 0.95 * beta).float().mean((1, 2, 3)),
                            "grad_c": gc, "grad_b": gb,
                            "corr_c_radial": pearson(c, radial.expand_as(c)), "corr_b_radial": pearson(b, radial.expand_as(b)),
                            "corr_c_highlight": pearson(c, hl), "corr_b_highlight": pearson(b, hl),
                            "highlight_frac": hl.mean((1, 2, 3)), "automask_frac": am.mean((1, 2, 3)),
                            "res_before": ((o["warped"] - target).abs().mean(1, True) * am).sum((1, 2, 3)) / (am.sum((1, 2, 3)) + 1e-6),
                            "res_after": ((refined - target).abs().mean(1, True) * am).sum((1, 2, 3)) / (am.sum((1, 2, 3)) + 1e-6),
                            "res_oracle16": ((c_or * o["warped"] + b_or - target).abs().mean(1, True) * am).sum((1, 2, 3)) / (am.sum((1, 2, 3)) + 1e-6),
                            "tz_pred": o["T"][:, 2, 3],
                        }
                        stats = {k: v.detach().cpu().numpy() for k, v in stats.items()}
                        for j in range(B):
                            li = idx[bi * 8 + j]
                            row = {"run": run, "seed": seed, "sequence": lines[li].split()[0], "frame": int(lines[li].split()[1]), "source": f}
                            row.update({k: float(v[j]) for k, v in stats.items()})
                            if gt_T is not None:
                                row["tz_gt"] = float(gt_T[li if f == 1 else li - 1][2, 3])
                            rows.append(row)
                    # temporal consistency and forward/backward reciprocity between consecutive frames
                    c_fwd, c_bwd = outs[1]["c"], outs[-1]["c"]
                    for j in range(B):
                        li = idx[bi * 8 + j]
                        cur = c_fwd[j:j + 1]
                        if prev_c_fwd is not None and prev_frame == int(lines[li].split()[1]) - 1:
                            tcorr = pearson(prev_c_fwd, cur).item()
                            recip = (prev_c_fwd * c_bwd[j:j + 1] - 1).abs().mean().item()
                            for r in rows[-2 * B:]:
                                if r["frame"] == int(lines[li].split()[1]):
                                    r["temporal_corr_c"], r["reciprocity_err"] = tcorr, recip
                        prev_c_fwd, prev_frame = cur, int(lines[li].split()[1])
                    hist["c"] += np.histogram(c_fwd.detach().cpu().numpy(), bins=c_edges)[0]
                    hist["b"] += np.histogram(outs[1]["b"].detach().cpu().numpy(), bins=b_edges)[0]
                fields = sorted(set(k for r in rows for k in r), key=lambda k: (k not in ("run", "seed", "sequence", "frame", "source"), k))
                write_csv(path, rows, fields)
                print("[illum-params] wrote {} ({} rows)".format(path, len(rows)))
            np.savez(out_path(cfg, "illum", "hist_{}_s{}.npz".format(run, seed)), c=hist["c"], b=hist["b"],
                     c_edges=c_edges, b_edges=b_edges, alpha=alpha, beta=beta)


def illum_loss_terms(refined, target, mask, iif_mask, iif_weight, iif_eps, ssim_map_fn, alpha=0.85):
    """The loss terms that see the illumination-calibrated warp, as scalars.

    Same formulas as Trainer.compute_losses: alpha*SSIM + (1-alpha)*L1 averaged over the
    automask-valid non-specular pixels, plus the illumination-invariant term on the eroded mask.
    """
    from utils.layers import get_illumination_invariant_features, get_illumination_invariant_l2
    denom = mask.sum() + 1e-6
    terms = {"ssim": alpha * (ssim_map_fn(refined, target) * mask).sum() / denom,
             "l1": (1.0 - alpha) * ((refined - target).abs().mean(1, True) * mask).sum() / denom}
    ft = get_illumination_invariant_features(target, eps=iif_eps)
    fp = get_illumination_invariant_features(refined, eps=iif_eps)
    d = get_illumination_invariant_l2(fp, ft)
    terms["iif"] = iif_weight * (d * iif_mask).sum() / (iif_mask.sum() + 1e-6)
    return terms


def stage_illum_grad(cfg, args):
    """How much gradient each loss term gives to the illumination calibration.

    The illumination-invariant descriptors are invariant to I -> c*I + b by construction, so that
    term cannot constrain (c, b); SSIM is largely insensitive to them as well. If almost all the
    gradient comes from the 0.15-weighted L1 inside the highlight mask, the calibration module is
    under-determined by the objective, which would explain why it does not track injected
    illumination changes (stage illum-sens).
    """
    from torch.utils.data import DataLoader
    from datasets.scared_dataset import SCAREDRAWDataset
    from utils.layers import SSIM, disp_to_depth, get_feature_oclution_mask, transformation_from_parameters
    device = device_of(args)
    ic = cfg["illum"]
    warper, ssim = Warper(device), SSIM().to(device)
    try:
        import kornia
        def ssim_map_fn(a, b):
            return kornia.losses.ssim_loss(a, b, window_size=7, reduction="none").mean(1, True)
        ssim_kind = "kornia window 7 (as in training)"
    except ImportError:
        ssim_map_fn = lambda a, b: ssim(a, b).mean(1, True)
        ssim_kind = "repo SSIM window 3 (kornia not installed)"

    rows = []
    for run in (args.only or ic["runs"]):
        seed = ic["seed"]
        models = load_run_models(cfg, run, seed, device)
        if "lighting" not in models:
            print("[illum-grad] {} has no illumination decoder, skipped".format(run))
            continue
        opt = models["_opt"]
        params = [p for p in models["lighting"].parameters()]
        lines = readlines(os.path.join(SPLITS, "test_files.txt"))
        ds = SCAREDRAWDataset(cfg["data"]["scared"], lines, H, W, [0, -1, 1], 4, is_train=False)
        loader = DataLoader(ds, 4, shuffle=False, num_workers=cfg["num_workers"], drop_last=True)
        for bi, batch in enumerate(loader):
            if bi >= ic.get("grad_batches", 20):
                break
            target = batch[("color", 0, 0)].to(device)
            tgt_aug = batch[("color_aug", 0, 0)].to(device)
            for f in (-1, 1):
                source = batch[("color", f, 0)].to(device)
                src_aug = batch[("color_aug", f, 0)].to(device)
                with torch.no_grad():                      # geometry is fixed: only (c, b) carry grad
                    disp = F.interpolate(models["depth"](tgt_aug)[("disp", 0)], [H, W],
                                         mode="bilinear", align_corners=True)
                    depth = disp_to_depth(disp, *DEPTH_RANGE)[1]
                    feats = models["pose_encoder"](torch.cat([src_aug, tgt_aug], 1))
                    axisangle, translation, inter = models["pose"]([feats])
                    if "intrinsics" in models:
                        K = models["intrinsics"](inter, W, H)
                    else:
                        K = batch[("K", 0)].to(device)
                    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
                    warped, valid, _ = warper.warp(source, depth, K, torch.inverse(K), T)
                    automask = (photometric_map(ssim, warped, target) <
                                photometric_map(ssim, source, target)).to(target.dtype) * valid
                    non_spec = 1.0 - highlight_mask(target)
                    mask = automask * non_spec
                    iif_mask = get_feature_oclution_mask(mask)
                out = models["lighting"](feats)
                c = F.interpolate(out[("contrast", 0)], [H, W], mode="bilinear", align_corners=False)
                b = F.interpolate(out[("brightness", 0)], [H, W], mode="bilinear", align_corners=False)
                refined = c * warped + b
                terms = illum_loss_terms(refined, target, mask, iif_mask,
                                         opt.get("illumination_invariant", 0.1),
                                         opt.get("iif_eps", 1e-4), ssim_map_fn)
                row = {"run": run, "seed": seed, "batch": bi, "source": f}
                # is the calibration input-dependent at all? compare how much the maps change
                # across the images of the batch with how much they vary inside one image
                row["c_across_inputs"] = float(c.mean((1, 2, 3)).std()) if c.shape[0] > 1 else float("nan")
                row["b_across_inputs"] = float(b.mean((1, 2, 3)).std()) if b.shape[0] > 1 else float("nan")
                row["c_within_image"] = float(c.flatten(1).std(1).mean())
                row["b_within_image"] = float(b.flatten(1).std(1).mean())
                for name, term in terms.items():
                    gc, gb = torch.autograd.grad(term, [c, b], retain_graph=True, allow_unused=True)
                    gp = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                    row[name + "_value"] = float(term)
                    row[name + "_grad_c"] = float(gc.abs().mean()) if gc is not None else 0.0
                    row[name + "_grad_b"] = float(gb.abs().mean()) if gb is not None else 0.0
                    row[name + "_grad_params"] = float(torch.sqrt(sum((x ** 2).sum() for x in gp if x is not None))) \
                        if any(x is not None for x in gp) else 0.0
                rows.append(row)
        del models
    if not rows:
        print("[illum-grad] nothing to measure")
        return
    fields = ["run", "seed", "batch", "source"] + sorted(k for k in rows[0] if k not in ("run", "seed", "batch", "source"))
    write_csv(out_path(cfg, "illum", "grad.csv"), rows, fields)
    print("[illum-grad] SSIM: {}".format(ssim_kind))
    print("[illum-grad] mean |gradient| delivered to the calibration maps, per loss term:")
    print("    {:<6} {:>12} {:>14} {:>14} {:>16}".format("term", "value", "d/dc", "d/db", "d/dparams"))
    for name in ("ssim", "l1", "iif"):
        print("    {:<6} {:>12.6f} {:>14.3e} {:>14.3e} {:>16.3e}".format(
            name, float(np.mean([r[name + "_value"] for r in rows])),
            float(np.mean([r[name + "_grad_c"] for r in rows])),
            float(np.mean([r[name + "_grad_b"] for r in rows])),
            float(np.mean([r[name + "_grad_params"] for r in rows]))))
    tot = sum(float(np.mean([r[n + "_grad_params"] for r in rows])) for n in ("ssim", "l1", "iif")) + 1e-12
    print("    share of the gradient reaching the decoder: " + ", ".join(
        "{} {:.1%}".format(n, float(np.mean([r[n + "_grad_params"] for r in rows])) / tot) for n in ("ssim", "l1", "iif")))
    print("[illum-grad] wrote {}".format(out_path(cfg, "illum", "grad.csv")))


def stage_illum_sens(cfg, args):
    from torch.utils.data import DataLoader
    from datasets.scared_dataset import SCAREDRAWDataset
    from utils.layers import SSIM, disp_to_depth
    device = device_of(args)
    ic = cfg["illum"]
    warper, ssim = Warper(device), SSIM().to(device)
    runs = args.only or ic["runs"]
    for run in runs:
        seed = ic["seed"]
        path = out_path(cfg, "illum", "sens_{}_s{}.csv".format(run, seed))
        if os.path.exists(path) and not args.force:
            print("[illum-sens] {} exists, skipping".format(path))
            continue
        try:
            models = load_run_models(cfg, run, seed, device)
        except Exception as e:
            print("[illum-sens] cannot load {} s{}: {}".format(run, seed, e))
            continue
        rows = []
        # (a) calibration response: perturb the source frame, read the predicted (c, b)
        seq = ic["sequences"][0]
        lines, pairs, _ = sequence_pairs(cfg, seq)
        sub = [lines[i] for i in pairs[::ic["sens_stride"]]]
        ds = SCAREDRAWDataset(cfg["data"]["scared"], sub, H, W, [0, 1], 4, is_train=False)
        loader = DataLoader(ds, 8, shuffle=False, num_workers=cfg["num_workers"])
        for data in loader:
            target, source = data[("color", 0, 0)].to(device), data[("color", 1, 0)].to(device)
            K_in, inv_K_in = data[("K", 0)].to(device), data[("inv_K", 0)].to(device)
            for g in ic["gains"]:
                for bias in ic["biases"]:
                    if g != 1.0 and bias != 0.0:
                        continue  # one factor at a time
                    o = run_pair(models, warper, ssim, target, (g * source + bias).clamp(0, 1), K_in, inv_K_in)
                    am = o["automask"]
                    lt = models.get("lighting")
                    alpha, beta = getattr(lt, "alpha", ic["alpha"]), getattr(lt, "beta", ic["beta"])
                    for j in range(target.shape[0]):
                        rows.append({"run": run, "seed": seed, "kind": "calib", "gain": g, "bias": bias,
                                     "alpha": alpha, "beta": beta,
                                     "frame": int(data["frame_id"][j]),
                                     "c_mean": o["c"][j].mean().item(), "b_mean": o["b"][j].mean().item(),
                                     "res_before": ((o["warped"][j] - target[j]).abs().mean(0, True) * am[j]).sum().item() / (am[j].sum().item() + 1e-6),
                                     "res_after": ((o["c"][j] * o["warped"][j] + o["b"][j] - target[j]).abs().mean(0, True) * am[j]).sum().item() / (am[j].sum().item() + 1e-6)})
        # (b) depth robustness: perturb the frame fed to the depth network, score against GT
        gts = np.load(os.path.join(SPLITS, "gt_depths.npz"), fix_imports=True, encoding="latin1")["data"]
        tlines = readlines(os.path.join(SPLITS, "test_files.txt"))
        tds = SCAREDRAWDataset(cfg["data"]["scared"], tlines, H, W, [0], 4, is_train=False)
        tl = DataLoader(tds, 1, shuffle=False, num_workers=cfg["num_workers"])
        settings = [(g, 0.0) for g in ic["gains"]] + [(1.0, b) for b in ic["biases"] if b != 0.0]
        for i, data in enumerate(tl):
            color = data[("color", 0, 0)].to(device)
            for g, bias in settings:
                with torch.no_grad():
                    disp = models["depth"]((g * color + bias).clamp(0, 1))[("disp", 0)]
                    scaled, _ = disp_to_depth(disp, *DEPTH_RANGE)
                m, _ = evaluate_prediction(scaled[0, 0].cpu().numpy(), "disp", "median", gts[i].astype(np.float32), MAX_DEPTH["scared"])
                row = {"run": run, "seed": seed, "kind": "depth", "gain": g, "bias": bias,
                       "sequence": tlines[i].split()[0], "frame": int(tlines[i].split()[1])}
                row.update(m)
                rows.append(row)
        fields = ["run", "seed", "kind", "gain", "bias", "alpha", "beta", "sequence", "frame",
                  "c_mean", "b_mean", "res_before", "res_after"] + METRICS
        write_csv(path, rows, fields)
        print("[illum-sens] wrote {} ({} rows)".format(path, len(rows)))


# --------------------------------------------------------------------------------------
# stage: da3 (zero-shot Depth Anything 3)
# --------------------------------------------------------------------------------------

def _da3_package_infer(model, img_uint8):
    """Single-view inference through the depth_anything_3 package -> depth (h, w) float32."""
    pred = model.inference([img_uint8])
    depth = getattr(pred, "depth", None)
    if depth is None and isinstance(pred, dict):
        depth = pred["depth"]
    depth = np.asarray(depth.cpu().numpy() if hasattr(depth, "cpu") else depth, dtype=np.float32)
    return depth[0] if depth.ndim == 3 else depth


def _da3_predictor(cfg, variant, device):
    """Zero-shot DA3 predictor: color (1,3,h,w) in [0,1] -> depth (h', w') numpy, or None.

    "da3-base" runs the ported encoder + DualDPT main branch (models/endodac/da3_zeroshot.py,
    verified bit-exact against the original code) from <pretrained_path>/da3_base.safetensors,
    so it needs no extra package. Any other id goes through the depth_anything_3 package
    (Python >= 3.9, torch >= 2) and is skipped when that is not installed.
    """
    if variant in ("da3-base", "depth-anything/da3-base", "depth-anything/DA3-BASE"):
        path = da3_weights_path(cfg)
        if not ensure_da3_weights(cfg):
            return None
        from models.endodac.da3_zeroshot import DA3BaseMono
        model = DA3BaseMono(weights_path=path).to(device).eval()

        def predict(color):
            with torch.no_grad():
                return model(color.to(device))[0, 0].float().cpu().numpy()
        return predict
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        print("[da3] {} needs the depth_anything_3 package, not installed here: skipped".format(variant))
        return None
    model = DepthAnything3.from_pretrained(variant).to(device).eval()

    def predict(color):
        return _da3_package_infer(model, (color[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    return predict


def stage_da3(cfg, args):
    device = device_of(args)
    per_frame = out_path(cfg, "per_frame.csv")
    done = {(r["method"], r["seed"], r["dataset"]) for r in read_csv(per_frame)}
    fields = ["method", "seed", "dataset", "sequence", "frame"] + METRICS + ["ratio", "infer_ms", "git"]
    g = git_hash()
    datasets = args.datasets or cfg["datasets"]
    for variant in cfg["da3"]["models"]:
        method = "DA3_" + variant.split("/")[-1].replace("-", "_").lower()
        todo = [d for d in datasets if (method, "0", d) not in done or args.force]
        if not todo:
            continue
        print("[da3] loading {}".format(variant))
        predict = _da3_predictor(cfg, variant, device)
        if predict is None:
            continue
        for ds in todo:
            rows, rows_aff, preds = [], [], []
            for i, color, seq, frame, gt in iterate_dataset(cfg, ds):
                t0 = time.time()
                depth = predict(color)
                ms = (time.time() - t0) * 1000.0
                preds.append(depth.astype(np.float16))
                for align, target, name in (("median", rows, method), ("affine", rows_aff, method + "_affine")):
                    m, ratio = evaluate_prediction(depth, "depth", align, gt, MAX_DEPTH[ds])
                    if m is None:
                        continue
                    row = {"method": name, "seed": 0, "dataset": ds, "sequence": seq, "frame": frame,
                           "ratio": ratio, "infer_ms": ms, "git": g}
                    row.update(m)
                    target.append(row)
            append_rows(per_frame, rows, fields)
            if cfg["da3"].get("affine_rows", True):
                append_rows(per_frame, rows_aff, fields)
            if cfg["save_pred"]:
                np.save(out_path(cfg, "pred", "{}_s0_{}.npy".format(method, ds)), np.stack(preds))
            print("[da3] {:<24} {:<7} abs_rel={:.4f}".format(method, ds, float(np.mean([r["abs_rel"] for r in rows]))))


# --------------------------------------------------------------------------------------
# stage: stats
# --------------------------------------------------------------------------------------

def group_mean(rows, keys, values):
    """Mean of `values` over rows sharing the same `keys` (insertion ordered)."""
    acc = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        a = acc.setdefault(k, {v: [] for v in values})
        for v in values:
            a[v].append(float(r[v]))
    out = []
    for k, a in acc.items():
        row = dict(zip(keys, k))
        row.update({v: float(np.mean(a[v])) for v in values})
        row["n"] = len(a[values[0]])
        out.append(row)
    return out


def bootstrap_ci(x, n_boot, alpha, rng, stat=np.mean):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    s = stat(x[idx], axis=1)
    return float(np.percentile(s, 100 * alpha / 2)), float(np.percentile(s, 100 * (1 - alpha / 2)))


def t_ci(x, alpha):
    from scipy import stats as st
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    h = st.t.ppf(1 - alpha / 2, len(x) - 1) * st.sem(x)
    return float(x.mean() - h), float(x.mean() + h)


def paired_tests(d):
    """d = method - proposed per sequence. Returns dict of p-values and effect size."""
    from scipy import stats as st
    d = np.asarray(d, dtype=float)
    res = {"n": len(d), "mean_diff": float(d.mean()), "dz": float(d.mean() / (d.std(ddof=1) + 1e-12)) if len(d) > 1 else float("nan"),
           "wins": int((d < 0).sum()), "losses": int((d > 0).sum())}
    if len(d) > 1 and np.any(d != 0):
        res["p_t"] = float(st.ttest_rel(d, np.zeros_like(d)).pvalue)
        try:
            res["p_wilcoxon"] = float(st.wilcoxon(d, method="exact").pvalue)
        except TypeError:
            res["p_wilcoxon"] = float(st.wilcoxon(d, mode="exact").pvalue)
        except ValueError:
            res["p_wilcoxon"] = float("nan")
        k, n = int((d != 0).sum()), int(min((d < 0).sum(), (d > 0).sum()))
        try:
            res["p_sign"] = float(st.binomtest(n, k, 0.5).pvalue) if k else float("nan")
        except AttributeError:
            res["p_sign"] = float(st.binom_test(n, k, 0.5)) if k else float("nan")
    else:
        res.update({"p_t": float("nan"), "p_wilcoxon": float("nan"), "p_sign": float("nan")})
    return res


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return adj
    idx = np.argsort(p[ok])
    m = ok.sum()
    sorted_p = p[ok][idx]
    a = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * sorted_p))
    tmp = np.empty(m)
    tmp[idx] = a
    adj[ok] = tmp
    return adj


def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else "{:.{}f}".format(x, nd)


def stage_stats(cfg, args):
    sc = cfg["stats"]
    rows = read_csv(out_path(cfg, "per_frame.csv"))
    if not rows:
        print("[stats] per_frame.csv is empty; run predict first")
        return
    # rows written before the no-ground-truth guard can carry NaN metrics; they would turn
    # their whole sequence mean into NaN, so drop them here as well
    clean, dropped = [], 0
    for r in rows:
        try:
            vals = [float(r[k]) for k in METRICS]
        except (KeyError, ValueError):
            dropped += 1
            continue
        if any(np.isnan(v) for v in vals):
            dropped += 1
            continue
        clean.append(r)
    if dropped:
        print("[stats] ignoring {} frame(s) without valid ground truth".format(dropped))
    rows = clean

    # A dataset with fewer than min_sequences videos (e.g. a Hamlyn copy holding a single
    # rectified sequence) cannot support sequence-level statistics. Use contiguous blocks of
    # frames as the unit instead: temporal correlation inside a block is absorbed by averaging,
    # and every method gets the same partition, so the paired tests stay aligned.
    min_seq, block = sc.get("min_sequences", 3), sc.get("block_frames", 100)
    seqs_per_ds = collections.defaultdict(set)
    for r in rows:
        seqs_per_ds[r["dataset"]].add(r["sequence"])
    units = {}

    def _frame_key(x):
        try:
            return (0, int(x), "")
        except (TypeError, ValueError):
            return (1, 0, str(x))

    for ds, seqs in sorted(seqs_per_ds.items()):
        if len(seqs) >= min_seq:
            continue
        frames = collections.defaultdict(set)
        for r in rows:
            if r["dataset"] == ds:
                frames[r["sequence"]].add(r["frame"])
        block_of = {}
        for s, fs in frames.items():
            for j, f in enumerate(sorted(fs, key=_frame_key)):
                block_of[(s, f)] = "{}#{:04d}".format(s, j // block)
        for r in rows:
            if r["dataset"] == ds:
                r["sequence"] = block_of[(r["sequence"], r["frame"])]
        units[ds] = {"sequences": len(frames), "blocks": len(set(block_of.values())), "block_frames": block}
        print("[stats] {}: only {} sequence(s) available; using {} blocks of {} frames as the unit".format(
            ds, len(frames), units[ds]["blocks"], block))
    with open(out_path(cfg, "stats_units.json"), "w", encoding="utf-8") as f:
        json.dump(units, f, indent=2)
    rng = np.random.default_rng(0)
    # 1) frame -> (method, seed, dataset, sequence); 2) -> (method, dataset, sequence) over seeds
    per_seed_seq = group_mean(rows, ["method", "seed", "dataset", "sequence"], METRICS)
    per_seq = group_mean(per_seed_seq, ["method", "dataset", "sequence"], METRICS)
    write_csv(out_path(cfg, "per_sequence.csv"), per_seq, ["method", "dataset", "sequence", "n"] + METRICS)
    # seed variability of the overall mean (multi-seed runs)
    per_seed = group_mean(per_seed_seq, ["method", "seed", "dataset"], METRICS)
    seed_sd = {}
    for (m, d), grp in _groupby(per_seed, ["method", "dataset"]).items():
        if len(grp) > 1:
            seed_sd[(m, d)] = {k: float(np.std([g[k] for g in grp], ddof=1)) for k in METRICS}
    # 3) summary per (method, dataset)
    summary = []
    for (m, d), grp in _groupby(per_seq, ["method", "dataset"]).items():
        row = {"method": m, "dataset": d, "n_seq": len(grp), "n_seeds": len({r["seed"] for r in rows if r["method"] == m and r["dataset"] == d})}
        for k in METRICS:
            x = np.array([g[k] for g in grp])
            row[k] = float(x.mean())
            row[k + "_sd"] = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
            row[k + "_tlo"], row[k + "_thi"] = t_ci(x, sc["alpha"])
            row[k + "_blo"], row[k + "_bhi"] = bootstrap_ci(x, sc["n_boot"], sc["alpha"], rng)
            row[k + "_seed_sd"] = seed_sd.get((m, d), {}).get(k, float("nan"))
        summary.append(row)
    sfields = ["method", "dataset", "n_seq", "n_seeds"] + [k + s for k in METRICS for s in ("", "_sd", "_tlo", "_thi", "_blo", "_bhi", "_seed_sd")]
    write_csv(out_path(cfg, "summary.csv"), summary, sfields)
    # 4) paired comparisons against the proposed method, per dataset and metric
    prop = cfg["proposed"]
    paired = []
    by_md = _groupby(per_seq, ["method", "dataset"])
    for d in sorted({r["dataset"] for r in per_seq}):
        if (prop, d) not in by_md:
            print("[stats] proposed method {} has no rows on {}".format(prop, d))
            continue
        ref = {r["sequence"]: r for r in by_md[(prop, d)]}
        block = []
        for (m, dd), grp in by_md.items():
            if dd != d or m == prop:
                continue
            common = [r for r in grp if r["sequence"] in ref]
            if len(common) < 2:
                continue
            for k in METRICS:
                diff = [r[k] - ref[r["sequence"]][k] for r in common]
                if not LOWER_IS_BETTER[k]:
                    diff = [-x for x in diff]  # so that negative = proposed is better, consistently
                res = paired_tests(diff)
                res["diff_blo"], res["diff_bhi"] = bootstrap_ci(diff, sc["n_boot"], sc["alpha"], rng)
                res.update({"dataset": d, "method": m, "metric": k})
                block.append(res)
        for k in METRICS:  # Holm across methods within (dataset, metric)
            sel = [r for r in block if r["metric"] == k]
            for key in ("p_t", "p_wilcoxon", "p_sign"):
                adj = holm([r[key] for r in sel])
                for r, a in zip(sel, adj):
                    r[key + "_holm"] = float(a)
        paired += block
    pfields = ["dataset", "method", "metric", "n", "mean_diff", "diff_blo", "diff_bhi", "dz", "wins", "losses",
               "p_t", "p_wilcoxon", "p_sign", "p_t_holm", "p_wilcoxon_holm", "p_sign_holm"]
    write_csv(out_path(cfg, "paired.csv"), paired, pfields)
    write_tables(cfg, summary, paired, per_seq)
    print("[stats] wrote summary.csv, per_sequence.csv, paired.csv and tables/*.tex")


def _groupby(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def _method_order(cfg, methods):
    """Row order of the tables: external methods, the ablation chain, then anything else.

    Every method with results must appear: grid runs outside ABLATION_ORDER (C1, C2-sup, entries
    added through the config) used to be dropped silently.
    """
    order = cfg["stats"].get("table_methods")
    prop = cfg["proposed"]
    if order:
        return [m for m in order if m in methods]
    grid = set(all_runs(cfg))
    known = [m for m in ABLATION_ORDER if m in methods and m != prop]
    rest = sorted(m for m in methods if m in grid and m not in known and m != prop)
    ext = sorted(m for m in methods if m not in grid and m != prop)
    return ext + known + rest + ([prop] if prop in methods else [])


def _seq_index(per_seq, metric):
    """{(method, dataset): {sequence: value}} from the per-sequence rows."""
    idx = {}
    for r in per_seq:
        try:
            v = float(r[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if v == v:                                    # drop NaN
            idx.setdefault((r["method"], r["dataset"]), {})[r["sequence"]] = v
    return idx


def paired_contrast(idx, a, b, ds, n_boot, alpha, rng):
    """Paired difference a - b over the sequences both share, or None if either is missing."""
    A, B = idx.get((a, ds)), idx.get((b, ds))
    if not A or not B:
        return None
    keys = sorted(set(A) & set(B))
    if len(keys) < 2:
        return None
    d = np.array([A[k] - B[k] for k in keys], dtype=float)
    res = paired_tests(d)
    res["blo"], res["bhi"] = bootstrap_ci(d, n_boot, alpha, rng)
    return res


def write_contrast_tables(cfg, per_seq, metric="abs_rel"):
    """The comparisons that do not go through cfg["proposed"]; see CONTRASTS."""
    tdir = ensure_dir(os.path.join(cfg["out_dir"], "tables"))
    sc = cfg["stats"]
    idx = _seq_index(per_seq, metric)
    datasets = [d for d in cfg["datasets"] if any(dd == d for (_, dd) in idx)]
    if not datasets:
        return
    for fname, caption, rows in CONTRASTS:
        rng = np.random.default_rng(0)                # same draws for every table
        lines = ["% {} ({}), paired by sequence, {} bootstrap draws".format(caption, metric, sc["n_boot"]),
                 "\\begin{tabular}{l" + "cc" * len(datasets) + "}", "\\toprule",
                 " & " + " & ".join("\\multicolumn{{2}}{{c}}{{{}}}".format(d) for d in datasets) + " \\\\",
                 "Variant & " + " & ".join("$\\Delta$ [95\\% CI] & wins" for _ in datasets) + " \\\\",
                 "\\midrule"]
        any_row = False
        for label, a, b in rows:
            cells = []
            for d in datasets:
                r = paired_contrast(idx, a, b, d, sc["n_boot"], sc["alpha"], rng)
                if r is None:
                    cells += ["--", "--"]
                    continue
                any_row = True
                clean = (r["blo"] > 0) == (r["bhi"] > 0)          # CI excludes zero
                cell = "{} [{}, {}]".format(fmt(r["mean_diff"], 4), fmt(r["blo"], 4), fmt(r["bhi"], 4))
                cells += ["\\textbf{{{}}}".format(cell) if clean else cell,
                          "{}/{}".format(r["wins"], r["n"])]
            lines.append("{} & {} \\\\".format(label.replace("_", "\\_"), " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}"]
        if not any_row:
            continue                                   # nothing trained yet for this contrast
        with open(os.path.join(tdir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def write_tables(cfg, summary, paired, per_seq):
    tdir = ensure_dir(os.path.join(cfg["out_dir"], "tables"))
    write_contrast_tables(cfg, per_seq)
    by = {(r["method"], r["dataset"]): r for r in summary}
    pb = {(r["dataset"], r["method"], r["metric"]): r for r in paired}
    sc = cfg["stats"]
    for d in sorted({r["dataset"] for r in summary}):
        methods = _method_order(cfg, [m for (m, dd) in by if dd == d])
        best = {}
        for k in METRICS:
            vals = [(by[(m, d)][k], m) for m in methods]
            best[k] = (min if LOWER_IS_BETTER[k] else max)(vals)[1]
        lines = ["% sequence-level results on {}: mean $\\pm$ SD over n={} video sequences".format(d, by[(methods[0], d)]["n_seq"]),
                 "\\begin{tabular}{l" + "c" * len(METRICS) + "}", "\\toprule",
                 "Method & " + " & ".join(k.replace("_", "\\_") for k in METRICS) + " \\\\", "\\midrule"]
        for m in methods:
            r = by[(m, d)]
            cells = []
            for k in METRICS:
                s = "{} $\\pm$ {}".format(fmt(r[k]), fmt(r[k + "_sd"]))
                cells.append("\\textbf{" + s + "}" if best[k] == m else s)
            lines.append("{} & {} \\\\".format(m.replace("_", "\\_"), " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(tdir, "main_{}.tex".format(d)), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        # paired table
        pm = sc["paired_metrics"]
        lines = ["% paired differences (method $-$ {0}) per sequence, signed so that positive favours {0}; "
                 "wins = sequences where that method beats {0}; bootstrap 95\\% CI; Holm-adjusted exact Wilcoxon".format(cfg["proposed"]),
                 "\\begin{tabular}{l" + "c" * len(pm) + "}", "\\toprule",
                 "Method & " + " & ".join("$\\Delta$" + k.replace("_", "\\_") + " [CI], $p_W$" for k in pm) + " \\\\", "\\midrule"]
        for m in methods:
            if m == cfg["proposed"]:
                continue
            cells = []
            for k in pm:
                r = pb.get((d, m, k))
                if r is None:
                    cells.append("--")
                else:
                    cells.append("{} [{}, {}], {} ({}/{})".format(fmt(r["mean_diff"]), fmt(r["diff_blo"]), fmt(r["diff_bhi"]),
                                                                  fmt(r.get("p_wilcoxon_holm"), 3), r["wins"], r["n"]))
            lines.append("{} & {} \\\\".format(m.replace("_", "\\_"), " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(tdir, "paired_{}.tex".format(d)), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        # supplementary per-sequence abs_rel
        seqs = sorted({r["sequence"] for r in per_seq if r["dataset"] == d})
        ps = {(r["method"], r["sequence"]): r["abs_rel"] for r in per_seq if r["dataset"] == d}
        lines = ["% per-sequence Abs Rel on {}".format(d), "\\begin{tabular}{l" + "c" * len(seqs) + "}", "\\toprule",
                 "Method & " + " & ".join(s.replace("_", "\\_") for s in seqs) + " \\\\", "\\midrule"]
        for m in methods:
            lines.append("{} & {} \\\\".format(m.replace("_", "\\_"), " & ".join(fmt(ps.get((m, s))) for s in seqs)))
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(tdir, "per_sequence_{}.tex".format(d)), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    # ablation and calibration tables (SCARED full metrics + Abs Rel on the other sets)
    runs = all_runs(cfg)
    others = [d for d in cfg["datasets"] if d != "scared"]
    for fname, order in (("ablation.tex", [(r, runs[r]["desc"]) for r in ABLATION_ORDER if r in runs]),
                         ("calibration.tex", CALIB_ORDER),
                         ("backbones.tex", BACKBONE_ORDER),
                         ("lambda_sweep.tex", LAMBDA_ORDER),
                         ("monoii_calib.tex", MONOII_CALIB_ORDER),
                         ("lambda_sweep_resnet.tex", LAMBDA_RES_ORDER),
                         ("calib_capacity.tex", BASIS_ORDER),
                         ("lambda_sweep_basis.tex", LAMBDA_BAS_ORDER),
                         ("ii_comparator.tex", COMPARATOR_ORDER),
                         ("lambda_sweep_ssim.tex", LAMBDA_SSIM_ORDER)):
        lines = ["\\begin{tabular}{ll" + "c" * (len(METRICS) + len(others)) + "}", "\\toprule",
                 "Run & Variant & " + " & ".join(k.replace("_", "\\_") for k in METRICS) + "".join(" & {} Abs Rel".format(d) for d in others) + " \\\\", "\\midrule"]
        for run, desc in order:
            if (run, "scared") not in by:
                continue
            r = by[(run, "scared")]
            cells = ["{} $\\pm$ {}".format(fmt(r[k]), fmt(r[k + "_sd"])) for k in METRICS]
            cells += [fmt(by[(run, d)]["abs_rel"]) if (run, d) in by else "--" for d in others]
            lines.append("{} & {} & {} \\\\".format(run, desc, " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(tdir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------
# stage: report
# --------------------------------------------------------------------------------------

def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _try_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def stage_report(cfg, args):
    ic = cfg["illum"]
    md = ["# MonoIIF CVIU revision: results", "", "git: `{}`  generated: {}".format(git_hash(), time.strftime("%Y-%m-%d %H:%M")), ""]
    plt = _try_plt()
    fdir = ensure_dir(os.path.join(cfg["out_dir"], "figures"))
    manifest = out_path(cfg, "train_manifest.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            mf = json.load(f)
        md += ["Training grid launched at git `{}` ({} jobs).".format(mf["git"], len(mf["jobs"])), ""]
    alive = running_cviu_runs()
    missing = [run_name(r, s) for r in all_runs(cfg) for s in run_seeds(cfg, r) if not run_is_done(cfg, run_name(r, s), alive)]
    if missing:
        md += ["**Grid runs not finished (absent from every table): {}.** Rerun "
               "`python cviu_revision.py full` to complete them.".format(", ".join(missing)), ""]

    summary = read_csv(out_path(cfg, "summary.csv"))
    paired = read_csv(out_path(cfg, "paired.csv"))
    per_seq = read_csv(out_path(cfg, "per_sequence.csv"))
    units = {}
    if os.path.exists(out_path(cfg, "stats_units.json")):
        with open(out_path(cfg, "stats_units.json"), encoding="utf-8") as f:
            units = json.load(f)
    if summary:
        md += ["## Sequence-level results (mean ± SD over sequences; bootstrap 95% CI of the mean)", ""]
        for d in sorted({r["dataset"] for r in summary}):
            rows = [r for r in summary if r["dataset"] == d]
            order = _method_order(cfg, [r["method"] for r in rows])
            by = {r["method"]: r for r in rows}
            u = units.get(d)
            if u:
                md += ["### {} (n = {} blocks of {} frames; this copy of the dataset holds only {} "
                       "sequence(s), so blocks are the unit of analysis and the intervals are "
                       "slightly optimistic)".format(d, rows[0]["n_seq"], u["block_frames"], u["sequences"]), ""]
            else:
                md += ["### {} (n = {} sequences)".format(d, rows[0]["n_seq"]), ""]
            md.append(_md_table(["method", "seeds"] + METRICS, [
                [m, by[m]["n_seeds"]] + ["{} ± {} [{}, {}]".format(fmt(float(by[m][k])), fmt(float(by[m][k + "_sd"])), fmt(float(by[m][k + "_blo"])), fmt(float(by[m][k + "_bhi"]))) for k in METRICS]
                for m in order]))
            md.append("")
            pr = [r for r in paired if r["dataset"] == d and r["metric"] in cfg["stats"]["paired_metrics"]]
            if pr:
                md += ["Paired vs {0}: difference is method $-$ {0} per sequence, signed so that **positive means "
                       "{0} is better**, and wins counts the sequences where that method beats {0}.".format(cfg["proposed"]), ""]
                md.append(_md_table(["method", "metric", "diff [CI]", "p_t", "p_W (Holm)", "p_sign", "wins/n", "d_z"], [
                    [r["method"], r["metric"], "{} [{}, {}]".format(fmt(float(r["mean_diff"])), fmt(float(r["diff_blo"])), fmt(float(r["diff_bhi"]))),
                     fmt(float(r["p_t"])), fmt(float(r["p_wilcoxon_holm"])), fmt(float(r["p_sign"])), "{}/{}".format(r["wins"], r["n"]), fmt(float(r["dz"]), 2)]
                    for r in pr]))
                md.append("")
            if plt is not None and per_seq:
                fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(order)), 3.2))
                data = [[float(r["abs_rel"]) for r in per_seq if r["dataset"] == d and r["method"] == m] for m in order]
                ax.boxplot(data, labels=order, showmeans=True)
                ax.set_ylabel("Abs Rel per sequence")
                ax.set_title(d)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
                fig.tight_layout()
                fig.savefig(os.path.join(fdir, "per_sequence_{}.pdf".format(d)))
                plt.close(fig)
                md += ["![per-sequence {}](figures/per_sequence_{}.pdf)".format(d, d), ""]

    # backbone x components (R3): is the gain the architecture or the two components?
    bb_pairs = [("R1", "MonoII", "ResNet-18 (Monodepth2 / MonoII)"),
                ("MonoViT", "MonoViT-II", "MPViT-small + HR decoder (MonoViT)"),
                ("E3", "M-local", "Depth Anything v1 + DV-LoRA (EndoDAC / MonoIIF)")]
    have = {r["method"] for r in per_seq}
    if any(b in have and f in have for b, f, _ in bb_pairs):
        md += ["## Backbone × components (R3): local calibration + II loss (λ₁ = 0.5) on three depth networks", "",
               "Every row is the same pair of components on a different architecture, with everything "
               "else equal (shared recipe: same pose net, same splits, 20 epochs, learned intrinsics). "
               "The difference is (with components) − (plain) per sequence, so **negative means the "
               "components help**.", ""]
        rows = []
        for d in sorted({r["dataset"] for r in per_seq}):
            ps = {(r["method"], r["sequence"]): float(r["abs_rel"]) for r in per_seq if r["dataset"] == d}
            for base, full, label in bb_pairs:
                seqs = sorted({s for (m, s) in ps if m == base} & {s for (m, s) in ps if m == full})
                if not seqs:
                    continue
                diff = np.array([ps[(full, s)] - ps[(base, s)] for s in seqs])
                t = paired_tests(diff)
                lo, hi = bootstrap_ci(diff, cfg["stats"]["n_boot"], cfg["stats"]["alpha"], np.random.default_rng(0))
                rows.append([d, label,
                             fmt(float(np.mean([ps[(base, s)] for s in seqs])), 4),
                             fmt(float(np.mean([ps[(full, s)] for s in seqs])), 4),
                             "{} [{}, {}]".format(fmt(t["mean_diff"], 4), fmt(lo, 4), fmt(hi, 4)),
                             "{}/{}".format(int((diff < 0).sum()), len(seqs)),
                             fmt(t.get("p_wilcoxon", float("nan")))])
        md.append(_md_table(["dataset", "backbone", "plain", "+ components", "diff [CI]", "helped/n", "p_W"], rows))
        md.append("")

    # illum-fit
    fits = {seq: read_csv(out_path(cfg, "illum", "fit_{}.csv".format(seq))) for seq in ic["sequences"]}
    fits = {k: v for k, v in fits.items() if v}
    if fits:
        sources = sorted({r.get("depth_source", "gt") for fr in fits.values() for r in fr})
        geometry = ("ground-truth depth and pose" if sources == ["gt"] else
                    "ground-truth pose with depth from {} (this SCARED copy has no per-frame depth; "
                    "depth error is therefore part of the residual and slightly favours the local model)".format(
                        ", ".join(s.split(":")[-1] for s in sources)))
        md += ["## Illumination model test (cross-validated L1 residual, geometry from {})".format(geometry), "",
               "Explained fraction = 1 - r_model / r_none. Bounded = clipped to the decoder's ranges and blurred.", ""]
        models = ["global"] + ["local{}".format(P) for P in ic["patch_sizes"]] + ["bounded{}".format(P) for P in ic["patch_sizes"]]
        rows = []
        curves = {}
        for seq, fr in fits.items():
            r_none = np.array([float(r["r_none"]) for r in fr])
            r_id = np.array([float(r["r_identity"]) for r in fr])
            row = [seq, len(fr), fmt(float(np.mean([float(r["zbuffer_pass"]) for r in fr])), 2), fmt(r_id.mean(), 4), fmt(r_none.mean(), 4)]
            for m in models:
                rm = np.array([float(r["r_" + m]) for r in fr])
                ef = 1 - rm / r_none
                lo, hi = bootstrap_ci(ef, 2000, 0.05, np.random.default_rng(0))
                row.append("{} [{}, {}]".format(fmt(ef.mean(), 3), fmt(lo, 3), fmt(hi, 3)))
                curves[(seq, m)] = ef.mean()
            oob = np.mean([float(r.get("out_of_bounds_local16", "nan")) for r in fr])
            row.append(fmt(oob, 3))
            rows.append(row)
        md.append(_md_table(["sequence", "pairs", "z-buffer pass", "r identity", "r none"] + ["EF " + m for m in models] + ["oracle16 outside bounds"], rows))
        md.append("")
        if plt is not None:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            for seq in fits:
                ax.plot(ic["patch_sizes"], [curves[(seq, "local{}".format(P))] for P in ic["patch_sizes"]], "o-", label="{} local".format(seq))
                ax.plot(ic["patch_sizes"], [curves[(seq, "bounded{}".format(P))] for P in ic["patch_sizes"]], "s--", label="{} bounded".format(seq))
                ax.axhline(curves[(seq, "global")], ls=":", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("patch size (px)")
            ax.set_ylabel("explained fraction of residual")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(fdir, "explained_vs_patch.pdf"))
            plt.close(fig)
            md += ["![explained vs patch](figures/explained_vs_patch.pdf)", ""]

    # illum-params
    pfiles = sorted(glob.glob(os.path.join(cfg["out_dir"], "illum", "params_*.csv")))
    if pfiles:
        md += ["## Learned calibration parameters (per-frame statistics, mean over frames)", ""]
        cols = ["c_mean", "c_std", "c_p1", "c_p99", "b_mean", "b_std", "b_p1", "b_p99", "sat_c", "sat_b", "grad_c", "grad_b",
                "corr_c_radial", "corr_b_radial", "corr_c_highlight", "corr_b_highlight", "temporal_corr_c", "reciprocity_err",
                "res_before", "res_after", "res_oracle16"]
        rows = []
        for p in pfiles:
            fr = read_csv(p)
            name = os.path.basename(p)[7:-4]
            vals = []
            for c in cols:
                x = [float(r[c]) for r in fr if r.get(c) not in (None, "")]
                vals.append(fmt(float(np.mean(x)), 4) if x else "--")
            tz = [(float(r["tz_gt"]), float(r["b_mean"])) for r in fr if r.get("tz_gt") not in (None, "") and r["source"] == "1"]
            corr_tz = fmt(float(np.corrcoef([t for t, _ in tz], [b for _, b in tz])[0, 1]), 3) if len(tz) > 2 else "--"
            rows.append([name, len(fr)] + vals + [corr_tz])
        md.append(_md_table(["run/seq", "rows"] + cols + ["corr(b, gt tz)"], rows))
        md.append("")
        if plt is not None:
            for hp in sorted(glob.glob(os.path.join(cfg["out_dir"], "illum", "hist_*.npz"))):
                h = np.load(hp)
                fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
                for ax, key, bounds in ((axes[0], "c", (1 - float(h["alpha"]), 1 + float(h["alpha"]))), (axes[1], "b", (-float(h["beta"]), float(h["beta"])))):
                    e = h[key + "_edges"]
                    ax.bar(0.5 * (e[1:] + e[:-1]), h[key] / max(h[key].sum(), 1), width=e[1] - e[0])
                    for bnd in bounds:
                        ax.axvline(bnd, color="k", ls="--", lw=0.8)
                    ax.set_xlabel(key)
                fig.tight_layout()
                name = os.path.basename(hp)[5:-4]
                fig.savefig(os.path.join(fdir, "cb_hist_{}.pdf".format(name)))
                plt.close(fig)
                md += ["![c/b histogram {}](figures/cb_hist_{}.pdf)".format(name, name), ""]

    # illum-sens
    sfiles = sorted(glob.glob(os.path.join(cfg["out_dir"], "illum", "sens_*.csv")))
    if sfiles:
        md += ["## Sensitivity to synthetic gain / bias", ""]
        rows = []
        for p in sfiles:
            fr = read_csv(p)
            name = os.path.basename(p)[5:-4]
            cal = [r for r in fr if r["kind"] == "calib"]
            # the warp is c * (g * I + bias) + b, so the ideal response is c = 1/g and b = -bias.
            # Only perturbations the decoder can express (c in 1+-alpha, b in +-beta) are fitted:
            # outside that range clipping flattens the slope whatever the module does.
            alpha = float(cal[0].get("alpha", 0.10)) if cal else 0.10
            beta = float(cal[0].get("beta", 0.05)) if cal else 0.05
            g_all = [(float(r["gain"]), float(r["c_mean"])) for r in cal if float(r["bias"]) == 0.0]
            b_all = [(float(r["bias"]), float(r["b_mean"])) for r in cal if float(r["gain"]) == 1.0]
            g_rows = [(g, c) for g, c in g_all if abs(1.0 / g - 1.0) <= alpha + 1e-9]
            b_rows = [(bb, bm) for bb, bm in b_all if abs(bb) <= beta + 1e-9]
            dropped = "{}/{} gain and {}/{} bias settings outside the decoder's range".format(
                len(g_all) - len(g_rows), len(g_all), len(b_all) - len(b_rows), len(b_all))
            sg = np.polyfit([1 / g for g, _ in g_rows], [c for _, c in g_rows], 1)[0] if len(set(g for g, _ in g_rows)) > 1 else float("nan")
            sb = np.polyfit([-b for b, _ in b_rows], [bm for _, bm in b_rows], 1)[0] if len(set(b for b, _ in b_rows)) > 1 else float("nan")
            dep = [r for r in fr if r["kind"] == "depth"]
            base = np.mean([float(r["abs_rel"]) for r in dep if float(r["gain"]) == 1.0 and float(r["bias"]) == 0.0]) if dep else float("nan")
            worst = max([np.mean([float(r["abs_rel"]) for r in dep if float(r["gain"]) == g and float(r["bias"]) == b])
                         for g, b in {(float(r["gain"]), float(r["bias"])) for r in dep}]) if dep else float("nan")
            rows.append([name, fmt(sg, 3), fmt(sb, 3), fmt(base, 4), fmt(worst, 4), dropped])
        md.append(_md_table(["run", "slope c vs 1/gain", "slope b vs -bias", "Abs Rel clean",
                             "Abs Rel worst perturbation", "excluded"], rows))
        md += ["", "Slopes near 1 mean the calibration head tracks the injected illumination change. "
                   "Settings whose ideal response falls outside the decoder's bounds are excluded from "
                   "the fit, since clipping would flatten the slope regardless of the module's behaviour.", ""]

    with open(out_path(cfg, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("[report] wrote {}".format(out_path(cfg, "report.md")))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

STAGES = {"train": stage_train, "predict": stage_predict, "illum-fit": stage_illum_fit,
          "illum-params": stage_illum_params, "illum-sens": stage_illum_sens,
          "illum-grad": stage_illum_grad, "da3": stage_da3,
          "stats": stage_stats, "report": stage_report}
ALL_ORDER = ["predict", "illum-fit", "illum-params", "illum-sens", "da3", "stats", "report"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["all", "full"])
    ap.add_argument("--config", default=os.path.join(ROOT, "cviu_config.yaml"))
    ap.add_argument("--python", help="interpreter for the training subprocesses (overrides the config)")
    ap.add_argument("--gpus", nargs="*", help="GPU ids for train (one subprocess per GPU)")
    ap.add_argument("--only", nargs="*", help="restrict to these runs / methods")
    ap.add_argument("--seeds", nargs="*", type=int, help="train: override the seeds of every selected run")
    ap.add_argument("--datasets", nargs="*", help="restrict predict/da3 to these datasets")
    ap.add_argument("--extra_flags", default="", help="appended to every training command (e.g. \"--num_epochs 1\")")
    ap.add_argument("--dry_run", action="store_true", help="train: print the commands only")
    ap.add_argument("--skip_preflight", action="store_true", help="train: launch even if the pre-flight checks fail")
    ap.add_argument("--use_busy_gpus", action="store_true", help="train: also use GPUs that already hold memory")
    ap.add_argument("--force", action="store_true", help="redo work whose outputs exist")
    ap.add_argument("--all_seeds", action="store_true", help="illum-params: every seed of the run")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.python:
        cfg["python"] = args.python
    if args.seeds:
        cfg["train"]["seeds"] = cfg["train"]["multi_seeds"] = list(args.seeds)
    ensure_dir(cfg["out_dir"])
    if args.stage == "all":
        stages = ALL_ORDER
    elif args.stage == "full":
        stages = ["train"] if args.dry_run else ["train"] + ALL_ORDER
    else:
        stages = [args.stage]
    failed, n_train_failed = [], 0
    for s in stages:
        print("=" * 20, s, "=" * 20)
        try:
            ret = STAGES[s](cfg, args)
            if s == "train" and ret:
                n_train_failed = ret
                if len(stages) > 1:
                    print("[full] {} training job(s) failed; the analysis goes on with the finished runs "
                          "(the report lists what is missing). Fix them and rerun 'full': finished runs "
                          "and finished analysis are skipped.".format(n_train_failed))
        except Exception:
            traceback.print_exc()
            failed.append(s)
            if len(stages) == 1:
                sys.exit(1)
    if n_train_failed:
        print("TRAINING: {} job(s) failed, see {}".format(n_train_failed, out_path(cfg, "train_failures.json")))
    if failed:
        print("FAILED stages: {}".format(failed))
    if failed or n_train_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
