# MonoIIF – CVIU revision experiment plan

One script, `cviu_revision.py`, runs every new experiment the reviewers asked for. It reuses the
existing trainer, model factory and datasets; the only repo edits are flag-gated switches whose
defaults reproduce the current behaviour.

**Status (2026-09-15). The grid is finished.** 41 training jobs (the §3 grid plus the extra
seeds and the later additions `E8-IIF`, `E8-DVLoRA`, `C2-sup`, `C1-lora` at 3 seeds each), all
evaluated on SCARED, Hamlyn and C3VD. Every stage has run: `predict`, `illum-fit`,
`illum-params`, `illum-sens`, `illum-grad`, `da3`, `stats`, `report`. Outputs in
`results/cviu/`: `per_frame.csv`, `summary.csv`, `per_sequence.csv`, `paired.csv`, `report.md`,
`tables/*.tex`. Findings in §8; response-letter drafts in `CVIU_RESPONSE_LETTER.md`.

Still open:

- **External baselines are absent from Tables 4-6.** `EndoDAC_MICCAI` and `AF_SfMLearner`
  failed to load: the paths in `cviu_config.yaml` (`logs/endodac_fullmodel/models/weights_19`,
  `logs/Model_MIA`) do not exist on the server. HADepth and Endo-SfMLearner were never
  configured. Until these are pointed at real checkpoints, the tables compare only this grid's
  runs. **Partly addressed 2026-09-15 (§12):** Monodepth2 (`R1`), MonoII and MonoViT are now
  trained here from scratch under the shared recipe, so they enter the tables as grid runs
  instead of as foreign checkpoints — 9 jobs still to run.
- DA3-Mono-Large zero-shot: skipped, needs the `depth_anything_3` package (DA3-Base zero-shot
  did run, through the in-repo port).
- Pose evaluation: never measured.
- `save_pred` was turned off and `results/cviu/pred/` deleted when the shared 7 TB overlay hit
  100%; re-run `predict --force --only <method>` to regenerate disparities for figures.

The DA3 rows (D3, D3-EndoDAC, zero-shot `da3-base`; §6) use a port of DA3's code in the repo, so
they run in the same Python 3.8 environment as everything else. They only need the checkpoint:

```
wget -O pretrained_model/da3_base.safetensors https://huggingface.co/depth-anything/DA3-BASE/resolve/main/model.safetensors
python cviu_revision.py train --gpus 0 --only D3 --extra_flags "--num_epochs 1"   # dry run, then delete logs/cviu_D3_s314
```

Reviewer asks → experiments:

| Reviewer ask | Experiment | Script stage |
|---|---|---|
| R2: validate the local-affine illumination model (none vs global vs local), analyse the estimated parameters | C-grid (3 trained calibration variants), model-free affine fit on GT geometry, learned-parameter statistics, perturbation sensitivity | `train`, `illum-fit`, `illum-params`, `illum-sens` |
| R3: isolate each component (Depth Anything backbone, DV-LoRA, calibration, IIF loss) | E-grid (build-up + leave-one-out), R-grid (same losses on a ResNet-18 backbone) and B-grid (§12: the same two components on ResNet-18 / MPViT / Depth Anything, each also trained plain) | `train`, `predict`, `stats` |
| Q2: why not Depth Anything 3 | DA3 zero-shot rows on all test sets; optional DA3-Base backbone inside MonoIIF; limitations paragraph | `da3`, `predict`, `stats` |
| Q6: uncertainty + paired tests at the video-sequence level | Per-frame → per-sequence aggregation, bootstrap CIs, paired tests for Tables 4-6 | `predict`, `stats`, `report` |

---

## 1. The script: `cviu_revision.py`

```
python cviu_revision.py <stage> [--config cviu_config.yaml] [--gpus 0 1 2 3] [--dry_run] [--only RUN ...]
stages: train | predict | illum-fit | illum-params | illum-sens | da3 | stats | report | all
```

Design rules:

- **One config file** (`cviu_config.yaml`) lists data paths, checkpoints of competing methods,
  the run grid (name → CLI flags → seeds) and the output root (`results/cviu/`).
- **Every stage is idempotent.** It skips a run/prediction whose output file already exists, so
  the script can be re-launched after a crash or run stage-by-stage on different days.
- **`train` launches `train_end_to_end.py` as subprocesses** (one per run, round-robin over
  `--gpus`, `CUDA_VISIBLE_DEVICES` set per process). Subprocesses keep CUDA memory and the
  wandb run isolated; the script just builds the command line from the grid and waits.
- **`predict` never prints tables.** It writes one long-format CSV with one row per
  (method, seed, dataset, sequence, frame): `abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, ratio`
  plus the raw disparity `.npy` per method/dataset. All statistics come from that CSV.
- Models are loaded through `DepthModelFactory` (evaluate_depth_all.py); external
  predictions (Monodepth2, DA v1/v2, DA3) enter as `.npy` disparity/depth files, exactly like
  the existing `--ext_disp_to_eval` path.
- Sequence identity is derived inside the script: SCARED from the split-file folder
  (`datasetX/keyframeY`, the loader is not shuffled), Hamlyn from `inputs["sequence"]`, C3VD from
  the scene name (needs the one-line dataset edit in §2).
- `stats` and `report` are pure numpy/scipy/pandas and run on the laptop from the CSV.

Output layout:

```
results/cviu/
  runs/<run>_s<seed>/            # symlink or path to logs/<run>/models/weights_*
  pred/<method>_<dataset>.npy    # disparities, frame order = dataset order
  per_frame.csv                  # everything predict produced
  per_sequence.csv               # aggregated by stats
  illum/fit_<seq>.csv, params_<run>.csv, sens_<run>.csv
  tables/*.tex                   # Tables 4-6 (revised), ablation, calibration, DA3
  figures/*.pdf
  report.md
```

---

## 2. Repo edits required (all flag-gated, defaults = current behaviour)

| File | Edit | Why |
|---|---|---|
| options.py | `--illum_calib {none,global,local}` (default `local`) | R2 comparison |
| options.py | `--photometric {highlight,standard}` (default `highlight`) | R3 leave-one-out of the highlight-aware term |
| options.py | `--depth_backbone {endodac,resnet18}` (default `endodac`) | R3 backbone control |
| options.py | `--seed` (default 314), `--wandb_mode {online,offline,disabled}` | multi-seed runs; eval stages must not open wandb runs |
| trainer_end_to_end.py | move `wandb.init` out of module import into `Trainer.__init__`, honour `--wandb_mode` | the script imports the trainer for `illum-*` stages |
| trainer_end_to_end.py `predict_poses` | `none`: skip the lighting decoder, `c=1, b=0`; `global`: new `GlobalLightingHead` (global-pool the pose bottleneck → FC → 2 scalars → same tanh/α/β bounds → broadcast); `local`: unchanged | the global model must be a genuinely global head, not a pooled local map, so the comparison tests the hypothesis and not the decoder |
| trainer_end_to_end.py `compute_losses` | `standard` photometric = existing `compute_reprojection_loss` × automask; skip the IIF branch when `--illumination_invariant 0` | today IIF is still computed at weight 0 (wasted time, and the IIF mask still runs) |
| trainer_end_to_end.py | `resnet18` backbone = `encoders.ResnetEncoder(18)` + `decoders.DepthDecoder` (same `("disp", s)` keys); bypass `mark_only_part_as_trainable` for it | R-grid |
| train_end_to_end.py | `random_seeds(opt.seed)` | seeds |
| datasets/c3vd_dataset.py | `inputs["sequence"] = scan["sequence"]` in `__getitem__` | sequence-level stats on C3VD |
| datasets/ (new `__init__.py`) | export the four dataset classes | `evaluate_depth.py` / `evaluate_pose.py` already assume this and currently cannot import |

Loss implementation for the grid (fixed 2026-09-10, verified with a CPU unit test): the
highlight-aware loss now masks its SSIM and L1 terms together with automask × highlight mask
and returns the pure highlight mask; the IIF loss is supervised on the 3×3 erosion of that joint
mask, because a Robinson descriptor is only trustworthy where its whole 3×3 support is valid.
The eroded-mask coverage is logged per scale as `iif_mask_coverage/<scale>`; check it in the
1-epoch dry runs (it must stay well above zero, otherwise the automask is too noisy for erosion).
Every grid run must be trained from the same commit; record the hash in `cviu_config.yaml`;
`report` prints it.

---

## 3. Training grid (stage `train`)

Common: SCARED train split, 20 epochs, batch 8, lr 1e-4, `--learn_intrinsics True`,
`--lora_type dvlora --residual_block_indexes 2 5 8 11` unless stated; best-RMSE epoch as today.
Every run is evaluated on SCARED, Hamlyn and C3VD test sets by `predict`.

### E-grid: component build-up and leave-one-out (R3)

| Run | Backbone / adapter | Calib | IIF | Highlight | Flags beyond common |
|---|---|---|---|---|---|
| E0 DA-v1 zero-shot | frozen, no training | – | – | – | `.npy` from `mytest_da.py` (also DA-v2) |
| E1 head-only | DA frozen + DPT output heads | none | no | no | `--lora_type none --residual_block_indexes --illum_calib none --illumination_invariant 0 --photometric standard` |
| E2 +Conv-neck | + residual blocks | none | no | no | `--lora_type none --illum_calib none --illumination_invariant 0 --photometric standard` |
| E3 EndoDAC (baseline) | + DV-LoRA | none | no | no | `--illum_calib none --illumination_invariant 0 --photometric standard` |
| E4 +calib | E3 | local | no | no | `--illumination_invariant 0 --photometric standard` |
| E5 +IIF | E3 | none | yes | no | `--illum_calib none --photometric standard` |
| E6 +highlight | E3 | none | no | yes | `--illum_calib none --illumination_invariant 0` |
| E7 +calib +IIF | E3 | local | yes | no | `--photometric standard` |
| E8 MonoIIF (full) | E3 | local | yes | yes | (defaults) |
| E8−IIF | E3 | local | no | yes | `--illumination_invariant 0` |
| E8−calib (= C0) | E3 | none | yes | yes | `--illum_calib none` |
| E8−DVLoRA | plain LoRA | local | yes | yes | `--lora_type lora` |

E3 must reproduce the MICCAI EndoDAC numbers (Abs Rel ≈ 0.052 on SCARED); it is the fairness
anchor for every gain reported.

### R-grid: same methodology on a weak backbone (R3, "backbone vs method")

| Run | Flags |
|---|---|
| R1 ResNet-18 + standard loss (≈ Monodepth2/AF-SfMLearner setting) | `--depth_backbone resnet18 --illum_calib none --illumination_invariant 0 --photometric standard` |
| R2 ResNet-18 + calib + IIF + highlight | `--depth_backbone resnet18` |

Report (R2 − R1) next to (E8 − E3): the method's gain on a weak backbone vs on Depth Anything.
Report (E3 − E1) and (E1 − E0) as the backbone/adaptation share of the total gain.

### C-grid: illumination model (R2)

| Run | `--illum_calib` |
|---|---|
| C0 none | `none` (same run as E8−calib) |
| C1 global affine | `global` |
| C2 local affine (MonoIIF) | `local` (same run as E8) |
| C2-α/β sweep (optional) | `local` with decoder bounds α,β ∈ {0.05/0.025, 0.10/0.05, 0.20/0.10} |

### D-grid: Depth Anything 3 encoder and no-foundation control (Q2, R3; see §6)

| Run | Flags beyond common |
|---|---|
| D3 MonoIIF on DA3-Base encoder | `--backbone_weights da3` |
| D3-EndoDAC (E3 recipe on DA3-Base) | `--backbone_weights da3 --illum_calib none --illumination_invariant 0 --photometric standard` |
| N0 MonoIIF on a random-init ViT-B | `--backbone_weights none` |

### Seeds

3 seeds (314, 1, 2) for E3, E8, C0, C1, R1, R2, D3; 1 seed for the rest. Seed SD of the overall
mean is reported as training variability, separately from the sequence-level CI.

Count: 17 grid runs, 31 jobs with the extra seeds (`train.skip_runs` in the config drops
entries). Measure one 1-epoch dry run first and multiply; on 8 GPUs expect a few days.
Minimum viable subset if budget is short (9 runs): E1, E3, E8, E8−IIF, C0, C1, R1, R2, D3.

---

## 4. Illumination validation (R2)

### 4a. `illum-fit`: model-free test of the affine assumption (no training needed, run first)

Data: `splits/endovis/test_files_sequence1.txt` (dataset5/keyframe4, 410 frames) and
`test_files_sequence2.txt` (dataset3/keyframe4, 832 frames): consecutive frames with per-frame
GT depth (`scene_points`), GT relative pose (`frame_data` json, as in export_gt_pose.py) and K.

For each pair (t, t+1): warp I_{t+1} into frame t with GT depth/pose/K (`BackprojectDepth`,
`Project3D`, `grid_sample`, same as the trainer), keep valid pixels (in-bounds, valid GT depth,
depth-consistency occlusion check). Then compare residual |I_t − model(Ŵ)| for:

1. **none**: Ŵ as is
2. **global affine**: closed-form least-squares (c, b) over valid pixels
3. **local affine, patch P ∈ {64, 32, 16, 8}**: per-patch least squares with a ridge toward
   c=1, b=0 (avoids degenerate fits in flat patches)
4. **bounded local affine**: fit 3 then clip to the decoder's ranges (c ∈ [1−α, 1+α],
   b ∈ [−β, β]) and Gaussian-blur with the decoder's kernel (kmax = 11) → what the
   LightingDecoder is actually allowed to express

Cross-validated: fit on a checkerboard half of each patch's pixels, evaluate on the other half,
so extra degrees of freedom cannot win by construction. Outputs: mean L1, SSIM and explained
fraction 1 − r_model / r_none per model, per sequence, with bootstrap CI; a curve of explained
fraction vs patch size. Expected evidence: local ≫ global ≫ none, saturating around P = 16–32
(low-frequency spatially-varying illumination), and the bounded variant retaining most of the
gain.

### 4b. `illum-params`: statistics of the learned parameters

Load `pose_encoder.pth` + `lighting.pth` from E8/C2 (all seeds), run all test pairs of SCARED
(and Hamlyn/C3VD neighbours where available) and log per frame:

- distribution of c and b: mean, SD, p1/p50/p99; fraction saturated at the tanh bounds
  (|c−1| > 0.95α, |b| > 0.95β) → is the range adequate?
- spatial smoothness: mean |∇c|, |∇b| → low-frequency as intended?
- temporal consistency: correlation of c_t with c_{t+1}; forward/backward pair agreement
- physical plausibility: correlation of b and c with radial distance from the image centre
  (vignetting / light-source fall-off) and with the highlight mask; sign of mean b under
  camera approach vs retreat (from GT translation)
- residual reduction: photometric error of the warped image before vs after applying (c, b)
  with the model's own geometry, and the gap to the least-squares oracle from 4a.

### 4c. `illum-sens`: robustness / sensitivity

Inject synthetic global gain g ∈ {0.8 … 1.2} and bias ∈ {−0.05 … 0.05} into the source frame at
inference. Regress mean predicted c on g and mean b on the bias (slope ≈ 1 → the head reads
illumination); report Abs Rel change of the depth output under the same perturbation for E8 vs
C0 (depth robustness to illumination change). Also evaluate C0/C1/C2 on the Hamlyn and C3VD
test sets (different scopes and light sources) as the cross-domain robustness row.

Paper outputs: Table "none / global / local" (sequence-level, 3 seeds), Figure "explained
fraction vs patch size", Figure "c/b histograms + example maps + sensitivity slopes",
one paragraph of parameter statistics.

---

## 5. Sequence-level statistics (Q6) — stage `stats`

Unit of analysis = video sequence:

| Dataset | Sequences (n) |
|---|---|
| SCARED | 7 keyframe videos: dataset1/keyframe3, dataset2–7/keyframe4 (71–87 frames each) |
| Hamlyn | one per `rectifiedXX` folder in the test root (n printed by the script) |
| C3VD | 7 scenes: trans_t1_a, t1_b, t2_a, t2_b, t2_c, t3_a, t3_b |

Per-frame median scaling stays as it is (it is the field's protocol); the frame metric is
averaged within a sequence, then across seeds, giving one value per (method, sequence).

Reported per method and metric:

- mean ± SD over sequences; 95% CI by cluster bootstrap (resample sequences, 10 000 draws) and by
  t-interval with n−1 df (the current frame-level `st.t.interval` is anti-conservative because
  frames within a video are correlated — say so in the paper)
- paired against MonoIIF for every competing method: mean paired difference with bootstrap CI,
  paired t-test, exact Wilcoxon signed-rank, sign test (wins/losses), Cohen's d_z;
  Holm correction across methods within a table
- training-seed SD for the 3-seed runs (second uncertainty source)
- supplementary table with the per-sequence numbers

State in the text that with n = 7 sequences the exact Wilcoxon can only reach p = 0.016, so the
bootstrap CI of the paired difference is the primary evidence and differences of 0.001 in Abs Rel
are reported as statistically indistinguishable when the CI covers zero.

Methods entering Tables 4-6: MonoIIF (E8), EndoDAC (E3), HADepth checkpoint, AF-SfMLearner,
Endo-SfMLearner, MonoViT, Monodepth2 (`.npy`), DA v1/v2 zero-shot, DA3 zero-shot — all through
`predict`, all from the same per-frame CSV.

---

## 6. Depth Anything 3 (Q2) — stage `da3`

DA3 (ByteDance-Seed, 2025) is a multi-view "depth-ray" model on a plain DINOv2-style
transformer; monocular use is one input view. Public weights: DA3-Small/Base/Large/Giant and
DA3-Mono-Large (relative depth), on Hugging Face under `depth-anything/`.

1. **Zero-shot rows** (cheap, do first): DA3-Mono-Large and DA3-Base in single-view mode on
   the three test sets. DA3 predicts depth, not disparity, so use median scaling on depth like
   every other row; also give the affine-aligned number used for DA v1 in `mytest_da.py` for
   parity. Stage `da3` → `per_frame.csv` → `stats`.
2. **Adapted rows (D-grid, implemented and verified)**: DA3-Base's encoder is *not* a vanilla
   DINOv2: from block 4 on it uses QK-norm, 2D RoPE, a camera token and alternating
   local/global attention, and it outputs 1536-d features (last local state | normalised
   current state) at layers 5/7/9/11. Its weights therefore cannot be copied onto the repo's ViT.
   `models/endodac/da3_vit.py` ports DA3's encoder (Apache-2.0, commit 3d835ec) to plain
   Python 3.8 / torch, single view, with EndoDAC's Conv-neck inline, and reads the official
   `model.safetensors` directly. The repo's DPT head gains DA3's pre-norm and uv pos-embed
   options and is initialised with **DA3's own trained neck** (all projections, resize layers
   and fusion blocks), so D3 trains exactly like E8: frozen neck, DV-LoRA in the encoder MLPs,
   Conv-neck and conv_depth heads. Trainable parameters are identical (8,961,540 in both), so
   D3 vs E8 isolates the foundation model. DA3's ImageNet input normalisation is kept (the DA v1
   path feeds [0, 1], EndoDAC's convention). Selected with `--backbone_weights da3`
   (`--da3_weights`, default `pretrained_model/da3_base.safetensors`).
   - Verification (local, against the original DA3 code with shared random weights): encoder
     features and camera tokens identical (max |diff| = 0) at 224x280, 406x504 and 518x518;
     zero-shot depth of the port (encoder + DualDPT main branch, `da3_zeroshot.py`) identical
     to the original; all 207 encoder tensors match the real DA3-BASE checkpoint header; the
     zero-initialised Conv-neck and DV-LoRA leave DA3's features unchanged at step 0.
   - **D3** = full MonoIIF recipe on the DA3 encoder (3 seeds); **D3-EndoDAC** = E3 recipe on
     the DA3 encoder. (D3 − D3-EndoDAC) vs (E8 − E3) shows whether the method's gain transfers
     to the newer foundation model; D3 vs E8 is the backbone comparison the reviewer asked for.
   - **N0** = full recipe with a randomly initialised ViT-B (`--backbone_weights none`): the
     "no foundation weights" control for the backbone-share question of R3. Its DA v1 neck does
     not match a random encoder, so its whole head trains (report its larger trainable count).
   - The zero-shot `da3-base` row uses the same port (no package). DA3-Mono-Large (ViT-L,
     different head) still needs the `depth_anything_3` package and is skipped without it.
3. **Limitations text**: DA3's monocular model is Large-only (~0.3 B params), so it enters only
   as a zero-shot row; DA3-Base is compared both zero-shot and adapted (D3). D3 keeps DA3's
   neck but replaces its depth-ray output head with EndoDAC's multi-scale disparity heads, as
   E8 does for Depth Anything v1, and runs single-view at the grid's 224x280 resolution.

---

## 7. Order of work

| Week | Work |
|---|---|
| 1 | Repo edits (§2) + decision on the pending highlight-loss fixes; 1-epoch dry runs of E1, E3, E8, C1, R1; `illum-fit` (needs no training) and DA3 zero-shot; sanity-check E3 against MICCAI numbers |
| 2–3 | `train` grid on the server; `predict` as checkpoints land |
| 3 | `illum-params`, `illum-sens`; `stats`; DA3 adapted row if the gate passes |
| 4 | `report`: tables (revised 4-6, ablation, calibration, DA3), figures, response letter text |

Risks to watch: E3 not reproducing EndoDAC (then the whole grid is uninterpretable — stop and
debug before launching the rest); the `none` calibration run changing the automask balance; the
`global` head collapsing to c=1, b=0 (check its statistics before interpreting C1); Hamlyn
sequence count small enough that only bootstrap CIs are meaningful.

---

## 8. Results (2026-09-15)

All numbers are SCARED Abs Rel, **sequence-level** (n = 7 keyframe videos), from
`results/cviu/summary.csv` and `results/cviu/paired.csv`. "vs C1" is the paired difference
`method − C1` per sequence with a 10 000-draw cluster bootstrap CI, signed so that **positive
means C1 is better**; `wins` counts the sequences where the method beats C1.

### 8a. On SCARED the best configuration is C1 (global affine calibration)

**Read §8g before quoting anything from this subsection.** C1 is the best configuration *on
SCARED, the training domain*. Out of domain the ranking changes completely and neither of the
two claims below replicates.

| Run | seeds | Abs Rel | seed SD | vs C1 | CI | wins | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|
| **C1** global affine | 3 | **0.04968** | 0.0021 | — | — | — | — |
| C1-lora global + plain LoRA | 3 | 0.04987 | 0.0012 | +0.00019 | [−0.00127, +0.00155] | 4/7 | 0.94 |
| E8-DVLoRA local + plain LoRA | 3 | 0.05031 | 0.0008 | +0.00063 | [−0.00089, +0.00238] | 3/7 | 0.81 |
| E8−IIF local, no IIF loss | 3 | 0.05052 | 0.0009 | +0.00084 | [−0.00073, +0.00236] | 2/7 | 0.38 |
| C2-sup local, LS-supervised | 3 | 0.05114 | 0.0014 | +0.00146 | [+0.00047, +0.00257] | 1/7 | 0.031 |
| E8 MonoIIF (local affine) | 3 | 0.05116 | 0.0014 | +0.00148 | [+0.00050, +0.00319] | **0/7** | 0.016 |
| E3 EndoDAC baseline | 3 | 0.05166 | 0.0009 | +0.00198 | [+0.00043, +0.00416] | 2/7 | 0.078 |
| C0 no calibration | 3 | 0.05184 | 0.0029 | +0.00216 | [+0.00064, +0.00397] | 1/7 | 0.047 |
| D3 MonoIIF on DA3-Base | 3 | 0.05448 | 0.0002 | +0.00480 | [+0.00157, +0.00840] | 1/7 | 0.031 |
| R2 ResNet-18 + full method | 3 | 0.05926 | 0.0015 | +0.00958 | [+0.00453, +0.01534] | 0/7 | 0.016 |
| R1 ResNet-18 + standard loss | 3 | 0.05929 | 0.0015 | +0.00961 | [+0.00561, +0.01357] | 0/7 | 0.016 |
| N0 random-init ViT-B | 1 | 0.10606 | — | +0.05638 | [+0.04329, +0.07027] | 0/7 | 0.016 |

Single-seed rows, for the build-up only (their differences are of the same size as the seed SD
of ~0.001-0.002, so they do not support claims): E1 0.08693, E2 0.06056, E4 0.05284,
E5 0.05201, E6 0.05135, E7 0.05350, D3-EndoDAC 0.05272.

**C1 beats C0 (calibration helps) and beats E8 (global beats local, in 7 of 7 sequences)**, both
with CIs excluding zero. `C1-lora`, added to test whether the two winning settings compose, is a
dead tie with C1 (d_z = 0.09): the gain comes from the calibration, not from the adapter type.

### 8b. Multiplicity: Holm over 20 comparisons is powerless by construction

With n = 7 the smallest attainable two-sided exact Wilcoxon p is 2/2⁷ = 0.0156, so Holm over the
20 rows of the table cannot go below 0.31 — and indeed no `p_wilcoxon_holm` in `paired.csv` does.
The fix is to pre-specify the two primary comparisons and declare the rest exploratory:

| Primary comparison | question | p | Holm (family = 2) |
|---|---|---|---|
| C1 vs E8 | global or local affine? | 0.0156 | **0.031** |
| C1 vs C0 | does calibration help at all? | 0.0469 | **0.047** |

Both below 0.05. Everything else is reported with bootstrap CIs and no correction, stated as
such.

### 8c. Most of the performance is the foundation model, not the method

| Step | Abs Rel | Δ | share of N0 → C1 |
|---|---|---|---|
| N0 random-init ViT-B | 0.10606 | — | — |
| E1 + DA v1 frozen weights, DPT heads trained | 0.08693 | 0.01913 | 34% |
| E2 + Conv-neck | 0.06056 | 0.02637 | 47% |
| E3 + DV-LoRA (= EndoDAC) | 0.05166 | 0.00890 | 16% |
| C0 + HADepth's highlight loss + our IIF loss | 0.05184 | -0.00018 | **-0.3%** |
| C1 + our global affine calibration | 0.04968 | 0.00216 | **3.8%** |

**Corrected 2026-09-15.** This table previously ended with a single row "+ our calibration,
0.00198", which was wrong: `C1` is `--illum_calib global` alone, so it inherits the defaults
`--illumination_invariant 0.1` and `--photometric highlight` and that 0.00198 bundled three
components, one of them prior work. `C0` (`--illum_calib none`, IIF and highlight on) is the run
that isolates them: E3 -> C0 is the highlight term (HADepth Eq. 6-11, implemented in
`trainer_end_to_end.py::get_highlight_mask` / `compute_highlight_aware_loss`) plus our IIF loss,
together worth **nothing** (-0.00018), and C0 -> C1 is our calibration alone, +0.00216 with the
CI excluding zero (§8a). Attribution of the whole N0 -> C1 span: 34% the foundation weights,
63% EndoDAC's adaptation machinery, -0.3% HADepth's highlight term plus our IIF loss, **3.8% our
calibration**.

The pretrained weights and EndoDAC's adaptation machinery (Conv-neck + DV-LoRA) account for
~96% of the span; our contribution on top of a faithfully reproduced EndoDAC is a 3.8% relative
reduction in Abs Rel whose CI excludes zero. E3 = 0.0517 reproduces the published EndoDAC
number (0.052), so the anchor holds.

**RETRACTED 2026-09-15.** This paragraph read: "the R-grid is a negative result worth stating: on
a ResNet-18 backbone the full method is R2 − R1 = −0.00003, i.e. nothing; the benefit of the
photometric machinery appears only on top of a strong foundation backbone." **That is not a test
of MonoII.** `R2` is ResNet-18 with λ₁ = **0.1** *and* HADepth's highlight term; MonoII is
ResNet-18 with the calibration and the II loss at λ₁ = **0.5** and monodepth2's photometric loss.
Two different configurations.

The paper's own Table 2 points the other way on the transformer backbone: line 14 (no calibration,
II at 0.5) gives 0.057 and line 11 (calibration, II at 0.5) gives 0.055 — the same order as the
+0.0022 we measure for the calibration on the foundation backbone. And Table 2 contains no row at
all that isolates the calibration on ResNet, so the paper does not evidence it there either.

Whether the two components help independently of the architecture is therefore an **open
question**, and the B-grid of §12 is the design that answers it: R1/MonoII, MonoViT/MonoViT-II and
E3/M-local, three seeds each, every pair differing only in the calibration and the II loss at the
published λ₁.

### 8d. The IIF loss does not help on SCARED

`E8-IIF` is E8 with `--illumination_invariant 0`. With 3 seeds it is **better** than E8:
0.05052 vs 0.05116. Against C1, E8's difference is +0.00148 with a CI excluding zero while
E8-IIF's is +0.00084 with a CI covering zero. The single-seed rows point the same way
(E5, +IIF only, 0.05201 vs E3 0.05166; E7, calib+IIF, 0.05350 vs E4, calib only, 0.05284).
`illum-grad` independently measured the IIF term as contributing 2.1% of the gradient reaching
the calibration decoder (SSIM 44.4%, L1 53.5%).

Consequence for the paper: the IIF term cannot be presented as a source of accuracy. Either it
is reframed (e.g. as robustness under illumination change, which `illum-sens` can support) or
dropped from the claims. This is a framing decision, not a bug.

### 8e. Illumination model (R2)

- `illum-fit`, model-free least squares on GT geometry: local affine explains ~10× more
  photometric residual than global (0.040 vs 0.004 of the residual), so the local model is the
  better *photometric* model — and yet it gives worse depth (E8 loses to C1 in 7/7 sequences).
- `illum-params`: the learned local maps do not track illumination (slope −0.006 against
  injected gain). They are not collapsed either (c SD across inputs 0.0136 vs 0.0287 within an
  image), so the module is active but fitting something other than illumination.
- `C2-sup`, supervising the local maps towards their per-patch least-squares fit
  (`calib_supervision`, `utils/layers.py:calibration_supervision_loss`), collapses them to the
  identity (c mean 0.9995, SD 0.0026) and does not recover C1's accuracy (0.05114 vs 0.04968).

Reading: the extra degrees of freedom of the dense maps absorb photometric residual that is not
illumination — most plausibly geometric error — and so weaken the depth gradient. The bounded
global head, with 2 parameters per pair, cannot do that. This is consistent with every
measurement above, but it is an interpretation; the falsifying experiment would be a local head
constrained to a low-order spatial basis, which was not run.

**This reading is confounded — see §9.** Every run above was trained with a colour jitter that
is re-sampled per frame, so on half the training items the illumination relation the lighting
head observes is dominated by augmentation noise. Nothing in §8a-§8c depends on it (all runs
share it), but §8e's claim that the module "does not track illumination" cannot be separated
from the possibility that it was never shown a usable signal. The A-grid of §9 settles it.

### 8g. Cross-dataset: the calibration result does not replicate

Abs Rel per dataset, the seven 3-seed configurations that differ only in the photometric
machinery (SCARED n=7 sequences; Hamlyn n=58 blocks of 100 frames from its single sequence, so
its intervals are optimistic; C3VD n=7 scenes). Best per column in bold.

| Run | what it is | SCARED | Hamlyn | C3VD | rank sum |
|---|---|---|---|---|---|
| E3 | EndoDAC baseline, no calibration / IIF / highlight | 0.0517 | **0.1565** | 0.2929 | 6+1+7 |
| C0 | E8 without calibration | 0.0518 | 0.1593 | 0.2658 | 7+2+3 |
| C1 | global affine calibration | **0.0497** | 0.1595 | 0.2717 | 1+3+5 |
| C1-lora | global affine + plain LoRA | 0.0499 | 0.1598 | 0.2751 | 2+4+6 |
| C2-sup | local affine supervised by the LS fit | 0.0511 | 0.1599 | 0.2611 | 5+5+2 |
| E8 | local affine (the submitted MonoIIF) | 0.0512 | 0.1608 | 0.2679 | 4+6+4 |
| E8-IIF | E8 without the IIF loss | 0.0505 | 0.1623 | **0.2509** | 3+7+1 |

Paired against C1, the two claims of §8a **fail to replicate**:

| Claim | SCARED | Hamlyn | C3VD |
|---|---|---|---|
| calibration helps (C1 vs C0) | +0.00216 [+0.00064,+0.00397] | -0.0002 [-0.0023,+0.0020] tie | -0.0059 [-0.0147,+0.0028] tie |
| global beats local (C1 vs E8) | +0.00148 [+0.00050,+0.00319] 7/7 | +0.0013 [-0.0004,+0.0030] tie | -0.0038 [-0.0138,+0.0094] tie |

And two results run the other way out of domain:

- **Hamlyn: the plain EndoDAC baseline E3 wins**, -0.0031 [-0.0048,-0.0013] against C1, 37/58
  blocks, p = 0.002. None of the photometric machinery earns its place there.
- **C3VD: E8-IIF wins**, -0.0208 [-0.0322,-0.0103] against C1, 6/7 scenes, p = 0.031. Removing
  the IIF loss buys 0.017 Abs Rel, ~6% relative — the largest single effect any component has on
  that dataset, and it is negative for the component the method is named after.

What does replicate on all three datasets, with CIs excluding zero: the ResNet-18 control loses
(R1, R2), the DA3 backbone loses (D3), and the random-init encoder loses by a wide margin. In
other words **only the large, backbone-level effects survive a change of domain; every
photometric-component effect is within noise of zero somewhere.**

Consequences for the paper:

1. §8a must be quoted as "on SCARED", never as "the best configuration".
2. The R2 answer cannot claim that calibration helps in general. What the data supports is: it
   helps in-domain, it is neutral out of domain, and the *global* form is never worse than the
   local one by more than noise while being clearly better in-domain. That is a defensible but
   much narrower claim.
3. §8d (the IIF loss does not help) gets stronger, not weaker: removing it is neutral on SCARED,
   costs 0.0015 on Hamlyn and gains 0.017 on C3VD.
4. Reporting only SCARED would be the reviewer's worst suspicion confirmed. All three datasets
   go in the paper with this table.

### 8h. Leave-one-out of every component of the pipeline (all three datasets)

`diff = variant - E8` per sequence, so **negative means removing the component improved
accuracy**. Bootstrap CI, 10 000 draws, from `results/cviu/per_sequence.csv`.

| Removed | run | SCARED (n=7) | Hamlyn (n=58) | C3VD (n=7) | seeds |
|---|---|---|---|---|---|
| affine calibration *(ours)* | C0 | +0.0007 [-0.0010,+0.0028] | **-0.0015 [-0.0028,-0.0003]** | -0.0020 [-0.0111,+0.0087] | 3 |
| IIF loss *(ours)* | E8-IIF | -0.0006 [-0.0021,+0.0008] | +0.0015 [-0.0001,+0.0030] | **-0.0170 [-0.0270,-0.0073]** | 3 |
| highlight loss *(HADepth)* | E7 | +0.0023 [-0.0017,+0.0082] | **-0.0048 [-0.0071,-0.0025]** | **+0.0342 [+0.0257,+0.0422]** | 1 |
| DV-LoRA -> LoRA *(EndoDAC)* | E8-DVLoRA | -0.0008 [-0.0017,+0.0001] | **+0.0030 [+0.0016,+0.0044]** | **+0.0219 [+0.0149,+0.0295]** | 3 |

Build-up one component at a time on top of E3 (delta Abs Rel, negative = better):

| Variant | SCARED | Hamlyn | C3VD |
|---|---|---|---|
| E4 = E3 + calibration | +0.0012 | -0.0014 | -0.0057 |
| E5 = E3 + IIF | +0.0004 | +0.0015 | +0.0056 |
| E6 = E3 + highlight | -0.0003 | +0.0036 | -0.0003 |
| E7 = E3 + calibration + IIF | +0.0018 | -0.0005 | +0.0091 |
| E8 = E3 + all three | -0.0005 | +0.0043 | **-0.0250** |

Readings:

1. **No component of ours improves accuracy on more than one dataset.** In-domain not even the
   individual additions help: only the three together beat E3, by 0.0005.
2. The **highlight-aware loss is the only component with a large effect anywhere**: +0.0342 on
   C3VD (0/7 scenes for the variant without it), which is essentially the whole -0.0250 that E8
   gains over E3 there. **It is HADepth's, not ours** -- `trainer_end_to_end.py::get_highlight_mask`
   implements HADepth Eq. 6-8 and `compute_highlight_aware_loss` its Eq. 9-11, and it is the
   default (`--photometric highlight`) in every run of the C-, D- and R-grids. It is negative on
   Hamlyn and has **one seed**, so two more seeds of E7 (~21 h on two GPUs) are needed to close
   the attribution between HADepth's term and ours, not to build a claim of our own.
3. **Retract the earlier "plain LoRA beats DV-LoRA" note** (memory and the 2026-09-13 results):
   it held on SCARED alone with the interval touching zero, and out of domain DV-LoRA wins on
   both datasets with intervals excluding zero. DV-LoRA is EndoDAC's component, not ours, and it
   is the one whose removal costs most.

### 8i. The method as the authors define it is E7, and it is a tie with EndoDAC (2026-09-15)

MonoIIF is EndoDAC's components plus **two** things: the illumination-invariant (ILL) loss and the
local affine calibration. The highlight-aware term is HADepth's and is not part of the method.
The grid run that matches that definition is therefore **E7** (`--photometric standard`, with the
default local calibration and IIF weight), *not* E8, and not C1.

| Dataset | E3 (EndoDAC) | E7 (the method) | E7 - E3 | wins | p |
|---|---|---|---|---|---|
| SCARED | 0.0517 | 0.0535 | +0.0018 [-0.0022,+0.0077] | 3/7 | 0.94 |
| Hamlyn | 0.1565 | 0.1559 | -0.0005 [-0.0022,+0.0012] | 36/58 | 0.21 |
| C3VD | 0.2929 | 0.3020 | +0.0091 [-0.0008,+0.0205] | 2/7 | 0.30 |

**All three intervals cover zero: the method is statistically indistinguishable from the EndoDAC
baseline on all three datasets.** Its two components separately, on top of E3: local calibration
alone (E4) +0.0012 / -0.0014 / -0.0057, IIF alone (E5) +0.0004 / +0.0015 / +0.0056.

**Design error in this grid, to fix before anything else.** The three runs that describe the
method (E4, E5, E7) each have **one seed**, while all eleven three-seed runs (E8, C0, C1,
E8-IIF, E8-DVLoRA, C2-sup, C1-lora, D3, R1, R2, E3) include HADepth's highlight term. E3's
seed SD on SCARED is 0.0009, so a single-seed difference of 0.0018 carries no information. The
seed budget went to the wrong runs.

Runs added 2026-09-15: E4, E5, E7 join `multi_seed_runs` (seeds 1 and 2 to train), and **C1-std**
(`--illum_calib global --photometric standard`, 3 seeds) re-tests "global beats local" inside the
method's own definition, since the C1-vs-E8 result of §8a was measured with HADepth's term
present in both arms.

**Question of fact for the authors.** The trainer applied the highlight-aware loss
unconditionally from commit `e65b446` ("ill + hlam", 2026-03-06) until `--photometric` was added
during this revision, with `highlight` as its default. Any model trained after that date includes
it. If the submitted paper's numbers came from such a run, the paper's model is E8, not E7, and
includes a component of HADepth's without crediting it. Check the checkpoint that produced the
submitted tables:

```
python -c "import json;print(json.load(open('logs/<paper_model>/models/opt.json')))"
ls -l --time-style=long-iso logs/<paper_model>/models/
```

### 8f. Illumination calibration: code review (2026-09-15)

Checked and clean, so these are not alternative explanations:

- The automask is computed from the **uncalibrated** warp and the raw source frame
  (`trainer_end_to_end.py:709-714`), so the calibration cannot widen the mask it is scored on.
- The highlight mask is computed from the **target** (`trainer_end_to_end.py:448`), so the
  calibration cannot dodge specular pixels by darkening its own output.
- `("ch"/"bh", scale, f_i)` are interpolated to full resolution and `("color", f_i, scale)` is
  full resolution too (`source_scale = 0` with `v1_multiscale` off), so the broadcast is right.
- `GlobalLightingHead`'s zero-initialised head starts exactly at c=1, b=0 with live gradients.

One structural point that does bear on local vs global: the photometric loss is
0.85·SSIM(window 7) + 0.15·L1, and SSIM factors into luminance × contrast × structure, both
normalised inside the window. A **local** affine map, bounded and Gaussian-blurred at a scale
comparable to that window, can drive the luminance and contrast factors towards 1 patch by patch
**whether or not the geometry is right**, leaving only the structure term to carry depth
information; two global scalars cannot. This matches `illum-grad`, where L1 supplies 53.5% of the
gradient reaching the calibration while weighing 0.15.

---

## 9. The colour-augmentation defect and the A-grid (found 2026-09-15)

`datasets/mono_dataset.py` built the jitter as `transforms.ColorJitter(...)` and called it **once
per frame** (`preprocess`, one call per image of the item). A constructed `ColorJitter`
re-samples its factors inside every call, so each frame of an item received a *different*
jitter. Verified on a constant grey image: five calls of one object give means
109 / 120 / 143 / 123 / 150. The docstring promised the opposite ("apply the same augmentation
to all images in this item … so that all images input to the pose network receive the same
augmentation"); monodepth2 achieved that with the *static* `ColorJitter.get_params`, which
returns fixed factors. The line comes from this repo's `Initial` commit, i.e. it is inherited
from EndoDAC, not introduced by the revision.

Why it matters here specifically: the lighting head reads `color_aug`
(`trainer_end_to_end.py:606`) while the (c, b) it predicts is applied to the **un-augmented**
warp (`trainer_end_to_end.py:909`). With `do_color_aug` true for half the items and brightness
and contrast drawn independently from [0.8, 1.2] per frame, the apparent gain between the two
frames the head observes carries random noise of up to 1.2/0.8 = 1.5, while the head can only
express c ∈ [0.9, 1.1] — and that noise is independent of the relation it must correct. The
optimal predictor under such an input shrinks to a constant.

That single defect predicts four measurements we had been treating as separate findings:

| Measurement | What the defect predicts |
|---|---|
| c ≈ 1.043, b ≈ −0.018 nearly constant; temporal corr. 0.92-0.94 | shrinkage to the prior mean |
| sensitivity slope −0.006 instead of ≈ 1 | the input does not inform the target |
| local (E8) worse than global (C1), 0/7 sequences | the dense map has the capacity to fit the noise; two scalars degrade gracefully |
| C2-sup collapses to the identity (c 0.9995 ± 0.0026) | its LS target is computed un-augmented, so it is unpredictable from an augmented input, and the L1 minimiser is the median |

**Fix and test.** `--color_aug_consistent` (default `False`, so the 41 runs above stay
reproducible and E3 still reproduces EndoDAC) makes `MonoDataset._sample_color_aug` draw the
factors once and apply them to every frame. Verified locally: five frames of an item get an
identical, non-identity jitter; different items still differ; the legacy path is unchanged.

A-grid, one seed each, ~11 h on three GPUs:

| Run | Flags |
|---|---|
| A-C0 | `--illum_calib none --color_aug_consistent True` |
| A-C1 | `--illum_calib global --color_aug_consistent True` |
| A-E8 | `--color_aug_consistent True` |

Then `illum-params` and `illum-sens` on `A-E8`. **Decision rule fixed in advance:** if the
sensitivity slope moves from −0.006 towards 1, the module was starved of signal and §8e must be
rewritten — the paper's claim would become "local calibration works once it is trained on
coherent data", and the local-vs-global comparison would have to be re-run at 3 seeds. If the
slope stays flat, §8e stands and is now defended against exactly this objection from a reviewer.

---

## 10. Paper vs code: discrepancies found on reading the submission (2026-09-15)

The submitted PDF was read after the grid finished. Three of its settings do not match the code
the grid was trained with, and one promised experiment does not exist.

| Paper | Code / grid | Consequence |
|---|---|---|
| λ₁ = **0.5** for the II loss (§3.3, Table 2 sweeps {0.25…10}) | `--illumination_invariant` default **0.1** (options.py:266), used by every E/C/D/R run | **0.1 is below the whole range the paper tested.** No run of the 41 is the published method. §8d ("the IIF loss does not help") is measured at one fifth of the published weight and is **retracted**. |
| Depth Anything **V2** encoder (abstract, §2.2.4, §3.3, §5, conclusion) | loads `depth_anything_vitb14.pth` (endodac.py:267), i.e. Depth Anything **v1** | **Resolved by the authors 2026-09-15: the code is right, the paper is wrong.** The backbone is Depth Anything **v1** (ViT-B/14). Every "V2" claim about the *proposed* model must be corrected — see §10c. Internal evidence agrees: the paper itself says MonoIIF "shares its adapted backbone with EndoDAC", and EndoDAC uses v1. |
| batch size **12**, encoder frozen the first **5 epochs** then DVLoRA + calibration fine-tuned 15 (§3.3) | `--batch_size 8`; freezing is governed by `--warm_up_step 20000`, which at batch 8 over 15 351 images is ≈10.4 epochs (≈15.6 at batch 12) | Neither matches 5 epochs. Check what `warm_up_step` actually does before quoting §3.3. |
| §2.1.3: "a quantitative comparison under an identical training framework is given in Section 3.4" for Robinson-8 **vs Sobel, Scharr and Census** | no such experiment exists, and the three descriptors are not implemented | A promised comparison with no data behind it. Either implement the three descriptors (≈1 day of work plus 3-4 runs) or remove the sentence. |

### 10c. "Depth Anything V2" must be corrected to v1 throughout the paper

The proposed model uses **Depth Anything v1, ViT-B/14** (`pretrained_model/depth_anything_vitb14.pth`).
Places where the submission claims V2 *for the proposed model* and must be changed:

- **Abstract**: "a Depth Anything V2 foundation model encoder adapted via domain-specific low-rank
  adaptation (MonoIIF)".
- **§2.2.4, "Foundation model backbone (MonoIIF)"**: "builds upon the Depth Anything V2 foundation
  model (Yang et al., 2024b)".
- **§3.3**: "For MonoIIF, the Depth Anything V2 encoder …" (the same sentence carries the TODO on
  encoder size: it is ViT-B/14, 8 961 540 trainable parameters under DV-LoRA).
- **§3.4, "Foundation model backbone"**: "the pretrained geometric priors of the Depth Anything V2
  backbone", and the Table 3 TODO's "zero-shot DA-V2" row.
- **§3.5, "MonoIIF model"**: "leverages the Depth Anything encoder (Yang et al., 2024b)" — the text
  is fine, the citation points at v2.
- **§5**: "The experiments of this work rely on the Depth Anything V2 encoder."
- **§6 Conclusion**: "within a Depth Anything V2 encoder adapted with low-rank updates (MonoIIF)".
- **References**: `Yang et al. 2024a` and `2024b` are two entries for the same v2 paper. One of
  them must become the v1 reference (Yang, Kang, Huang, Xu, Feng, Zhao, *Depth Anything:
  Unleashing the Power of Large-Scale Unlabeled Data*, CVPR 2024).

The mentions of Depth Anything V2 in §1.2 and §1.3, which describe **related work**, stay as they
are.

One consequence worth weighing: with the text corrected, a reviewer who already asks "why not
Depth Anything 3" will equally ask "why not v2". Two ways to answer, in increasing cost:

1. State it in the limitations: v1 was chosen to share EndoDAC's backbone exactly, so that the
   comparison isolates the losses rather than the foundation model. That is true and is the reason
   the whole E-grid is interpretable.
2. Run it: DA v2 ViT-B is the same DINOv2-based architecture as v1, so a `--backbone_weights da2`
   option is a few lines pointing at `depth_anything_v2_vitb.pth`, plus one zero-shot row and one
   adapted run at 3 seeds (~32 h). This would give the paper v1 / v2 / v3 on the same recipe with
   identical trainable-parameter counts, which is a strong answer to Q2 and cheap relative to what
   has already been spent.

### 10a. M-grid: the published method

| Run | Flags | What it is |
|---|---|---|
| M-local | `--photometric standard --illumination_invariant 0.5` | **MonoIIF as published**: local calibration + II at λ₁ = 0.5 |
| M-none | `--illum_calib none --photometric standard --illumination_invariant 0.5` | II only, no calibration |
| M-global | `--illum_calib global --photometric standard --illumination_invariant 0.5` | global calibration + II, to re-test §8a at the published λ₁ |

Three seeds each; `E3` (PML only) and `E4` (calibration only, λ₁ = 0) are the remaining arms of the
Table-3 ablation the paper asks for and already have, or will have, three seeds. `E5` and `E7` stay
single-seed: at λ₁ = 0.1 they are λ-sensitivity points, not the method. `C1-std` was dropped in
favour of `M-global`.

Everything measured at λ₁ = 0.1 remains valid **as a comparison between its own arms** (both sides
share λ₁), so §8a's calibration result, §8c's backbone attribution and §8g's cross-dataset table
still stand as statements about that configuration. What none of them can be called is "the
method".

### 10b. Paper TODOs the existing results already answer

- §3.1 test counts: SCARED 551 frames / **7** keyframe sequences; Hamlyn 5 737 usable frames /
  **1** sequence (analysed as 58 blocks of 100 frames); C3VD 1 757 frames / **7** scenes.
- §3.3 seed policy: 3 seeds (314, 1, 2) for every claim-bearing run, seed SD reported separately
  from the sequence-level CI (0.0008-0.0029 Abs Rel on SCARED).
- §2.2.4 / §3.3 parameter counts: ViT-B/14 encoder, **8 961 540** trainable parameters.
- Table 4 / §4.1.1 / §4.1.2 per-sequence statistics, bootstrap CIs and paired tests: `paired.csv`,
  `per_sequence.csv`, `tables/paired_*.tex`. Still missing the external checkpoints (HADepth,
  EndoDAC, MonoPCC) — see the Status block.
- §5 / §22 Depth Anything 3: zero-shot DA3-Base (0.0718 median / 0.0692 affine on SCARED) and
  adapted D3 (0.0545) are measured; §6 of this plan has the text.
- Discussion item (iv), the statistical caveat on few test sequences: §8b.

---

## 11. λ₁ sweep for the II loss (lam-da grid), and the checkpoint-selection bias

### 11a. Why the paper's Table 2 is not the answer

Table 2 of the submission sweeps λ₁ ∈ {0.25, 0.5, 1, 2, 3, 4, 5, 10} and picks 0.5, but it does
so **on the ResNet variant** (lines 2-10 have no transformer blocks), with one seed, on frame-level
means. MonoIIF, the paper's primary contribution, has no sweep at all: its λ₁ = 0.5 is inherited
from a different backbone. The differences between adjacent λ₁ in that table (0.058 / 0.060 /
0.063) are the size of the seed SD we measure here (0.0008-0.0029), so with one seed the optimum
is not resolvable.

### 11b. The design

Six points, all with the method's structure (local calibration, monodepth2 photometric loss),
differing only in λ₁, **three seeds each**:

| λ₁ | Run | Status |
|---|---|---|
| 0 | E4 | exists (1 seed), +2 queued |
| 0.1 | E7 | exists (1 seed), +2 to train — the repo default the whole grid used |
| 0.25 | **lam-da-025** | new |
| 0.5 | M-local | queued — the published value |
| 1.0 | **lam-da-100** | new |
| 2.0 | **lam-da-200** | new |

Three seeds, not one, because a λ curve drawn through single runs cannot separate the optimum from
seed noise — which is exactly the weakness of the paper's Table 2. Analysis uses the existing
sequence-level machinery: the six points share the same test sequences, so λ values are compared
**paired across sequences**, far more powerful than comparing means. `stats` writes
`tables/lambda_sweep.tex` (SCARED metrics plus Hamlyn and C3VD Abs Rel per λ).

Cost: 11 new jobs (lam-da-025, lam-da-100, lam-da-200 × 3 seeds, plus E7 seeds 1 and 2) at ~10.5 h. Two rounds on
7 GPUs, ≈ 22 h.

### 11c. The selection problem, which the sweep must not compound

`run_epoch_eval()` iterates **`self.test_loader`** (trainer_end_to_end.py:534) and `train()` keeps
the checkpoint only when that RMSE improves. **The best-epoch checkpoint of every run in this
project — and of every number in the submitted paper — is selected on the SCARED test set.** It is
inherited from EndoDAC's training loop and is common in this literature, but it biases every
reported figure optimistically, and a reviewer who asked for proper uncertainty quantification
(Q6) may well notice it.

Choosing λ₁ by test Abs Rel on top of that would be selecting a hyperparameter on the test set and
then reporting test numbers from it. Two defensible ways out, in order of preference:

1. **Select λ₁ on SCARED, report the held-out evidence on Hamlyn and C3VD.** No extra compute, and
   it is the structure the paper already has: SCARED is the development set, the other two are
   generalization tests. State in the text that λ₁ was chosen on SCARED.
2. **Split the SCARED sequences**: choose λ₁ on a fixed subset (e.g. datasets 1-3) and report on
   the remaining four. Honest and free, but it halves an already small n = 7.

The validation split (1 705 images) cannot be used for this as things stand: this SCARED copy has
no per-frame ground-truth depth outside the test split (§4a), and the self-supervised loss is not
comparable across λ₁ because λ₁ multiplies one of its terms — using the photometric term alone as
the criterion would mechanically favour λ₁ = 0.

Recommendation: option 1, plus one sentence of limitation in the paper about checkpoint selection.
Fixing the selection properly (a `--checkpoint_split val` flag and ground truth for the validation
frames) would invalidate the comparability of all 41 existing runs and is not worth it for this
revision.

---

## 12. Competing architectures trained here from scratch: MonoII and MonoViT (B-grid, added 2026-09-15)

The open item at the top of this plan — Tables 4-6 have no external baselines — had two causes:
the checkpoints of the published methods are not on the server, and the ones that are would have
been trained by somebody else, on another split, with another protocol. Both go away if the
competing architectures are trained **in this repo, from scratch, under the shared recipe**.

**Shared recipe** (the user's decision, 2026-09-15): same pose network, same splits, same 20
epochs, same batch size, same optimizer and learning rate, same learned intrinsics, same data
augmentation, same checkpoint selection. Only two things change per row: the depth network, and
whether the two components of the method (local affine calibration + II loss at the published
λ₁ = 0.5, on monodepth2's photometric loss — no HADepth highlight term) are switched on. A row
pair therefore isolates the components and a column pair isolates the architecture. The price is
that these are not the published numbers of MonoViT or Monodepth2; say so in the caption.

| Run | Depth network | Components | Flags on top of `common_flags` |
|---|---|---|---|
| `R1` | ResNet-18 U-Net (= **Monodepth2**) | none | `--depth_backbone resnet18 --illum_calib none --illumination_invariant 0 --photometric standard` |
| `MonoII` | ResNet-18 U-Net | local affine + II (λ₁ = 0.5) | `--depth_backbone resnet18 --photometric standard --illumination_invariant 0.5` |
| `MonoViT` | MPViT-small + HR decoder | none | `--depth_backbone monovit --illum_calib none --illumination_invariant 0 --photometric standard` |
| `MonoViT-II` | MPViT-small + HR decoder | local affine + II (λ₁ = 0.5) | `--depth_backbone monovit --photometric standard --illumination_invariant 0.5` |
| `E3` | Depth Anything v1 + DV-LoRA (= **EndoDAC**) | none | `--illum_calib none --illumination_invariant 0 --photometric standard` |
| `M-local` | Depth Anything v1 + DV-LoRA | local affine + II (λ₁ = 0.5) | `--photometric standard --illumination_invariant 0.5` |

`R1`, `E3` and `M-local` are already trained at 3 seeds, so the new compute is 9 jobs:
`MonoII`, `MonoViT`, `MonoViT-II` × 3 seeds.

**`MonoII` is not `R2`.** `R2` is `--depth_backbone resnet18` with the repo defaults, i.e.
λ₁ = 0.1 *and* HADepth's highlight-aware term. MonoII, like `M-local`, is the method as published:
λ₁ = 0.5, monodepth2 photometric loss, nothing of HADepth's.

### Repo edits (2026-09-15)

- `models/monovit/mpvit.py` was unusable: it imported `timm`, `einops`, `mmcv` and `mmengine`
  (none in `requirements.txt`, so `import models.monovit` raised ImportError, which is why the
  `monovit` entry of `DepthModelFactory` had never run), and `mpvit_small()` loaded a checkpoint
  from a hard-coded `/workspace/endo-manydepth/...` path. It is now self-contained (local
  `build_norm_layer`, `DropPath`, `trunc_normal_`, the two `rearrange` calls written out, and a
  plain `torch.load`), and every factory takes `pretrained=<path>`, falling back to random
  initialisation with a printed warning.
- `models/monovit_depth.py`: `MonoViTDepth` = `mpvit_small` + `DepthDecoderT` behind the endodac
  interface (`image -> {("disp", s)}`), the same wrapper trick `models/resnet_depth.py` uses.
  27.87 M parameters (22.60 M encoder, 5.27 M decoder), features at H/2…H/32 =
  [64, 128, 216, 288, 288], four disparity scales with `disp0` at full resolution.
- `--depth_backbone monovit` in `options.py` / `trainer_end_to_end.py`, with `--mpvit_weights`
  (default `<pretrained_path>/mpvit_small.pth`).
- `--depth_lr`: learning rate of the depth network alone; `None` (the default) keeps the single
  group, so nothing changes for the 41 existing runs. MonoViT's published recipe uses 5e-5 for
  the MPViT encoder against 1e-4 elsewhere — only needed if MonoViT underfits at the shared 1e-4.
- `cviu_revision.py`: the three B-grid entries, `load_depth_model_from_run` builds `MonoViTDepth`
  for `depth_backbone == "monovit"`, `stage_train` downloads `mpvit_small.pth` (and skips the
  MonoViT rows if it cannot, rather than training a random-init encoder and calling it MonoViT),
  `BACKBONE_ORDER` → `tables/backbones.tex`, and a "Backbone × components" section in
  `report.md` with the three paired differences (with-components − plain) per dataset.
- `evaluate_depth_all.py`: `--model_type monovit` now also accepts a checkpoint trained here
  (`depth_model.pth`), not only the official two-file layout.

**The ImageNet MPViT-small weights have to be found, not downloaded.** The link both the MPViT
and the MonoViT README give — `https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth` — is dead
(checked 2026-09-15: Dropbox retired the `/s/` links and it answers with an HTML page; there is no
Hugging Face mirror and the repo has no GitHub release). `ensure_mpvit_weights` therefore looks for
a copy already on the machine before trying the download, starting with
`/workspace/endo-manydepth/manydepth/mpvit/mpvit_small.pth` — **the path `models/monovit/mpvit.py`
used to hard-code, so a copy very likely exists on the DGX** — then `./ckpt/mpvit_small.pth` and
`~/mpvit_small.pth`, and copies what it finds into `pretrained_model/`. Anywhere else: set
`mpvit_weights: <path>` in `cviu_config.yaml` (or `--mpvit_weights` for a single run). If no copy
exists anywhere, the MonoViT rows are skipped rather than trained from random weights and
mislabelled; `--skip_preflight` overrides that, and the run then prints
`[mpvit] mpvit_small: ... random initialisation` — which must be said in the paper.

```
ls /workspace/endo-manydepth/manydepth/mpvit/mpvit_small.pth    # the likely copy on the server
python cviu_revision.py train --gpus 0 --only MonoViT --extra_flags "--num_epochs 1"   # smoke test, then delete logs/cviu_MonoViT_s314
python cviu_revision.py train --gpus 0 1 2 --only MonoII MonoViT MonoViT-II            # 9 jobs
python cviu_revision.py predict && python cviu_revision.py stats && python cviu_revision.py report
```

### What this does and does not answer

It answers R3 ("separate the backbone's contribution from the method's") across three
architectures instead of two, and it puts three rows in Tables 4-6 that were trained under a
protocol we control. It does **not** replace HADepth, AF-SfMLearner and Endo-SfMLearner: those
are method-level baselines whose losses are not implemented here, and they still need their
published checkpoints (`cviu_config.yaml::methods`, currently pointing at paths that do not exist
on the server).

## 13. R2 as a factorial on MonoII: calibration x colour augmentation (AM-grid, 2026-09-15)

Reviewer 2 asks whether the local affine calibration is validated. The cleanest way to answer is a
**full factorial on the cheapest backbone**: the ResNet-18 of MonoII trains in ~4-6 h against
~10.5 h for the adapted foundation encoder, so eighteen runs here cost about what six cost there.

| | jitter as shipped | consistent jitter |
|---|---|---|
| no calibration | `MonoII-none` | `A-MonoII-none` |
| **global** affine | `MonoII-glob` | `A-MonoII-glob` |
| **local** affine (the proposal) | `MonoII` | `A-MonoII` |

Three seeds per cell. Everything else is the published recipe: II loss at λ₁ = 0.5, monodepth2
photometric loss, no HADepth term, batch 8, 20 epochs. `stats` writes `tables/monoii_calib.tex`.

**Why the two factors together.** Reading **down a column** gives the none/global/local comparison
the reviewer asked for, on a backbone where the paper claims the mechanism works. Reading **across
a row** gives the effect of the colour-augmentation defect of §9 on that calibration model, which
is the open hypothesis for why the module looked inert: its input was noise on half the training
items. Run either factor alone and the other confounds it.

**Pre-registered reading**, fixed before any of these runs exists:

- *Calibration main effect.* Paired per sequence against the no-calibration cell of the same
  column, SCARED as the selection set, Hamlyn and C3VD held out, Abs Rel primary. Two primary
  comparisons per column (global vs none, local vs none), Holm over that family of two — the same
  rule as §8b, which is the only way to escape the n=7 Wilcoxon floor of 0.0156.
- *Augmentation main effect.* Each cell against its row partner.
- *Interaction.* If the calibration helps only in the right-hand column, the defect was starving
  the module and §8e must be rewritten. If it helps in both, the defect was never the issue and
  §8e stands. If it helps in neither, the mechanism does not work on this backbone either, and
  that is the answer to R2 whether we like it or not.

**What it cannot settle.** It is one architecture. A positive result here plus the B-grid of §12
would support "backbone-agnostic"; a positive result here alone supports "it works on ResNet-18",
which is still more than the submission evidences, since the paper's Table 2 has no row isolating
the calibration on that backbone at all.
