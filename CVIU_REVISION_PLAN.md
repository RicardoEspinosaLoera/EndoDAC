# MonoIIF – CVIU revision experiment plan

One script, `cviu_revision.py`, runs every new experiment the reviewers asked for. It reuses the
existing trainer, model factory and datasets; the only repo edits are flag-gated switches whose
defaults reproduce the current behaviour.

**Status (2026-09-10).** Script, `cviu_config.yaml` and all §2 repo edits are implemented.
Verified locally without data or GPU: config merging, every generated training command parses
with `options.py`, checkpoint selection, real split pairing, the affine fit and GT warper on
synthetic geometry, and the `stats`/`report` stages on synthetic per-frame rows. Not yet run
anywhere: `predict`, `illum-params`, `illum-sens`, `da3` (need the server, the datasets and the
checkpoints); `_da3_infer()` is the only function that touches the DA3 API and may need a
one-line adjustment to the installed version. The E/R/C grid was launched on the server on
2026-09-10. The DA3-encoder rows (D3, D3-EndoDAC, N0; §6) were added afterwards and need the
`depth_anything_3` package on the server plus a 1-epoch dry run
(`python cviu_revision.py train --gpus 0 --only D3 --extra_flags "--num_epochs 1"`) before
relaunching `train`, which skips the finished runs.

Reviewer asks → experiments:

| Reviewer ask | Experiment | Script stage |
|---|---|---|
| R2: validate the local-affine illumination model (none vs global vs local), analyse the estimated parameters | C-grid (3 trained calibration variants), model-free affine fit on GT geometry, learned-parameter statistics, perturbation sensitivity | `train`, `illum-fit`, `illum-params`, `illum-sens` |
| R3: isolate each component (Depth Anything backbone, DV-LoRA, calibration, IIF loss) | E-grid (build-up + leave-one-out) and R-grid (same losses on a ResNet-18 backbone) | `train`, `predict`, `stats` |
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
2. **Adapted rows (D-grid, implemented)**: DA3-Base's encoder is *not* a vanilla DINOv2 (from
   block 4 on it uses QK-norm, RoPE and alternating attention, and its DualDPT head takes
   1536-dim inputs), so its weights cannot be copied onto the repo's ViT. Instead
   `models/endodac/da3_backbone.py` wraps DA3's own `DinoV2` module (loaded through the
   `depth_anything_3` package, Apache-2.0) behind endodac's encoder interface: DV-LoRA is
   inserted into its `mlp.fc1/fc2` with the pretrained weights kept, the Conv-neck is re-created
   as forward hooks on blocks 2/5/8/11 (same `residual_` prefix, same trainable policy), and
   the repo's DPT head is initialised from DA v1 but fully trained because a head trained on
   DA v1 features does not match DA3 features (`--train_depth_head` is forced on; report the
   larger trainable count honestly). Inputs stay in [0, 1] like the DA v1 path. Selected with
   `--backbone_weights da3` (`--da3_model_id`, default `depth-anything/da3-base`).
   - **D3** = full MonoIIF recipe on the DA3 encoder (3 seeds); **D3-EndoDAC** = E3 recipe on
     the DA3 encoder. (D3 − D3-EndoDAC) vs (E8 − E3) shows whether the method's gain transfers
     to the newer foundation model; D3 vs E8 is the backbone comparison the reviewer asked for.
   - **N0** = full recipe with a randomly initialised ViT-B (`--backbone_weights none`): the
     "no foundation weights" control for the backbone-share question of R3.
   - Gate: the wrapper was verified only against a stand-in ViT; the first server dry run
     (`--only D3 --extra_flags "--num_epochs 1"`) must show the DA3 encoder printout, a sane
     trainable-parameter count and a decreasing loss. If the installed DA3 version differs in
     its ViT API, `DA3Encoder.get_intermediate_layers` is the place to adapt.
3. **Limitations text**: DA3's monocular model is Large-only (~0.3 B params), so it enters only
   as a zero-shot row; DA3-Base is compared both zero-shot and adapted (D3); DA3's multi-view
   depth-ray head is replaced by the repo's DPT head, so D3 measures the encoder, not DA3's head.

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
