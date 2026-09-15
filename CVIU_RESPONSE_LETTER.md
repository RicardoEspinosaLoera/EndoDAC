# Response to reviewers — CVIU

Drafts for the four experimental asks. Every number is SCARED Abs Rel at the **sequence level**
(n = 7 keyframe videos, 3 training seeds where stated), from `results/cviu/summary.csv` and
`results/cviu/paired.csv`; the derivation is in `CVIU_REVISION_PLAN.md` §8. Paired differences
are `method − ours` per sequence with a 10 000-draw cluster bootstrap CI.

Hamlyn and C3VD numbers are filled in (from the results pulled to the laptop on 2026-09-15).
**The cross-dataset picture is in `CVIU_REVISION_PLAN.md` §8g and it narrows the R2 claim
considerably: nothing about the calibration replicates outside SCARED.**

---

## Reviewer 2 — the illumination calibration module is not fully validated

> Second, the effectiveness of the illumination calibration module is not yet fully validated…

We agree, and the new experiments changed our own conclusion. We now report three trained
variants at three seeds each, plus a model-free analysis of the affine assumption and statistics
of the learned parameters.

**Draft text.**

We evaluate three photometric calibration models under an otherwise identical recipe: no
calibration (C0), one affine pair (c, b) per image predicted by a global head (C1), and the
spatially varying maps of the original submission (C2 = E8). At the sequence level on SCARED,
Abs Rel is 0.0518 ± 0.0029 for no calibration, **0.0497 ± 0.0021 for the global model** and
0.0512 ± 0.0014 for the local model. The paired differences against the global model are
+0.00216, CI [+0.00064, +0.00397], for no calibration and +0.00148, CI [+0.00050, +0.00319],
for the local model, the latter with the global model better in **all seven sequences**. Both
CIs exclude zero. Calibration therefore helps, but the *global* model is the one that helps:
the spatially varying variant of the original submission is worse than a two-parameter affine
correction.

To understand why, we fitted the affine model in closed form on ground-truth geometry, with no
network involved: per-patch least squares explains about ten times more photometric residual
than a single global fit (0.040 vs 0.004 of the residual). The dense model is thus the better
*photometric* model and the worse *depth* model. Inspecting the learned maps, their mean gain
does not track an injected illumination gain (regression slope −0.006, where a head reading
illumination would give ≈ 1), while the maps are demonstrably active (gain SD 0.0136 across
inputs vs 0.0287 within an image). Supervising them towards their own least-squares fit
collapses them to the identity (mean gain 0.9995, SD 0.0026) without recovering the global
model's accuracy (0.0511). We conclude that the extra degrees of freedom absorb photometric
residual that is not illumination — most plausibly geometric error — and thereby weaken the
gradient that should reach the depth network, whereas a two-parameter bounded correction cannot.
We have replaced the local module with the global one throughout and report the local variant as
an ablation.

**Out of domain the comparison is a tie, and we say so.** On Hamlyn (n = 58 blocks from its
single sequence) and C3VD (n = 7 scenes) neither claim replicates: against the global model, no
calibration gives −0.0002, CI [−0.0023, +0.0020] on Hamlyn and −0.0059, CI [−0.0147, +0.0028] on
C3VD, and the local model gives +0.0013, CI [−0.0004, +0.0030] and −0.0038, CI [−0.0138,
+0.0094]. Every interval covers zero. Two results even run the other way: on Hamlyn the plain
EndoDAC baseline is the best of the seven configurations (−0.0031, CI [−0.0048, −0.0013], better
in 37 of 58 blocks), and on C3VD the best is our model *without* the illumination-invariant loss
(−0.0208, CI [−0.0322, −0.0103], better in 6 of 7 scenes). We therefore claim that affine
photometric calibration helps **in the training domain** and is neutral outside it, and that the
global form is the one to use because it is clearly better in-domain and never worse than the
local one by more than noise. We do not claim a general improvement, and all three datasets are
reported with this table rather than SCARED alone.

**What is honest to claim and what is not.** The mechanism above is an interpretation consistent
with every measurement we made; the falsifying experiment — a local head restricted to a
low-order spatial basis — was not run, and we say so in the paper. Only the backbone-level
effects (the ResNet-18 control, the DA3 encoder, the random-init encoder) survive a change of
domain with intervals excluding zero; every photometric-component effect is within noise of zero
on at least one dataset.

---

## Reviewer 3 — separate the pretrained backbone's contribution from the method's

> …a component-wise ablation that separates what the pretrained backbone contributes from what
> the method contributes…

**Draft text.**

We built the model up from a randomly initialised ViT-B, adding one component at a time, all at
the same recipe and resolution:

| Configuration | SCARED Abs Rel | Δ | share |
|---|---|---|---|
| Random-init ViT-B, full method | 0.1061 | — | — |
| + Depth Anything v1 weights (frozen), DPT heads trained | 0.0869 | 0.0191 | 34% |
| + Conv-neck residual blocks | 0.0606 | 0.0264 | 47% |
| + DV-LoRA adapters (= EndoDAC) | 0.0517 | 0.0089 | 16% |
| + the highlight-aware loss of HADepth + our illumination-invariant loss | 0.0518 | -0.0002 | **-0.3%** |
| + our global affine calibration (full model) | 0.0497 | 0.0022 | **3.8%** |

The foundation weights and the adaptation machinery introduced by EndoDAC account for
approximately 96% of the improvement over a randomly initialised encoder, on top of an EndoDAC
reproduction that matches the published number (0.0517 here vs 0.052 as published). The two
photometric terms we adopted or proposed before the calibration -- the highlight-aware loss of
HADepth and our illumination-invariant loss -- are together worth nothing (-0.0002). The one
component of ours that earns its place is the global affine calibration, isolated by comparing
against the run that keeps both of those terms and drops only the calibration: +0.00216, CI
[+0.00064, +0.00397], better in 6 of 7 sequences, a 3.8% relative reduction in Abs Rel. We state
this proportion explicitly in the revised paper rather than reporting only the final number, and
we attribute each component to whoever introduced it.

**Leave-one-out of every component of the pipeline.** The table above isolates the backbone; this
one isolates each loss and adapter, credited to whoever introduced it -- our pipeline builds on
EndoDAC's adapters and adopts HADepth's highlight-aware loss. Each row removes one component from the full model and reports the paired
difference `variant − full` per sequence, so a **negative** number means the component was
costing accuracy. Sequence-level, 10 000-draw bootstrap CI.

| Component removed | whose | SCARED (n=7) | Hamlyn (n=58) | C3VD (n=7) | seeds |
|---|---|---|---|---|---|
| affine calibration | ours | +0.0007 [−0.0010, +0.0028] | **−0.0015 [−0.0028, −0.0003]** | −0.0020 [−0.0111, +0.0087] | 3 |
| illumination-invariant loss | ours | −0.0006 [−0.0021, +0.0008] | +0.0015 [−0.0001, +0.0030] | **−0.0170 [−0.0270, −0.0073]** | 3 |
| highlight-aware loss | HADepth | +0.0023 [−0.0017, +0.0082] | **−0.0048 [−0.0071, −0.0025]** | **+0.0342 [+0.0257, +0.0422]** | 1 |
| DV-LoRA → plain LoRA | EndoDAC | −0.0008 [−0.0017, +0.0001] | **+0.0030 [+0.0016, +0.0044]** | **+0.0219 [+0.0149, +0.0295]** | 3 |

Of the two losses, our illumination-invariant term is neutral in-domain, worth 0.0015 on Hamlyn
and **costs 0.0170 on C3VD**; it does not survive as an accuracy contribution. The two components
with a large effect out of domain are **both adopted, not ours**: HADepth's highlight-aware loss
(+0.0342 on C3VD, though negative on Hamlyn and on a single training seed) and EndoDAC's DV-LoRA
(+0.0030 and +0.0219, both intervals excluding zero). Our own calibration is the component with
the clearest in-domain effect and no measurable effect outside it. We report the attribution this
way rather than presenting the pipeline as one undifferentiated method.

Adding the components to the baseline one at a time tells the same story on the training domain:
+calibration +0.0012, +invariant loss +0.0004, +highlight −0.0003, +calibration+invariant
+0.0018, all three together −0.0005 Abs Rel. Only the combination is (marginally) better than
the baseline, and only in-domain. On C3VD the combination is worth −0.0250 and essentially all
of it comes from the highlight-aware term.

We also repeated the same losses on a ResNet-18 U-Net backbone. There the method changes
nothing: 0.0593 with the standard photometric loss against 0.0593 with the full method, a
difference of 0.00003. The benefit of photometric calibration appears only on top of a strong
foundation backbone, and we now say so as a limitation instead of presenting the losses as
backbone-agnostic.

Finally, with three seeds per configuration we report the training-seed SD (0.0008-0.0029 in
Abs Rel) separately from the sequence-level CI, so that readers can see which ablation gaps are
smaller than run-to-run variation. Several are, and we no longer draw conclusions from them.

---

## Question 2 — why was Depth Anything 3 not considered?

**Draft text.**

It now is, both zero-shot and adapted. DA3-Base's encoder is not a vanilla DINOv2 — from block 4
it uses QK-norm, 2D RoPE, a camera token and alternating local/global attention, and it emits
1536-d features at layers 5/7/9/11 — so its weights cannot be copied onto the encoder used in
the submission. We ported DA3's encoder and the main branch of its DualDPT head into our
codebase and verified the port against the reference implementation: encoder features and camera
tokens are bit-identical at three resolutions, all 207 encoder tensors match the released
DA3-BASE checkpoint, and the zero-initialised adapters leave DA3's features unchanged at step 0.

Zero-shot, DA3-Base reaches Abs Rel 0.0718 with median scaling and 0.0692 with affine disparity
alignment on SCARED, against 0.0497 for our adapted model. Adapted with exactly our recipe and
an identical trainable-parameter count (8,961,540 in both), the DA3 encoder reaches 0.0545,
i.e. worse than the same recipe on Depth Anything v1 (+0.00480 against our model, CI [+0.00157,
+0.00840]). Under the EndoDAC recipe the ordering is the same (0.0527 on DA3 vs 0.0517 on
DA v1). We therefore keep Depth Anything v1 as the backbone and report DA3 as an evaluated
alternative rather than an improvement, noting the two constraints we could not remove: DA3's
monocular model is released at Large size only, so it enters only as a zero-shot row, and our
adaptation replaces DA3's depth-ray output head with multi-scale disparity heads, which may not
be the best way to use that model.

---

## Question 6 — uncertainty and paired tests at the video-sequence level

**Draft text.**

Tables 4-6 are recomputed with the video sequence as the unit of analysis. Frame metrics are
averaged within a sequence and then across seeds, giving one value per (method, sequence); we
report mean ± SD over sequences with a 95% cluster-bootstrap CI (10 000 draws, resampling
sequences), and, for every competing method, the paired difference against ours with its
bootstrap CI, a paired t-test, the exact Wilcoxon signed-rank test, a sign test, and Cohen's
d_z. Per-sequence numbers are in the supplement. The frame-level t-intervals of the original
submission were anti-conservative, because frames within a video are strongly correlated, and we
say so.

Two consequences we report openly:

1. **Multiplicity.** With n = 7 sequences the smallest attainable two-sided exact Wilcoxon
   p-value is 2/2⁷ = 0.0156, so a family-wise correction across the ~20 rows of a table cannot
   yield a significant result no matter how large the effect. We therefore pre-specify two
   primary comparisons — ours vs no calibration, and global vs local calibration — and apply
   Holm within that family of two: adjusted p = 0.047 and 0.031 respectively. All remaining
   comparisons are reported as exploratory, with CIs and uncorrected p-values, and labelled as
   such.
2. **Differences of ~0.001 Abs Rel are not resolvable.** Several method pairs that the original
   tables separated are statistically indistinguishable at the sequence level; we now mark them
   as ties rather than ranking them. Concretely, our model and the same model with plain LoRA
   adapters differ by 0.00019, CI [−0.00127, +0.00155], d_z = 0.09.

On Hamlyn our copy of the dataset holds a single rectified sequence, so we use contiguous blocks
of 100 frames as the unit (n = 58) and state that these intervals are slightly optimistic,
because blocks within one video remain correlated; the partition is identical for every method,
so the paired tests stay aligned. C3VD keeps its 7 scenes as units. The recomputation changes
what the tables say: on SCARED (n = 7) the differences among our photometric variants span
0.0021 Abs Rel, on C3VD (n = 7) they span 0.042, and on Hamlyn 0.006, so the same method pair
can be significant on one dataset and a tie on another. We now report all three and mark per
dataset which differences their intervals actually resolve, instead of ranking methods on a
single pooled frame-level average.

---

## Open items before submission

0. **HOLD the R2 answer until the A-grid lands.** A defect found on 2026-09-15
   (`CVIU_REVISION_PLAN.md` §9) means every run of the grid was trained with a colour jitter
   re-sampled per frame, so on half the training items the illumination relation the lighting
   head reads is augmentation noise. It does not affect the accuracy comparisons — all runs share
   it — but it is a plausible single cause of all four measurements the R2 answer above leans on
   (near-constant c and b, zero sensitivity slope, local worse than global, C2-sup collapsing to
   the identity). `A-C0` / `A-C1` / `A-E8` retrain the calibration comparison with the jitter
   fixed (~11 h). If the sensitivity slope moves towards 1, the paragraph "we conclude that the
   extra degrees of freedom absorb photometric residual that is not illumination" must be
   rewritten and the local-vs-global comparison re-run at 3 seeds.

1. **Tables 4-6 have no external baselines.** `EndoDAC_MICCAI` and `AF_SfMLearner` failed to
   load (the checkpoint paths in `cviu_config.yaml` do not exist on the server); HADepth,
   MonoViT, Endo-SfMLearner and Monodepth2 were never configured. Fix the paths and re-run
   `predict` + `stats`.
2. **Decide the method's identity.** The IIF term does not improve accuracy: removing it is
   neutral on SCARED (0.05052 vs 0.05116, 3 seeds), costs 0.0015 on Hamlyn, and **gains 0.0208
   on C3VD with the interval excluding zero** — the largest single component effect measured on
   that dataset, in the wrong direction for the component the method is named after. Either it
   is reframed as robustness under illumination change (`illum-sens` is already computed) or it
   is dropped from the claims. The paper's title depends on this choice.
3. **Consider whether the headline claim should be the calibration at all.** Per §8g only the
   backbone-level effects replicate across the three datasets. An honest paper built on these
   results is closer to "a careful ablation of what actually drives self-supervised endoscopic
   depth, with a small in-domain gain from global photometric calibration" than to a new method
   claim. That is a strong CVIU contribution — the reviewers asked exactly these questions — but
   it is a different paper from the one submitted.
4. **`E7` needs two more seeds** -- not to build a claim of ours, but to close the attribution.
   It is the leave-one-out arm of **HADepth's** highlight-aware loss, which is the largest single
   component effect measured anywhere (+0.0342 Abs Rel on C3VD, 0/7 scenes without it) and is
   enabled by default in every run of the C-, D- and R-grids. With one seed the split between
   "HADepth's term" and "ours" rests on a single training run, which a reviewer asking exactly
   this question will notice. `train --only E7 --seeds 1 2`, two GPUs, ~21 h.
5. **Retract the "plain LoRA beats DV-LoRA" note.** That held on SCARED only, with the interval
   touching zero (−0.0008 [−0.0017, +0.0001]); out of domain DV-LoRA wins on both datasets with
   intervals excluding zero.
6. Pose evaluation was never run; `evaluate_pose.py` exists if a reviewer asks.
