# Response to Reviewers — CVIU

Manuscript: *Illumination-Invariant Foundation Model Adaptation for Self-Supervised Monocular
Depth Estimation in Endoscopy*

We thank the reviewers for four questions that turned out to be more productive than we expected.
Answering them required rebuilding the experimental section, and in the process we found and
corrected several errors of our own. Everything below is new work.

**How the numbers in this letter were produced.** Every configuration was retrained with **three
random seeds** under one recipe (same pose network, same splits, 20 epochs, same optimiser and
learning rate, same learned intrinsics, same augmentation). All models are trained on SCARED
only; Hamlyn and C3VD are evaluation-only. Metrics are aggregated with the **video sequence** as
the unit of analysis (Q6): frames are averaged within a sequence, then across seeds, giving one
value per (method, sequence). Intervals are 95% cluster bootstrap CIs over sequences, 10 000
draws. Paired differences are `variant − MonoIIF` per sequence, so a **positive** number means
MonoIIF is better. SCARED has n = 7 sequences, C3VD n = 7 scenes; our copy of Hamlyn holds a
single rectified sequence, so we use n = 58 contiguous blocks of 100 frames and say so.

---

## Reviewer 2 — the effectiveness of the illumination calibration module is not fully validated

> Second, the effectiveness of the illumination calibration module is not yet fully validated…

We agree. The submission compared the module against nothing, which cannot establish that its
*spatially varying* form is what produces the gain. We now report a **capacity family** in which
"no calibration" and the two forms the reviewer names are points on one axis, on **two different
depth backbones**, plus a model-free analysis of the affine assumption and statistics of the
learned parameters.

### The comparison the reviewer asks for

All three cells keep the rest of the method fixed and change only the calibration model. SCARED,
mean over 3 seeds ± seed SD, and the paired difference against the local model.

| Calibration | SCARED Abs Rel | paired vs local | sequences |
|---|---|---|---|
| none | 0.0518 ± 0.0029 | +0.0007 [−0.0010, +0.0028] | 3/7 |
| **global**, one affine pair (c, b) per image | **0.0497 ± 0.0021** | **−0.0015 [−0.0032, −0.0005]**, p = 0.016 | **0/7** |
| local, dense map (the submission) | 0.0512 ± 0.0014 | — | — |

**We report a result that does not favour our own implementation.** In the training domain the
global two-parameter model is better than the dense map in **all seven sequences**, with the
interval excluding zero, while the dense map is not distinguishable from no calibration at all
(+0.0007, interval covering zero). Out of domain the two forms are a tie: Hamlyn −0.0013
[−0.0030, +0.0004] and C3VD +0.0038 [−0.0094, +0.0138].

### A capacity sweep locates why

To see whether this is a property of the *form* or of its *capacity*, we parameterised the affine
field as a polynomial of degree d: degree 0 is exactly the global model, and the dense per-pixel
map is the limit. Trained with the calibration alone (no invariant loss), on the foundation
backbone:

| Spatial capacity | SCARED | Hamlyn | C3VD |
|---|---|---|---|
| none | 0.0517 | **0.1565** | 0.2929 |
| degree 0 (global) | 0.0505 | 0.1597 | **0.2672**, −0.0256 [−0.0297, −0.0222], 7/7 |
| degree 1 | **0.0502**, −0.0015 [−0.0024, −0.0004], 6/7 | 0.1597 | 0.2956 |
| degree 2 | 0.0510 | 0.1586 | 0.2870 |
| dense map | 0.0517 | 0.1594 | **0.2669**, −0.0260 [−0.0360, −0.0164], 7/7 |

Repeating the sweep on a ResNet-18 backbone gives a **different** optimum: there the dense map is
the single worst point on Hamlyn (+0.0059 [+0.0034, +0.0083] against no calibration, better in
only 15 of 58 blocks, p < 0.001) and degree 1 beats both the global model (−0.0037 [−0.0059,
−0.0017], p = 0.004) and the dense map (−0.0058 [−0.0078, −0.0037], p < 0.001) on Hamlyn and both
on C3VD (−0.0123, 7/7 and −0.0101, 6/7). **The capacity optimum is backbone-dependent**, which we
state as a finding rather than recommending a single degree.

### Analysis of the learned parameters

Three model-free checks, all in the revised supplement:

1. **Is the affine model adequate?** Fitting it in closed form on ground-truth geometry, per-patch
   least squares explains about ten times more photometric residual than a single global fit
   (0.040 against 0.004 of the residual). The dense model is therefore the better *photometric*
   model while being no better as a *depth* model.
2. **Does the head read illumination?** Injecting a known gain into the source frame and
   regressing the predicted mean gain on it gives a slope of −0.006, where a head that recovered
   illumination would give ≈ 1 — although the maps are demonstrably active (gain SD 0.0136 across
   inputs against 0.0287 within one image).
3. **What happens if we supervise it?** Training the head towards its own least-squares fit
   collapses it to the identity (mean gain 0.9995, SD 0.0026) without recovering the global
   model's accuracy (0.0511).

### What we now claim

We claim that **affine photometric calibration helps in the training domain**, with the global
form supported by an interval that excludes zero, and that the affine assumption itself is sound.
We **no longer claim that the spatial variation is what produces the gain**: the dense
parameterisation of the submission ties the global one out of domain and is worse in-domain, and
the capacity sweep shows the useful capacity is low and backbone-dependent. We keep the dense form
as the method for continuity with the submitted model and report the global variant, with its
numbers, as an ablation rather than burying it. The mechanism we propose — that surplus degrees of
freedom absorb photometric residual that is not illumination, most plausibly geometric error — is
an interpretation consistent with every measurement, and the capacity sweep is the intervention
that supports it.

---

## Reviewer 3 — separate the pretrained backbone's contribution from the method's

> …a component-wise ablation that separates what the pretrained backbone contributes from what the
> method contributes…

This was the most uncomfortable of the four questions and we answer it directly.

### Building the model up one component at a time

Each row adds one component to the row above, at an identical recipe and resolution. SCARED.

| Configuration | Abs Rel | Δ | share of the span |
|---|---|---|---|
| Randomly initialised ViT-B | 0.1061 | — | — |
| + Depth Anything v1 weights (frozen), DPT heads trained | 0.0869 | 0.0191 | 34% |
| + convolutional neck | 0.0606 | 0.0264 | **47%** |
| + DV-LoRA adapters (= EndoDAC) | 0.0517 | 0.0089 | 16% |
| + the highlight-aware loss of HADepth and our invariant loss | 0.0518 | −0.0002 | **−0.3%** |
| + our affine calibration (full model) | 0.0497–0.0512 | 0.0006–0.0021 | **1–4%** |

**Roughly 96% of the improvement over a randomly initialised encoder comes from the pretrained
weights and from adaptation machinery that is not ours.** We state this proportion in the revised
paper instead of reporting only the final number, and we attribute each component to whoever
introduced it. We also correct an attribution error of our own: the convolutional neck worth 47%
is a **large-kernel attention block inherited from HADepth**, not the bottleneck neck of EndoDAC
as the submission implies (see the corrections section).

### Removing one component at a time from the full model

`variant − MonoIIF` per sequence; **negative** means the component was costing accuracy.

| Component removed | introduced by | SCARED (n=7) | Hamlyn (n=58) | C3VD (n=7) |
|---|---|---|---|---|
| affine calibration | ours | +0.0007 [−0.0010, +0.0028] | **−0.0015 [−0.0028, −0.0003]** | −0.0020 [−0.0109, +0.0084] |
| illumination-invariant loss | ours | −0.0006 [−0.0021, +0.0009] | +0.0015 [−0.0000, +0.0030] | **−0.0170 [−0.0271, −0.0072]**, 1/7 |
| highlight-aware loss | HADepth | +0.0022 [+0.0006, +0.0040] | −0.0002 [−0.0022, +0.0019] | −0.0059 [−0.0142, +0.0030] |
| DV-LoRA → plain LoRA | EndoDAC | −0.0008 [−0.0017, +0.0002] | **+0.0030 [+0.0017, +0.0044]** | **+0.0219 [+0.0151, +0.0296]**, 7/7 |

The honest reading is that **our illumination-invariant term does not earn its place as an
accuracy contribution**: it is neutral in-domain and on Hamlyn, and removing it *improves* C3VD by
0.0170 with the interval excluding zero. We report this rather than omit it, and we reframe the
term in the revised paper as a robustness mechanism under illumination change — for which we give
the sensitivity analysis — rather than as an accuracy contribution. The component with the
clearest out-of-domain effect, DV-LoRA, is EndoDAC's, not ours.

### Where the method does win

Against a faithful EndoDAC baseline retrained under our recipe, the two components together are
worth **+0.0250 [+0.0189, +0.0316] on C3VD, better in 7 of 7 scenes** (p = 0.016), a tie in-domain
(+0.0005), and a loss on Hamlyn (−0.0043, EndoDAC better in 42 of 58 blocks). The same pattern
holds on a ResNet-18 backbone, where adding the components is a tie in-domain (0.0593 both) and
worth 0.0157 on C3VD and 0.0055 on Hamlyn. **The contribution of the method is out-of-domain
robustness under geometric shift, not in-domain accuracy**, and that is how the revised paper
states it.

---

## Question 2 — why was Depth Anything 3 not considered?

It now is, both zero-shot and adapted.

DA3-Base's encoder is not a vanilla DINOv2: from block 4 it uses QK-norm, 2D RoPE, a camera token
and alternating local/global attention, and it emits 1536-d features at layers 5/7/9/11, so its
weights cannot be copied onto the encoder used in the submission. We ported DA3's encoder and the
main branch of its DualDPT head into our codebase and verified the port against the reference
implementation: encoder features and camera tokens are bit-identical at three resolutions, all 207
encoder tensors match the released DA3-BASE checkpoint, and the zero-initialised adapters leave
DA3's features unchanged at step 0.

| Model | SCARED | Hamlyn | C3VD |
|---|---|---|---|
| DA3-Base zero-shot, median scaling | 0.0718 | 0.2066 | 0.3947 |
| DA3-Base zero-shot, affine disparity alignment | 0.0692 | 0.1811 | 0.6154 |
| DA3-Base adapted with our recipe | 0.0545 | 0.1735 | 0.3616 |
| DA3-Base adapted with the EndoDAC recipe | 0.0527 | 0.1762 | 0.3597 |
| **MonoIIF (Depth Anything v1)** | **0.0512** | **0.1608** | **0.2679** |

Adapted with an identical trainable-parameter count (8 961 540 in both), the DA3 encoder is worse
than the same recipe on Depth Anything v1 on all three datasets (+0.0033 in-domain, +0.0127 on
Hamlyn, +0.0938 on C3VD, the last two with intervals excluding zero). We therefore keep Depth
Anything v1 and report DA3 as an evaluated alternative rather than an improvement, noting two
constraints we could not remove: DA3's monocular model is released at Large size only, so it
enters only as a zero-shot row, and our adaptation replaces DA3's depth-ray output head with
multi-scale disparity heads, which may not be the best way to use that model.

---

## Question 6 — uncertainty estimates and paired comparisons at the video-sequence level

> The differences between the leading methods in Tables 4–6 are often very small, such as 0.049
> versus 0.050. Please report uncertainty estimates and paired statistical comparisons using video
> sequences, rather than individual frames.

The reviewer is right, and acting on it changed what the tables say.

Tables 4–6 are recomputed with the video sequence as the unit. We report mean ± SD over sequences
with a 95% cluster bootstrap CI (10 000 draws, resampling sequences) and, for every competing
method, the paired difference against MonoIIF with its bootstrap CI, a paired t-test, the exact
Wilcoxon signed-rank test, a sign test and Cohen's d_z. The frame-level intervals of the
submission were anti-conservative because frames within a video are strongly correlated, and we
say so.

**The comparison methods are the authors' own released checkpoints**, evaluated under our
protocol. They reproduce the published numbers in every cell we can check — EndoDAC 0.0507 against
0.051 published, for instance — which validates the evaluation protocol itself.

| Method | seeds | SCARED | paired vs MonoIIF | sequences |
|---|---|---|---|---|
| HADepth | 1 | 0.0489 | −0.0023 [−0.0068, +0.0016] | 3/7 |
| **MonoIIF (ours)** | 3 | **0.0512 ± 0.0014** | — | — |
| EndoDAC | 1 | 0.0507 | −0.0004 [−0.0048, +0.0041] | 3/7 |
| MonoPCC | 1 | 0.0505 | −0.0007 [−0.0073, +0.0053] | 4/7 |
| AF-SfMLearner | 1 | 0.0584 | +0.0072 [+0.0011, +0.0137] | 5/7 |
| Monodepth2 | 1 | 0.0933 | +0.0421 [+0.0266, +0.0587] | 7/7 |

**The reviewer's example is exactly right.** The four leading methods span 0.0489 to 0.0512 on
SCARED, and every pairwise interval among them covers zero. MonoIIF's own seed-to-seed spread is
0.0503 to 0.0528, so **the separation between the leading methods is smaller than the variation
between training runs of a single one of them**. We now mark these as ties instead of ranking
them, and we report the seed SD alongside the sequence-level CI so that readers can see which gaps
are smaller than run-to-run variation. Several are.

Three consequences we report openly:

1. **Multiplicity.** With n = 7 sequences the smallest attainable two-sided exact Wilcoxon p-value
   is 2/2⁷ = 0.0156, so a family-wise correction across the ~20 rows of a table cannot yield a
   significant result however large the effect. We pre-specify two primary comparisons — the
   method against no calibration, and the global against the local calibration — and apply Holm
   within that family of two; all other comparisons are reported as exploratory with intervals and
   uncorrected p-values, and labelled as such.
2. **The ranking is dataset-dependent.** On C3VD HADepth beats MonoIIF by 0.0466 [−0.0683,
   −0.0292] in 7 of 7 scenes, while on Hamlyn and SCARED the two are a tie. A single pooled
   frame-level average hid this. All three datasets are now reported per dataset with the
   differences their intervals actually resolve.
3. **Single-checkpoint baselines.** The comparison methods are released checkpoints, so they have
   no seed variation to report; we give their sequence-level CI and state that the seed SD column
   applies only to models we trained. We do not report a best-of-three number for our own model
   against single runs of theirs.

---

## Corrections to the manuscript we make on our own initiative

Re-running everything surfaced errors in the submission that no reviewer raised. We list them
because a reader checking our released code would find them.

- **The backbone is Depth Anything v1, not V2.** Every claim of "V2" about the proposed model is
  wrong: abstract, §2.2.4, §3.3, §3.4, §3.5, §5, §6 and one of the two duplicated *Yang et al.
  2024* references. The V2 mentions in the related-work sections are correct and stay.
- **The convolutional neck is not EndoDAC's.** The class carries the name and docstring of
  ViTDet's 1×1/3×3/1×1 bottleneck but implements a large-kernel attention block from HADepth.
  This is the component worth 47% of the improvement in the R3 table, so the attribution matters.
- **λ₁ = 0.1, not 0.5.** The text describes λ₁ = 0.5; the released code defaults to 0.1 and the
  published numbers were produced with 0.1.
- **The eight Robinson kernels come in opposite pairs.** Kernels 5–8 are the negatives of 1–4, so
  the descriptor holds each direction twice and, normalised over the eight, equals the
  four-direction descriptor scaled by $1/\sqrt{2}$ in each half; the L2 distance between two
  descriptors is therefore that of the four independent directions. The released code keeps the
  eight directions the paper describes; the two forms optimise the same objective (verified to
  $10^{-7}$), and we state the pairing in §2.1.3 rather than leave the redundancy implicit.
- **The descriptor comparator is L2, not SSIM.** Every number in the paper was produced with the
  L2 comparator; the SSIM form of Eqs. (14)–(15) is reported as an ablation, which ties in-domain.
- **The MonoIIT row does not contain the method.** The MonoViT-based variant was trained in a
  separate codebase that has no illumination calibration and no invariant loss, with an encoder
  starting from random weights rather than ImageNet. It is therefore neither MonoViT as published
  nor an instance of our method, and the row is relabelled and retrained accordingly.
- **The highlight-aware photometric term is HADepth's**, adopted here, and was not attributed.
- **The C3VD evaluation cap is 100 mm**, not the 300 mm stated in §3.2.
- **Checkpoint selection used the test split.** This is stated as a limitation in the revised §3.3.
- **§2.1.3 promises a comparison that does not exist** ("a quantitative comparison under an
  identical training framework is given in Section 3.4" for Robinson vs Sobel, Scharr and Census).
  The sentence is removed.
