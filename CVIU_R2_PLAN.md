# CVIU round 2 — plan for the corrections

> Execution plan with the scope agreed on 2026-11-18 (one cross-domain table, protocol rewrite, qualitative 3D section): `CVIU_R2_ACTIONS.md`. This file keeps the full analysis and the draft texts.

Reviewer comments received 2026-11-04. Manuscript under correction: `cas-dc-template_new.tex` in
`Elsevier_Ricardo_Depth_Prediction_Flattened.zip` (round-1 changes in blue; the "umarked" file is the
same text in black). Line numbers below refer to that file. Numbers come from
`results/cviu/per_frame.csv` / `per_sequence.csv` (grid of 2026-09-15/21, three seeds unless stated),
recomputed on 2026-11-04 with `tools/paired.py`, `tools/components_crossdomain.py` and
`tools/selection_stability.py`.

Sign convention everywhere: (with the component) − (without it), **negative = the component helps**.

---

## 0. Priorities at a glance

| # | Item | Needs | Effort |
|---|---|---|---|
| A | Decide once which run **is** MonoIIF in the statistics tables (§1.1) | your decision, possibly server | 1 h + eval |
| B | Fix the two mislabelled rows of the component table and extend it to Hamlyn and C3VD (R2-1) | done locally, table generated | 2 h writing |
| C | Protocol paragraph on test-set selection + selection-stability result (R2-2) | done locally | 1 h |
| D | Remove every "equivalent / on par" wording; state CI bounds instead (R2-2, R3-3) | nothing | 1 h |
| E | Fig. 4 caption, MonoII paragraph in §2.2.4 (R3-1, R3-2) | nothing | 30 min |
| F | Quantitative surface metrics for the 3D section (R2-3) | server: predictions on the Hamlyn 3D frames | 1 day |
| G | Consistency sweep: abstract, captions, V2, λ₁, depth caps, promises of the round-1 letter (R2-3) | the submitted round-1 letter | 2 h |
| H | Optional server runs that strengthen B (seeds of E7/E5, Hamlyn 4-sequence evaluation) | server | ~1 day GPU |

Everything except F and H can be finished from this machine.

---

## 1. Things found while checking the manuscript against the data (fix before anything else)

### 1.1 "MonoIIF" is not one model across the tables — decision needed

- Table `tab:paired-c1` is built on a reference called `MonoIIF_published` that does not exist in the
  local results. Its SCARED block is numerically **C1** (global calibration, λ₁ = 0.1, HADepth
  highlight term): 0.0497 [0.0400, 0.0594], HADepth −0.0008 [−0.0053, +0.0026], 4/7, p = 0.938 — all
  identical to `table7_ref.py --ref C1`. Its C3VD block is numerically **E8-IIF** (dense local, λ₁ = 0,
  highlight term): 0.2509, HADepth −0.0296. Its Hamlyn block uses four sequences that are not in
  the local csv. Either `MonoIIF_published` is a run I cannot see, or the three blocks come from
  three different reference runs. **Check on the server** (`grep MonoIIF_published results/cviu/per_frame.csv | head`, and `ls logs | grep -i publ`).
- The method as described in §2 (dense local calibration + L_II, λ₁ = 0.5, Eq. 12–13 SSIM
  comparator, no highlight-aware term) matches grid run **`M-local`** (`--photometric standard
  --illumination_invariant 0.5`), which was trained but is not in the local csv. The component
  table (`tab:components-isolated-scared`) and the ladder use **E8** (λ₁ = 0.1, plus HADepth's
  highlight term, which §2 never mentions).
- **Recommendation:** pick one of
  - **Option B (cleaner):** MonoIIF := `M-local`. Evaluate `M-local`, `M-none`, `M-global` (3 seeds)
    on the three datasets if not done, then regenerate `tab:paired-c1` with `--ref M-local` and the
    component table with the contrasts `M-local − E3`, `M-none − E3`, `M-local − M-none`, `E4 − E3`.
    No change to §2, λ₁ stays 0.5, no highlight term to attribute.
  - **Option A (no server):** MonoIIF := `E8`. Then §2 must add the highlight-aware photometric
    term with attribution to HADepth, λ₁ becomes 0.1 in §3.3/§3.4 and Table 2's caption, and the
    old single-run rows (Table 2 line 16, Tables 4–6) must be labelled as the submission's run.
- Whichever option: the same run must be the reference in `tab:paired-c1`, the component table,
  the "Training runs and seeds" paragraph (seed SD 0.0023, 0.0485–0.0528) and the discussion.

### 1.2 Two rows of `tab:components-isolated-scared` are the wrong contrasts (verified)

| row label in the manuscript | printed | actually is | correct value for the label |
|---|---|---|---|
| local calibration alone | −0.0012 [−0.0028, +0.0002], 6/7, p = 0.219 | **global** calibration alone (`da-bas-0 − E3`) | `E4 − E3` = **+0.0000** [−0.0018, +0.0022], 3/7, p = 1.000 |
| illumination-invariant loss | −0.0023 [−0.0081, +0.0017], 3/7, p = 0.938 | HADepth **highlight term** (`E8 − E7`, E7 one seed) | `E8 − E8-IIF` = **+0.0006** [−0.0009, +0.0021], 3/7, p = 0.469 |

The two correct rows are weaker than the printed ones. The text built on them (§3.4 "Illumination
correction model", "Isolated contribution of each component", Discussion first paragraph) has to be
rewritten; drafts in §3 below. Say in the response letter that the two rows were corrected.

### 1.3 Statements that contradict the round-1 corrections or the data

| line | text | problem | fix |
|---|---|---|---|
| 111 | `MonoIIF performs \textcolor{blue}{on par?}` | placeholder left in the abstract | rewrite abstract (§6) |
| 655–656, 762–763, 792 | "DA v2", "Depth~Anything~v2" | the backbone is v1 (round-1 letter says so); §2.2.4 and the abstract already say v1 | "DA v1" / "Depth Anything v1"; keep the V2 citations of §1.2–1.3 |
| 587 | caps 150/150/300 mm | the grid evaluation (`cviu_revision.py:65`) uses 150 / **200** / **100** mm; Tables 4–6 (old single runs) may use the old caps | state the caps per table, or re-evaluate Tables 4–6 with the grid caps |
| 1122–1124 | "working distances up to 300 mm" (C3VD) | C3VD ground truth is clamped at 100 mm | 100 mm |
| 623–632 | "Training runs and seeds": seed SD of MonoIIF 0.0023, 0.0485–0.0528 | must refer to the run chosen in §1.1 | recompute from that run |
| — | round-1 letter promised a checkpoint-selection limitation in §3.3 | absent from the manuscript | §3 R2-2 below |
| — | round-1 letter promised L2 comparator, Robinson-pair sentence, neck attribution to HADepth, highlight-term attribution | none is in the manuscript (Eq. 12–13 still SSIM; no "large-kernel"; no HADepth term) | decide per §1.1; at least the Robinson-pair sentence and the comparator must match the trained models |
| 1073 (Fig. `fig:qual_gen`) | "MonoIIF produces the most uniform and low-error depth maps" on C3VD | `tab:paired-c1`: HADepth better in 6/7 C3VD scenes | "MonoIIF and EndoDAC produce the most uniform maps among the methods without explicit highlight handling; HADepth remains more accurate on this dataset" |
| 1112 | "confirming that illumination-invariant supervision produces depth maps with higher geometric consistency" | the isolated contrasts show L_II does not improve accuracy; attribution is unsupported | "MonoIIF depth maps", and only after the metrics of §3 R2-3 |
| 227 | "competitive or superior accuracy compared to flow-based and foundation model baselines" | not superior to HADepth/EndoDAC | "accuracy that cannot be distinguished from the strongest foundation-model baselines on SCARED and Hamlyn, at less than half their training time" |
| 1150 | Conclusion: "can improve the accuracy of the depth prediction in complex scenes" | not shown for the foundation backbone | align with the abstract |
| Fig. 1 caption | "the proposed method (MonoIIT)" | fine, but the figure is MonoIIT; say so once in §1 | optional |

I do not have the round-1 letter that was actually submitted (the docx in the repo lacks the Fig. 4 /
§2.2.4 acknowledgements the reviewers quote). Put it in the repo and I will check every promise.

---

## 2. Reviewer 3

### R3-1 Figure 4 caption (line 417)
Replace "Architecture of the transformer-based depth estimation framework." by
"Architecture of MonoIIT, the transformer-based depth estimation network." Also §2.2.4 line 409:
"\paragraph{Transformer-based backbone (MonoIIT).}" is fine; add "(Fig.~\ref{MonoViT})" after
"MonoIIT" in its first sentence.

### R3-2 MonoII in §2.2.4 (insert before line 409)
```latex
\paragraph{Convolutional backbone (MonoII).}
MonoII keeps the depth network of Monodepth2~\citep{Godard_ICCV_2019}: a ResNet-18 encoder
pre-trained on ImageNet~\citep{Deng_CVPR_2009} and a DispNet-style decoder with skip connections
that predicts disparity at four scales. The pose network is a separate ResNet-18 encoder taking the
concatenated pair $(I_s, I_t)$, followed by the pose decoder of Monodepth2; its last feature map
also feeds the lighting change decoder of Section~\ref{sec:lit}. MonoII therefore differs from
Monodepth2 only by the two proposed components, the per-pixel lighting calibration in
$\mathcal{L}_\textrm{PML}$ and the illumination-invariant loss $\mathcal{L}_\textrm{II}$, and is
the reference against which the transformer-based (MonoIIT) and foundation-model (MonoIIF)
encoders are compared.
```
Then make §3.5 (lines 846–848) consistent: same wording for encoder/decoder/pose network, and drop
"This architecture of this model" (typo).

### R3-3 Equivalence wording (shared with R2-2)
Lines to change: 111 (abstract: "statistically indistinguishable", "generalizes comparably"),
1024 ("MonoIIF should therefore be considered equivalent"), 1046 ("statistically equivalent"),
1119 ("statistically on par"). Replacement pattern, with the CI bounds of `tab:paired-c1`:

> "No significant difference was detected between MonoIIF and EndoDAC on any dataset, nor between
> MonoIIF and HADepth on SCARED and Hamlyn ($n = 7$, $7$ and $4$ units). With these sample sizes the
> absence of a significant difference does not establish equivalence; what the intervals exclude is
> a deficit of MonoIIF larger than 0.0053 (SCARED) and 0.0042 (Hamlyn) in $\varepsilon_\textrm{AbsRel}$
> with respect to HADepth, and larger than 0.0058 (SCARED), 0.0016 (Hamlyn) and 0.0189 (C3VD) with
> respect to EndoDAC. On C3VD HADepth is more accurate than MonoIIF (6 of 7 scenes, $p = 0.031$)."

(Bounds = the CI end that is unfavourable to MonoIIF, read from `tab:paired-c1`; recompute if the
reference run changes per §1.1.) Add one sentence to §3.2: "Throughout, 'cannot be distinguished'
means that the 95\% interval of the paired difference includes zero; it is not a claim of
equivalence."

---

## 3. Reviewer 2

### R2-1 Contribution of the components beyond the backbone, isolated on every dataset

**Data: done.** `results/cviu/tables/components_crossdomain.tex` (regenerate with
`python tools/components_crossdomain.py`, or `--markdown` for a text version). Replace
`tab:components-isolated-scared` (lines 733–749) by it and cite it from §3.4 and the Discussion.
Key rows (Abs Rel, with − without; * = interval excludes zero):

| component | SCARED (n=7) | Hamlyn (n=58 blocks) | C3VD (n=7) | seeds |
|---|---|---|---|---|
| pretrained DA weights (E8 − N0) | −0.0549* 7/7 | −0.0840* 55/58 | −0.1372* 7/7 | 1 |
| DV-LoRA adapters (E3 − E2) | −0.0089* 6/7 | −0.0031, 34/58 | +0.0323* 0/7 | 1 |
| dense local calibration alone (E4 − E3) | +0.0000, 3/7 | +0.0029* 18/58 | **−0.0260*** 7/7 | 3 |
| global calibration alone (da-bas-0 − E3) | −0.0012, 6/7 | +0.0032* 22/58 | **−0.0256*** 7/7 | 3 |
| L_II alone (E5 − E3) | +0.0004, 5/7 | +0.0015* 21/58 | +0.0056, 2/7 | 1 |
| calibration + L_II, no highlight term (E7 − E3) | +0.0018, 3/7 | −0.0005, 36/58 | +0.0091, 2/7 | 1 |
| L_II inside MonoIIF (E8 − E8-IIF) | +0.0006, 3/7 | −0.0015, 38/58 | **+0.0170*** 1/7 | 3 |
| highlight term inside MonoIIF (E8 − E7) | −0.0023, 3/7 | +0.0048* 16/58 | −0.0342* 7/7 | 1 |
| all three (E8 − E3) | −0.0005, 5/7 | +0.0043* 16/58 | −0.0250* 7/7 | 3 |
| ResNet-18: calibration + L_II (MonoII − R1) | +0.0015, 3/7 | −0.0007, 28/58 | +0.0100* 0/7 | 3 |
| ResNet-18: global calibration alone (cal0-glob − R1) | +0.0016, 2/7 | −0.0108* 49/58 | −0.0002, 3/7 | 1 |
| ResNet-18: all three (R2 − R1) | −0.0000, 4/7 | −0.0055* 42/58 | −0.0156* 7/7 | 3 |
| MPViT-S: calibration + L_II given highlight term | −0.0008, 1/7 | −0.0214* 50/58 | +0.0094* 1/7 | 1 |

**What to write (drafts).**

*§3.4, replace the "Illumination correction model" and "Isolated contribution" paragraphs:*
> On the foundation backbone, neither illumination component changes the in-domain accuracy: the
> dense calibration alone is +0.0000 [−0.0018, +0.0022] and $\mathcal{L}_\textrm{II}$ alone
> +0.0004 [−0.0029, +0.0048], against a contribution of −0.0549 for the pretrained weights and
> −0.0089 for the DV-LoRA adapters. The in-domain accuracy of MonoIIF is therefore that of the
> adapted foundation model. Outside the training domain the two components act in opposite
> directions and depend on the type of shift (Table~X). Under the photometric shift of Hamlyn the
> calibration costs 0.0029 [0.0013, 0.0045] and $\mathcal{L}_\textrm{II}$ is neutral
> (−0.0015 [−0.0029, +0.0001]); under the geometric shift of C3VD the calibration is worth
> −0.0260 [−0.0360, −0.0160], better in all seven scenes, whereas $\mathcal{L}_\textrm{II}$ costs
> 0.0170 [0.0070, 0.0270]. On the ResNet-18 backbone the same two components improve Hamlyn
> (−0.0055 [−0.0074, −0.0036], 42 of 58 blocks) and C3VD (−0.0156 [−0.0212, −0.0106], 7 of 7).
> The benefit of the illumination components is thus largest for an encoder without large-scale
> pretraining, and on the foundation backbone it is confined to the calibration under geometric
> shift.

*Discussion "Cross-dataset generalization" (lines 1122–1124):* delete "Since the illumination
components were isolated only on SCARED ... can only be assessed indirectly" and the three
"observations"; replace by the direct result above. Keep the sentence on MonoII vs AF-SfMLearner
if you want, labelled as indirect.

*Abstract and §1.3:* state the contribution as "a lighting calibration that improves transfer under
geometric domain shift (C3VD, all seven scenes) and an illumination-invariant loss whose benefit is
measurable only for lightweight encoders"; drop "complementary photometric robustness mechanisms"
as a claim of accuracy.

**Decision:** whether to keep $\mathcal{L}_\textrm{II}$ in the headline of MonoIIF. The honest
options are (i) keep it with the sentence above, or (ii) present MonoIIF as calibration-only on the
foundation backbone and $\mathcal{L}_\textrm{II}$ as the lightweight-backbone component. (i) needs
no retraining.

**Optional server work that strengthens this section** (in order of value):
1. Two more seeds of `E7` (calibration + L_II, no highlight term) — the row that is literally "the
   two proposed components", at one seed. If Option B of §1.1 is taken, `M-local`/`M-none` replace it.
2. Evaluate the grid runs used in the table (`N0 E2 E3 E4 E5 E7 E8 E8-IIF C0 da-bas-0 R1 R2 cal0-*
   MonoII* MonoIIT-*`) on Hamlyn `rectified06/08/11`, so the Hamlyn column uses the same four
   sequences as `tab:paired-c1` (evaluation only; the script switches to sequence units
   automatically). Until then the caption must say the Hamlyn unit is a 100-frame block of
   rectified14.

### R2-2 Evaluation protocol: did the test set influence model selection?

**Facts to state (new paragraph "Model selection and the test split", after "Training runs and
seeds", line 632):**
1. The λ₁ sweep and the choice of configuration (Table 2) were made on the 551 SCARED test frames,
   because the SCARED validation split of our copy has no depth ground truth [confirm]. The SCARED
   numbers are therefore development-set numbers for the proposed models. Hamlyn and C3VD were
   never used for any choice and are the held-out evidence.
2. For every model trained here, the reported checkpoint is the epoch with the lowest RMSE on the
   SCARED test split (the EndoDAC training loop, inherited); the released checkpoints of the
   comparison methods were not re-selected. This favours the models trained here on SCARED only.
3. Result of `tools/selection_stability.py` (all C(7,k) splits of the seven test sequences,
   k = 3 and 4; select on the subset, score on the complement):

| family | range on all 7 | selected = best on held-out | mean / max held-out regret |
|---|---|---|---|
| DA calibration form, λ₁ = 0 (5 configs) | 0.0015 | 12/35 (k=3), 12/35 (k=4) | 0.0004 / 0.0024 |
| DA λ₁ ∈ {0, 0.1, 0.25, 0.5} at degree 1 | 0.0024 | 27/35, 27/35 | 0.0006 / 0.0043 |
| MonoIIF candidates (E3, C0, C1, E8-IIF, E8) | 0.0022 | 21/35, 21/35 | 0.0005 / 0.0029 |
| ResNet-18 calibration form, λ₁ = 0 | 0.0033 | 5/35, 5/35 | 0.0019 / 0.0060 |
| ResNet-18 λ₁ ∈ {0, 0.1, 0.25, 1, 2} | 0.0048 | 6/35, 6/35 | 0.0015 / 0.0048 |
| MonoII family (none/global/dense, λ₁ = 0.5) | 0.0009 | 0/35, 0/35 | 0.0020 / 0.0054 |

   Reading for the paper: the configurations differ by at most 0.005 Abs Rel in-domain, the
   winner on a subset of sequences is the winner on the remaining ones in a minority of splits,
   and choosing on the full test set can have flattered the chosen configuration by at most
   0.002–0.006 Abs Rel (max regret). This is why no in-domain claim rests on the choice, and why
   the paper now reports the components' effect out of domain, where no selection took place.
   Two caveats to write: the retained λ₁ = 0.5 is not the in-domain optimum of either backbone
   (λ₁ = 0 is), and the ResNet-18 sweep of Table 2 (one seed, frame-pooled) is superseded by the
   three-seed sweep (`results/cviu/tables/lambda_sweep_resnet.tex`).
4. Wording: see R3-3. Never "equivalent"; "no difference detected" plus the interval bound.

**Optional server work:** evaluate the last-epoch checkpoint (`weights_last`, if it exists for the
grid runs) of E8/C1/E3 to quantify the checkpoint-selection optimism on SCARED; if it is below the
seed SD (0.001–0.002), say so in one sentence.

### R2-3 3D surface reconstruction: quantitative support, and the consistency check

**Metrics to add (new Table in §4.3, three methods: HADepth, EndoDAC, MonoIIF × 3 seeds, same
frames as Figs. 3D_Single and 3D_10Frames):**
- per-frame, after median scaling to the ground truth: Chamfer accuracy (pred→GT) and
  completeness (GT→pred) in mm, F-score at τ = 5 mm, mean angular error of surface normals (deg),
  and a roughness ratio (mean |Laplacian| of the predicted depth over that of the GT depth, which
  captures the staircase/comb artefacts described in the text);
- for the 10-frame mosaic: Chamfer distance between the fused mesh and the mesh fused from GT
  depth with the same poses, and the multi-view depth inconsistency between consecutive frames
  (|D_i→j − D_j| / D_j, as in SC-SfMLearner).
- Tool: `tools/surface_metrics.py` (to write; inputs: per-frame predicted depth `.npy`, GT depth
  PNG in mm (`depth01/`) or GT `.ply`, intrinsics txt, optional 4×4 poses; numpy/scipy only,
  TSDF via open3d when available). Server: dump predictions for the Hamlyn frames of the figures
  for the three models (~minutes each). I can write the tool on request.
- Text: replace "qualitative superiority ... confirming that illumination-invariant supervision
  produces depth maps with higher geometric consistency" (lines 1111–1112) by the measured
  differences, attributed to MonoIIF's depth maps, not to $\mathcal{L}_\textrm{II}$. If the
  metrics do not separate the methods, keep the figures as illustrations and say so.

**Consistency check:** work through the table of §1.3 plus every promise of the submitted round-1
letter. Mark every round-2 change in a new colour (e.g. `\textcolor{red}`, `\rev` is already
defined at line 67) and produce the black version by the same substitution as before.

---

## 4. Order of work

1. §1.1 decision (which run is MonoIIF) and, if Option B, launch the `M-*` evaluations on the server.
2. R3-1, R3-2, R3-3 edits (mechanical).
3. R2-1: paste `components_crossdomain.tex`, rewrite the three paragraphs, abstract, §1.3, conclusion.
4. R2-2: protocol paragraph, selection-stability sentence, wording sweep.
5. §1.3 consistency table; captions.
6. R2-3: predictions on the server, `tools/surface_metrics.py`, table and text.
7. Response letter: one block per comment with the new table/paragraph quoted; state the two
   corrected rows and the resolved inconsistencies explicitly.
8. Regenerate the black version, diff against the previous submission, final read.

## 5. Open questions for you

1. Which run is `MonoIIF_published` in `tab:paired-c1`, and are `M-local`/`M-none`/`M-global`
   evaluated on the server (3 seeds, 3 datasets)?
2. Did the submitted models use λ₁ = 0.5 (manuscript) or 0.1 (round-1 letter draft)?
3. Which depth caps produced Tables 4–6 (old scripts: 150/150/300?) — they must be stated per table.
4. Does the SCARED validation split of your copy have depth ground truth? (Determines the wording
   of R2-2 point 1.)
5. The submitted round-1 response letter (not the docx in the repo) — needed for the consistency check.
