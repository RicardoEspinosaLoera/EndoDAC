# CVIU round 2 — action plan (agreed scope, 2026-11-18)

Scope fixed by the co-authors' annotations on the reviewer comments:

- **(1)** one new table isolating the components on Hamlyn and C3VD (inference only, no training);
- **(2)** Sections 3.1 and 3.4 rewritten to say how the public split was used;
- **(3)** the 3D section stays qualitative (continuity of texture and geometry), with careful statements.

Background analysis and all numbers: `CVIU_R2_PLAN.md`. Line numbers refer to
`cas-dc-template_new.tex` in `Elsevier_Ricardo_Depth_Prediction_Flattened.zip`. Mark every round-2
change with `\textcolor{red}{}` (`\rev` is defined at line 67 and unused) and produce the black
version by the same substitution used for the blue round-1 marks.

---

## Day 0 — three decisions (block everything else)

| # | Decision | Default if nobody objects | Consequence |
|---|---|---|---|
| D1 | Which run is "MonoIIF" in the statistics tables | **E8** (dense local calibration + L_II, λ₁ = 0.1, HADepth highlight term): fully evaluated, 3 seeds, 3 datasets | caption of every statistics table names the run; §2 must then mention the highlight-aware term with attribution to HADepth, or the λ₁/term of §2 must be reconciled (see `CVIU_R2_PLAN.md` §1.1). If `M-local` is preferred, evaluate `M-local`, `M-none`, `M-global` on the server first (inference only if trained). |
| D2 | Hamlyn unit in the new table | 58 blocks of 100 frames of rectified14 (what the local data has) | caption states it; if the co-authors want the four sequences of `tab:paired-c1`, run the server evaluation of task 1.2 first |
| D3 | One continuity statistic for the 3D section, or none | none (co-authors' preference) | with none, every conclusion about surfaces becomes an observation about the shown examples (task 3) |

---

## Task 1 — the cross-domain component table (reviewer 2, point 1)

| step | what | where | status |
|---|---|---|---|
| 1.1 | Generate the table: `python -B tools/components_crossdomain.py --markdown` → `results/cviu/tables/components_crossdomain.tex`. Rows = (with − without) per component on SCARED, Hamlyn, C3VD, 95 % bootstrap CI, k/n, exact p. | — | done 2026-11-04; rerun after any new evaluation |
| 1.2 | *(only if D2 = four sequences)* server: re-run the `predict` and `stats` stages of `cviu_revision.py` for `N0 E2 E3 E4 E5 E7 E8 E8-IIF C0 da-bas-0 R1 R2 cal0-glob cal0-deg1 cal0-local MonoII MonoII-none MonoIIT-repro MonoIIT-base MonoIIT-components` with the Hamlyn folder holding rectified06/08/11/14; the script switches to sequence units by itself. Inference only, a few hours. | server | optional |
| 1.3 | Replace `tab:components-isolated-scared` (lines 733–749) by the new table. It subsumes the SCARED-only table and corrects its two wrong rows ("local calibration alone" was the global contrast; "illumination-invariant loss" was the highlight-term contrast). Rename the label to `tab:components-isolated` and update the references at lines 619, 621, 752, 767, 778, 1119, 1122. | §3.4 | |
| 1.4 | Rewrite the paragraphs that quote the old rows: "Illumination correction model" (765–767: delete the −0.0012 argument), "Isolated contribution of each component" (777–781), Discussion first paragraph (1119) and "Cross-dataset generalization" (1122–1124: delete "isolated only on SCARED ... assessed indirectly" and the three indirect observations). Drafts: `CVIU_R2_PLAN.md` §3 R2-1. | §3.4, §5 | |
| 1.5 | Align the claims: abstract (line 111), §1.3 (227: no "superior"), conclusion (1150). The sentence to carry everywhere: on the foundation backbone the components do not change in-domain accuracy; the calibration improves transfer under the geometric shift of C3VD (7/7 scenes) and costs accuracy under the photometric shift of Hamlyn; L_II has no measurable benefit on this backbone and costs accuracy on C3VD; both components help the lightweight backbones out of domain. | abstract, §1.3, §6 | |
| 1.6 | "Training runs and seeds" (618–632): list the new table among the three-seed tables; recompute the seed SD and range of MonoIIF from the D1 run. | §3.2 | |

Acceptance: every number quoted in §3.4 and §5 is in the new table; no sentence claims a consistent
benefit of the two components on the foundation backbone; the old −0.0012 and −0.0023 no longer appear.

---

## Task 2 — evaluation protocol and wording (reviewer 2, point 2)

Facts verified in `splits/endovis/` (the AF-SfMLearner split adopted by EndoDAC):

| subset | frames in the split files | keyframe videos |
|---|---|---|
| train | 15 350 | keyframes 1–2 of datasets 1–7, keyframes 1–4 of datasets 8–9 |
| validation | 1 705 | keyframe 3 of datasets 2–7 |
| test | 550 | keyframe 3 of dataset 1, keyframe 4 of datasets 2–7 (the seven units of the paired tables) |

The manuscript says 15 351 / 1 705 / 551; check which count is right before quoting.

| step | what | where |
|---|---|---|
| 2.1 | §3.1 (line 556): name the split (AF-SfMLearner/EndoDAC, files released with the code), give the table above in words, and say in one sentence what each subset was used for: training; validation not used for depth selection (no depth ground truth in our copy — **confirm on the server**); test = ablation, λ₁ and configuration choice, checkpoint choice, and the reported SCARED numbers. | §3.1 |
| 2.2 | New paragraph "Model selection and the test split" after "Training runs and seeds" (after line 632): (i) λ₁ and the configuration were chosen on the SCARED test frames, so the SCARED numbers of the proposed models are development-set numbers; Hamlyn and C3VD were never used for any choice; (ii) for every model trained here the reported checkpoint is the epoch with the lowest RMSE on the SCARED test split (EndoDAC's training loop); the released checkpoints of the comparison methods were not re-selected; (iii) the split-half check (`tools/selection_stability.py`): over all C(7,3) and C(7,4) subsets of the test sequences, the configuration selected on a subset is also the best on the held-out sequences in a minority of splits, and the held-out regret is at most 0.006 Abs Rel, below or at the seed-to-seed variation; hence no in-domain claim rests on the choice, and the components are judged on the held-out datasets. | §3.2 |
| 2.3 | §3.4 ablation introduction (line 685): say the ablation runs on the test frames and was used for the choices above, referring to 2.2; add that in the three-seed sweeps (`results/cviu/tables/lambda_sweep_resnet.tex`, `contrast_da_lambda.tex`) λ₁ = 0 is the in-domain optimum on both backbones and the retained λ₁ = 0.5 was kept for continuity with the submitted models. | §3.4 |
| 2.4 | Wording sweep. Replace at lines 111 ("statistically indistinguishable", "generalizes comparably"), 1024 ("should therefore be considered equivalent"), 1046 ("statistically equivalent"), 1119 ("statistically on par") with: "no significant difference was detected (n = 7 / 4 units); this does not establish equivalence; the intervals exclude a deficit of MonoIIF larger than X" with X = the unfavourable CI bound of `tab:paired-c1` (HADepth: 0.0053 SCARED, 0.0042 Hamlyn; EndoDAC: 0.0058 SCARED, 0.0016 Hamlyn, 0.0189 C3VD; recompute if D1 changes the reference). Add to §3.2 the definition: "cannot be distinguished" = the 95 % interval of the paired difference includes zero. | abstract, §4.1, §5 |
| 2.5 | Optional server check: evaluate `weights_last` of E8, C1, E3 if kept, to quantify the checkpoint-selection optimism; one sentence if it is below the seed SD. | server |

Acceptance: `grep -n "equivalen\|on par\|comparabl\|indistinguish"` finds no sentence that asserts
equivalence; §3.1 and §3.2 state every use of the test split explicitly.

---

## Task 3 — 3D section kept qualitative (reviewer 2, point 3)

Principle: the figures show continuity of texture and geometry in examples; nothing in the text may
state a conclusion about accuracy or attribute the difference to the illumination-invariant loss.

| step | what | where |
|---|---|---|
| 3.1 | Abstract, last sentence: "Qualitative surface reconstruction and 3D mosaicing examples on Hamlyn illustrate the use of the depth maps; in these examples the surfaces obtained from MonoIIF show fewer geometric discontinuities than those obtained from HADepth." | 111 |
| 3.2 | §4.3 text (1093–1112): keep the descriptions of what is visible ("staircase", "comb-like"), prefixed by "in this example"; delete "confirming that illumination-invariant supervision produces depth maps with higher geometric consistency that translates directly to improved 3D mosaicing quality" (1112) and "reveals a practical advantage not fully captured by the quantitative benchmarks"; replace by: "These examples illustrate a property that the per-pixel criteria of Table 1 do not measure, the continuity of the surface across neighbouring pixels and viewpoints; they are illustrations, not a quantitative comparison, and are consistent with the paired analysis in which MonoIIF and HADepth cannot be distinguished on Hamlyn." | §4.3 |
| 3.3 | Discussion (1126–1127): remove "This explains why the performance difference between MonoIIF and HADepth is clearly perceptible ..."; keep the methodological remark that pixel-wise criteria do not measure surface continuity, as a limitation of the evaluation, not as evidence for MonoIIF. | §5 |
| 3.4 | Add EndoDAC to Figs. 3D_Single and 3D10 (it cannot be distinguished from MonoIIF on any dataset), or one sentence stating why only HADepth is shown (best competitor in-domain and on C3VD). | §4.3 |
| 3.5 | *(only if D3 = yes)* one number per reconstruction, no ground truth needed: fraction of the fused surface lying on depth discontinuities above a threshold, and number of disconnected mesh fragments, for HADepth, EndoDAC and MonoIIF; needs the predictions already used for the figures and a 100-line script (`tools/surface_continuity.py`, to write on request). | server + tools |
| 3.6 | Captions: `fig:qual_gen` (1036) C3VD sentence → "MonoIIF and EndoDAC produce the most uniform maps among the methods without explicit highlight handling; HADepth remains more accurate on this dataset (Table~\ref{tab:paired-c1})"; `fig:qual_scared` "consistently produces the lowest error" → "produces the lowest error in the ROIs of these five examples"; Fig. 1 caption: fine. | captions |

Acceptance: no sentence in the abstract, §4.3 or §5 uses "confirm", "demonstrate", "more accurate",
"superior" or attributes surface quality to L_II.

---

## Task 4 — consistency sweep (reviewer 2, point 3, second half)

Work through `CVIU_R2_PLAN.md` §1.3 in full. Minimum list: "on par?" placeholder (111); "DA v2" →
"DA v1" (655–656, 762–763, 792); depth caps 150/200/100 mm stated per table (587, and C3VD "up to
300 mm" at 1122–1124 → 100 mm); seed SD paragraph consistent with D1; every promise of the
submitted round-1 letter present in the text (needs that letter — the docx in the repo is not it).

---

## Task 5 — response letter and build

| step | what |
|---|---|
| 5.1 | One block per comment, in the reviewers' order; each block names the section, table and lines changed and quotes the new key sentence. For point 1, say explicitly that two rows of the previous component table were computed from the wrong configuration pairs and give the corrected values. For point 2, quote the new protocol paragraph and the split-half result. For point 3, state that the 3D results are presented as illustrations and that every conclusion drawn from them was removed or reworded, and list the consistency corrections. |
| 5.2 | Build the red-marked PDF, produce the black version, diff against the previous submission, check that every `\ref` resolves (`tab:components-isolated`), and that the abstract, captions and tables agree on every number. |

---

## Sequence

| day | work | parallel server work |
|---|---|---|
| 0 | D1–D3; if D1 = `M-local`, start its evaluations | `M-*` evaluations; task 1.2 if D2 = four sequences |
| 1 | Task 1 (table in, four paragraphs, abstract/§1.3/conclusion) | |
| 2 | Task 2 (§3.1, new protocol paragraph, §3.4 intro, wording sweep) | 2.5 if `weights_last` exists |
| 3 | Task 3 and Task 4 | 3.5 if D3 = yes |
| 4 | Task 5 (letter, build, diff, final read) | |

Total: four working days without server work; the optional server items add inference time only.
