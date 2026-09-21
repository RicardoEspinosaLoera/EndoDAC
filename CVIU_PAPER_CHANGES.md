# Changes to the manuscript, point by point

Every number below is from `results/cviu/summary.csv` and `results/cviu/per_sequence.csv`,
sequence-level, three training seeds where the model was trained here. Paired differences are
`variant − MonoIIF` per sequence: **positive means MonoIIF is better**. LaTeX blocks are ready to
paste and need only `\usepackage{booktabs}`.

Legend used throughout: `[a, b]` is a 95% cluster bootstrap CI over sequences (10 000 draws);
`k/n` is the number of sequences in which MonoIIF is better; exact Wilcoxon p is given for the
two datasets with n = 7 and omitted for Hamlyn, where n = 58 blocks and the CI is the informative
quantity.

---

## 1. Title, abstract and contribution claims

**1.1 — "Depth Anything V2" → "Depth Anything".** The backbone is **v1**. Fix in the abstract,
§2.2.4, §3.3, §3.4, §3.5, §5 and §6, and delete the duplicated *Yang et al. 2024* reference that
points at V2. The V2 mentions inside the related-work sections §1.2 and §1.3 are correct and stay.

**1.2 — Reframe the illumination-invariant loss.** The abstract and §1 present it as an accuracy
contribution. It is not: removing it is neutral in-domain (−0.0006 [−0.0021, +0.0009]) and on
Hamlyn (+0.0015 [−0.0000, +0.0030]), and **improves** C3VD by 0.0170 [−0.0271, −0.0072], better
in 6 of 7 scenes. Replace "improves accuracy" with a robustness claim supported by the
sensitivity analysis, or state the trade-off explicitly.

**1.3 — Soften the headline.** The four leading methods are statistically indistinguishable
in-domain (§7 below). The defensible claim is out-of-domain robustness under geometric shift,
where MonoIIF beats a retrained EndoDAC by 0.0250 in 7 of 7 C3VD scenes.

---

## 2. §2.1.3 — the illumination-invariant descriptor

**2.1 — "eight Robinson kernels" → "four".** The remaining four are the exact negatives of the
first four and add no information; the code uses four.

**2.2 — Delete the promised comparison.** The sentence *"a quantitative comparison under an
identical training framework is given in Section 3.4"* (Robinson vs Sobel, Scharr, Census) refers
to an experiment that does not exist. Remove it, or restrict it to the comparator ablation that
does exist (L2 vs SSIM, §5.2 below).

**2.3 — The comparator is L2, not SSIM.** Eqs. (14)–(15) describe the SSIM form; every published
number was produced with the L2 comparator. State L2 as the method and cite SSIM as an ablation
that ties in-domain.

---

## 3. §2.2 — architecture and attribution

**3.1 — The convolutional neck is not EndoDAC's.** The block credited to EndoDAC is a
**large-kernel attention module inherited from HADepth** (three gated depthwise branches with
dilated companions), not ViTDet's 1×1/3×3/1×1 bottleneck. This is the component worth 47% of the
improvement over a random encoder, so the attribution is material. Rewrite the sentence that
credits it and cite HADepth.

**3.2 — Attribute the highlight-aware photometric term to HADepth.** It is adopted, not proposed,
and currently appears unattributed.

**3.3 — λ₁ = 0.1, not 0.5.** The text states 0.5; the code defaults to 0.1 and every published
number was produced with 0.1. Fix §3.3 and Table 2's caption.

---

## 4. §3 — experimental protocol

**4.1 — C3VD depth cap is 100 mm**, not the 300 mm stated in §3.2.

**4.2 — Checkpoint selection used the test split.** Add a limitation sentence in §3.3: the best
epoch was chosen by evaluating on the test sequences, so in-domain numbers are optimistic. This
affects every method in the tables equally, ours included.

**4.3 — Add the statistical protocol** (new paragraph in §3.2, text for the response letter is in
`CVIU_RESPONSE_LETTER.md`): sequence as the unit, cluster bootstrap, paired tests, the n = 7
Wilcoxon floor of 2/2⁷ = 0.0156, and the two pre-specified primary comparisons.

**4.4 — Report three seeds.** Every model trained here is now mean ± SD over seeds 314, 1 and 2.

---

## 5. Tables 4–6 — keep them, and add what Question 6 asks for

The reviewer asks for *uncertainty estimates and paired statistical comparisons using video
sequences*. That is an addition, not a replacement: the point estimates of Tables 4–6 are not what
is being questioned and do not change. Aggregating by sequence instead of pooling frames moves
every number by 0.0005–0.0006 — the seven SCARED sequences hold 71 to 87 frames each, so the two
weightings almost coincide — which is not a difference worth rewriting a published table for.

**5.0 — What actually changes in Tables 4–6.** Only two things:

1. Add `±` to the rows of models trained here, the SD across the three training seeds. The
   external rows are the authors' released checkpoints (n = 1) and get no `±`; say so in the
   caption. Our reproduction of all five reproduces their published values (Monodepth2 0.0939 vs
   0.094, AF-SfMLearner 0.0590 vs 0.059, MonoPCC 0.0513 vs 0.051, EndoDAC 0.0513 vs 0.051,
   HADepth 0.0494 vs 0.049), which is worth one sentence because it validates the protocol.
2. Stop bolding differences the data cannot resolve. On SCARED the four leading rows span 0.0489
   to 0.0512 and every interval among them covers zero (§5.4).

The new material goes in a companion table, §5.4, immediately after them.

**Why the sequence-level analysis matters even though the means barely move.** The uncertainty
changes by an order of magnitude. For MonoIIF on SCARED, treating the 551 frames as independent
observations gives a 95% interval of [0.0507, 0.0528], width 0.0022; resampling the 7 sequences
gives [0.0408, 0.0616], width 0.0208 — **9.5 times wider**. Frames within one video are nearly the
same observation, so the frame-level interval is not an interval at all. Under the naive version
HADepth [0.0478, 0.0511] and MonoIIF [0.0507, 0.0528] look almost disjoint; at the sequence level
they overlap almost entirely. The reviewer's example resolves the moment the unit is right: the
per-sequence differences between them are +0.0005, −0.0018, −0.0136, +0.0035, −0.0064, +0.0040,
−0.0019, so MonoIIF is better in 3 of 7 sequences and nearly the whole aggregate gap comes from
one sequence.

### (optional) Sequence-level versions of Tables 4–6

Only if the editor prefers the main tables recomputed rather than annotated. The values differ
from the published ones by 0.0005–0.0006.

#### 5.1 Table 4 — SCARED (n = 7 sequences)

```latex
\begin{table}[t]
\centering
\caption{SCARED, sequence level ($n=7$ keyframe videos). Metrics are averaged within a sequence
and then over sequences. For models trained here, $\pm$ is the standard deviation across three
training seeds; the comparison methods are the authors' released checkpoints and have no seed
variation. Best in bold.}
\label{tab:scared}
\small
\begin{tabular}{lccccc}
\toprule
Model & $\varepsilon_\text{AbsRel}$\,$\downarrow$ & $\varepsilon_\text{SqRel}$\,$\downarrow$ &
        $\varepsilon_\text{RMSE}$\,$\downarrow$ & $\varepsilon_\text{RMSELog}$\,$\downarrow$ &
        $\delta_{1.25}$\,$\uparrow$ \\
\midrule
Monodepth2      & 0.093 & 1.178 & 8.128 & 0.133 & 0.907 \\
AF-SfMLearner   & 0.058 & 0.469 & 5.067 & 0.083 & 0.968 \\
MonoPCC         & 0.051 & 0.348 & 4.507 & 0.072 & 0.983 \\
EndoDAC         & 0.051 & 0.350 & 4.384 & 0.072 & 0.979 \\
HADepth         & \textbf{0.049} & \textbf{0.320} & \textbf{4.270} & \textbf{0.069} & \textbf{0.983} \\
\midrule
MonoIIF (ours)  & 0.051\,$\pm$\,0.001 & 0.381\,$\pm$\,0.025 & 4.619\,$\pm$\,0.143 &
                  0.074\,$\pm$\,0.002 & 0.980\,$\pm$\,0.002 \\
\bottomrule
\end{tabular}
\end{table}
```

#### 5.2 Table 5 — Hamlyn (evaluation only, n = 58 blocks of 100 frames)

```latex
\begin{table}[t]
\centering
\caption{Hamlyn, evaluation only; all models are trained on SCARED. Our copy holds a single
rectified sequence, so contiguous blocks of 100 frames are the unit of analysis ($n=58$); the
partition is identical for every method, so the paired tests stay aligned, and the intervals are
slightly optimistic because blocks within one video remain correlated.}
\label{tab:hamlyn}
\small
\begin{tabular}{lccccc}
\toprule
Model & $\varepsilon_\text{AbsRel}$\,$\downarrow$ & $\varepsilon_\text{SqRel}$\,$\downarrow$ &
        $\varepsilon_\text{RMSE}$\,$\downarrow$ & $\varepsilon_\text{RMSELog}$\,$\downarrow$ &
        $\delta_{1.25}$\,$\uparrow$ \\
\midrule
Monodepth2      & 0.223 & 6.024 & 17.427 & 0.296 & 0.637 \\
AF-SfMLearner   & 0.183 & 5.561 & 15.240 & 0.233 & 0.748 \\
MonoPCC         & 0.172 & 4.960 & 15.139 & 0.232 & 0.757 \\
HADepth         & 0.163 & 4.414 & 14.046 & 0.217 & 0.768 \\
EndoDAC         & \textbf{0.161} & \textbf{4.028} & \textbf{13.507} & \textbf{0.213} & 0.767 \\
\midrule
MonoIIF (ours)  & \textbf{0.161}\,$\pm$\,0.003 & 4.246\,$\pm$\,0.122 & 13.785\,$\pm$\,0.195 &
                  0.215\,$\pm$\,0.002 & \textbf{0.772}\,$\pm$\,0.002 \\
\bottomrule
\end{tabular}
\end{table}
```

#### 5.3 Table 6 — C3VD (evaluation only, n = 7 scenes)

```latex
\begin{table}[t]
\centering
\caption{C3VD, evaluation only; all models are trained on SCARED. Depth is capped at 100\,mm.}
\label{tab:c3vd}
\small
\begin{tabular}{lccccc}
\toprule
Model & $\varepsilon_\text{AbsRel}$\,$\downarrow$ & $\varepsilon_\text{SqRel}$\,$\downarrow$ &
        $\varepsilon_\text{RMSE}$\,$\downarrow$ & $\varepsilon_\text{RMSELog}$\,$\downarrow$ &
        $\delta_{1.25}$\,$\uparrow$ \\
\midrule
MonoPCC         & 0.357 & 5.755 & 15.933 & 0.438 & 0.355 \\
Monodepth2      & 0.342 & 5.529 & 15.745 & 0.428 & 0.386 \\
AF-SfMLearner   & 0.336 & 5.671 & 15.827 & 0.425 & 0.402 \\
EndoDAC         & 0.257 & 3.690 & 13.233 & 0.338 & 0.523 \\
HADepth         & \textbf{0.221} & \textbf{3.096} & \textbf{12.704} & \textbf{0.309} & \textbf{0.584} \\
\midrule
MonoIIF (ours)  & 0.268\,$\pm$\,0.011 & 3.908\,$\pm$\,0.217 & 13.892\,$\pm$\,0.319 &
                  0.354\,$\pm$\,0.012 & 0.489\,$\pm$\,0.016 \\
\bottomrule
\end{tabular}
\end{table}
```

### 5.4 New Table 7 — the uncertainty and paired comparisons Question 6 asks for

This is the table that answers Q6 and it should sit immediately after Tables 4–6.

```latex
\begin{table}[t]
\centering
\caption{Uncertainty and paired comparisons for the methods of Tables~4--6, with the video
sequence as the unit of analysis. Column~3 gives each method's Abs Rel with a 95\% cluster
bootstrap confidence interval over sequences (10\,000 draws). Column~4 gives the paired difference
(method $-$ MonoIIF) per sequence, signed so that a positive value means MonoIIF is better, with
its own bootstrap interval; $k/n$ counts the sequences in which MonoIIF is better. The exact
two-sided Wilcoxon signed-rank $p$ is given for the two datasets with $n=7$; its smallest
attainable value there is $2/2^{7}=0.016$, so no family-wise correction across a table of this
size can reach significance, and these comparisons are exploratory. In column~3 the best value per
dataset is in bold and the second best underlined; in column~4 paired differences whose interval
excludes zero are in bold. SCARED and C3VD use $n=7$ sequences; our copy of Hamlyn holds a single
rectified sequence, so $n=58$ contiguous blocks of 100 frames are the unit there, and its
intervals are slightly optimistic because blocks of one video remain correlated.}
\label{tab:paired}
\small
\begin{tabular}{llcccc}
\toprule
Dataset & Method & Abs Rel [95\% CI] & $\Delta$ vs MonoIIF [95\% CI] & $k/n$ & exact $p$ \\
\midrule
\multirow{6}{*}{SCARED}
 & HADepth       & \textbf{0.0489} $[0.0399, 0.0570]$ & $-0.0023$ $[-0.0068, +0.0016]$ & 3/7 & 0.578 \\
 & MonoPCC       & \underline{0.0505} $[0.0399, 0.0592]$ & $-0.0007$ $[-0.0073, +0.0053]$ & 4/7 & 0.938 \\
 & EndoDAC       & 0.0507 $[0.0399, 0.0607]$ & $-0.0004$ $[-0.0048, +0.0041]$ & 3/7 & 0.688 \\
 & MonoIIF (ours) & 0.0512 $[0.0407, 0.0616]$ & --- & --- & --- \\
 & AF-SfMLearner & 0.0584 $[0.0442, 0.0727]$ & $\mathbf{+0.0072}$ $[+0.0011, +0.0137]$ & 5/7 & 0.109 \\
 & Monodepth2    & 0.0933 $[0.0751, 0.1104]$ & $\mathbf{+0.0421}$ $[+0.0266, +0.0587]$ & 7/7 & 0.016 \\
\midrule
\multirow{6}{*}{Hamlyn}
 & EndoDAC       & \textbf{0.1608} $[0.1426, 0.1807]$ & $-0.0000$ $[-0.0037, +0.0036]$ & 31/58 & --- \\
 & MonoIIF (ours) & \underline{0.1608} $[0.1413, 0.1826]$ & --- & --- & --- \\
 & HADepth       & 0.1627 $[0.1425, 0.1846]$ & $+0.0019$ $[-0.0002, +0.0040]$ & 38/58 & --- \\
 & MonoPCC       & 0.1721 $[0.1499, 0.1957]$ & $\mathbf{+0.0113}$ $[+0.0067, +0.0158]$ & 41/58 & --- \\
 & AF-SfMLearner & 0.1828 $[0.1583, 0.2089]$ & $\mathbf{+0.0220}$ $[+0.0151, +0.0296]$ & 47/58 & --- \\
 & Monodepth2    & 0.2231 $[0.2034, 0.2438]$ & $\mathbf{+0.0624}$ $[+0.0498, +0.0745]$ & 55/58 & --- \\
\midrule
\multirow{6}{*}{C3VD}
 & HADepth       & \textbf{0.2213} $[0.2017, 0.2425]$ & $\mathbf{-0.0466}$ $[-0.0683, -0.0292]$ & 0/7 & 0.016 \\
 & EndoDAC       & \underline{0.2568} $[0.2216, 0.2926]$ & $-0.0111$ $[-0.0405, +0.0141]$ & 3/7 & 0.813 \\
 & MonoIIF (ours) & 0.2679 $[0.2447, 0.2893]$ & --- & --- & --- \\
 & AF-SfMLearner & 0.3356 $[0.2811, 0.3847]$ & $\mathbf{+0.0678}$ $[+0.0207, +0.1126]$ & 5/7 & 0.078 \\
 & Monodepth2    & 0.3419 $[0.2942, 0.3778]$ & $\mathbf{+0.0740}$ $[+0.0410, +0.1016]$ & 6/7 & 0.031 \\
 & MonoPCC       & 0.3568 $[0.3177, 0.3865]$ & $\mathbf{+0.0889}$ $[+0.0621, +0.1116]$ & 7/7 & 0.016 \\
\bottomrule
\end{tabular}
\end{table}
```

Needs `\usepackage{multirow}`.

**Text to accompany it.** *"On SCARED the four leading methods span 0.0489 to 0.0512 in Abs Rel
and every pairwise interval among them covers zero, while MonoIIF's own spread across training
seeds is 0.0503 to 0.0528. The separation between the leading methods is therefore smaller than
the variation between training runs of a single one of them, and we report these as ties rather
than ranking them. The ordering is also dataset-dependent: on C3VD HADepth is better in all seven
scenes with the interval excluding zero, while on SCARED and Hamlyn the two are indistinguishable."*



### 5.4b Table 7 with the best own model as the reference (`C1`)

Same table as §5.4 with `C1` --- MonoIIF-G --- in place of the submitted MonoIIF. Generated by `table7_ref.py`; rerun it for any other reference run.

```latex
\begin{table}[t]
\centering
\caption{Uncertainty and paired comparisons for the methods of Tables~4--6, with the video sequence as the unit of analysis. Column~3 gives each method's Abs Rel with a 95\% cluster bootstrap confidence interval over sequences (10\,000 draws). Column~4 gives the paired difference (method $-$ MonoIIF-G) per sequence, signed so that a positive value means MonoIIF-G is better, with its own bootstrap interval; $k/n$ counts the sequences in which MonoIIF-G is better. The exact two-sided Wilcoxon signed-rank $p$ is given for the two datasets with $n=7$; its smallest attainable value there is $2/2^{7}=0.016$, so no family-wise correction across a table of this size can reach significance, and these comparisons are exploratory. In column~3 the best value per dataset is in bold and the second best underlined; in column~4 paired differences whose interval excludes zero are in bold. SCARED and C3VD use $n=7$ sequences; our copy of Hamlyn holds a single rectified sequence, so $n=58$ contiguous blocks of 100 frames are the unit there, and its intervals are slightly optimistic because blocks of one video remain correlated. MonoIIF-G is the mean of three training seeds; the comparison methods are the authors' released checkpoints.}
\label{tab:paired-c1}
\small
\begin{tabular}{llcccc}
\toprule
Dataset & Method & Abs Rel [95\% CI] & $\Delta$ vs MonoIIF-G [95\% CI] & $k/n$ & exact $p$ \\
\midrule
\multirow{6}{*}{SCARED}
 & HADepth & \textbf{0.0489} $[0.0398, 0.0569]$ & $-0.0008$ $[-0.0053, +0.0026]$ & 4/7 & 0.938 \\
 & MonoIIF-G (ours) & \underline{0.0497} $[0.0400, 0.0594]$ & --- & --- & --- \\
 & MonoPCC & 0.0505 $[0.0396, 0.0591]$ & $+0.0008$ $[-0.0057, +0.0063]$ & 4/7 & 0.688 \\
 & EndoDAC & 0.0507 $[0.0401, 0.0607]$ & $+0.0011$ $[-0.0036, +0.0058]$ & 4/7 & 0.688 \\
 & AF-SfMLearner & 0.0584 $[0.0446, 0.0724]$ & $\mathbf{+0.0087}$ $[+0.0018, +0.0160]$ & 6/7 & 0.078 \\
 & Monodepth2 & 0.0933 $[0.0750, 0.1104]$ & $\mathbf{+0.0436}$ $[+0.0287, +0.0598]$ & 7/7 & 0.016 \\
\midrule
\multirow{6}{*}{Hamlyn}
 & MonoIIF-G (ours) & \textbf{0.1595} $[0.1401, 0.1809]$ & --- & --- & --- \\
 & EndoDAC & \underline{0.1608} $[0.1424, 0.1807]$ & $+0.0012$ $[-0.0020, +0.0044]$ & 28/58 & --- \\
 & HADepth & 0.1627 $[0.1424, 0.1849]$ & $\mathbf{+0.0032}$ $[+0.0005, +0.0059]$ & 36/58 & --- \\
 & MonoPCC & 0.1721 $[0.1499, 0.1968]$ & $\mathbf{+0.0125}$ $[+0.0071, +0.0183]$ & 40/58 & --- \\
 & AF-SfMLearner & 0.1828 $[0.1581, 0.2096]$ & $\mathbf{+0.0233}$ $[+0.0154, +0.0318]$ & 45/58 & --- \\
 & Monodepth2 & 0.2231 $[0.2041, 0.2442]$ & $\mathbf{+0.0636}$ $[+0.0508, +0.0759]$ & 55/58 & --- \\
\midrule
\multirow{6}{*}{C3VD}
 & HADepth & \textbf{0.2213} $[0.2026, 0.2420]$ & $\mathbf{-0.0504}$ $[-0.0629, -0.0388]$ & 0/7 & 0.016 \\
 & EndoDAC & \underline{0.2568} $[0.2204, 0.2925]$ & $-0.0149$ $[-0.0468, +0.0123]$ & 4/7 & 0.812 \\
 & MonoIIF-G (ours) & 0.2717 $[0.2485, 0.2953]$ & --- & --- & --- \\
 & AF-SfMLearner & 0.3356 $[0.2805, 0.3836]$ & $\mathbf{+0.0639}$ $[+0.0136, +0.1109]$ & 5/7 & 0.078 \\
 & Monodepth2 & 0.3419 $[0.2947, 0.3786]$ & $\mathbf{+0.0702}$ $[+0.0289, +0.1043]$ & 6/7 & 0.047 \\
 & MonoPCC & 0.3568 $[0.3184, 0.3863]$ & $\mathbf{+0.0851}$ $[+0.0514, +0.1148]$ & 7/7 & 0.016 \\
\bottomrule
\end{tabular}
\end{table}
```

### 5.5 The same comparison, one column per test sequence

Table 7 summarises; these show *where* each comparison is decided. They are the per-sequence values that Table 7's intervals and paired tests are computed from, so they carry the uncertainty estimate (the spread across sequences and the mean's CI) and the paired comparison (the difference block) in the raw. SCARED and C3VD have seven units each and go one column per unit; Hamlyn's unit is a 100-frame block of its single sequence, 58 of them, so that table is transposed --- one row per block --- and is a full-page float for the supplement. Needs `\usepackage{multirow}` and the `table*` environment.

```latex
\begin{table*}[t]
\centering
\caption{SCARED, per keyframe video. Upper block: Abs Rel of every method on each of the $n=7$ test keyframe videos, then the mean over keyframe videos with its 95\% cluster bootstrap CI (10\,000 draws) --- the uncertainty estimate. Lower block: the paired difference (method $-$ MonoIIF) on the same keyframe video, signed so that \textbf{positive means MonoIIF is better}, then its mean with bootstrap CI, the count of keyframe videos in which MonoIIF is better, and the exact two-sided Wilcoxon $p$ --- the paired comparison. d$i$k$j$ is keyframe $j$ of dataset $i$ of the test split. Best per column in bold and second best underlined in the upper block, also on the mean; paired means whose interval excludes zero in bold in the lower block. MonoIIF is the mean of three training seeds; the comparison methods are released checkpoints.}
\label{tab:perseq-scared}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\toprule
Method & d1k3 & d2k4 & d3k4 & d4k4 & d5k4 & d6k4 & d7k4 & mean [95\% CI] \\
\midrule
\multicolumn{9}{l}{\emph{Abs Rel per keyframe video}} \\
HADepth & 0.0373 & 0.0277 & \underline{0.0534} & 0.0524 & \underline{0.0659} & 0.0500 & \textbf{0.0557} & \textbf{0.0489} $[0.0399, 0.0570]$ \\
MonoPCC & 0.0405 & \textbf{0.0231} & \textbf{0.0517} & 0.0564 & \textbf{0.0640} & 0.0551 & 0.0625 & \underline{0.0505} $[0.0399, 0.0591]$ \\
EndoDAC & \textbf{0.0351} & \underline{0.0266} & 0.0561 & 0.0595 & 0.0715 & \underline{0.0483} & 0.0580 & 0.0507 $[0.0399, 0.0610]$ \\
MonoIIF (ours) & \underline{0.0369} & 0.0295 & 0.0670 & \textbf{0.0490} & 0.0722 & \textbf{0.0460} & \underline{0.0576} & 0.0512 $[0.0407, 0.0616]$ \\
AF-SfMLearner & 0.0414 & 0.0290 & 0.0630 & \underline{0.0497} & 0.0914 & 0.0610 & 0.0732 & 0.0584 $[0.0445, 0.0726]$ \\
Monodepth2 & 0.1195 & 0.0584 & 0.1237 & 0.0603 & 0.0931 & 0.0938 & 0.1040 & 0.0933 $[0.0750, 0.1110]$ \\
\midrule
\multicolumn{9}{l}{\emph{paired difference vs MonoIIF, positive $=$ MonoIIF better}} \\
Method & d1k3 & d2k4 & d3k4 & d4k4 & d5k4 & d6k4 & d7k4 & mean [95\% CI], $k/n$, $p$ \\
\midrule
HADepth & $+0.0005$ & $-0.0018$ & $-0.0136$ & $+0.0035$ & $-0.0064$ & $+0.0040$ & $-0.0019$ & $-0.0023$ $[-0.0068, +0.0016]$, 3/7, 0.578 \\
MonoPCC & $+0.0036$ & $-0.0064$ & $-0.0152$ & $+0.0074$ & $-0.0082$ & $+0.0091$ & $+0.0049$ & $-0.0007$ $[-0.0071, +0.0052]$, 4/7, 0.938 \\
EndoDAC & $-0.0018$ & $-0.0029$ & $-0.0108$ & $+0.0106$ & $-0.0007$ & $+0.0023$ & $+0.0004$ & $-0.0004$ $[-0.0048, +0.0041]$, 3/7, 0.688 \\
AF-SfMLearner & $+0.0046$ & $-0.0005$ & $-0.0040$ & $+0.0008$ & $+0.0192$ & $+0.0150$ & $+0.0156$ & $\mathbf{+0.0072}$ $[+0.0011, +0.0137]$, 5/7, 0.109 \\
Monodepth2 & $+0.0826$ & $+0.0289$ & $+0.0568$ & $+0.0113$ & $+0.0209$ & $+0.0479$ & $+0.0464$ & $\mathbf{+0.0421}$ $[+0.0266, +0.0587]$, 7/7, 0.016 \\
\bottomrule
\end{tabular}
\end{table*}
```

```latex
\begin{table*}[t]
\centering
\caption{C3VD, per scene. Upper block: Abs Rel of every method on each of the $n=7$ test scenes, then the mean over scenes with its 95\% cluster bootstrap CI (10\,000 draws) --- the uncertainty estimate. Lower block: the paired difference (method $-$ MonoIIF) on the same scene, signed so that \textbf{positive means MonoIIF is better}, then its mean with bootstrap CI, the count of scenes in which MonoIIF is better, and the exact two-sided Wilcoxon $p$ --- the paired comparison. t$i$\_$x$ is the C3VD transverse-colon scene of that name. Best per column in bold and second best underlined in the upper block, also on the mean; paired means whose interval excludes zero in bold in the lower block. MonoIIF is the mean of three training seeds; the comparison methods are released checkpoints.}
\label{tab:perseq-c3vd}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\toprule
Method & t1_a & t1_b & t2_a & t2_b & t2_c & t3_a & t3_b & mean [95\% CI] \\
\midrule
\multicolumn{9}{l}{\emph{Abs Rel per scene}} \\
HADepth & \textbf{0.2400} & \underline{0.2661} & \textbf{0.1882} & \textbf{0.1958} & 0.2107 & \textbf{0.2017} & \textbf{0.2465} & \textbf{0.2213} $[0.2024, 0.2419]$ \\
EndoDAC & 0.3195 & \textbf{0.2054} & \underline{0.2813} & 0.2515 & \textbf{0.1887} & \underline{0.2291} & 0.3220 & \underline{0.2568} $[0.2213, 0.2938]$ \\
MonoIIF (ours) & \underline{0.2922} & 0.2914 & 0.2924 & \underline{0.2308} & 0.2246 & 0.2409 & 0.3027 & 0.2679 $[0.2445, 0.2893]$ \\
AF-SfMLearner & 0.3843 & 0.4355 & 0.3402 & 0.3142 & \underline{0.2009} & 0.3792 & \underline{0.2951} & 0.3356 $[0.2815, 0.3854]$ \\
Monodepth2 & 0.4051 & 0.3716 & 0.3828 & 0.3456 & 0.2133 & 0.3345 & 0.3402 & 0.3419 $[0.2947, 0.3778]$ \\
MonoPCC & 0.4117 & 0.3748 & 0.3923 & 0.3482 & 0.2527 & 0.3625 & 0.3553 & 0.3568 $[0.3177, 0.3865]$ \\
\midrule
\multicolumn{9}{l}{\emph{paired difference vs MonoIIF, positive $=$ MonoIIF better}} \\
Method & t1_a & t1_b & t2_a & t2_b & t2_c & t3_a & t3_b & mean [95\% CI], $k/n$, $p$ \\
\midrule
HADepth & $-0.0522$ & $-0.0252$ & $-0.1042$ & $-0.0350$ & $-0.0139$ & $-0.0392$ & $-0.0562$ & $\mathbf{-0.0466}$ $[-0.0683, -0.0292]$, 0/7, 0.016 \\
EndoDAC & $+0.0273$ & $-0.0860$ & $-0.0112$ & $+0.0206$ & $-0.0359$ & $-0.0118$ & $+0.0193$ & $-0.0111$ $[-0.0412, +0.0133]$, 3/7, 0.812 \\
AF-SfMLearner & $+0.0921$ & $+0.1441$ & $+0.0478$ & $+0.0833$ & $-0.0237$ & $+0.1383$ & $-0.0076$ & $\mathbf{+0.0678}$ $[+0.0217, +0.1117]$, 5/7, 0.078 \\
Monodepth2 & $+0.1129$ & $+0.0802$ & $+0.0904$ & $+0.1147$ & $-0.0113$ & $+0.0936$ & $+0.0375$ & $\mathbf{+0.0740}$ $[+0.0407, +0.1019]$, 6/7, 0.031 \\
MonoPCC & $+0.1195$ & $+0.0835$ & $+0.0999$ & $+0.1174$ & $+0.0281$ & $+0.1216$ & $+0.0526$ & $\mathbf{+0.0889}$ $[+0.0631, +0.1119]$, 7/7, 0.016 \\
\bottomrule
\end{tabular}
\end{table*}
```

```latex
\begin{table*}[p]
\centering
\caption{Hamlyn, per block. Our copy of Hamlyn holds a single rectified sequence (\texttt{rectified14}), so the unit of analysis is a contiguous block of 100 frames, $n=58$, identical for every method. Left: Abs Rel of each method on each block, best per row in bold and second best underlined, with the mean over blocks and its 95\% cluster bootstrap CI at the foot --- the uncertainty estimate. Right: the paired difference (method $-$ MonoIIF) on the same block, signed so that \textbf{positive means MonoIIF is better}, with the mean, its CI and the count of blocks in which MonoIIF is better at the foot --- the paired comparison. Blocks of one video remain correlated, so the intervals are slightly optimistic. MonoIIF is the mean of three training seeds; the comparison methods are released checkpoints.}
\label{tab:perseq-hamlyn}
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccc|ccccc}
\toprule
 & \multicolumn{6}{c|}{Abs Rel} & \multicolumn{5}{c}{$\Delta$ vs MonoIIF} \\
block & EndoDAC & MonoIIF (ours) & HADepth & MonoPCC & AF-SfMLearner & Monodepth2 & EndoDAC & HADepth & MonoPCC & AF-SfMLearner & Monodepth2 \\
\midrule
b00 & \underline{0.3719} & \textbf{0.3680} & 0.3825 & 0.4062 & 0.4215 & 0.3811 & $+0.0039$ & $+0.0145$ & $+0.0382$ & $+0.0535$ & $+0.0130$ \\
b01 & 0.2092 & 0.2090 & 0.2092 & \underline{0.2073} & 0.2137 & \textbf{0.2038} & $+0.0002$ & $+0.0002$ & $-0.0017$ & $+0.0048$ & $-0.0051$ \\
b02 & \textbf{0.2044} & \underline{0.2067} & 0.2212 & 0.2384 & 0.2693 & 0.2495 & $-0.0023$ & $+0.0145$ & $+0.0317$ & $+0.0626$ & $+0.0428$ \\
b03 & \textbf{0.2337} & \underline{0.2397} & 0.2622 & 0.2805 & 0.3366 & 0.2951 & $-0.0060$ & $+0.0225$ & $+0.0408$ & $+0.0969$ & $+0.0554$ \\
b04 & 0.1057 & 0.1088 & 0.1113 & \underline{0.1035} & \textbf{0.1007} & 0.1621 & $-0.0030$ & $+0.0025$ & $-0.0053$ & $-0.0081$ & $+0.0533$ \\
b05 & \textbf{0.1287} & 0.1419 & \underline{0.1354} & 0.1599 & 0.1637 & 0.2459 & $-0.0132$ & $-0.0066$ & $+0.0180$ & $+0.0217$ & $+0.1039$ \\
b06 & \textbf{0.1948} & \underline{0.2163} & 0.2211 & 0.2624 & 0.2692 & 0.3343 & $-0.0215$ & $+0.0048$ & $+0.0461$ & $+0.0529$ & $+0.1180$ \\
b07 & \textbf{0.2177} & \underline{0.2413} & 0.2515 & 0.2740 & 0.3278 & 0.3067 & $-0.0235$ & $+0.0103$ & $+0.0327$ & $+0.0865$ & $+0.0654$ \\
b08 & \textbf{0.1765} & 0.1856 & \underline{0.1850} & 0.1945 & 0.2228 & 0.2234 & $-0.0091$ & $-0.0006$ & $+0.0089$ & $+0.0372$ & $+0.0378$ \\
b09 & 0.1200 & \underline{0.1137} & 0.1236 & \textbf{0.1022} & 0.1211 & 0.1376 & $+0.0063$ & $+0.0099$ & $-0.0115$ & $+0.0074$ & $+0.0239$ \\
b10 & \underline{0.2232} & 0.2253 & 0.2290 & 0.2272 & \textbf{0.2165} & 0.2421 & $-0.0021$ & $+0.0037$ & $+0.0019$ & $-0.0088$ & $+0.0168$ \\
b11 & 0.2381 & 0.2381 & 0.2427 & \textbf{0.2363} & \underline{0.2378} & 0.2588 & $-0.0000$ & $+0.0046$ & $-0.0018$ & $-0.0003$ & $+0.0207$ \\
b12 & 0.2282 & \textbf{0.2200} & \underline{0.2212} & 0.2248 & 0.2334 & 0.2416 & $+0.0082$ & $+0.0012$ & $+0.0049$ & $+0.0134$ & $+0.0217$ \\
b13 & 0.2145 & \textbf{0.2025} & \underline{0.2044} & 0.2052 & 0.2157 & 0.2207 & $+0.0120$ & $+0.0020$ & $+0.0027$ & $+0.0133$ & $+0.0182$ \\
b14 & 0.1957 & \textbf{0.1893} & 0.1968 & \underline{0.1911} & 0.1984 & 0.2026 & $+0.0065$ & $+0.0076$ & $+0.0018$ & $+0.0091$ & $+0.0134$ \\
b15 & 0.1824 & \underline{0.1787} & 0.1888 & \textbf{0.1774} & 0.1874 & 0.1825 & $+0.0038$ & $+0.0101$ & $-0.0013$ & $+0.0088$ & $+0.0038$ \\
b16 & \underline{0.1789} & 0.1839 & 0.1869 & 0.1831 & \textbf{0.1772} & 0.1934 & $-0.0051$ & $+0.0030$ & $-0.0008$ & $-0.0067$ & $+0.0095$ \\
b17 & \textbf{0.2896} & \underline{0.3129} & 0.3144 & 0.3325 & 0.3407 & 0.3294 & $-0.0233$ & $+0.0015$ & $+0.0196$ & $+0.0278$ & $+0.0165$ \\
b18 & \textbf{0.3904} & \underline{0.4210} & 0.4301 & 0.4534 & 0.4961 & 0.4520 & $-0.0307$ & $+0.0091$ & $+0.0324$ & $+0.0751$ & $+0.0310$ \\
b19 & \textbf{0.3561} & \underline{0.3887} & 0.3989 & 0.4331 & 0.4744 & 0.4190 & $-0.0326$ & $+0.0102$ & $+0.0444$ & $+0.0857$ & $+0.0302$ \\
b20 & \textbf{0.2547} & \underline{0.2679} & 0.2765 & 0.3132 & 0.3417 & 0.3112 & $-0.0131$ & $+0.0087$ & $+0.0453$ & $+0.0739$ & $+0.0433$ \\
b21 & \textbf{0.2601} & \underline{0.2805} & 0.2819 & 0.3332 & 0.3830 & 0.3744 & $-0.0204$ & $+0.0014$ & $+0.0527$ & $+0.1025$ & $+0.0939$ \\
b22 & 0.1264 & \underline{0.1178} & 0.1295 & 0.1218 & 0.1399 & \textbf{0.0892} & $+0.0085$ & $+0.0117$ & $+0.0040$ & $+0.0221$ & $-0.0286$ \\
b23 & \underline{0.1705} & 0.1826 & \textbf{0.1693} & 0.2086 & 0.1998 & 0.3593 & $-0.0121$ & $-0.0133$ & $+0.0260$ & $+0.0171$ & $+0.1767$ \\
b24 & \underline{0.0677} & \textbf{0.0674} & 0.0918 & 0.0842 & 0.1105 & 0.0756 & $+0.0003$ & $+0.0244$ & $+0.0168$ & $+0.0431$ & $+0.0082$ \\
b25 & \textbf{0.0907} & \underline{0.0987} & 0.1025 & 0.1079 & 0.1342 & 0.1157 & $-0.0080$ & $+0.0038$ & $+0.0092$ & $+0.0355$ & $+0.0170$ \\
b26 & \textbf{0.1935} & 0.2189 & \underline{0.2082} & 0.2599 & 0.2799 & 0.3372 & $-0.0254$ & $-0.0107$ & $+0.0411$ & $+0.0611$ & $+0.1183$ \\
b27 & \textbf{0.0940} & 0.1068 & \underline{0.1028} & 0.1240 & 0.1198 & 0.2226 & $-0.0128$ & $-0.0040$ & $+0.0172$ & $+0.0130$ & $+0.1158$ \\
b28 & \textbf{0.0787} & \underline{0.0903} & 0.0913 & 0.1122 & 0.1084 & 0.2018 & $-0.0116$ & $+0.0010$ & $+0.0219$ & $+0.0181$ & $+0.1115$ \\
b29 & \textbf{0.0721} & \underline{0.0821} & 0.0829 & 0.1031 & 0.0989 & 0.1936 & $-0.0100$ & $+0.0008$ & $+0.0211$ & $+0.0169$ & $+0.1115$ \\
b30 & \underline{0.0951} & \textbf{0.0950} & 0.0961 & 0.1046 & 0.0971 & 0.1972 & $+0.0002$ & $+0.0011$ & $+0.0096$ & $+0.0021$ & $+0.1022$ \\
b31 & 0.1109 & 0.1009 & \textbf{0.0880} & \underline{0.0929} & 0.1038 & 0.1788 & $+0.0100$ & $-0.0129$ & $-0.0080$ & $+0.0029$ & $+0.0779$ \\
b32 & 0.1339 & 0.1069 & \underline{0.1039} & \textbf{0.0977} & 0.1226 & 0.1834 & $+0.0270$ & $-0.0030$ & $-0.0091$ & $+0.0157$ & $+0.0765$ \\
b33 & 0.1532 & 0.1237 & \textbf{0.1153} & \underline{0.1157} & 0.1409 & 0.2152 & $+0.0295$ & $-0.0084$ & $-0.0081$ & $+0.0172$ & $+0.0915$ \\
b34 & 0.1792 & 0.1442 & \textbf{0.1321} & \underline{0.1353} & 0.1456 & 0.2312 & $+0.0349$ & $-0.0122$ & $-0.0090$ & $+0.0013$ & $+0.0870$ \\
b35 & 0.1389 & \underline{0.1189} & \textbf{0.1178} & 0.1372 & 0.1447 & 0.2218 & $+0.0199$ & $-0.0011$ & $+0.0182$ & $+0.0257$ & $+0.1028$ \\
b36 & 0.1255 & \textbf{0.1000} & \underline{0.1082} & 0.1152 & 0.1250 & 0.1832 & $+0.0255$ & $+0.0082$ & $+0.0152$ & $+0.0250$ & $+0.0832$ \\
b37 & 0.0943 & \textbf{0.0766} & \underline{0.0834} & 0.0837 & 0.0983 & 0.1515 & $+0.0177$ & $+0.0068$ & $+0.0071$ & $+0.0217$ & $+0.0749$ \\
b38 & 0.0836 & \underline{0.0785} & 0.0801 & \textbf{0.0705} & 0.0797 & 0.1361 & $+0.0050$ & $+0.0016$ & $-0.0080$ & $+0.0012$ & $+0.0576$ \\
b39 & 0.0777 & 0.0734 & 0.0722 & \textbf{0.0642} & \underline{0.0697} & 0.1268 & $+0.0043$ & $-0.0012$ & $-0.0092$ & $-0.0037$ & $+0.0534$ \\
b40 & 0.1011 & \textbf{0.0900} & \underline{0.1008} & 0.1090 & 0.1223 & 0.1789 & $+0.0111$ & $+0.0108$ & $+0.0190$ & $+0.0323$ & $+0.0889$ \\
b41 & 0.1415 & \textbf{0.1293} & \underline{0.1308} & 0.1358 & 0.1558 & 0.2457 & $+0.0122$ & $+0.0014$ & $+0.0064$ & $+0.0265$ & $+0.1164$ \\
b42 & 0.1377 & 0.1226 & \underline{0.1213} & \textbf{0.1208} & 0.1412 & 0.2513 & $+0.0150$ & $-0.0014$ & $-0.0019$ & $+0.0186$ & $+0.1287$ \\
b43 & 0.1215 & \underline{0.1026} & 0.1037 & \textbf{0.0957} & 0.1087 & 0.1950 & $+0.0189$ & $+0.0010$ & $-0.0070$ & $+0.0061$ & $+0.0923$ \\
b44 & 0.1396 & \textbf{0.1351} & \underline{0.1382} & 0.1480 & 0.1428 & 0.2252 & $+0.0045$ & $+0.0032$ & $+0.0129$ & $+0.0077$ & $+0.0901$ \\
b45 & 0.1780 & 0.1678 & 0.1675 & \textbf{0.1656} & \underline{0.1665} & 0.2036 & $+0.0103$ & $-0.0003$ & $-0.0022$ & $-0.0012$ & $+0.0359$ \\
b46 & 0.0710 & \underline{0.0709} & \textbf{0.0655} & 0.0980 & 0.0979 & 0.1541 & $+0.0001$ & $-0.0054$ & $+0.0272$ & $+0.0270$ & $+0.0832$ \\
b47 & \underline{0.0778} & 0.0780 & \textbf{0.0743} & 0.0998 & 0.0957 & 0.1541 & $-0.0001$ & $-0.0037$ & $+0.0218$ & $+0.0177$ & $+0.0761$ \\
b48 & 0.0769 & \underline{0.0768} & \textbf{0.0731} & 0.0938 & 0.0769 & 0.1526 & $+0.0001$ & $-0.0037$ & $+0.0170$ & $+0.0001$ & $+0.0758$ \\
b49 & 0.1015 & \textbf{0.0988} & 0.1023 & 0.1137 & \underline{0.0988} & 0.1633 & $+0.0028$ & $+0.0035$ & $+0.0149$ & $+0.0001$ & $+0.0645$ \\
b50 & 0.1100 & \underline{0.1097} & 0.1134 & 0.1205 & \textbf{0.1065} & 0.1580 & $+0.0003$ & $+0.0037$ & $+0.0108$ & $-0.0032$ & $+0.0483$ \\
b51 & \underline{0.1551} & \textbf{0.1524} & 0.1560 & 0.1639 & 0.1612 & 0.1970 & $+0.0027$ & $+0.0036$ & $+0.0116$ & $+0.0088$ & $+0.0446$ \\
b52 & 0.2008 & 0.1951 & 0.1934 & \textbf{0.1749} & \underline{0.1831} & 0.2247 & $+0.0057$ & $-0.0017$ & $-0.0202$ & $-0.0120$ & $+0.0297$ \\
b53 & 0.1090 & 0.1106 & \underline{0.0978} & 0.1155 & \textbf{0.0905} & 0.1793 & $-0.0016$ & $-0.0128$ & $+0.0049$ & $-0.0201$ & $+0.0687$ \\
b54 & \underline{0.1144} & 0.1192 & 0.1145 & 0.1221 & \textbf{0.1135} & 0.2309 & $-0.0047$ & $-0.0047$ & $+0.0029$ & $-0.0057$ & $+0.1118$ \\
b55 & \textbf{0.1124} & \underline{0.1138} & 0.1197 & 0.1191 & 0.1316 & 0.2641 & $-0.0015$ & $+0.0059$ & $+0.0052$ & $+0.0178$ & $+0.1503$ \\
b56 & \textbf{0.1154} & \underline{0.1160} & 0.1195 & 0.1207 & 0.1284 & 0.2510 & $-0.0006$ & $+0.0035$ & $+0.0047$ & $+0.0125$ & $+0.1351$ \\
b57 & 0.1999 & 0.2150 & 0.1967 & \underline{0.1846} & 0.2127 & \textbf{0.1261} & $-0.0151$ & $-0.0183$ & $-0.0304$ & $-0.0023$ & $-0.0888$ \\
\midrule
mean & \textbf{0.1608} & \underline{0.1608} & 0.1627 & 0.1721 & 0.1828 & 0.2231 & $-0.0000$ & $+0.0019$ & $\mathbf{+0.0113}$ & $\mathbf{+0.0220}$ & $\mathbf{+0.0624}$ \\
95\% CI / $k/n$ & $[0.1426, 0.1801]$ & $[0.1409, 0.1827]$ & $[0.1423, 0.1856]$ & $[0.1493, 0.1962]$ & $[0.1581, 0.2095]$ & $[0.2032, 0.2436]$ & 31/58 & 38/58 & 41/58 & 47/58 & 55/58 \\
\bottomrule
\end{tabular}
\end{table*}
```

**Text.** *"The per-sequence view shows that the aggregate gap between the leading methods on SCARED is not a consistent advantage: the sign of the difference between HADepth and MonoIIF changes from sequence to sequence, MonoIIF is better on three of the seven, and a single sequence (d3k4) contributes most of the aggregate. On C3VD, by contrast, HADepth is better on every scene, which is why that comparison resolves and the SCARED one does not. On Hamlyn the picture is the SCARED one again at larger $n$: EndoDAC and MonoIIF split the 58 blocks almost evenly and HADepth's edge is within the interval, while the three older methods lose on most blocks."*

---

## 6. Section 4.x — the calibration ablation (Reviewer 2)

New tables, replacing the ablation that compared the module only against nothing. The calibration
is isolated three ways: on the method's own backbone with **nothing else switched on** (§6.1), on
a second backbone within the MonoII loss configuration (§6.2), and inside the method's full loss
configuration (§6.3).

### 6.1 The calibration alone, on the method's backbone

This is the cleanest form of the comparison the reviewer asks for: the invariant loss is switched
off ($\lambda_1 = 0$) and the photometric loss is monodepth2's, so the **only** thing that differs
across rows is the calibration model.

```latex
\begin{table}[t]
\centering
\caption{Photometric calibration models on the Depth Anything backbone with the
illumination-invariant loss switched off ($\lambda_1=0$) and monodepth2's photometric loss, so
that the rows differ only in the calibration. Three seeds. The lower block is the paired
difference (variant $-$ no calibration) in Abs Rel per sequence, so a \textbf{negative} value
means the calibration helps; bold marks intervals excluding zero.}
\label{tab:calib-lam0}
\small
\begin{tabular}{lccc}
\toprule
Calibration model & SCARED & Hamlyn & C3VD \\
\midrule
no calibration                  & 0.0517\,$\pm$\,0.0009 & \textbf{0.1565\,$\pm$\,0.0043} & 0.2929\,$\pm$\,0.0136 \\
global, one $(c,b)$ pair        & 0.0505\,$\pm$\,0.0020 & 0.1597\,$\pm$\,0.0030 & \textbf{0.2672\,$\pm$\,0.0170} \\
linear field (degree 1)         & \textbf{0.0502} & 0.1597 & 0.2956 \\
quadratic field (degree 2)      & 0.0510 & 0.1586 & 0.2870 \\
dense map (original submission) & 0.0517\,$\pm$\,0.0193 & 0.1594 & \textbf{0.2669} \\
\midrule
\multicolumn{4}{l}{\emph{paired against no calibration}} \\
global     & $-0.0012$ $[-0.0028, +0.0002]$ & $\mathbf{+0.0032}$ $[+0.0012, +0.0052]$ &
             $\mathbf{-0.0256}$ $[-0.0296, -0.0222]$, \textbf{7/7} \\
degree 1   & $\mathbf{-0.0015}$ $[-0.0024, -0.0004]$, 6/7, $p=0.047$ &
             $\mathbf{+0.0033}$ $[+0.0020, +0.0046]$ & $+0.0028$ $[-0.0054, +0.0110]$ \\
degree 2   & $-0.0006$ $[-0.0025, +0.0009]$ & $\mathbf{+0.0021}$ $[+0.0009, +0.0033]$ &
             $-0.0059$ $[-0.0167, +0.0032]$ \\
dense map  & $+0.0000$ $[-0.0018, +0.0022]$ & $\mathbf{+0.0029}$ $[+0.0014, +0.0045]$ &
             $\mathbf{-0.0260}$ $[-0.0360, -0.0160]$, \textbf{7/7} \\
\bottomrule
\end{tabular}
\end{table}
```

**Text.** *"With the invariant loss switched off, the calibration is isolated completely. Three
things follow. In the training domain only the degree-1 field separates from no calibration
($-0.0015$ $[-0.0024, -0.0004]$, better in 6 of 7 sequences); the other forms are ties. Under
photometric shift with unchanged geometry (Hamlyn) every calibration model hurts, all four
intervals excluding zero, which we report as a limitation. Under geometric shift (C3VD) the two
\emph{ends} of the capacity axis are what work --- the global model and the dense map each beat no
calibration by about 0.026 in all seven scenes, while the intermediate fields do not separate ---
and the dense map also beats the degree-1 and degree-2 fields there ($+0.0288$ and $+0.0202$,
0 of 7 scenes each). Intensity fall-off carries depth information in a tubular organ, and a
calibration model either has to be coarse enough not to touch it or expressive enough to model it;
the middle of the axis does neither."*

The C3VD result is why the dense parameterisation is retained on this backbone despite §6.2: on
ResNet-18 the optimum is in the middle of the axis, on Depth Anything it is at the ends, and the
two backbones disagree with intervals excluding zero in both directions. §7.1 states this
explicitly, and the paper must not present a single recommended capacity.

**Missing cells.** The same $\lambda_1=0$ family on ResNet-18 exists only at two of the five
points (no calibration, and degree 2). Completing it needs
`--depth_backbone resnet18 --illum_calib global --illumination_invariant 0 --photometric standard`
and the same with `--illum_calib local` — six jobs at three seeds, a few hours each.

### 6.2 The same axis on a second backbone, ResNet-18

```latex
\begin{table}[t]
\centering
\caption{Photometric calibration models on the ResNet-18 backbone. The five calibration rows share
one recipe --- monodepth2's photometric loss, illumination-invariant loss at $\lambda_1=0.5$,
three seeds --- and differ \emph{only} in how the affine field $(c,b)$ is parameterised, from a
single pair per image to a free per-pixel map. The first row is a reference outside that family:
it has \emph{neither} of the method's components and is therefore the Monodepth2 architecture
under our recipe, which is why its number differs from the released Monodepth2 checkpoint of
Table~\ref{tab:scared} (0.0933); that checkpoint was trained by its authors, not here. The lower
block is the paired difference (variant $-$ degree 1) in Abs Rel per sequence, so a
\textbf{positive} value means the degree-1 field is better; bold marks intervals excluding zero.}
\label{tab:calib-resnet}
\small
\begin{tabular}{lccc}
\toprule
Calibration model & SCARED & Hamlyn & C3VD \\
\midrule
\emph{neither component} ($=$ Monodepth2, our recipe) & 0.0593\,$\pm$\,0.0015 & 0.1764\,$\pm$\,0.0070 & 0.3280\,$\pm$\,0.0117 \\
\midrule
no calibration, invariant loss kept & 0.0599\,$\pm$\,0.0016 & 0.1698\,$\pm$\,0.0041 & 0.3363\,$\pm$\,0.0116 \\
global, one $(c,b)$ pair      & 0.0600\,$\pm$\,0.0009 & 0.1758\,$\pm$\,0.0065 & 0.3290\,$\pm$\,0.0072 \\
\textbf{linear field (degree 1)} & \textbf{0.0596\,$\pm$\,0.0002} & 0.1700\,$\pm$\,0.0014 & \textbf{0.3278\,$\pm$\,0.0101} \\
quadratic field (degree 2)    & 0.0608\,$\pm$\,0.0008 & \textbf{0.1690\,$\pm$\,0.0010} & 0.3383\,$\pm$\,0.0040 \\
dense map (original submission) & 0.0608\,$\pm$\,0.0007 & 0.1757\,$\pm$\,0.0021 & 0.3379\,$\pm$\,0.0057 \\
\midrule
\multicolumn{4}{l}{\emph{paired against the degree-1 field}} \\
neither component  & $-0.0003$ $[-0.0041, +0.0038]$ & $\mathbf{+0.0064}$ $[+0.0034, +0.0097]$, 40/58 &
                     $+0.0002$ $[-0.0089, +0.0081]$ \\
no calibration     & $+0.0002$ $[-0.0037, +0.0047]$ & $-0.0001$ $[-0.0017, +0.0015]$ &
                     $\mathbf{+0.0085}$ $[+0.0009, +0.0153]$, 6/7 \\
global             & $+0.0004$ $[-0.0023, +0.0030]$ & $\mathbf{+0.0058}$ $[+0.0036, +0.0081]$, 43/58 &
                     $+0.0012$ $[-0.0066, +0.0101]$ \\
degree 2           & $+0.0012$ $[-0.0024, +0.0049]$ & $-0.0009$ $[-0.0027, +0.0007]$ &
                     $\mathbf{+0.0105}$ $[+0.0035, +0.0171]$, 6/7 \\
dense map          & $+0.0012$ $[-0.0011, +0.0037]$ & $\mathbf{+0.0058}$ $[+0.0037, +0.0078]$, 47/58 &
                     $\mathbf{+0.0101}$ $[+0.0016, +0.0167]$, 6/7 \\
\bottomrule
\end{tabular}
\end{table}
```

**Text.** *"The spatially varying affine model is validated, but at low capacity. A linear field
--- six coefficients against the two of the global model and the $2HW$ of a dense map --- is the
best of the five on two of the three datasets and is never beaten with an interval excluding zero.
It beats the global model of Ozyoruk et al.\ on Hamlyn ($+0.0058$ $[+0.0036, +0.0081]$, better in
43 of 58 blocks), the free per-pixel map of the original submission on Hamlyn ($+0.0058$, 47/58)
and on C3VD ($+0.0101$ $[+0.0016, +0.0167]$, 6 of 7 scenes), and not calibrating at all on C3VD
($+0.0085$, 6/7). In the training domain the five are indistinguishable, with all means inside
0.0012 of one another. The capacity of the field, not the presence of spatial variation, is what
decides whether the calibration helps: the two ends of the axis --- a single global pair and a
free per-pixel map --- are both worse than a linear gradient."*

This is the positive answer to Reviewer 2: five comparisons in favour of the low-order field with
intervals excluding zero, none against it, and the two forms the reviewer named are both on the
axis as special cases.

**Three rows that are easy to confuse, and must be distinguished in the caption.** *Neither
component* (0.0593) is the Monodepth2 architecture with no calibration and no invariant loss.
*No calibration* (0.0599) keeps the invariant loss at $\lambda_1=0.5$ and removes only the
lighting model — it is the control for the calibration, not a baseline. And the *Monodepth2* row
of Table~\ref{tab:scared} (0.0933) is the checkpoint released by its authors, trained elsewhere.
The gap between the first and the third is the training recipe, not the architecture; the gap
between the first and the second is the invariant loss alone, which is a tie in-domain
($+0.0006$ $[-0.0021, +0.0037]$) and costs 0.0066 on Hamlyn.

### 6.2 Confirmation on the method's own backbone

Same three-cell design on Depth Anything, where the method's loss configuration is used
($\lambda_1 = 0.1$ with the highlight-aware term). Within each table the contrast is clean; across
the two tables the loss configurations differ, so they corroborate rather than pool.

```latex
\begin{table}[t]
\centering
\caption{Photometric calibration models under an otherwise identical recipe. \textbf{All three rows
are the full method on the Depth Anything backbone and keep the illumination-invariant loss at
$\lambda_1=0.1$ and the highlight-aware photometric term}; only the calibration changes, which is
what makes the contrast attributable to it. The first row is therefore \emph{not} a plain
baseline: the model without any of the three components is EndoDAC, at 0.0517 (Table~\ref{tab:ladder}).
The paired difference is (variant $-$ local) in Abs Rel per sequence on SCARED, so a negative value
means the variant is better than the dense map of the original submission.}
\label{tab:calib-da}
\small
\begin{tabular}{lcccc}
\toprule
Calibration model & SCARED & Hamlyn & C3VD & paired vs local (SCARED) \\
\midrule
no lighting model        & 0.0518\,$\pm$\,0.0029 & 0.1593 & 0.2658 & $+0.0007$ $[-0.0010, +0.0028]$, 3/7 \\
global, one $(c,b)$ pair & \textbf{0.0497\,$\pm$\,0.0021} & \textbf{0.1595} & 0.2717 &
                           $\mathbf{-0.0015}$ $[-0.0032, -0.0005]$, \textbf{0/7}, $p=0.016$ \\
local, dense map (ours)  & 0.0512\,$\pm$\,0.0014 & 0.1608 & \textbf{0.2679} & --- \\
\bottomrule
\end{tabular}
\end{table}
```

**Text.** *"In the training domain the global two-parameter model is better than the dense map in
all seven sequences with the interval excluding zero, while the dense map is not distinguishable
from no calibration at all. Out of domain the two forms tie (Hamlyn $-0.0013$ $[-0.0030, +0.0004]$,
C3VD $+0.0038$ $[-0.0094, +0.0138]$). We therefore no longer claim that the spatial variation is
what produces the gain."*

**Note on what this table can and cannot contain.** The three cells above share the method's loss
configuration (invariant loss at $\lambda_1=0.1$ and the highlight-aware photometric term) and
differ only in the calibration, which is what makes the contrast clean. The polynomial-basis
family was trained with the calibration alone ($\lambda_1=0$, monodepth2's photometric loss), so
its rows belong in the capacity table of §7 rather than here; mixing the two would confound the
calibration with two loss terms.

---

## 7. Section 4.x — the capacity sweep (Reviewer 2)

This is the new experiment and the strongest part of the revision: it turns "none vs global vs
local" into one axis with the two forms as endpoints.

```latex
\begin{table}[t]
\centering
\caption{Spatial capacity of the affine calibration field, parameterised as a polynomial of
degree $d$: degree 0 is exactly the global model and the dense per-pixel map is the limit. Trained
with the calibration alone ($\lambda_1=0$) on the foundation backbone. Differences in the last
column are against no calibration on C3VD; $\dagger$ marks intervals excluding zero.}
\label{tab:capacity}
\small
\begin{tabular}{lcccl}
\toprule
Spatial capacity & SCARED & Hamlyn & C3VD & C3VD vs no calibration \\
\midrule
none              & 0.0517 & \textbf{0.1565} & 0.2929 & --- \\
degree 0 (global) & 0.0505 & 0.1597 & \textbf{0.2672} & $-0.0256$ $[-0.0297, -0.0222]$, 7/7$^\dagger$ \\
degree 1          & \textbf{0.0502} & 0.1597 & 0.2956 & $+0.0028$, tie \\
degree 2          & 0.0510 & 0.1586 & 0.2870 & $-0.0059$, tie \\
dense map         & 0.0517 & 0.1594 & 0.2669 & $-0.0260$ $[-0.0360, -0.0164]$, 7/7$^\dagger$ \\
\bottomrule
\end{tabular}
\end{table}
```

### 7.1 Does a low-order field beat the global model?

This is the comparison that decides whether spatial variation is worth anything at all, and the
answer differs by backbone. Each cell is the paired difference (basis $-$ global) in Abs Rel per
sequence, so a **negative** value means the polynomial field beats the global model.

```latex
\begin{table}[t]
\centering
\caption{Polynomial calibration field against the global model, on both backbones. The difference
is (basis $-$ global) per sequence, so negative favours the spatially varying field. Bold marks
intervals excluding zero. Both families are trained with the calibration alone, so the two rows of
a backbone differ only in the degree of the field.}
\label{tab:basis-vs-global}
\small
\begin{tabular}{llccc}
\toprule
Backbone & Field & SCARED ($n=7$) & Hamlyn ($n=58$) & C3VD ($n=7$) \\
\midrule
\multirow{2}{*}{Depth Anything}
 & degree 1 & $-0.0003$ $[-0.0009, +0.0004]$ & $+0.0000$ $[-0.0024, +0.0027]$ &
             $\mathbf{+0.0284}$ $[+0.0214, +0.0349]$, 0/7 \\
 & degree 2 & $+0.0005$ $[-0.0004, +0.0016]$ & $-0.0011$ $[-0.0031, +0.0010]$ &
             $\mathbf{+0.0198}$ $[+0.0128, +0.0258]$, 0/7 \\
\midrule
\multirow{2}{*}{ResNet-18}
 & degree 1 & $-0.0004$ $[-0.0033, +0.0027]$ & $\mathbf{-0.0037}$ $[-0.0059, -0.0016]$ &
             $\mathbf{-0.0123}$ $[-0.0172, -0.0068]$, \textbf{7/7} \\
 & degree 2 & $+0.0007$ $[-0.0008, +0.0022]$ & $\mathbf{-0.0047}$ $[-0.0079, -0.0017]$ &
             $-0.0018$ $[-0.0089, +0.0053]$ \\
\bottomrule
\end{tabular}
\end{table}
```

**Text.** *"On the ResNet-18 backbone a low-order field is the better calibration model: degree 1
beats the global model on both generalisation sets, by $-0.0037$ $[-0.0059, -0.0016]$ on Hamlyn
and $-0.0123$ $[-0.0172, -0.0068]$ on C3VD, where it wins all seven scenes, and degree 2 beats it
on Hamlyn by $-0.0047$ $[-0.0079, -0.0017]$. It also beats the dense map there ($-0.0058$
$[-0.0078, -0.0037]$, $p<0.001$), so on that backbone the capacity optimum lies strictly between
the two extremes and the submission's free per-pixel parameterisation is the wrong end of the
axis. On the foundation backbone the ordering does not carry over: neither degree improves on the
global model, and on C3VD the global model is better in all seven scenes against both. In the
training domain every comparison is a tie on both backbones. We therefore report the capacity
axis as a property of the calibration that must be tuned per backbone, and we do not recommend a
single degree."*

### 7.2 The balance over the whole axis

Of the eighteen paired contrasts among the calibration models (six contrasts on three datasets),
only eight have intervals excluding zero, and they do not crown a single form.

```latex
\begin{table}[t]
\centering
\caption{Every calibration contrast whose 95\% interval excludes zero. All remaining contrasts,
including the whole of SCARED except one, are ties. A form is named as the winner only where the
interval separates it from its rival.}
\label{tab:calib-balance}
\small
\begin{tabular}{lllc}
\toprule
Backbone & Dataset & Contrast & Winner \\
\midrule
Depth Anything & C3VD   & global vs no calibration & global, $-0.0256$, 7/7 \\
Depth Anything & C3VD   & global vs degree 1       & global, $-0.0284$, 7/7 \\
Depth Anything & C3VD   & global vs degree 2       & global, $-0.0198$, 7/7 \\
Depth Anything & SCARED & global vs dense map      & global, $-0.0015$, 7/7 \\
\midrule
Depth Anything & Hamlyn & global vs no calibration & \textbf{no calibration}, $-0.0032$ \\
ResNet-18      & Hamlyn & global vs no calibration & \textbf{no calibration}, $-0.0039$ \\
ResNet-18      & Hamlyn & global vs degree 1       & degree 1, $-0.0037$ \\
ResNet-18      & C3VD   & global vs degree 1       & degree 1, $-0.0123$, 7/7 \\
\bottomrule
\end{tabular}
\end{table}
```

**Text.** *"Three patterns organise this. In the training domain nothing separates: every
calibration model ties every other on SCARED, with the single exception of the global model
against the dense map. On Hamlyn the best choice is not to calibrate at all, and this is the one
result that repeats on both backbones with intervals excluding zero. On C3VD the winner reverses
with the backbone --- the global model wins all seven scenes on Depth Anything, and the degree-1
field wins all seven on ResNet-18 --- which is why we report the capacity axis rather than
recommending a degree."*

**Text for the discussion.** *"The one claim that survives on both backbones is the negative one:
the free per-pixel field of the original submission is never the best point of the axis. On
ResNet-18 it is the worst point on Hamlyn; on Depth Anything it ties the global model in-domain
while costing parameters. Whatever the calibration contributes, it is not contributed by the
spatial degrees of freedom."*

---

## 8. Section 4.x — component attribution (Reviewer 3)

Two new tables. The first separates the backbone from the method, which is exactly what was asked.

```latex
\begin{table}[t]
\centering
\caption{Building the model up one component at a time on SCARED, at an identical recipe and
resolution. The last column is each component's share of the span between a randomly initialised
encoder and the full model. Components are attributed to whoever introduced them.}
\label{tab:ladder}
\small
\begin{tabular}{llcc}
\toprule
Configuration & introduced by & Abs Rel & share \\
\midrule
Randomly initialised ViT-B                       & ---      & 0.1061 & --- \\
$+$ Depth Anything weights (frozen), DPT heads   & Yang et al. & 0.0869 & 34\% \\
$+$ convolutional neck (large-kernel attention)  & HADepth  & 0.0606 & \textbf{47\%} \\
$+$ DV-LoRA adapters ($=$ EndoDAC)               & EndoDAC  & 0.0517 & 16\% \\
$+$ highlight-aware loss and invariant loss      & HADepth / ours & 0.0518 & $\mathbf{-0.3\%}$ \\
$+$ affine calibration (full model)              & ours     & 0.0497--0.0512 & 1--4\% \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\centering
\caption{Removing one component at a time from the full model. The difference is
(variant $-$ MonoIIF) per sequence, so a \textbf{negative} value means the component was costing
accuracy. Bold marks intervals excluding zero.}
\label{tab:loo}
\small
\begin{tabular}{llccc}
\toprule
Component removed & by & SCARED ($n=7$) & Hamlyn ($n=58$) & C3VD ($n=7$) \\
\midrule
affine calibration      & ours    & $+0.0007$ $[-0.0010, +0.0028]$ &
                                    $\mathbf{-0.0015}$ $[-0.0028, -0.0003]$ &
                                    $-0.0020$ $[-0.0109, +0.0084]$ \\
invariant loss          & ours    & $-0.0006$ $[-0.0021, +0.0009]$ &
                                    $+0.0015$ $[-0.0000, +0.0030]$ &
                                    $\mathbf{-0.0170}$ $[-0.0271, -0.0072]$ \\
highlight-aware loss    & HADepth & $\mathbf{+0.0022}$ $[+0.0006, +0.0040]$ &
                                    $-0.0002$ $[-0.0022, +0.0019]$ &
                                    $-0.0059$ $[-0.0142, +0.0030]$ \\
DV-LoRA $\to$ plain LoRA & EndoDAC & $-0.0008$ $[-0.0017, +0.0002]$ &
                                    $\mathbf{+0.0030}$ $[+0.0017, +0.0044]$ &
                                    $\mathbf{+0.0219}$ $[+0.0151, +0.0296]$ \\
\bottomrule
\end{tabular}
\end{table}
```

**Text.** *"Roughly 96\% of the improvement over a randomly initialised encoder comes from the
pretrained weights and from adaptation machinery that is not ours. Our invariant term does not
earn its place as an accuracy contribution: removing it improves C3VD by 0.0170 with the interval
excluding zero. Where the method does win is against a faithful EndoDAC baseline retrained under
our recipe, by $+0.0250$ $[+0.0189, +0.0316]$ on C3VD in 7 of 7 scenes, at a tie in-domain and a
loss on Hamlyn. The contribution of this work is out-of-domain robustness under geometric shift,
not in-domain accuracy."*

---

## 9. Table 2 — the λ₁ sweep

The submitted Table 2 sweeps λ₁ on the ResNet variant, one seed, frame-level, and reports that
adding the invariant loss at 0.5 improves Abs Rel from 0.062 to 0.058. **That reverses with three
seeds and paired statistics.** Replace it.

```latex
\begin{table}[t]
\centering
\caption{Weight $\lambda_1$ of the illumination-invariant loss, three seeds, sequence level, on
the ResNet-18 backbone with the calibration held fixed. The difference is against $\lambda_1=0$,
so a positive value means the loss costs accuracy. Bold marks intervals excluding zero.}
\label{tab:lambda}
\small
\begin{tabular}{lccc}
\toprule
$\lambda_1$ & SCARED & Hamlyn & C3VD \\
\midrule
0            & \textbf{0.0576} & 0.1706 & 0.3381 \\
0.1          & $+0.0005$, tie & $-0.0031$ $[-0.0067, +0.0002]$ & $\mathbf{+0.0040}$ $[+0.0001, +0.0076]$ \\
0.25         & $+0.0017$, tie & $+0.0026$ $[+0.0000, +0.0051]$ & $-0.0020$, tie \\
0.5          & $\mathbf{+0.0032}$ $[+0.0005, +0.0063]$, 1/7 & $-0.0016$, tie & $+0.0002$, tie \\
1.0          & $\mathbf{+0.0041}$ $[+0.0017, +0.0069]$, $p=0.031$ & $+0.0009$, tie & $-0.0040$, tie \\
2.0          & $\mathbf{+0.0048}$ $[+0.0009, +0.0102]$ & $+0.0012$, tie & $-0.0053$, tie \\
\bottomrule
\end{tabular}
\end{table}
```

**Text.** *"In the training domain the invariant loss degrades accuracy monotonically with its
weight, and from $\lambda_1=0.5$ upward every interval excludes zero. The same holds with the SSIM
comparator of Eqs.~(14)--(15), so the finding does not depend on how the descriptors are compared.
Out of domain no weight is distinguishable from zero."*

## 9b. The ablation table (`tab:ablation`) extended with the backbones and the MonoIIF components

The submission's Table~\ref{tab:ablation} with a backbone column, a seeds column, the Depth Anything v1 component rows and the Depth Anything 3 rows. Lines 1--14 are the original rows verbatim; lines 15 onward come from `results/cviu/summary.csv` (SCARED, sequence level). Best/second best per metric are recomputed over the whole table. Generated by `ablation_table.py`. Needs `pifont` or the paper's existing `\cmark`/`\xmark` macros.

```latex
\begin{table*}[!t]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{|r|l|c|c|c|c|c|c|c|c|c|c|}
\hline
 & & Transformer & Use of & Lighting & Use of & {\large $\varepsilon_\textrm{AbsRel}$} ($\downarrow$) & {\large $\varepsilon_\textrm{SqRel}$} ($\downarrow$) & {\large $\varepsilon_\textrm{RMSE}$} ($\downarrow$) & {\large $\varepsilon_\textrm{RMSELog}$} ($\downarrow$) & $\delta_{1.25}$ ($\uparrow$) & \\
\textcolor{gray}{Line} & Backbone & blocks & loss $\mathcal{L}_\textrm{PML}$ & correction & loss $\mathcal{L}_\textrm{II}$ & & & & & & seeds \\
\hline
\textcolor{gray}{1} & ResNet-18 & \xmark & \xmark & \xmark & \cmark $(\lambda_1 = 1)$ & 0.072 & 0.713 & 6.027 & 0.098 & 0.942 & 1 \\
\textcolor{gray}{2} & ResNet-18 & \xmark & \cmark & local & \xmark & 0.062 & 0.486 & 5.093 & 0.084 & 0.968 & 1 \\
\textcolor{gray}{3} & ResNet-18 & \xmark & \cmark & local & \cmark $(\lambda_1 = 0.25)$ & 0.063 & 0.490 & 5.100 & 0.088 & 0.956 & 1 \\
\textcolor{gray}{4} & ResNet-18 & \xmark & \cmark & local & \cmark $(\lambda_1 = 0.5)$ & 0.058 & 0.438 & 4.850 & 0.082 & 0.966 & 1 \\
\textcolor{gray}{5} & ResNet-18 & \xmark & \cmark & local & \cmark $(\lambda_1 = 1)$ & 0.060 & 0.448 & 4.864 & 0.083 & 0.964 & 1 \\
\textcolor{gray}{6} & ResNet-18 & \xmark & \cmark & local & \cmark $(\lambda_1 = 2)$ & 0.063 & 0.492 & 5.072 & 0.088 & 0.957 & 1 \\
\textcolor{gray}{7} & ResNet-18 & \xmark & \cmark & local & \cmark $(\lambda_1 = 3)$ & 0.064 & 0.508 & 5.176 & 0.088 & 0.956 & 1 \\
\textcolor{gray}{8} & ResNet-18 & \xmark & \cmark & local & \cmark $(\lambda_1 = 4)$ & 0.064 & 0.506 & 5.193 & 0.088 & 0.960 & 1 \\
\textcolor{gray}{9} & ResNet-18 & \xmark & \cmark & local & \cmark $(\lambda_1 = 5)$ & 0.063 & 0.478 & 4.992 & 0.086 & 0.963 & 1 \\
\textcolor{gray}{10} & ResNet-18 & \xmark & \cmark & local & \cmark $(\lambda_1 = 10)$ & 0.061 & 0.469 & 4.997 & 0.084 & 0.966 & 1 \\
\textcolor{gray}{11} & MPViT-S & \cmark & \cmark & local & \cmark $(\lambda_1 = 0.5)$ & 0.055 & 0.412 & 4.614 & 0.077 & 0.969 & 1 \\
\textcolor{gray}{12} & MPViT-S & \cmark & \cmark & local & \xmark & 0.059 & 0.502 & 5.221 & 0.085 & 0.968 & 1 \\
\textcolor{gray}{13} & MPViT-S & \cmark & \cmark & \xmark & \xmark & 0.059 & 0.492 & 5.108 & 0.084 & 0.964 & 1 \\
\textcolor{gray}{14} & MPViT-S & \cmark & \cmark & \xmark & \cmark $(\lambda_1 = 0.5)$ & 0.057 & 0.439 & 4.737 & 0.080 & 0.966 & 1 \\
\hline
\textcolor{gray}{15} & ViT-B, random init & \cmark & \cmark$^{\dagger}$ & local & \cmark $(\lambda_1 = 0.1)$ & 0.106 & 1.391 & 8.751 & 0.147 & 0.892 & 1 \\
\textcolor{gray}{16} & DA v1 frozen, DPT heads & \cmark & \cmark$^{\dagger}$ & local & \cmark $(\lambda_1 = 0.1)$ & 0.087 & 0.968 & 7.392 & 0.118 & 0.933 & 1 \\
\textcolor{gray}{17} & DA v1 $+$ conv.\ neck & \cmark & \cmark$^{\dagger}$ & local & \cmark $(\lambda_1 = 0.1)$ & 0.061 & 0.565 & 5.518 & 0.088 & 0.968 & 1 \\
\textcolor{gray}{18} & DA v1 $+$ DV-LoRA (EndoDAC) & \cmark & \cmark & \xmark & \xmark & 0.052 & 0.404 & 4.737 & 0.075 & 0.979 & 3 \\
\textcolor{gray}{19} & DA v1 $+$ DV-LoRA & \cmark & \cmark & local & \xmark & 0.052 & 0.406 & 4.745 & 0.075 & 0.980 & 3 \\
\textcolor{gray}{20} & DA v1 $+$ DV-LoRA & \cmark & \cmark & global & \xmark & 0.050 & 0.383 & 4.594 & 0.073 & 0.981 & 3 \\
\textcolor{gray}{21} & DA v1 $+$ DV-LoRA & \cmark & \cmark & poly.\,1 & \xmark & \underline{0.050} & 0.377 & 4.556 & \underline{0.072} & \textbf{0.981} & 3 \\
\textcolor{gray}{22} & DA v1 $+$ DV-LoRA & \cmark & \cmark & poly.\,2 & \xmark & 0.051 & 0.390 & 4.683 & 0.074 & 0.980 & 3 \\
\textcolor{gray}{23} & DA v1 $+$ DV-LoRA & \cmark & \cmark & poly.\,1 & \cmark $(\lambda_1 = 0.1)$ & 0.051 & 0.399 & 4.705 & 0.074 & 0.980 & 3 \\
\textcolor{gray}{24} & DA v1 $+$ DV-LoRA & \cmark & \cmark & poly.\,1 & \cmark $(\lambda_1 = 0.25)$ & 0.052 & 0.418 & 4.805 & 0.075 & 0.980 & 3 \\
\textcolor{gray}{25} & DA v1 $+$ DV-LoRA & \cmark & \cmark & poly.\,1 & \cmark $(\lambda_1 = 0.5)$ & 0.053 & 0.429 & 4.872 & 0.076 & 0.979 & 3 \\
\textcolor{gray}{26} & DA v1 $+$ DV-LoRA & \cmark & \cmark & local & \cmark $(\lambda_1 = 0.1)$ & 0.054 & 0.444 & 4.923 & 0.078 & 0.978 & 1 \\
\textcolor{gray}{27} & DA v1 $+$ DV-LoRA & \cmark & \cmark$^{\dagger}$ & \xmark & \cmark $(\lambda_1 = 0.1)$ & 0.052 & 0.405 & 4.727 & 0.075 & 0.980 & 3 \\
\textcolor{gray}{28} & DA v1 $+$ DV-LoRA & \cmark & \cmark$^{\dagger}$ & global & \cmark $(\lambda_1 = 0.1)$ & \textbf{0.050} & \underline{0.368} & \textbf{4.521} & \textbf{0.072} & \underline{0.981} & 3 \\
\textcolor{gray}{29} & DA v1 $+$ DV-LoRA & \cmark & \cmark$^{\dagger}$ & local & \xmark & 0.051 & \textbf{0.367} & \underline{4.529} & 0.073 & 0.981 & 3 \\
\textcolor{gray}{30} & DA v1 $+$ DV-LoRA (MonoIIF) & \cmark & \cmark$^{\dagger}$ & local & \cmark $(\lambda_1 = 0.1)$ & 0.051 & 0.381 & 4.619 & 0.074 & 0.980 & 3 \\
\hline
\textcolor{gray}{31} & DA3-Base, zero-shot & \cmark & --- & --- & --- & 0.072 & 0.720 & 6.472 & 0.102 & 0.953 & 1 \\
\textcolor{gray}{32} & DA3 $+$ DV-LoRA (EndoDAC recipe) & \cmark & \cmark & \xmark & \xmark & 0.053 & 0.448 & 4.939 & 0.077 & 0.977 & 1 \\
\textcolor{gray}{33} & DA3 $+$ DV-LoRA (MonoIIF recipe) & \cmark & \cmark$^{\dagger}$ & local & \cmark $(\lambda_1 = 0.1)$ & 0.054 & 0.441 & 4.904 & 0.078 & 0.974 & 3 \\
\hline
\end{tabular}
}
\caption{Overview of the ablation study results on SCARED. Six columns define the model configuration: i) the backbone of the depth network; ii) whether transformer-based depth estimation blocks (Section~\ref{sec:Architectures}) are used (\cmark) or not (\xmark) --- for the Depth Anything rows the encoder itself is a vision transformer; iii) whether the photometric similarity loss $\mathcal{L}_\textrm{PML}$ is used, where \cmark$^{\dagger}$ denotes its highlight-aware variant adopted from HADepth; iv) the illumination change model with which the synthesized target images $\hat{I}_t$ are corrected (Section~\ref{sec:lit}): none (\xmark), one global affine pair per image, a polynomial field of degree 1 or 2, or the dense per-pixel map (local); v) the weight of the illumination-invariant loss $\mathcal{L}_\textrm{II}$ in Eq.~\eqref{eq:global_loss}, where \xmark\ is equivalent to $\lambda_1=0$. Lines 1--14 are the configurations of the original study, one training seed, metrics averaged over frames. Lines 15--33 are averaged within each of the seven test sequences and then across sequences, and over the number of training seeds in the last column. Line 31 is Depth Anything 3 without any training. The best and second best values per metric are in bold and underlined respectively.}
\label{tab:ablation}
\end{table*}
```

---

## 10. Section 4.x — Depth Anything 3 (Question 2)

```latex
\begin{table}[t]
\centering
\caption{Depth Anything 3 as a backbone, zero-shot and adapted with our recipe at an identical
trainable-parameter count (8\,961\,540 in both). DA3's monocular model is released at Large size
only, so it enters as a zero-shot row.}
\label{tab:da3}
\small
\begin{tabular}{lccc}
\toprule
Model & SCARED & Hamlyn & C3VD \\
\midrule
DA3-Base zero-shot, median scaling           & 0.0718 & 0.2066 & 0.3947 \\
DA3-Base zero-shot, affine disparity aligned & 0.0692 & 0.1811 & 0.6154 \\
DA3-Base adapted, our recipe                 & 0.0545 & 0.1735 & 0.3616 \\
DA3-Base adapted, EndoDAC recipe             & 0.0527 & 0.1762 & 0.3597 \\
\midrule
MonoIIF (Depth Anything v1)                  & \textbf{0.0512} & \textbf{0.1608} & \textbf{0.2679} \\
\bottomrule
\end{tabular}
\end{table}
```

**Text.** *"DA3-Base's encoder is not a vanilla DINOv2 --- from block 4 it uses QK-norm, 2D RoPE, a
camera token and alternating local/global attention, and it emits 1536-d features at layers
5/7/9/11 --- so its weights cannot be copied onto the encoder used here. We ported its encoder and
the main branch of its DualDPT head and verified the port against the reference implementation:
encoder features and camera tokens are bit-identical at three resolutions and all 207 encoder
tensors match the released checkpoint. Adapted under our recipe it is worse on all three datasets,
so we keep Depth Anything v1 and report DA3 as an evaluated alternative."*

---

## 11. The MonoII / MonoIIT / MonoIIF table

**11.1 — The MonoIIT row does not contain the method.** It was trained in a separate codebase with
no illumination calibration, no invariant loss, and an encoder starting from **random** weights
rather than ImageNet. It is therefore neither MonoViT as published nor an instance of the method.
Either relabel it honestly or replace it with the retrained row.

**11.2 — Make the three rows one experiment.** The retrained version uses the repo defaults and
changes only the depth network, so the column isolates the architecture:

```latex
\begin{table}[t]
\centering
\caption{The same method on three depth networks, changing only the backbone: identical pose
network, splits, schedule, optimiser and augmentation. The plain row of each pair has neither
component. The MPViT encoder starts from random weights in both of its cells, because the ImageNet
MPViT checkpoint is no longer retrievable from the authors' distribution; this is also how the
original MonoIIT row was trained, and part of the gap to the other two columns is that rather than
the architecture.}
\label{tab:backbones}
\small
\begin{tabular}{llccc}
\toprule
Backbone & Configuration & SCARED & Hamlyn & C3VD \\
\midrule
\multirow{2}{*}{ResNet-18}
 & plain              & 0.0593 & 0.1764 & 0.3280 \\
 & $+$ components (MonoII)  & 0.0593 & 0.1709 & 0.3123 \\
\midrule
\multirow{2}{*}{MPViT-small}
 & plain              & TBD & TBD & TBD \\
 & $+$ components (MonoIIT) & TBD & TBD & TBD \\
\midrule
\multirow{2}{*}{Depth Anything}
 & plain ($=$ EndoDAC) & 0.0517 & \textbf{0.1565} & 0.2929 \\
 & $+$ components (MonoIIF) & \textbf{0.0512} & 0.1608 & \textbf{0.2679} \\
\bottomrule
\end{tabular}
\end{table}
```

The MPViT rows are training (`MonoIIT-repro`, `MonoIIT-components`); fill them from
`results/cviu/summary.csv` when they finish.

**11.3 — Explain why Monodepth2 appears twice with different numbers.** Table 4 lists Monodepth2
at 0.093 and the table above lists a plain ResNet-18 at 0.0593. They are the same architecture and
the gap is the training, not the network: Table 4's row is the authors' released checkpoint,
evaluated unchanged, while the row here is that architecture retrained on SCARED under our shared
recipe (learned intrinsics, 20 epochs, our augmentation). Add one sentence saying so, otherwise the
two rows read as a contradiction. The same distinction applies to EndoDAC, which appears as a
released checkpoint (0.0507) and as a retrained baseline (`E3`, 0.0517) — there the two agree,
which is the evidence that the recipe is faithful.

**11.4 — Do not call the first row of the calibration table a baseline.** It keeps the invariant
loss and the highlight term and removes only the lighting model; the model with none of the three
components is EndoDAC at 0.0517. The caption in §6 now says this explicitly, because the row label
"none" invites exactly the opposite reading.

---

## 12. Figures

**12.1** — Add the capacity curve of §7 as a figure: Abs Rel against polynomial degree, one line
per dataset, two panels for the two backbones. It is the visual answer to Reviewer 2.

**12.2** — Add a per-sequence dot plot for SCARED showing MonoIIF and the four leading baselines,
one column per sequence. It shows at a glance why the ranking is not resolvable and supports Q6
better than any table.

**12.3** — The qualitative figures keep their captions, but any that claims "V2" needs the same
fix as §1.1.
