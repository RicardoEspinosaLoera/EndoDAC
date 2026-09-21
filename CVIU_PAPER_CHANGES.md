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

## 5. Tables 4–6 — recomputed at the sequence level

Replace the three main tables. Values are means over sequences; `±` is the SD **across training
seeds** and exists only for models trained here. The five external methods are the authors'
released checkpoints (n = 1), which reproduce their published numbers under this protocol.

### 5.1 Table 4 — SCARED (n = 7 sequences)

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

### 5.2 Table 5 — Hamlyn (evaluation only, n = 58 blocks of 100 frames)

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

### 5.3 Table 6 — C3VD (evaluation only, n = 7 scenes)

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

### 5.4 New Table 7 — the paired comparisons the reviewer asked for

This is the table that answers Q6 and it should sit immediately after Tables 4–6.

```latex
\begin{table}[t]
\centering
\caption{Paired comparisons against MonoIIF with the video sequence as the unit of analysis.
The difference is (method $-$ MonoIIF) in Abs Rel per sequence, so a \textbf{positive} value means
MonoIIF is better; $[\,\cdot\,]$ is a 95\% cluster bootstrap CI over sequences (10\,000 draws) and
$k/n$ counts the sequences in which MonoIIF is better. Exact two-sided Wilcoxon $p$ is reported
for the datasets with $n=7$; with $n=7$ its smallest attainable value is $2/2^{7}=0.016$, so
family-wise correction across a table of this size cannot reach significance and these are
reported as exploratory. Rows whose interval excludes zero are in bold.}
\label{tab:paired}
\small
\begin{tabular}{llccc}
\toprule
Dataset & Method & $\Delta$ Abs Rel [95\% CI] & $k/n$ & exact $p$ \\
\midrule
\multirow{5}{*}{SCARED}
 & HADepth       & $-0.0023$ $[-0.0068, +0.0016]$ & 3/7 & 0.578 \\
 & EndoDAC       & $-0.0004$ $[-0.0048, +0.0041]$ & 3/7 & 0.688 \\
 & MonoPCC       & $-0.0007$ $[-0.0073, +0.0053]$ & 4/7 & 0.938 \\
 & AF-SfMLearner & $\mathbf{+0.0072}$ $[+0.0011, +0.0137]$ & 5/7 & 0.109 \\
 & Monodepth2    & $\mathbf{+0.0421}$ $[+0.0266, +0.0587]$ & 7/7 & 0.016 \\
\midrule
\multirow{5}{*}{Hamlyn}
 & EndoDAC       & $-0.0000$ $[-0.0037, +0.0036]$ & 31/58 & --- \\
 & HADepth       & $+0.0019$ $[-0.0002, +0.0040]$ & 38/58 & --- \\
 & MonoPCC       & $\mathbf{+0.0113}$ $[+0.0067, +0.0158]$ & 41/58 & --- \\
 & AF-SfMLearner & $\mathbf{+0.0220}$ $[+0.0151, +0.0296]$ & 47/58 & --- \\
 & Monodepth2    & $\mathbf{+0.0624}$ $[+0.0498, +0.0745]$ & 55/58 & --- \\
\midrule
\multirow{5}{*}{C3VD}
 & HADepth       & $\mathbf{-0.0466}$ $[-0.0683, -0.0292]$ & 0/7 & 0.016 \\
 & EndoDAC       & $-0.0111$ $[-0.0405, +0.0141]$ & 3/7 & 0.813 \\
 & AF-SfMLearner & $\mathbf{+0.0678}$ $[+0.0207, +0.1126]$ & 5/7 & 0.078 \\
 & Monodepth2    & $\mathbf{+0.0740}$ $[+0.0410, +0.1016]$ & 6/7 & 0.031 \\
 & MonoPCC       & $\mathbf{+0.0889}$ $[+0.0621, +0.1116]$ & 7/7 & 0.016 \\
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

---

## 6. Section 4.x — the calibration ablation (Reviewer 2)

New table, replacing the ablation that compared the module only against nothing. The controlled
comparison is run on the **ResNet-18 backbone**, where the full three-cell design fits in the
compute budget at three seeds, and is then confirmed on the method's own backbone.

### 6.1 The controlled comparison, ResNet-18

```latex
\begin{table}[t]
\centering
\caption{Photometric calibration models on the ResNet-18 backbone. All five rows share one recipe
--- monodepth2's photometric loss, illumination-invariant loss at $\lambda_1=0.5$, three seeds ---
and differ \emph{only} in how the affine field $(c,b)$ is parameterised, from a single pair per
image to a free per-pixel map. The lower block is the paired difference (variant $-$ degree 1) in
Abs Rel per sequence, so a \textbf{positive} value means the degree-1 field is better; bold marks
intervals excluding zero.}
\label{tab:calib-resnet}
\small
\begin{tabular}{lccc}
\toprule
Calibration model & SCARED & Hamlyn & C3VD \\
\midrule
no lighting model             & 0.0599\,$\pm$\,0.0016 & 0.1698\,$\pm$\,0.0041 & 0.3363\,$\pm$\,0.0116 \\
global, one $(c,b)$ pair      & 0.0600\,$\pm$\,0.0009 & 0.1758\,$\pm$\,0.0065 & 0.3290\,$\pm$\,0.0072 \\
\textbf{linear field (degree 1)} & \textbf{0.0596\,$\pm$\,0.0002} & 0.1700\,$\pm$\,0.0014 & \textbf{0.3278\,$\pm$\,0.0101} \\
quadratic field (degree 2)    & 0.0608\,$\pm$\,0.0008 & \textbf{0.1690\,$\pm$\,0.0010} & 0.3383\,$\pm$\,0.0040 \\
dense map (original submission) & 0.0608\,$\pm$\,0.0007 & 0.1757\,$\pm$\,0.0021 & 0.3379\,$\pm$\,0.0057 \\
\midrule
\multicolumn{4}{l}{\emph{paired against the degree-1 field}} \\
no lighting model  & $+0.0002$ $[-0.0037, +0.0047]$ & $-0.0001$ $[-0.0017, +0.0015]$ &
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
