"""Round-2 edits of the CVIU manuscript: round-1 marks (blue) to black, new text in red.

    python apply_r2_edits.py "<cas-dc-template_new (5).tex>" <components_crossdomain.tex> <out_dir>
"""
import os
import re
import sys

src, table_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(src, encoding="utf-8").read()
t = t.replace("\\textcolor{blue}", "\\textcolor{black}")


def rx(s):
    return r"\s+".join(re.escape(p) for p in s.split())


def R(s):
    return "\\textcolor{red}{" + s + "}"


def one(old, new):
    global t
    m = re.findall(rx(old), t)
    assert len(m) == 1, "%d matches for: %s" % (len(m), old[:70])
    t = re.sub(rx(old), lambda _: new, t)


def every(old, new, n):
    global t
    m = re.findall(rx(old), t)
    assert len(m) == n, "%d matches (expected %d) for: %s" % (len(m), n, old[:70])
    t = re.sub(rx(old), lambda _: new, t)


def between(start, end, new):
    global t
    pat = rx(start) + r".*?" + rx(end)
    m = re.findall(pat, t, flags=re.S)
    assert len(m) == 1, "%d matches for: %s ... %s" % (len(m), start[:50], end[:40])
    t = re.sub(pat, lambda _: new, t, flags=re.S)


def after(anchor, new):
    one(anchor, anchor + new)


E = r"$\varepsilon_\textrm{AbsRel}$"
LII = r"$\mathcal{L}_\textrm{II}$"
TAB = r"Table~\ref{tab:components-isolated}"

# ---------------------------------------------------------------- paired table, single reference (E8)
PAIRED = r"""\multirow{6}{*}{SCARED}
 & HADepth            & \textbf{0.0489} $[0.0398, 0.0569]$    & $-0.0023$ $[-0.0067, +0.0016]$          & 3/7   & 0.578 \\
 & MonoPCC            & \underline{0.0505} $[0.0396, 0.0591]$ & $-0.0007$ $[-0.0073, +0.0053]$          & 4/7   & 0.938 \\
 & EndoDAC            & 0.0507 $[0.0401, 0.0607]$             & $-0.0004$ $[-0.0047, +0.0041]$          & 3/7   & 0.688 \\
 & MonoIIF (proposed) & 0.0512 $[0.0408, 0.0619]$             & ---                                     & ---   & ---   \\
 & AF-SfMLearner      & 0.0584 $[0.0446, 0.0724]$             & $\mathbf{+0.0072}$ $[+0.0011, +0.0135]$ & 5/7   & 0.109 \\
 & Monodepth2         & 0.0933 $[0.0750, 0.1104]$             & $\mathbf{+0.0421}$ $[+0.0266, +0.0587]$ & 7/7   & 0.016 \\
\midrule
\multirow{6}{*}{Hamlyn}
 & EndoDAC            & \textbf{0.0904} $[0.0477, 0.1366]$    & $-0.0026$ $[-0.0076, +0.0023]$          & 1/4   & 0.375 \\
 & MonoIIF (proposed) & \underline{0.0930} $[0.0552, 0.1387]$ & ---                                     & ---   & ---   \\
 & HADepth            & 0.0934 $[0.0549, 0.1394]$             & $+0.0004$ $[-0.0033, +0.0038]$          & 2/4   & 1.000 \\
 & MonoPCC            & 0.1042 $[0.0645, 0.1523]$             & $\mathbf{+0.0112}$ $[+0.0068, +0.0148]$ & 4/4   & 0.125 \\
 & AF-SfMLearner      & 0.1106 $[0.0655, 0.1636]$             & $\mathbf{+0.0176}$ $[+0.0089, +0.0263]$ & 4/4   & 0.125 \\
 & Monodepth2         & 0.1392 $[0.0866, 0.2016]$             & $\mathbf{+0.0461}$ $[+0.0316, +0.0607]$ & 4/4   & 0.125 \\
\midrule
\multirow{6}{*}{C3VD}
 & HADepth            & \textbf{0.2213} $[0.2019, 0.2419]$    & $\mathbf{-0.0466}$ $[-0.0683, -0.0292]$ & 0/7   & 0.016 \\
 & EndoDAC            & \underline{0.2568} $[0.2215, 0.2938]$ & $-0.0111$ $[-0.0403, +0.0141]$          & 3/7   & 0.812 \\
 & MonoIIF (proposed) & 0.2679 $[0.2448, 0.2893]$             & ---                                     & ---   & ---   \\
 & AF-SfMLearner      & 0.3356 $[0.2822, 0.3839]$             & $\mathbf{+0.0678}$ $[+0.0216, +0.1117]$ & 5/7   & 0.078 \\
 & Monodepth2         & 0.3419 $[0.2948, 0.3775]$             & $\mathbf{+0.0740}$ $[+0.0404, +0.1016]$ & 6/7   & 0.031 \\
 & MonoPCC            & 0.3568 $[0.3184, 0.3865]$             & $\mathbf{+0.0889}$ $[+0.0628, +0.1119]$ & 7/7   & 0.016 \\"""
between(r"\multirow{6}{*}{SCARED}", r"$\mathbf{+0.1059}$ $[+0.0731, +0.1329]$ & 7/7 & 0.016 \\", PAIRED)
one(r"\centering \caption{\textcolor{black}{Uncertainty and paired comparisons",
    "\\centering\n\\rev\n" + r"\caption{\textcolor{red}{Uncertainty and paired comparisons")
one(r"SCARED is $0.0023$ (values between $0.0485$ and $0.0528$", "SCARED is " + R(r"$0.0014$ (values between $0.0508$ and $0.0534$"))
one(r"($0.050$) lies within this range.", R(r"($0.050$) lies 0.0008 below the lowest of these values."))
one(r"(0.002) is smaller than the seed-to-seed standard deviation of MonoIIF (0.0023,",
    R(r"(0.002) is of the order of the seed-to-seed standard deviation of MonoIIF (0.0014,"))
between(r"MonoIIF obtains the lowest $\varepsilon_\textrm{AbsRel}$ on both generalization datasets",
        r"while MonoIIF and EndoDAC cannot be distinguished.", R(
    r"MonoIIF obtains the lowest $\varepsilon_\textrm{AbsRel}$ on sequence rectified14 of Hamlyn (Table~\ref{tab:quant_hamlyn}) but not on C3VD (Table~\ref{tab:quant_c3vd}), and the Hamlyn margin does not hold in the paired analysis of Table~\ref{tab:paired-c1}: on the four Hamlyn sequences MonoIIF is slightly behind EndoDAC and level with HADepth, and on C3VD HADepth is more accurate than MonoIIF in all seven scenes, while MonoIIF and EndoDAC are not distinguished."))


def redrow(line, cells):
    return line + " & " + " & ".join(R(c) for c in cells) + r" \\"


one(r"\textcolor{gray}{6} &EndoDAC & 0.267 & 3.273 & \textbf{11.194} & 0.321 & 0.519 & \textbf{0.841} & 0.950 \\",
    redrow(r"\textcolor{gray}{6} &EndoDAC       ", [r"\underline{0.236}", r"\underline{3.072}", r"\textbf{11.804}", r"\textbf{0.309}", r"\textbf{0.558}", r"\textbf{0.843}", r"\underline{0.957}"]))
one(r"\textcolor{gray}{7} &HADepth & \underline{0.234} & \underline{3.052} & 12.080 & \textbf{0.312} & \textbf{0.555} & \underline{0.835} & \textbf{0.963} \\",
    redrow(r"\textcolor{gray}{7} &HADepth       ", [r"\textbf{0.234}", r"\textbf{3.052}", r"\underline{12.080}", r"\underline{0.312}", r"\underline{0.555}", r"\underline{0.835}", r"\textbf{0.963}"]))
one(r"\textcolor{gray}{10} &MonoIIF (Proposed) & \textbf{0.226} & \textbf{3.043} & \underline{12.030} & \textbf{0.312} & \underline{0.534} & \underline{0.835} & \textbf{0.963} \\",
    redrow(r"\textcolor{gray}{10} &MonoIIF (Proposed)", ["0.299", "4.406", "14.132", "0.384", "0.439", "0.739", "0.920"]))
between(r"On C3VD, MonoIIF obtains the lowest $\varepsilon_\textrm{AbsRel}$ (0.226)", r"HADepth is more accurate than MonoIIF in six of them.}", R(
    r"On C3VD (Table~\ref{tab:quant_c3vd}), HADepth and EndoDAC are the two most accurate methods on all criteria ($\varepsilon_\textrm{AbsRel}$ of 0.234 and 0.236). MonoIIF (0.299) is less accurate than both, and also than MonoIIT (0.259), MonoViT (0.281) and MonoII (0.293); it remains more accurate than Monodepth2, MonoPCC, AF-SfMLearner and EndoSfMLearner. The single training run of Table~\ref{tab:quant_c3vd} is less accurate on this dataset than the average of the three seeds of Table~\ref{tab:paired-c1} (0.268 when the seven scenes are given equal weight), which shows the seed sensitivity of the transfer to C3VD; in that analysis HADepth is more accurate than MonoIIF in all seven scenes, and MonoIIF and EndoDAC are not distinguished.") + "}")
one(r"lie within 0.002 of each other", "lie within " + R("0.0023") + " of each other")
one(r"MonoIIF is better in only four of the seven sequences against each of these methods",
    "MonoIIF is better in " + R("at most four of the seven sequences") + " against each of these methods")
one(r"($-0.0027$, $[-0.0069, +0.0016]$)", R(r"($-0.0026$, $[-0.0076, +0.0023]$)"))
one(r"($+0.0004$, $[-0.0034, +0.0042]$)", R(r"($+0.0004$, $[-0.0033, +0.0038]$)"))
one(r"($-0.0296$, $[-0.0494, -0.0105]$, in six of the seven scenes, $p=0.031$)",
    R(r"($-0.0466$, $[-0.0683, -0.0292]$, in all seven scenes, $p=0.016$)"))
one(r"($+0.0059$, $[-0.0189, +0.0285]$)", R(r"($-0.0111$, $[-0.0403, +0.0141]$)"))

# ---------------------------------------------------------------- Table 5 (Hamlyn, rectified14, 200 mm cap)
B, U = (lambda s: r"\textbf{%s}" % s), (lambda s: r"\underline{%s}" % s)
one(r"&Monodepth2 & 0.224 & 6.069 & 17.517 & 0.297 & 0.635 & 0.856 & 0.938 \\",
    redrow("&Monodepth2    ", ["0.228", "6.504", "18.315", "0.303", "0.632", "0.852", "0.933"]))
one(r"&AF-SfMLearner & 0.182 & 5.586 & 15.267 & 0.232 & 0.750 & 0.911 & 0.968 \\",
    redrow("&AF-SfMLearner ", ["0.187", "6.221", "16.112", "0.238", "0.747", "0.907", "0.964"]))
one(r"&MonoPCC & 0.172 & 4.984 & 15.176 & 0.232 & 0.758 & 0.905 & 0.965 \\",
    redrow("&MonoPCC       ", ["0.176", "5.480", "16.014", "0.238", "0.754", "0.901", "0.959"]))
one(r"&EndoDAC & 0.184 & 4.686 & 15.182 & 0.247 & 0.717 & 0.891 & 0.965 \\",
    redrow("&EndoDAC       ", [U("0.163"), B("4.377"), B("14.172"), B("0.217"), "0.766", B("0.927"), B("0.976")]))
one(r"&HADepth & \underline{0.162} & 4.432 & 14.071 & \underline{0.217} & \textbf{0.769} & \underline{0.926} & \textbf{0.976} \\",
    redrow("&HADepth       ", ["0.166", "4.938", "14.865", "0.222", U("0.767"), "0.921", "0.972"]))
one(r"\textcolor{black}{MonoII (Proposed)} & \underline{0.162} & \underline{4.216} & \underline{13.873} & 0.220 & \underline{0.765} & 0.918 & \textbf{0.976} \\",
    r"\textcolor{black}{MonoII (Proposed)}$^{\ast}$ & 0.162 & 4.216 & 13.873 & 0.220 & 0.765 & 0.918 & 0.976 \\")
one(r"&MonoIIF (Proposed) & \textbf{0.159} & \textbf{4.155} & \textbf{13.680} & \textbf{0.214} & 0.764 & \textbf{0.930} & \textbf{0.976} \\",
    redrow("&MonoIIF (Proposed)", [B("0.161"), U("4.497"), U("14.352"), U("0.218"), B("0.772"), U("0.926"), U("0.973")]))
one(r"&MonoViT & 0.195 &", r"&MonoViT$^{\ast}$ & 0.195 &")
one(r"&EndoSfMLearner & 0.238 &", r"&EndoSfMLearner$^{\ast}$ & 0.238 &")
one(r"&MonoIIT (Proposed) & 0.188 &", r"&MonoIIT (Proposed)$^{\ast}$ & 0.188 &")
one(r"\caption{Generalization results on the Hamlyn dataset.", r"\caption{Generalization results on the Hamlyn dataset " + R(
    r"(sequence rectified14). Depth is capped at 200\,mm, except for the rows marked with $^{\ast}$, which come from the original evaluation with a 150\,mm cap and are therefore not strictly comparable with the others; best and second-best values are marked among the unmarked rows only."))
one(r"\caption{Generalization results on the C3VD dataset.", r"\caption{Generalization results on the C3VD dataset. " + R(
    r"Rows 2, 3, 8 and 9 come from the original evaluation; the others were recomputed with the protocol of Section~\ref{sec:protocol}."))
between(r"MonoIIF achieves the best performance across most metrics", r"(0.769 vs.\ 0.764).", R(
    r"MonoIIF, EndoDAC and HADepth obtain close values ($\varepsilon_\textrm{AbsRel}$ of 0.161, 0.163 and 0.166, respectively). MonoIIF has the lowest $\varepsilon_\textrm{AbsRel}$ and the highest $\delta_{1.25}$ (0.772), and EndoDAC the lowest $\varepsilon_\textrm{SqRel}$, $\varepsilon_\textrm{RMSE}$ and $\varepsilon_\textrm{RMSELog}$; differences of this size are of the order of the seed-to-seed variability and do not rank the three methods.") + r" \textcolor{black}{")
one(r"and foundation model-based approaches achieve the highest accuracy.", R(
    r"and the foundation model-based approaches HADepth and EndoDAC are among the most accurate methods on both datasets, as is MonoIIF on Hamlyn."))
between(r"On Hamlyn, MonoII (ResNet-18, 2h\,6m of training) ties HADepth", r"whereas HADepth requires 26h\,13m of training.", R(
    r"On Hamlyn, MonoII (ResNet-18, 2h\,6m of training) obtains an $\varepsilon_\textrm{AbsRel}$ of 0.162, close to the values of the foundation-model methods (0.161 to 0.166), although its row was computed with the original 150\,mm cap and is not strictly comparable with theirs."))

# ---------------------------------------------------------------- abstract
between(r"\textcolor{black}{This work proposes a unified", r"than those of HADepth.}", R(
    r"This work proposes a self-supervised framework that integrates two photometric mechanisms: a per-pixel lighting calibration module and an illumination-invariant loss derived from Robinson kernel-based descriptors. These components are combined with three depth encoders: a CNN backbone (MonoII), a hybrid CNN--transformer backbone (MonoIIT), and a Depth~Anything foundation model encoder adapted through domain-specific low-rank adaptation (MonoIIF). All three variants are trained end-to-end in a single run, without auxiliary pretrained subnetworks such as optical flow networks. On the SCARED benchmark, no significant difference in " + E + r" was detected between MonoIIF and the strongest foundation-model baselines, HADepth and EndoDAC, while MonoIIF requires less than half of their training time (12h\,8m vs.\ more than 25h). On the Hamlyn and C3VD generalization datasets, a paired analysis detects no significant difference between MonoIIF and EndoDAC, nor between MonoIIF and HADepth on Hamlyn, whereas HADepth is more accurate on C3VD; MonoIIF is more accurate than the remaining methods on both datasets. With four to seven test sequences, the absence of a significant difference does not establish equivalence. A component-wise analysis on the three datasets shows that the accuracy of MonoIIF stems mainly from the pretrained encoder and its adaptation: the lighting calibration improves the transfer to C3VD, whose geometry differs most from the training data, but not to Hamlyn, and the illumination-invariant loss brings a measurable benefit only to the lightweight encoders. Qualitative surface reconstruction and 3D mosaicing examples on the Hamlyn dataset illustrate the continuity of the textured surfaces obtained from the depth maps; in these examples, the surfaces obtained with MonoIIF show fewer geometric discontinuities than those obtained with HADepth."))

# ---------------------------------------------------------------- introduction
between(r"As demonstrated in Section~\ref{sec:Experimental-Results}, this strategy yields",
        r"significantly reducing training time.", R(
    r"As shown in Section~\ref{sec:Experimental-Results}, with a foundation encoder this strategy reaches an accuracy for which no significant difference with the strongest foundation-model baselines is detected on SCARED and Hamlyn, at less than half of their training time, and it is more accurate than the flow-based baseline."))
between(r"Extensive evaluations on SOTA", r"setting a foundation for real-world deployment.", R(
    r"Evaluations on three public datasets assess the generalization of the models and isolate the contribution of each component. Additionally, the use of the predicted depth maps for 3D surface reconstruction is illustrated qualitatively on the HAMLYN dataset, by checking the continuity of the textures and of the geometry of the reconstructed surfaces."))

# ---------------------------------------------------------------- 2.2.4 MonoII, Fig. 4
one(r"\paragraph{Transformer-based backbone (MonoIIT).}", R(
    r"\paragraph{Convolutional backbone (MonoII).} MonoII keeps the depth network of Monodepth2~\citep{Godard_ICCV_2019}: a ResNet-18 encoder~\citep{He_CVPR_2016} pre-trained on ImageNet~\citep{Deng_CVPR_2009} and a DispNet-style decoder with skip connections that predicts disparity maps at four scales. The pose network is a separate ResNet-18 encoder taking the concatenated image pair $(I_s, I_t)$, followed by the pose decoder of Monodepth2; its last feature map also feeds the lighting change decoder of Section~\ref{sec:lit}. MonoII therefore differs from Monodepth2 only by the two proposed components, namely the per-pixel lighting calibration used in $\mathcal{L}_\textrm{PML}$ and the illumination-invariant loss " + LII + r", and is the reference with respect to which the transformer-based (MonoIIT) and foundation-model (MonoIIF) encoders are compared.")
    + "\n%\n" + r"\paragraph{Transformer-based backbone (MonoIIT).}")
one(r"\caption{Architecture of the transformer-based depth estimation framework.",
    r"\caption{" + R(r"Architecture of MonoIIT, the transformer-based depth estimation network."))

# ---------------------------------------------------------------- 3.1 datasets
between(r"The SCARED dataset is divided into three subsets", r"performance of \textcolor{black}{ten} models.", R(
    r"The public split of AF-SfMLearner~\citep{Shao_MIA_2022}, also used by EndoDAC~\citep{EndoDAC_MICCAI_2024}, is adopted: 15,351 training images (keyframes 1 and 2 of datasets 1 to 7 and keyframes 1 to 4 of datasets 8 and 9), 1,705 validation images (keyframe 3 of datasets 2 to 7) and 551 test images taken from seven keyframe videos (keyframe 3 of dataset 1 and keyframe 4 of datasets 2 to 7). The validation split was not used for depth evaluation. The 551 test images were used both for the ablation study of Section~\ref{sec:ablation}, which guided the choice of the weight $\lambda_1$ and of the model configurations, and for the quantitative evaluation of the ten models compared in Section~\ref{sec:Experimental-Results}; the consequences of this double use are examined in Section~\ref{sec:protocol}. The Hamlyn and C3VD datasets were not used for any choice."))
after(r"for which ground truth is provided in the form of 3D point clouds.}", " " + R(
    r"The component analysis of " + TAB + r" uses sequence rectified14, divided into 58 contiguous blocks of 100 frames."))

# ---------------------------------------------------------------- 3.2 protocol
one(r"\subsection{Performance evaluation protocol}", r"\subsection{Performance evaluation protocol\label{sec:protocol}}")
after(r"excluding extreme or unrealistic predictions.", "\n" + R(
    r"For the configurations evaluated with the sequence-level protocol (" + TAB + r" and Table~\ref{tab:paired-c1}), the caps are 150\,mm for SCARED, 200\,mm for Hamlyn (150\,mm would discard a fifth of the valid ground truth of sequence rectified06) and 100\,mm for C3VD, the range of its ground truth.") + "\n")
one(r"were trained with three random seeds and the reported values are averaged over these seeds.",
    r"were trained with three random seeds and the reported values are averaged over these seeds " + R(r"(except the rows of " + TAB + r" for which one seed is indicated)") + ".")
after(r"and the single-run tables should be read with this in mind.}", "\n" + R(
    r"\noindent{\bf Model selection and the test split.} "
    r"The weight $\lambda_1$ and the model configurations were chosen on the 551 SCARED test images (Section~\ref{sec:ablation}), which are also the images on which the SCARED results are reported. Moreover, for every model trained in this work, the reported checkpoint is the epoch with the lowest RMSE on these same images, as in the training procedure of EndoDAC, whereas the released checkpoints of the comparison methods were not re-selected. The SCARED results of the proposed models are therefore development-set results and may be optimistic; only the Hamlyn and C3VD results, obtained with datasets that were never used for any choice, are held-out results. To quantify the influence of the selection, the seven test sequences were split in all possible ways into a selection subset of three or four sequences and its complement, for each family of configurations compared on SCARED with three seeds (illumination correction model, weight $\lambda_1$, and candidate configurations of MonoIIF). The configuration selected on a subset is also the best one on the held-out sequences in a minority of the splits for the illumination correction model (12 of 35 with the foundation encoder), and the held-out " + E + r" of the selected configuration exceeds that of the best held-out configuration by 0.0005 on average and by at most 0.006. The configurations of a family differ by at most 0.005 in " + E + r" on the whole test set. The in-domain differences between configurations are thus of the order of the selection effect, no claim of this contribution rests on them, and the effect of the proposed components is assessed on the held-out datasets (" + TAB + r"). Throughout the paper, two methods are said to be \emph{not distinguished} when the 95\% interval of their paired difference includes zero; with four to seven test sequences this is not a statement of equivalence, and the bounds of the intervals indicate the largest difference that the data do not exclude.") + "\n")

# ---------------------------------------------------------------- V2 -> v1
every(r"DA v2 $+$", R("DA v1") + r" $+$", 2)
every(r"DA~v2", R(r"DA~v1"), 2)
one(r"Depth~Anything~v2 adapted", R(r"Depth~Anything~v1") + " adapted")

# ---------------------------------------------------------------- 3.3 implementation
after(r"An ablation justifying the choice of these values is presented in Section~\ref{sec:ablation}.", " " + R(
    r"The configurations retrained with three seeds for " + TAB + r" and Table~\ref{tab:paired-c1} use the default settings of the released training code, which differ from the above on two points: $\lambda_1 = 0.1$, and, unless stated otherwise, the highlight-aware weighting of $\mathcal{L}_\textrm{PML}$ adopted from HADepth~\citep{HAdepth_2025}, which excludes the pixels detected as specular highlights from the photometric loss. This term is not a contribution of this work and its effect is isolated in " + TAB + "."))

# ---------------------------------------------------------------- 3.4 ablation
after(r"five depth-prediction criteria reported in Table~\ref{tab:metrics}.", " " + R(
    r"Since this ablation is carried out on the test images, its role in the choice of $\lambda_1$ and of the configurations is discussed in Section~\ref{sec:protocol}. Lines~1--18 of Table~\ref{tab:ablation} are single training runs; in sweeps repeated with three seeds, $\lambda_1 = 0$ gave the lowest in-domain " + E + r" for both the ResNet-18 and the Depth~Anything encoders, the differences with $\lambda_1 = 0.5$ being at most 0.003, so that the value $\lambda_1 = 0.5$ retained for the three proposed models should not be read as an in-domain optimum."))

# the component table
tab = open(table_path, encoding="utf-8").read()
tab = tab.replace("\\centering", "\\centering\n\\rev", 1)
tab = tab.replace("tab:components-crossdomain", "tab:components-isolated")
tab = tab.replace("All models are trained on SCARED only.",
                  "All models are trained on SCARED only. The configurations use $\\lambda_1=0.1$ unless another value is given; "
                  "``MonoIIF'' denotes the configuration with dense local calibration, $\\mathcal{L}_\\textrm{II}$ and the highlight-aware "
                  "weighting of $\\mathcal{L}_\\textrm{PML}$ adopted from HADepth, and ``EndoDAC'' the same network trained with none of the three. "
                  "The MPViT-S encoder is randomly initialised.")
L2 = "$\\mathcal{L}_\\textrm{II}$"
SHORT = [
    ("pretrained Depth Anything weights (full model vs. random encoder)", "pretrained weights"),
    ("DV-LoRA adapters (EndoDAC vs. frozen encoder with neck)", "DV-LoRA adapters"),
    ("dense local calibration alone, on EndoDAC", "dense calibration alone"),
    ("global calibration alone, on EndoDAC", "global calibration alone"),
    (L2 + " ($\\lambda_1=0.1$) alone, on EndoDAC", L2 + " alone"),
    ("dense calibration $+$ " + L2 + ", on EndoDAC (the two proposed components)", "dense calibration $+$ " + L2),
    ("dense local calibration, inside MonoIIF", "dense calibration, within MonoIIF"),
    (L2 + " ($\\lambda_1=0.1$), inside MonoIIF", L2 + ", within MonoIIF"),
    ("highlight-aware term of HADepth, inside MonoIIF", "highlight weighting, within MonoIIF"),
    ("calibration $+$ " + L2 + " $+$ highlight term (MonoIIF vs. EndoDAC)", "all three (MonoIIF vs.\\ EndoDAC)"),
    ("ResNet-18: dense local calibration alone", "ResNet-18: dense calibration alone"),
    ("ResNet-18: " + L2 + " ($\\lambda_1=0.5$) alone", "ResNet-18: " + L2 + " ($\\lambda_1{=}0.5$) alone"),
    ("ResNet-18: dense calibration $+$ " + L2 + " ($\\lambda_1=0.5$) (MonoII)", "ResNet-18: MonoII (dense calib.\\ $+$ " + L2 + ", $\\lambda_1{=}0.5$)"),
    ("ResNet-18: calibration $+$ " + L2 + " ($\\lambda_1=0.1$) $+$ highlight term", "ResNet-18: all three"),
    ("MPViT-S (random init.): highlight term alone", "MPViT-S: highlight weighting alone"),
    ("MPViT-S (random init.): calibration $+$ " + L2 + ", given the highlight term", "MPViT-S: calibration $+$ " + L2),
]
for a_, b_ in SHORT:
    assert a_ in tab, a_
    tab = tab.replace(a_, b_)
tab = tab.replace("{lcccccccc}", "{@{}lccccccc@{}}").replace("multicolumn{9}", "multicolumn{8}")
tab = re.sub(r", \$p\$=([0-9.]+)", r" (\1)", tab)
tab = tab.replace("$k/n$, $p$ &", "$k/n$ ($p$) &")
tab = tab.replace("Component isolated (with $-$ without)", "Component (with $-$ without)")
tab = tab.replace("\\scriptsize\n", "").replace("\\setlength{\\tabcolsep}{3.5pt}\n", "")
tab = tab.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
tab = tab.replace("\\end{tabular}", "\\end{tabular}\n}", 1)
between(r"\begin{table*}[t] \centering \caption{\textcolor{black}{Contribution of each component",
        r"illumination-invariant loss & $-0.0023$ & $[-0.0081, +0.0017]$ & 3/7 & 0.938 \\ \bottomrule \end{tabular} \end{table*}",
        tab.strip())

after(r"where these errors concentrate in specular and poorly exposed regions.", " " + R(
    r"The contrasts repeated with three seeds do not confirm this single-run difference on SCARED: the two components together change " + E + r" by $+0.0018$ $[-0.0022, +0.0077]$, and by $-0.0005$ $[-0.0012, +0.0004]$ when the highlight-aware weighting is added (" + TAB + ")."))
between(r"Used alone, the correction is therefore a weak signal", r"(line~14 $\rightarrow$ line~11).}", R(
    r"With the foundation encoder, the correction alone does not change the in-domain accuracy either (" + TAB + r": $+0.0000$ $[-0.0018, +0.0022]$ for the dense field and $-0.0012$ $[-0.0028, +0.0002]$ for the global pair), and the two models are not distinguished from each other. The effect of the correction appears outside the training domain and depends on the dataset, as discussed below.") + "}")
one(r"confirming that the pretrained geometric priors of the Depth Anything backbone provide a complementary benefit to the illumination-invariant training strategy.",
    R(r"which reflects the pretrained geometric priors of the Depth Anything backbone, the main source of the accuracy of MonoIIF (" + TAB + ")."))
between(r"Table~\ref{tab:components-isolated-scared} quantifies the contribution", r"which is examined in the generalization tests below.}", R(
    TAB + r" isolates the contribution of each component on the three datasets as a paired difference between two configurations that differ only in that component. On SCARED, the pretrained Depth~Anything weights are by far the dominant factor ($-0.0549$, all seven sequences) and the DV-LoRA adapters add a smaller gain ($-0.0089$, six of seven sequences); both intervals exclude zero. Neither illumination component changes the in-domain accuracy of the foundation model: $+0.0000$ $[-0.0018, +0.0022]$ for the dense calibration alone and $+0.0004$ $[-0.0029, +0.0048]$ for " + LII + r" alone. The in-domain accuracy of MonoIIF is therefore that of the adapted foundation model. Outside the training domain, the two components act differently according to the dataset. On Hamlyn, the calibration alone increases the error by $0.0029$ $[0.0013, 0.0045]$ and " + LII + r", added to the full configuration, is not distinguished from zero ($-0.0015$ $[-0.0029, +0.0001]$). On C3VD, the calibration alone reduces the error by $0.0260$ $[0.0160, 0.0360]$, in all seven scenes, whereas " + LII + r" increases it by $0.0170$ $[0.0070, 0.0270]$. The two proposed components together, without the highlight-aware weighting, are not distinguished from the adapted foundation model on any dataset (one seed). With the highlight-aware weighting of HADepth, the full configuration is more accurate than the adapted foundation model on C3VD ($-0.0250$ $[-0.0315, -0.0190]$, seven of seven scenes) and less accurate on Hamlyn ($+0.0043$ $[+0.0026, +0.0060]$); the row isolating this weighting shows that it accounts for most of the C3VD gain. With the lightweight encoders, the components are more useful on Hamlyn: with ResNet-18, the calibration alone ($-0.0073$ and $-0.0108$ for the dense and global models, one seed) and " + LII + r" alone ($-0.0066$ $[-0.0101, -0.0034]$) reduce the error, and with MPViT-S the two components reduce it by $0.0214$ $[0.0159, 0.0268]$; on C3VD, in contrast, they do not help these encoders. In summary, a consistent benefit of the illumination components across datasets is not established: with a foundation encoder it is limited to the calibration on C3VD, and for the lightweight encoders to Hamlyn.") + "}")

# ---------------------------------------------------------------- results
one(r"MonoIIF should therefore be considered equivalent to the best competing methods on Hamlyn, and to EndoDAC but not to HADepth on C3VD.}",
    R(r"MonoIIF is therefore not distinguished from the best competing methods on Hamlyn, nor from EndoDAC on C3VD, where HADepth is more accurate.") + "}")
one(r"In summary, in terms of $\varepsilon_\textrm{AbsRel}$, MonoIIF is statistically equivalent to EndoDAC on the three datasets and to HADepth on SCARED and Hamlyn, whereas HADepth is more accurate on C3VD; MonoIIF requires less than half of the training time of either method.",
    R(r"In summary, in terms of " + E + r", no significant difference is detected between MonoIIF and EndoDAC on the three datasets, nor between MonoIIF and HADepth on SCARED and Hamlyn, whereas HADepth is more accurate on C3VD; MonoIIF requires less than half of the training time of either method. The absence of a significant difference does not establish equivalence: what the intervals of Table~\ref{tab:paired-c1} exclude is a deficit of MonoIIF larger than 0.0067 (SCARED) and 0.0033 (Hamlyn) with respect to HADepth, and larger than 0.0047 (SCARED), 0.0076 (Hamlyn) and 0.0403 (C3VD) with respect to EndoDAC."))
one(r"MonoIIF consistently produces the lowest error in the ROI regions", R(r"In these five examples, MonoIIF produces the lowest error in the ROI regions"))
between(r"In the C3VD rows, the tubular geometry and uniform mucosal texture present", r"than competing methods.", R(
    r"In the C3VD rows, the tubular geometry and uniform mucosal texture present a strong generalization challenge; in these two examples MonoIIF produces uniform error maps with few boundary artifacts, although HADepth is more accurate on this dataset when all scenes are considered (Table~\ref{tab:paired-c1})."))
one(r"MonoIIF produces the most consistent blue error maps in these regions, \textcolor{black}{consistent with the hypothesis that the illumination-invariant loss $\mathcal{L}_\text{II}$ provides} a supervision signal that generalizes across photometric domains without retraining.",
    R(r"In these examples, MonoIIF produces the most consistent blue error maps in these regions. This observation cannot be attributed to the illumination-invariant loss: with the foundation encoder, its isolated effect on Hamlyn is not distinguished from zero (" + TAB + ")."))
between(r"MonoIIF again achieves the most uniform error distribution", r"strongest foundation-model baselines}.", R(
    r"In the two examples shown, MonoIIF gives a uniform error distribution, with smooth transitions that follow the tubular structure of the colon wall. Over the seven C3VD scenes, however, HADepth is more accurate than MonoIIF, and MonoIIF and EndoDAC are not distinguished (Table~\ref{tab:paired-c1})."))

# ---------------------------------------------------------------- surface section
one(r"Another way to visually assess the accuracy of the proposed models is to use the depth maps they produce for an image to determine the 3D shape of a meshed surface carrying the textures seen in the image.",
    R(r"The depth maps can also be examined visually by using them to build a meshed surface carrying the textures seen in the image, and by checking the continuity of these textures and the presence of geometric discontinuities on the surface. This section is qualitative: its aim is not to measure the accuracy of the surfaces, but to illustrate, on examples, the suitability of the depth maps for 3D cartography."))
one(r"The accuracy of reconstructed surfaces can be visually evaluated by comparing", R(r"The continuity of the reconstructed surfaces can be visually checked by comparing"))
one(r"MonoIIF recovers a more faithful fold geometry", R(r"In this example,") + " MonoIIF recovers a more faithful fold geometry")
one(r"MonoIIF produces a substantially cleaner reconstruction:", R(r"In this example, the reconstruction obtained with MonoIIF is visibly cleaner:"))
one(r"The MonoIIF mosaic (right column) is substantially more geometrically coherent.", R(r"In this example, the MonoIIF mosaic (right column) shows fewer geometric discontinuities."))
one(r"the individual surface patches produced by MonoIIF constitute a substantially more suitable input for downstream 3D cartography algorithms.",
    R(r"its surface patches are visually continuous. EndoDAC, which is not distinguished from MonoIIF in Table~\ref{tab:paired-c1}, is not shown; HADepth was chosen for this comparison because it is the most accurate competing method on SCARED and C3VD."))
between(r"The qualitative superiority of", r"improved 3D mosaicing quality.", R(
    r"The examples of Figs.~\ref{Fig:3D_Single} and~\ref{Fig:3D10_HAMLYM} illustrate this property, which the pixelwise criteria do not measure. They are illustrations on a few images and not a quantitative comparison: they do not establish that MonoIIF yields more accurate surfaces than HADepth, from which it is not distinguished on Hamlyn (Table~\ref{tab:paired-c1}), and the observed differences cannot be attributed to a particular component of the model."))

# ---------------------------------------------------------------- discussion, conclusion
one(r"show that MonoIIF is statistically on par with EndoDAC on the three datasets and with HADepth on SCARED and Hamlyn, whereas HADepth is more accurate on C3VD (Table~\ref{tab:paired-c1}).",
    R(r"show no significant difference between MonoIIF and EndoDAC on the three datasets, nor between MonoIIF and HADepth on SCARED and Hamlyn, whereas HADepth is more accurate on C3VD (Table~\ref{tab:paired-c1}); given the small number of test sequences, this does not establish equivalence."))
between(r"According to the isolated ablation", r"robust to local brightness and contrast changes.}", R(
    r"According to the component analysis of " + TAB + r", the in-domain accuracy stems from the pretrained geometric priors of the Depth Anything foundation model, adapted to the endoscopic domain via DVLoRA; the lighting calibration module and the illumination-invariant loss " + LII + r" do not change it measurably.") + "}")
every(r"working distances up to 300\,mm", "working distances up to " + R(r"100\,mm in the ground truth"), 2)
between(r"Since the illumination components were isolated only on SCARED", r"since the compared methods also differ in training details.}", R(
    r"The component analysis of " + TAB + r" gives the role of the illumination components under these two shifts directly. With the foundation encoder, the calibration is useful under the shift of C3VD (error reduced by 0.026 in all seven scenes, for the dense as well as for the global model) and detrimental on Hamlyn (error increased by 0.003), while " + LII + r" is neutral on Hamlyn and detrimental on C3VD. A possible explanation, which was not tested, is that in the tubular C3VD scenes the intensity fall-off with distance is large and systematic, so that compensating it in $\mathcal{L}_\textrm{PML}$ removes a strong violation of brightness constancy, whereas on Hamlyn, whose scenes are close to the training data, the additional degrees of freedom of the calibration bring no such benefit. With the lightweight encoders, the components improve the transfer to Hamlyn (ResNet-18 and MPViT-S) but not to C3VD. The transferability of MonoIIF thus stems mainly from the pretrained geometric priors of the foundation encoder; the contribution of the illumination components is real but dataset- and encoder-dependent, and a consistent benefit is not established. The limited number of test units (seven sequences for SCARED and C3VD, blocks of a single sequence for Hamlyn in this analysis) and the single seed of some rows call for caution.") + "}")
one(r"However, as demonstrated in Section~\ref{sec:surface},", "However, " + R(r"in the examples of") + r" Section~\ref{sec:surface},")
one(r"This explains why the performance difference between MonoIIF and HADepth is clearly perceptible in the qualitative results of Section~\ref{sec:Experimental-Results}, but is not fully captured by the quantitative metrics of Section~\ref{sec:Experimental-Results}.",
    R(r"The pixelwise criteria therefore do not inform on the continuity of the reconstructed surfaces, which was only examined qualitatively here; a quantitative evaluation of the surfaces is left for future work."))
between(r"This contribution shows that an approach", r"showing the potential of the method.", R(
    r"This contribution proposes a self-supervised depth estimation framework integrating an illumination-invariant loss and a per-pixel lighting calibration, applied to three encoders. With a foundation encoder, no significant difference with the best published methods is detected on SCARED and Hamlyn, at less than half of their training time. The component-wise analysis shows that this accuracy stems mainly from the pretrained encoder, and that the illumination components contribute to the transfer to new acquisition conditions in a dataset- and encoder-dependent way. The depth maps led to reconstructions of endoscopic surfaces illustrating the potential of the method for 3D cartography."))

t = t.replace("tab:components-isolated-scared", "tab:components-isolated")

os.makedirs(out_dir, exist_ok=True)
open(os.path.join(out_dir, "cas-dc-template_r2_marked.tex"), "w", encoding="utf-8", newline="\n").write(t)
clean = t.replace("\\textcolor{red}", "\\textcolor{black}").replace("\\newcommand{\\rev}{\\color{red}}", "\\newcommand{\\rev}{}")
open(os.path.join(out_dir, "cas-dc-template_r2_clean.tex"), "w", encoding="utf-8", newline="\n").write(clean)
print("red blocks:", t.count("\\textcolor{red}"), "| blue left:", t.count("{blue}"),
      "| braces:", t.count("{") - t.count("}"), "(source:", open(src, encoding="utf-8").read().count("{") - open(src, encoding="utf-8").read().count("}"), ")")
for w in ("equivalen", "on par", "superior", "isolated-scared", "v2"):
    print(w, [m.start() for m in re.finditer(w, t)][:6])
