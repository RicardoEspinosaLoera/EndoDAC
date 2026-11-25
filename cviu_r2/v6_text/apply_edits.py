"""Round-2 text-only edits of the CVIU manuscript (version 6). No number, table or figure changes.

Round-1 marks (blue) become black; every new or reworded passage is in red.

    python -B apply_edits.py "<cas-dc-template_new (6).tex>" <out_dir>
"""
import os
import re
import sys

src, out_dir = sys.argv[1], sys.argv[2]
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


def between(start, end, new):
    global t
    pat = rx(start) + r".*?" + rx(end)
    m = re.findall(pat, t, flags=re.S)
    assert len(m) == 1, "%d matches for: %s ... %s" % (len(m), start[:50], end[:40])
    assert len(m[0]) < 1500, "span too long for: %s" % start[:50]
    t = re.sub(pat, lambda _: new, t, flags=re.S)


def after(anchor, new):
    one(anchor, anchor + new)


E = r"$\varepsilon_\textrm{AbsRel}$"
PAIRED = r"Table~\ref{tab:paired-c1}"

# ------------------------------------------------------------------ abstract
between(r"MonoIIF performs \textcolor{black}{on par?} with the strongest",
        r"smaller than the seed-to-seed variability, while requiring",
        R("no significant difference in " + E + " is detected between MonoIIF and")
        + " the strongest foundation-model baselines, HADepth and EndoDAC, "
        + R("the observed differences being") + " smaller than the seed-to-seed variability, "
        + R("while MonoIIF requires"))
between(r"a paired analysis shows that MonoIIF is statistically indistinguishable",
        r"at less than half of their training cost.",
        R(r"a paired analysis detects no significant difference between MonoIIF and HADepth or EndoDAC on Hamlyn, nor between MonoIIF and EndoDAC on C3VD, where HADepth is more accurate; MonoIIF is more accurate than the remaining methods on both datasets. With four to seven test sequences, the absence of a significant difference does not establish equivalence."))
between(r"Qualitative surface reconstruction and 3D mosaicing experiments on the Hamlyn dataset further",
        r"surfaces than those of HADepth.",
        R(r"Qualitative surface reconstruction and 3D mosaicing examples on the Hamlyn dataset illustrate the use of the depth maps for 3D cartography: in these examples, the textured surfaces obtained with MonoIIF are visually coherent and show fewer geometric discontinuities than those obtained with HADepth."))

# ------------------------------------------------------------------ introduction
between(r"As demonstrated in Section~\ref{sec:Experimental-Results}, this strategy yields",
        r"significantly reducing training time.",
        R(r"As shown in Section~\ref{sec:Experimental-Results}, with a foundation encoder this strategy is more accurate than the flow-based baseline, and no significant difference with the strongest foundation-model baselines is detected on SCARED and Hamlyn, at less than half of their training time."))
between(r"Extensive evaluations on SOTA", r"setting a foundation for real-world deployment.",
        R(r"Evaluations on the Hamlyn and C3VD datasets assess the generalization capabilities of the models. Additionally, the predicted depth maps are used for 3D surface reconstructions on the Hamlyn dataset, in order to visually check the continuity of the textures and of the geometry of the reconstructed surfaces."))

# ------------------------------------------------------------------ section 2
one(r"the colors of frame {$\hat{I}_{t}$ to make", r"the colors of frame $\hat{I}_{t}$ to make")

one(r"\paragraph{Transformer-based backbone (MonoIIT).}",
    r"\paragraph{\textcolor{red}{CNN backbone (MonoII).}} " + R(r"The MonoII model uses the convolutional encoder--decoder of Monodepth2 \citep{Godard_ICCV_2019}: a ResNet-18 encoder \citep{He_CVPR_2016} pretrained on ImageNet \citep{Deng_CVPR_2009}, followed by a DispNet-like decoder with skip connections that predicts the disparity maps. The pose estimation module is also based on a ResNet-18 encoder. This lightweight variant allows the two illumination components to be assessed with a purely convolutional depth network (see also Section~\ref{sec:model_architectures}).")
    + "\n%\n" + r"\paragraph{Transformer-based backbone (MonoIIT).}")
one(r"\subsection{MonoIIT, MonoII and MonoIIF model architectures}",
    r"\subsection{MonoIIT, MonoII and MonoIIF model architectures\label{sec:model_architectures}}")

between(r"\caption{Architecture of the transformer-based depth estimation framework. The network begins",
        r"(Conv-steam) that processes",
        r"\caption{" + R("Architecture of MonoIIT.") + " The network begins with a convolutional "
        + R("stem (Conv-stem)") + " that processes")

# ------------------------------------------------------------------ section 3: protocol
one(r"presents the architecture of the two proposed models.",
    "presents the architecture of the " + R("three") + " proposed models.")

between(r"The SCARED dataset is divided into three subsets: 15,351 training images",
        r"the performance of \textcolor{black}{ten} models.",
        R(r"The public split introduced with AF-SfMLearner \citep{Shao_MIA_2022} and also adopted by EndoDAC \citep{EndoDAC_MICCAI_2024} is used without modification.")
        + " The SCARED dataset is divided into three subsets: 15,351 training images, 1,705 validation images, and 551 test images"
        + R(r", the three subsets being taken from different keyframe videos. Only the training images are used to optimize the network weights.")
        + r" The 551 test images" + R(", which belong to seven video sequences,")
        + r" were used for the ablation study described in Section~\ref{sec:ablation} and the quantitative evaluations given in Section~\ref{sec:Experimental-Results} allowing for a fair comparison of the performance of ten models"
        + R(r" under the evaluation protocol of the compared methods. Since the ablation study also served to choose the loss weight $\lambda_1$ and the retained configurations, the consequences of this double use of the test images are stated in Section~\ref{sec:ablation}."))

one(r"The Hamlyn and C3VD datasets were used in the generalization tests detailed in Section~\ref{sec:Experimental-Results}.",
    r"The Hamlyn and C3VD datasets were used in the generalization tests detailed in Section~\ref{sec:Experimental-Results}. "
    + R("These two datasets were used for inference only: no network weight, hyperparameter or model configuration was chosen with them."))

one(r"Five metrics used to assess model performances", R("Metrics") + " used to assess model performances")
one(r"within progressively tighter accuracy bounds.",
    "within progressively " + R("looser") + " accuracy bounds.")

after(r"and the single-run tables should be read with this in mind.",
      " " + R(r"In the paired analyses, a difference is said to be not significant, and the two methods are said not to be distinguished, when the 95\% interval of the paired difference includes zero. No equivalence test was performed: with four to seven test sequences, the absence of a significant difference does not establish that two methods are equivalent, but only that the available test data do not allow them to be separated."))

after(r"to investigate the importance of the different loss components of the proposed model.",
      " " + R(r"The 15,351 training images of the public split were used to optimize the network weights, and the 1,705 validation images were used only to monitor a validation loss during training. The ablation experiments were evaluated on the 551 test images, and their results were used to choose the weight $\lambda_1$ and the three retained configurations. In addition, MonoIIF is trained with the publicly released training code of EndoDAC \citep{EndoDAC_MICCAI_2024}, which evaluates the model on the test split after each epoch and retains the epoch with the lowest $\varepsilon_\textrm{RMSE}$; this checkpoint selection was kept unchanged, so that MonoIIF and EndoDAC are compared under the same training and selection procedure. The test images therefore influenced the model selection, and the SCARED values of the proposed models in Tables~\ref{tab:ablation}, \ref{tab:backbone_comparison} and~\ref{tab:quant_scared} may be slightly optimistic. Two elements limit the consequences of this choice. First, the differences between the candidate configurations are of the order of the seed-to-seed variability (standard deviation of $0.0023$ in $\varepsilon_\textrm{AbsRel}$ for MonoIIF, Section~\ref{sec:experimental-setup}), so that no in-domain conclusion of this work relies on the selected values. Second, the Hamlyn and C3VD datasets were used for inference only: no network weight, hyperparameter or configuration was chosen with them, so that the generalization results are independent of this selection."))

one(r"used in the loss $\mathcal{L}_\textrm{PML}$ weighted by $\lambda_2 = 10^{-3}$ in Eq.~\eqref{eq:global_loss}, and iii)",
    r"used in the loss $\mathcal{L}_\textrm{PML}$ " + R(r"(see Eq.~\eqref{eq:global_loss})") + ", and iii)")

# ------------------------------------------------------------------ section 4: quantitative
one(r"shows that MonoIIF, HADepth, MonoPCC and EndoDAC cannot be statistically distinguished on SCARED;",
    R("detects no significant difference between MonoIIF and HADepth, MonoPCC or EndoDAC on SCARED;"))
one(r"MonoIIF should therefore be considered equivalent to the best competing methods on Hamlyn, and to EndoDAC but not to HADepth on C3VD.",
    R("No significant difference is therefore detected between MonoIIF and the best competing methods on Hamlyn, nor between MonoIIF and EndoDAC on C3VD, where HADepth is more accurate."))
one(r"MonoIIF is statistically equivalent to EndoDAC on the three datasets and to HADepth on SCARED and Hamlyn, whereas",
    R("no significant difference is detected between MonoIIF and EndoDAC on the three datasets, nor between MonoIIF and HADepth on SCARED and Hamlyn, whereas"))
one(r"would require a larger number of test sequences. }",
    "would require a larger number of test sequences.\n")

# ------------------------------------------------------------------ captions of the error-map figures
one(r"MonoIIF consistently produces the lowest error in the ROI regions, particularly",
    R("In these five examples, MonoIIF produces low errors in the ROI regions, particularly"))
between(r"MonoIIF produces the most uniform and low-error depth maps,", r"than competing methods.",
        R(r"in the two examples shown, the error maps of MonoIIF are among the most uniform, although HADepth is more accurate on this dataset as a whole (" + PAIRED + ")."))

# ------------------------------------------------------------------ section 4: qualitative text
one(r"MonoIIF produces the most uniformly low-error maps, with the ROI",
    R("In these examples, MonoIIF produces uniformly low-error maps, with the ROI"))
one(r"the advantage of illumination-aware methods becomes even more pronounced.",
    R("the differences between the three categories of methods remain visible."))
between(r"MonoIIF produces the most consistent blue error maps in these regions,",
        r"photometric domains without retraining.",
        R(r"In the two Hamlyn examples, MonoIIF produces mostly blue error maps in these regions; these examples alone do not allow this behaviour to be attributed to a particular component of the model."))
between(r"MonoIIF again achieves the most uniform error distribution,",
        r"that of the strongest foundation-model baselines}.",
        R(r"In the two examples shown, MonoIIF gives a uniform error distribution, with smooth transitions that follow the tubular structure of the colon wall. These examples are illustrative: over the whole C3VD dataset, HADepth is more accurate than MonoIIF, and no significant difference is detected between MonoIIF and EndoDAC (" + PAIRED + ")."))

# ------------------------------------------------------------------ section 4.3: surfaces
between(r"Another way to visually assess the accuracy of the proposed models", r"textures seen in the image.",
        R(r"The depth maps produced by a model for an image can also be inspected visually by using them to determine the 3D shape of a meshed surface carrying the textures seen in the image. The aim of this section is not to measure the accuracy of the reconstructed surfaces, but to visually check the continuity of their textures and the presence of geometric discontinuities."))
one(r"The accuracy of reconstructed surfaces can be visually evaluated by comparing 3D textured surfaces with their ground truth, shown",
    R("The reconstructed 3D textured surfaces can be visually compared with the surfaces obtained with the ground truth depths, shown"))
between(r"MonoIIF recovers a more faithful fold geometry", r"visible in the ground truth surface.",
        R("In this example, the surface obtained with MonoIIF shows smoother depth transitions at the fold boundaries, and the shape of the fold is visually closer to that of the ground truth surface."))
between(r"indicating that the depth predictions in this region contain systematic errors",
        r"bleeding artifacts near specular highlights.",
        R("which suggests that the depth predictions in this region contain errors that propagate into the mesh. In this example, the reconstruction obtained with MonoIIF is visually cleaner: the lower edge is free of such discontinuities, the global shape is visually closer to that of the ground truth surface, and the texture of the mucosal surface remains visually coherent, without bleeding artifacts near specular highlights."))
between(r"also correlates with surface distortions, suggesting that", r"coherent multi-frame fusion.",
        "also correlates with surface distortions, "
        + R("which may indicate that the depth estimates of HADepth in this photometrically degraded area are less suited to multi-frame fusion."))
one(r"The MonoIIF mosaic (right column) is substantially more geometrically coherent.",
    R("In this example, the MonoIIF mosaic (right column) shows fewer geometric discontinuities and a visually coherent texture."))
between(r"the individual surface patches produced by MonoIIF constitute", r"downstream 3D cartography algorithms.",
        R("the individual surface patches obtained with MonoIIF in this sequence appear to be a suitable input for downstream 3D cartography algorithms."))
between(r"The qualitative superiority of", r"improved 3D mosaicing quality.",
        R(r"The examples of Figs.~\ref{Fig:3D_Single} and~\ref{Fig:3D10_HAMLYM} therefore illustrate a property that the pixelwise criteria do not measure, namely the continuity of the textures and of the geometry of the reconstructed surfaces. They are visual illustrations on a few images and not a quantitative comparison: they do not establish that the surfaces obtained with MonoIIF are more accurate than those obtained with HADepth, and the observed differences cannot be attributed to a particular component of the model."))

# ------------------------------------------------------------------ discussion
one(r"show that MonoIIF is statistically on par with EndoDAC on the three datasets and with HADepth on SCARED and Hamlyn, whereas",
    "show that " + R("no significant difference is detected between MonoIIF and EndoDAC on the three datasets, nor between MonoIIF and HADepth on SCARED and Hamlyn, whereas"))
between(r"However, as demonstrated in Section~\ref{sec:surface}, HADepth and MonoIIF", r"for 3D surface reconstruction.",
        R(r"However, in the examples of Section~\ref{sec:surface}, the surfaces reconstructed with the depth maps of HADepth and of MonoIIF differ visually."))
between(r"This explains why the performance difference between MonoIIF and HADepth",
        r"quantitative metrics of Section~\ref{sec:Experimental-Results}.",
        R(r"This may explain why differences between the surfaces obtained with MonoIIF and with HADepth are visible in the examples of Section~\ref{sec:surface}, whereas the quantitative criteria do not separate the two methods on Hamlyn. A quantitative evaluation of the reconstructed surfaces, which was not the aim of this work, would be required to confirm this observation."))
one(r"as reflected in the depth error effects visible in Section~\ref{sec:surface}.",
    "as reflected in the depth error effects visible in "
    + R(r"Figs.~\ref{fig:qual_scared} and~\ref{fig:qual_gen}."))
one(r"a prerequisite that the proposed MonoIIF framework is well positioned to fulfill.",
    "a prerequisite " + R("to which the proposed framework aims to contribute."))

one(r"while achieving \textcolor{black}{comparable} accuracy, and MonoIIT",
    "while achieving " + R("an accuracy that is not significantly different from theirs on SCARED and Hamlyn") + ", and MonoIIT")

# ------------------------------------------------------------------ conclusion
between(r"This contribution shows that an approach integrating an illumination-invariant loss",
        r"showing the potential of the method.",
        R(r"This contribution proposes a self-supervised framework that integrates an illumination-invariant loss and a per-pixel lighting calibration mechanism into three depth estimation architectures (CNN, hybrid CNN--transformer and adapted foundation model), all trained end-to-end in a single run and without auxiliary pretrained subnetworks. On recognized benchmarks of the depth prediction community, the foundation-model variant, MonoIIF, reaches the accuracy level of the strongest foundation-model baselines: no significant difference with HADepth and EndoDAC is detected on SCARED and Hamlyn, HADepth remaining more accurate on C3VD, while MonoIIF requires less than half of their training time. MonoIIF is also more accurate than the flow-based and brightness-constancy methods on the two generalization datasets, and the lightweight MonoII variant, trained in about two hours, reaches on the Hamlyn sequence of Table~\ref{tab:quant_hamlyn} an accuracy close to that of the best foundation-model baseline. Most of the accuracy of MonoIIF comes from the pretrained encoder and its adaptation; the illumination components act as a refinement, whose clearest effect is the improved transfer to colonoscopic scenes brought by the lighting calibration (all seven C3VD scenes). The depth maps led to reconstructions of endoscopic surfaces whose textures and geometry are visually coherent in the examples shown, which makes the framework a promising and computationally affordable basis for 3D cartography in endoscopy."))

# ------------------------------------------------------------------ cross-domain component table (R2, point 1)
XT = r"Table~\ref{tab:components-crossdomain}"
table_tex = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "components_crossdomain_v6.tex"),
                 encoding="utf-8").read()
PAIRED_HEAD = r"\begin{table*}[t] \centering \caption{\textcolor{black}{Uncertainty and paired comparisons"
one(PAIRED_HEAD, table_tex + "\n%\n" + "\\begin{table*}[t]\n\\centering\n\\caption{\\textcolor{black}{Uncertainty and paired comparisons")

FIG_HEAD = r"\begin{figure*} \centering \includegraphics[width=\linewidth]{MonoIIF - SCARED.png}"
one(FIG_HEAD,
    R(r"\noindent{\bf Isolated contribution of the illumination components outside the training domain.} "
      + XT + r" isolates the two illumination components on Hamlyn and C3VD, by inference only, with models trained on SCARED: each row compares two configurations that differ only in the component considered. With the foundation encoder, the dense lighting calibration lowers " + E + r" on C3VD in all seven scenes ($-0.0260$), whereas it has no measurable effect on the four Hamlyn sequences ($+0.0002$). The loss $\mathcal{L}_\textrm{II}$ with $\lambda_1=0.5$ brings no measurable benefit with this encoder on either dataset ($+0.0001$ on Hamlyn; $-0.0034$ on C3VD, four of seven scenes, not significant). With the ResNet-18 encoder, the differences on Hamlyn are slightly favourable for the calibration alone ($-0.0013$), for $\mathcal{L}_\textrm{II}$ alone ($-0.0012$) and for their combination (MonoII, $-0.0017$), but none of them is significant with four sequences; on C3VD, the calibration alone and the combination increase the error ($+0.0149$ and $+0.0100$), and the effect of $\mathcal{L}_\textrm{II}$ alone ($+0.0084$) is not significant. The cross-domain benefit of the illumination components is therefore not consistent: the only significant gain is that of the lighting calibration on C3VD with the foundation encoder. These contrasts rely on a limited number of test units, in particular on four sequences for Hamlyn, and should be read as exploratory.")
    + "\n\n" + "\\begin{figure*}\n    \\centering\n    \\includegraphics[width=\\linewidth]{MonoIIF - SCARED.png}")

one(r"which is examined in the generalization tests below.}",
    r"which is examined in the generalization tests below " + R("(" + XT + "); as shown there, this expectation is only partly confirmed.") + "}")

after(r"and the reported values are averaged over these seeds.",
      " " + R("The number of training seeds of each configuration of " + XT + " is given in that table."))

between(r"Since the illumination components were isolated only on SCARED",
        r"with the published method closest to it.",
        R("The illumination components were isolated directly on these two datasets in " + XT + ", which shows no consistent cross-domain benefit. This direct analysis can be complemented by comparing each proposed variant with the published method closest to it."))
after(r"This evidence remains indirect, since the compared methods also differ in training details.",
      " " + R("The direct contrasts of " + XT + r" refine this picture: $\mathcal{L}_\textrm{II}$ brings no measurable benefit with the foundation encoder, the lighting calibration improves the transfer to C3VD with this encoder, and, with the ResNet-18 encoder, the differences on Hamlyn are favourable but not significant, so that the transfer benefit suggested for lightweight encoders by the indirect comparison is not confirmed by the direct one."))

one(r"the absence of a significant difference does not establish equivalence.}",
    r"the absence of a significant difference does not establish equivalence.} "
    + R("A component-wise analysis on Hamlyn and C3VD shows that the lighting calibration improves the transfer to C3VD but not to Hamlyn, and that the illumination-invariant loss brings no measurable cross-domain benefit with the foundation encoder."))

# ------------------------------------------------------------------ relative change of the three proposed models (from Table 2)
REL = r"Table~\ref{tab:components-relative}"
rel_table = "\n".join([
    r"\begin{table}[t]",
    r"\centering",
    r"\color{red}",
    r"\caption{\textcolor{red}{Change brought by the illumination components to the three proposed models on SCARED, computed from the values of Table~\ref{tab:ablation}: each model is compared with the configuration of the same backbone given in the second column. Relative change for the four error criteria (negative: lower error) and absolute change for $\delta_{1.25}$. Each configuration corresponds to a single training run.}}",
    r"\label{tab:components-relative}",
    r"\resizebox{\columnwidth}{!}{%",
    r"\begin{tabular}{llccccc}",
    r"\toprule",
    r"Model & Reference configuration & $\varepsilon_\text{AbsRel}$ & $\varepsilon_\text{SqRel}$ & $\varepsilon_\text{RMSE}$ & $\varepsilon_\text{RMSELog}$ & $\delta_{1.25}$ \\",
    r"\midrule",
    r"MonoII  & line~2 (ResNet-18, without $\mathcal{L}_\textrm{II}$) & $-6.5\%$ & $-9.9\%$ & $-4.8\%$ & $-2.4\%$ & $-0.002$ \\",
    r"MonoIIT & line~13 (MPViT-S, without both components) & $-6.8\%$ & $-17.3\%$ & $-8.3\%$ & $-8.3\%$ & $+0.007$ \\",
    r"MonoIIF & line~15 (DA v2 $+$ DV-LoRA, without both components) & $-3.8\%$ & $-15.6\%$ & $-8.7\%$ & $-6.7\%$ & $+0.003$ \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"}",
    r"\end{table}",
]) + "\n"

ABL_FIG = r"\begin{figure}[] \centering \includegraphics[scale=.23]{ablation_image.png}"
one(ABL_FIG,
    rel_table + "%\n"
    + R(r"\noindent{\bf Summary for the three proposed models.} " + REL
        + r" summarizes, from the values of Table~\ref{tab:ablation}, the change brought by the illumination components to each of the three proposed models with respect to the same backbone trained without them (without $\mathcal{L}_\textrm{II}$ only for the ResNet-18 backbone, since line~2 keeps the lighting correction). The pattern is the same for the three backbones: the squared-error criteria decrease by 10\% to 17\% for $\varepsilon_\textrm{SqRel}$ and by 5\% to 9\% for $\varepsilon_\textrm{RMSE}$, whereas " + E
        + r" decreases by 0.002 to 0.004, which is of the order of the seed-to-seed variability. Since each configuration corresponds to a single training run, these reductions indicate a trend that is consistent across the three backbones rather than an established effect; they suggest that the two components mainly act on the largest depth errors.")
    + "\n\n" + "\\begin{figure}[]\n    \\centering\n    \\includegraphics[scale=.23]{ablation_image.png}")

# ------------------------------------------------------------------ output
os.makedirs(out_dir, exist_ok=True)
open(os.path.join(out_dir, "cas-dc-template_r2_marked.tex"), "w", encoding="utf-8", newline="\n").write(t)
clean = t.replace("\\textcolor{red}", "\\textcolor{black}")
open(os.path.join(out_dir, "cas-dc-template_r2_clean.tex"), "w", encoding="utf-8", newline="\n").write(clean)


def balance(s):
    s = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", s, flags=re.S)
    s = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in s.split("\n"))
    s = s.replace("\\{", "").replace("\\}", "")
    return s.count("{") - s.count("}")


print("red blocks:", t.count("\\textcolor{red}"), "| blue left:", t.count("{blue}"),
      "| brace balance:", balance(t), "(source:", balance(open(src, encoding="utf-8").read()), ")")
for w in ("equivalen", "on par", "superior", "indistinguish", "comparabl", "confirm", "demonstrat"):
    print(w, [t[max(0, m.start() - 40):m.start() + 40].replace("\n", " ") for m in re.finditer(w, t)])
