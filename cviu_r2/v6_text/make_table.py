"""Cross-domain component table (Hamlyn, C3VD) for the round-2 manuscript, from stored inference.

    python -B make_table.py <repo_root> <out.tex>

Rows are (with the component) - (without it) on Abs Rel, from results/cviu/per_frame.csv.
"""
import os
import sys

import numpy as np

root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "tools"))
import components_crossdomain as m  # noqa: E402

LII = r"$\mathcal{L}_\textrm{II}$"
BLOCKS = [
    ("Depth~Anything $+$ DV-LoRA encoder (MonoIIF)", [
        ("Lighting calibration alone", "E4", "E3"),
        (LII + r" ($\lambda_1=0.5$) added to the calibration$^{\dagger}$", "da-lam-050", "da-bas-1"),
        (LII + r" ($\lambda_1=0.1$) added to the calibration$^{\dagger}$", "da-lam-010", "da-bas-1"),
        (LII + r" ($\lambda_1=0.1$) added to the dense calibration", "E7", "E4"),
        (LII + r" ($\lambda_1=0.1$) alone", "E5", "E3"),
    ]),
    ("ResNet-18 encoder (MonoII)", [
        ("Lighting calibration alone", "cal0-local", "R1"),
        (LII + r" ($\lambda_1=0.5$) alone", "MonoII-none", "R1"),
        ("Calibration $+$ " + LII + r" ($\lambda_1=0.5$) (MonoII)", "MonoII", "R1"),
    ]),
]
DATASETS = ["hamlyn", "c3vd"]

runs = {r for _, rows in BLOCKS for _, a, b in rows for r in (a, b)}
rng = np.random.default_rng(0)
res, seeds, units = {}, {}, {}
for ds in DATASETS:
    acc = m.load(os.path.join(root, "results", "cviu", "per_frame.csv"), ds, runs)
    for (meth, s, _, _) in acc:
        seeds.setdefault((ds, meth), set()).add(s)
    by, unit, nseq = m.per_unit(acc, 100)
    units[ds] = (unit, len(next(iter(by.values()))))
    for _, rows in BLOCKS:
        for lab, a, b in rows:
            res[(a, b, ds)] = m.contrast(by, a, b, rng, 10000)


def num(x):
    s = "%+.4f" % x
    return "0.0000" if s in ("+0.0000", "-0.0000") else s


caption = (
    r"Isolated contribution of the two illumination components on the generalization datasets "
    r"(inference only, models trained on SCARED). $\Delta$ is the paired difference in "
    r"$\varepsilon_\textrm{AbsRel}$ between two configurations that differ only in the component, "
    r"(with the component) $-$ (without it): a \textbf{negative} value means that the component helps. "
    r"CI: 95\% cluster bootstrap interval; $k/n$: units in which the component lowers "
    r"$\varepsilon_\textrm{AbsRel}$. Units: HAMUNIT for "
    r"Hamlyn ($n=NHAM$) and scenes for C3VD ($n=NC3VD$, with the exact two-sided Wilcoxon signed-rank $p$); "
    r"same evaluation pipeline as Table~\ref{tab:paired-c1}. "
    r"Seeds: training seeds averaged with/without the component. Bold: interval excluding "
    r"zero. $^{\dagger}$Both configurations use a spatially varying affine calibration field restricted to a "
    r"first-order polynomial of the pixel coordinates."
).replace("NC3VD", str(units["c3vd"][1])).replace("NHAM", str(units["hamlyn"][1])).replace(
    "HAMUNIT", "blocks of 100 consecutive frames of sequence rectified14" if units["hamlyn"][0].startswith("block")
    else "the rectified sequences of Table~\\ref{tab:paired-c1}")
if not units["hamlyn"][0].startswith("block"):
    caption = caption.replace("Seeds: training seeds",
                              "With four sequences, the Hamlyn intervals are only indicative. "
                              "Seeds: training seeds")

# every run of a contrast must be evaluated on the same Hamlyn units, otherwise the pairing is empty
for (a, b, ds), c in res.items():
    assert c is not None and c[4] == units[ds][1], "run pair %s/%s has %s units on %s, expected %d" % (
        a, b, None if c is None else c[4], ds, units[ds][1])

L = []
L.append(r"\begin{table*}[t]")
L.append(r"\centering")
L.append(r"\color{red}")
L.append(r"\caption{\textcolor{red}{" + caption + "}}")
L.append(r"\label{tab:components-crossdomain}")
L.append(r"\small")
L.append(r"\setlength{\tabcolsep}{5pt}")
L.append(r"\begin{tabular}{l rcc rccc c}")
L.append(r"\toprule")
L.append(r" & \multicolumn{3}{c}{Hamlyn} & \multicolumn{4}{c}{C3VD} & \\")
L.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}")
L.append(r"Component isolated & \multicolumn{1}{c}{$\Delta$} & 95\% CI & $k/n$ & "
         r"\multicolumn{1}{c}{$\Delta$} & 95\% CI & $k/n$ & $p$ & Seeds \\")
for title, rows in BLOCKS:
    L.append(r"\midrule")
    L.append(r"\multicolumn{9}{l}{\emph{%s}} \\" % title)
    for lab, a, b in rows:
        cells = []
        for ds in DATASETS:
            mean, lo, hi, k, n, p = res[(a, b, ds)]
            s = num(mean)
            if lo > 0 or hi < 0:
                s = r"\mathbf{%s}" % s
            cells += ["$%s$" % s, "$[%s,\\ %s]$" % (num(lo), num(hi)), "%d/%d" % (k, n)]
            if ds == "c3vd":
                cells.append("%.3f" % p)
        sa = min(len(seeds[(ds, a)]) for ds in DATASETS)
        sb = min(len(seeds[(ds, b)]) for ds in DATASETS)
        L.append(r"\quad %s & %s & %d/%d \\" % (lab, " & ".join(cells), sa, sb))
L.append(r"\bottomrule")
L.append(r"\end{tabular}")
L.append(r"\end{table*}")
open(out, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("units:", units)
