"""Isolated contribution of each component on every evaluation dataset (CVIU round 2, R2 (1)).

Each row is the paired difference in Abs Rel per unit between the two grid runs that differ only
in that component, (with the component) - (without it), so a NEGATIVE value means the component
helps. The unit is the test sequence when a dataset has at least three of them, otherwise
contiguous blocks of --block frames of its single sequence (same rule as hamlyn_table.py and the
`stats` stage), so the table regenerates itself with sequences as units once the Hamlyn
rectified06/08/11 evaluations of the grid runs are in per_frame.csv.

    python tools/components_crossdomain.py                       # LaTeX to results/cviu/tables/
    python tools/components_crossdomain.py --markdown            # also print a markdown version
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hamlyn_table import load, per_unit, wilcoxon  # noqa: E402

# (label, run WITH the component, run WITHOUT it, note).  Flags are those of cviu_revision.py.
DA_ROWS = [
    ("pretrained Depth Anything weights (full model vs. random encoder)", "E8", "N0", ""),
    ("DV-LoRA adapters (EndoDAC vs. frozen encoder with neck)", "E3", "E2", ""),
    ("dense local calibration alone, on EndoDAC", "E4", "E3", ""),
    ("global calibration alone, on EndoDAC", "da-bas-0", "E3", ""),
    ("$\\mathcal{L}_\\textrm{II}$ ($\\lambda_1=0.1$) alone, on EndoDAC", "E5", "E3", ""),
    ("dense calibration $+$ $\\mathcal{L}_\\textrm{II}$, on EndoDAC (the two proposed components)", "E7", "E3", ""),
    ("dense local calibration, inside MonoIIF", "E8", "C0", ""),
    ("$\\mathcal{L}_\\textrm{II}$ ($\\lambda_1=0.1$), inside MonoIIF", "E8", "E8-IIF", ""),
    ("highlight-aware term of HADepth, inside MonoIIF", "E8", "E7", ""),
    ("calibration $+$ $\\mathcal{L}_\\textrm{II}$ $+$ highlight term (MonoIIF vs. EndoDAC)", "E8", "E3", ""),
]
LIGHT_ROWS = [
    ("ResNet-18: dense local calibration alone", "cal0-local", "R1", ""),
    ("ResNet-18: global calibration alone", "cal0-glob", "R1", ""),
    ("ResNet-18: $\\mathcal{L}_\\textrm{II}$ ($\\lambda_1=0.5$) alone", "MonoII-none", "R1", ""),
    ("ResNet-18: dense calibration $+$ $\\mathcal{L}_\\textrm{II}$ ($\\lambda_1=0.5$) (MonoII)", "MonoII", "R1", ""),
    ("ResNet-18: calibration $+$ $\\mathcal{L}_\\textrm{II}$ ($\\lambda_1=0.1$) $+$ highlight term", "R2", "R1", ""),
    ("MPViT-S (random init.): highlight term alone", "MonoIIT-base", "MonoIIT-repro", ""),
    ("MPViT-S (random init.): calibration $+$ $\\mathcal{L}_\\textrm{II}$, given the highlight term", "MonoIIT-components", "MonoIIT-base", ""),
]
DATASETS = [("scared", "SCARED"), ("hamlyn", "Hamlyn"), ("c3vd", "C3VD")]


def contrast(by, with_run, without_run, rng, boot):
    if with_run not in by or without_run not in by:
        return None
    units = sorted(set(by[with_run]) & set(by[without_run]))
    d = np.array([by[with_run][u] - by[without_run][u] for u in units])
    m = d[rng.integers(0, len(d), size=(boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(m, 2.5), np.percentile(m, 97.5)
    return d.mean(), lo, hi, int((d < 0).sum()), len(d), wilcoxon(d)


def fmt_cell(c, with_p):
    if c is None:
        return "---", "---"
    mean, lo, hi, k, n, p = c
    s = "%+.4f" % mean
    if lo > 0 or hi < 0:
        s = r"\mathbf{%s}" % s
    cell = "$%s$ $[%+.4f, %+.4f]$" % (s, lo, hi)
    kn = "%d/%d" % (k, n)
    if with_p and p == p:
        kn += ", $p$=%.3f" % p
    return cell, kn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/cviu/per_frame.csv")
    ap.add_argument("--out", default="results/cviu/tables/components_crossdomain.tex")
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = DA_ROWS + LIGHT_ROWS
    runs = {r for _, w, wo, _ in rows for r in (w, wo)}
    per_ds = {}
    for ds, _ in DATASETS:
        acc = load(a.csv, ds, runs)
        seeds = {}
        for (m, s, _, _) in acc:
            seeds.setdefault(m, set()).add(s)
        by, unit, nseq = per_unit(acc, a.block)
        per_ds[ds] = (by, unit, nseq, {m: len(v) for m, v in seeds.items()})

    def seeds_of(run):
        return min(per_ds[ds][3].get(run, 0) for ds, _ in DATASETS)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    units_txt = "; ".join("%s: %s (%d)" % (lab, per_ds[ds][1], len(next(iter(per_ds[ds][0].values()))))
                          for ds, lab in DATASETS)
    lines.append(r"\caption{Isolated contribution of each component on the three evaluation datasets. "
                 r"Each row is the paired difference in $\varepsilon_\textrm{AbsRel}$ per unit of analysis between the two "
                 r"configurations that differ only in that component, computed as (with the component) $-$ (without it), "
                 r"so that a \textbf{negative} value means the component helps, with its 95\% cluster bootstrap interval "
                 r"(10\,000 draws); $k/n$ is the number of units in which the component helps, and $p$ the exact two-sided "
                 r"Wilcoxon signed-rank test where $n \le 18$ (its smallest attainable value is $2/2^{7}=0.016$ for $n=7$). "
                 r"Bold marks intervals excluding zero. All models are trained on SCARED only. Units: " + units_txt + ". "
                 r"The last column gives the number of training seeds averaged (the smaller of the two configurations).}")
    lines.append(r"\label{tab:components-crossdomain}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{SCARED} & \multicolumn{2}{c}{Hamlyn} & \multicolumn{2}{c}{C3VD} & \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(r"Component isolated (with $-$ without) & $\Delta\varepsilon_\textrm{AbsRel}$ [95\% CI] & $k/n$, $p$ & "
                 r"$\Delta\varepsilon_\textrm{AbsRel}$ [95\% CI] & $k/n$ & $\Delta\varepsilon_\textrm{AbsRel}$ [95\% CI] & $k/n$, $p$ & seeds \\")
    md = ["| Component (with - without) | SCARED | k/n, p | Hamlyn | k/n | C3VD | k/n, p | seeds |",
          "|---|---|---|---|---|---|---|---|"]

    def emit(block_title, block_rows):
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{9}{l}{\emph{%s}} \\" % block_title)
        md.append("| **%s** | | | | | | | |" % block_title)
        for label, w, wo, _ in block_rows:
            cells = []
            mdc = []
            for ds, _ in DATASETS:
                c = contrast(per_ds[ds][0], w, wo, rng, a.boot)
                cell, kn = fmt_cell(c, with_p=(ds != "hamlyn"))
                cells += [cell, kn]
                if c is None:
                    mdc += ["---", "---"]
                else:
                    mean, lo, hi, k, n, p = c
                    star = "*" if (lo > 0 or hi < 0) else ""
                    mdc += ["%+.4f [%+.4f, %+.4f]%s" % (mean, lo, hi, star),
                            "%d/%d" % (k, n) + ((", p=%.3f" % p) if (ds != "hamlyn" and p == p) else "")]
            s = min(seeds_of(w), seeds_of(wo))
            lines.append("%s & %s & %d \\\\" % (label, " & ".join(cells), s))
            md.append("| %s | %s | %d |" % (label.replace("$", "").replace("\\mathcal{L}_\\textrm{II}", "L_II")
                                            .replace("\\lambda_1", "lambda1").replace("$+$", "+"),
                                            " | ".join(mdc), s))

    emit("Depth Anything v1 $+$ DV-LoRA backbone (MonoIIF)", DA_ROWS)
    emit("Lightweight backbones", LIGHT_ROWS)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", a.out)
    if a.markdown:
        print()
        print("\n".join(md))


if __name__ == "__main__":
    main()
