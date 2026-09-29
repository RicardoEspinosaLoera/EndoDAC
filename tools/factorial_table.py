"""2x2x2 factorial of the method's photometric components on the EndoDAC backbone.

Factors: C = illumination calibration, I = illumination-invariant loss (lambda1 = 0.1),
H = highlight-aware photometric term. For every dataset the script reports the eight cells, the
main effect of each factor (the change it produces averaged over the four settings of the other
two), the two-way interactions, and the full model against the references that matter. Every
effect is computed per test sequence and bootstrapped over sequences, as in the paired tables.

    python tools/factorial_table.py                  # the degree-2 design, II cells with 8 directions
    python tools/factorial_table.py --design local   # the same design with the dense map (existing runs)
"""
import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hamlyn_table import load, per_unit, wilcoxon  # noqa: E402

# (C, I, H) -> run
DESIGNS = {
    "deg2": {(0, 0, 0): "E3", (0, 0, 1): "E6", (0, 1, 0): "f8-ii", (0, 1, 1): "f8-ii-hl",
             (1, 0, 0): "da-bas-2", (1, 0, 1): "f8-cal-hl", (1, 1, 0): "f8-cal-ii", (1, 1, 1): "f8-full"},
    "local": {(0, 0, 0): "E3", (0, 0, 1): "E6", (0, 1, 0): "E5", (0, 1, 1): "C0",
              (1, 0, 0): "E4", (1, 0, 1): "E8-IIF", (1, 1, 0): "E7", (1, 1, 1): "E8"},
}
CALIB = {"deg2": "basis, degree 2", "local": "dense map"}
REFS = [("E3", "EndoDAC (retrained)"), ("C1", "global calib. + II + highlight"),
        ("E8-IIF", "dense calib. + highlight, no II"), ("HADepth", "HADepth"),
        ("EndoDAC_MICCAI", "EndoDAC (released)")]
DATASETS = [("scared", "SCARED"), ("hamlyn", "Hamlyn"), ("c3vd", "C3VD")]
FACT = {"C": 0, "I": 1, "H": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/cviu/per_frame.csv")
    ap.add_argument("--design", default="deg2", choices=sorted(DESIGNS))
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    cells = DESIGNS[a.design]
    runs = set(cells.values()) | {m for m, _ in REFS}
    rng = np.random.default_rng(0)

    def boot(x):
        i = rng.integers(0, len(x), size=(a.boot, len(x)))
        m = x[i].mean(axis=1)
        return np.percentile(m, 2.5), np.percentile(m, 97.5)

    def fmt(d, n_units):
        lo, hi = boot(d)
        s = "%+.4f" % d.mean()
        if lo > 0 or hi < 0:
            s = r"\mathbf{%s}" % s
        p = wilcoxon(d)
        out = "$%s$ $[%+.4f, %+.4f]$, %d/%d" % (s, lo, hi, int((d < 0).sum()), n_units)
        return out + ((", $p$=%.3f" % p) if p == p else "")

    results = {}
    for ds, label in DATASETS:
        acc = load(a.csv, ds, runs)
        seqs = collections.defaultdict(set)
        seeds = collections.defaultdict(set)
        for (m, s, seq, _) in acc:
            seqs[m].add(seq)
            seeds[m].add(s)
        missing = [r for r in cells.values() if r not in seqs]
        counts = {r: len(seqs[r]) for r in cells.values() if r in seqs}
        if missing:
            print("%% %s: cells not evaluated yet, skipped: %s" % (label, ", ".join(missing)))
            continue
        if len(set(counts.values())) > 1:
            print("%% %s: WARNING cells evaluated on different sequence sets %s -- re-run "
                  "`predict --datasets %s --force` for the cells with fewer; skipped" % (label, counts, ds))
            continue
        by, unit, nseq = per_unit(acc, a.block)
        units = sorted(set.intersection(*[set(by[r]) for r in cells.values()]))
        V = {k: np.array([by[r][u] for u in units]) for k, r in cells.items()}
        results[ds] = dict(label=label, unit=unit, n=len(units), V=V, by=by, units=units,
                           seeds={r: len(seeds[r]) for r in cells.values()})

    if not results:
        return

    # --- the eight cells ---------------------------------------------------------------------
    C, X = r"\cmark", r"\xmark"
    print()
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Factorial ablation of the photometric components on the EndoDAC backbone "
          r"(Depth~Anything v1 $+$ DV-LoRA $+$ convolutional neck): illumination calibration (%s), "
          r"illumination-invariant loss $\mathcal{L}_\textrm{II}$ ($\lambda_1=0.1$, eight Robinson "
          r"directions) and the highlight-aware photometric term. Abs Rel averaged within each test "
          r"sequence, then across sequences and training seeds; best per column in bold, second "
          r"underlined.}" % CALIB[a.design])
    print(r"\label{tab:factorial-%s}" % a.design)
    print(r"\small")
    print(r"\begin{tabular}{ccc%sc}" % ("c" * len(results)))
    print(r"\toprule")
    print("Calib. & $\\mathcal{L}_\\textrm{II}$ & Highlight & " +
          " & ".join(results[d]["label"] for d in results) + r" & seeds \\")
    print(r"\midrule")
    keys = sorted(cells)
    means = {d: {k: results[d]["V"][k].mean() for k in keys} for d in results}
    marks = {}
    for d in results:
        o = sorted(keys, key=lambda k: means[d][k])
        marks[d] = {o[0]: r"\textbf{%s}", o[1]: r"\underline{%s}"}
    for k in keys:
        cols = [marks[d].get(k, "%s") % ("%.4f" % means[d][k]) for d in results]
        sd = min(results[d]["seeds"][cells[k]] for d in results)
        print("%s & %s & %s & %s & %d \\\\" % (C if k[0] else X, C if k[1] else X, C if k[2] else X,
                                             " & ".join(cols), sd))
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # --- effects -----------------------------------------------------------------------------
    def main_effect(V, f):
        i = FACT[f]
        pairs = [(k, tuple(1 if j == i else k[j] for j in range(3))) for k in V if k[i] == 0]
        return np.mean([V[on] - V[off] for off, on in pairs], axis=0)

    def interaction(V, f, g):
        """(effect of f with g on) - (effect of f with g off), averaged over the third factor."""
        i, j = FACT[f], FACT[g]
        t = 3 - i - j
        out = []
        for lt in (0, 1):
            def cell(li, lj):
                k = [0, 0, 0]
                k[i], k[j], k[t] = li, lj, lt
                return V[tuple(k)]
            out.append((cell(1, 1) - cell(0, 1)) - (cell(1, 0) - cell(0, 0)))
        return np.mean(out, axis=0)

    rows = [("calibration (main effect)", lambda V: main_effect(V, "C")),
            (r"$\mathcal{L}_\textrm{II}$ (main effect)", lambda V: main_effect(V, "I")),
            ("highlight term (main effect)", lambda V: main_effect(V, "H")),
            (r"calib. $\times$ $\mathcal{L}_\textrm{II}$", lambda V: interaction(V, "C", "I")),
            (r"calib. $\times$ highlight", lambda V: interaction(V, "C", "H")),
            (r"$\mathcal{L}_\textrm{II}$ $\times$ highlight", lambda V: interaction(V, "I", "H")),
            ("all three vs none (full $-$ EndoDAC)", lambda V: V[(1, 1, 1)] - V[(0, 0, 0)])]
    print()
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Effects of the three components, from the factorial of Table~\ref{tab:factorial-%s}. "
          r"A main effect is the change in Abs Rel produced by switching the component on, averaged "
          r"over the four settings of the other two; an interaction is the effect of the first "
          r"component with the second on minus with it off. Computed per test sequence, so negative "
          r"values mean the component helps; 95\%% bootstrap interval over sequences, the number of "
          r"sequences where the change is an improvement, and the exact Wilcoxon $p$ where $n\le 18$. "
          r"Bold marks intervals excluding zero.}" % a.design)
    print(r"\label{tab:factorial-effects-%s}" % a.design)
    print(r"\small")
    print(r"\begin{tabular}{l%s}" % ("c" * len(results)))
    print(r"\toprule")
    print("Effect & " + " & ".join("%s ($n=%d$)" % (results[d]["label"], results[d]["n"]) for d in results) + r" \\")
    print(r"\midrule")
    for lab, fn in rows:
        if lab.startswith("all three"):
            print(r"\midrule")
        print("%s & %s \\\\" % (lab, " & ".join(fmt(fn(results[d]["V"]), results[d]["n"]) for d in results)))
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # --- the full model against the references ------------------------------------------------
    print()
    print("%% full model (%s) against the references: full - reference, negative = full better" % cells[(1, 1, 1)])
    for m, lab in REFS:
        cols = []
        for d in results:
            by, units = results[d]["by"], results[d]["units"]
            if m not in by:
                cols.append("n/a")
                continue
            u = [x for x in units if x in by[m]]
            full = np.array([by[cells[(1, 1, 1)]][x] for x in u])
            ref = np.array([by[m][x] for x in u])
            cols.append(fmt(full - ref, len(u)))
        print("%%   vs %-34s %s" % (lab, "  |  ".join(cols)))
    print("%% units: " + "; ".join("%s = %s, n = %d" % (results[d]["label"], results[d]["unit"], results[d]["n"])
                                   for d in results))


if __name__ == "__main__":
    main()
