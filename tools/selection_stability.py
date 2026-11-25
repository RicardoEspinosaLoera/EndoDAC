"""Does choosing a configuration on the SCARED test sequences hold up on sequences not used for the
choice?  (CVIU round 2, R2 (2): "it is unclear whether the test set influenced model selection".)

For every family of configurations that was compared on SCARED, every split of the seven test
sequences into a selection subset of k sequences and its held-out complement is enumerated. The
configuration with the lowest mean Abs Rel on the subset is "selected"; the script reports how
often each configuration is selected, how often the selected one is also the best on the held-out
complement, the mean held-out regret (held-out Abs Rel of the selected configuration minus that of
the configuration that is best on the complement) and the selection optimism (Abs Rel of the
selected configuration on the held-out complement minus on the subset, positive = the subset flattered it).
Values are read from per_sequence.csv (frames averaged within sequence, then over seeds).

    python tools/selection_stability.py            # markdown to stdout
    python tools/selection_stability.py --k 3 4    # subset sizes (default 3 and 4)
"""
import argparse
import collections
import csv
import itertools

import numpy as np

FAMILIES = [
    ("Calibration form on Depth Anything v1, lambda1 = 0 (Table calibration family)",
     [("none (EndoDAC)", "E3"), ("global", "da-bas-0"), ("degree 1", "da-bas-1"),
      ("degree 2", "da-bas-2"), ("dense local", "E4")]),
    ("lambda1 of L_II on Depth Anything v1, degree-1 calibration",
     [("0", "da-bas-1"), ("0.1", "da-lam-010"), ("0.25", "da-lam-025"), ("0.5", "da-lam-050")]),
    ("MonoIIF candidates (Depth Anything v1 + DV-LoRA, highlight recipe)",
     [("EndoDAC (none)", "E3"), ("no calibration", "C0"), ("global calibration", "C1"),
      ("dense local, no L_II", "E8-IIF"), ("MonoIIF (dense local + L_II)", "E8")]),
    ("Calibration form on ResNet-18, lambda1 = 0",
     [("none (Monodepth2)", "R1"), ("global", "cal0-glob"), ("degree 1", "cal0-deg1"),
      ("degree 2", "lam-bas-000"), ("dense local", "cal0-local")]),
    ("lambda1 of L_II on ResNet-18, degree-2 calibration",
     [("0", "lam-bas-000"), ("0.1", "lam-bas-010"), ("0.25", "lam-bas-025"),
      ("1", "lam-bas-100"), ("2", "lam-bas-200")]),
    ("Calibration form on ResNet-18, lambda1 = 0.5 (MonoII family)",
     [("none", "MonoII-none"), ("global", "MonoII-glob"), ("dense local (MonoII)", "MonoII")]),
]


def read(path, dataset):
    by = collections.defaultdict(dict)   # method -> sequence -> abs_rel
    seeds = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] != dataset:
                continue
            by[r["method"]][r["sequence"]] = float(r["abs_rel"])
            seeds[r["method"]] = int(r["n"])
    return by, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/cviu/per_sequence.csv")
    ap.add_argument("--dataset", default="scared")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 4])
    a = ap.parse_args()
    by, seeds = read(a.csv, a.dataset)

    print("# Stability of configuration selection on the %s test sequences\n" % a.dataset.upper())
    print("Unit: test sequence, Abs Rel averaged over frames and training seeds (per_sequence.csv). "
          "For each subset size k, all C(7, k) selection subsets are enumerated; the configuration "
          "with the lowest mean Abs Rel on the subset is selected and then scored on the complement.\n")
    for title, members in FAMILIES:
        members = [(lab, run) for lab, run in members if run in by]
        if len(members) < 2:
            print("## %s\n\n(not enough runs in the csv)\n" % title)
            continue
        seqs = sorted(set.intersection(*[set(by[run]) for _, run in members]))
        M = np.array([[by[run][s] for s in seqs] for _, run in members])   # configs x sequences
        full = M.mean(axis=1)
        full_best = int(np.argmin(full))
        print("## %s\n" % title)
        print("| configuration | run | seeds | Abs Rel, all %d sequences | " % len(seqs)
              + " | ".join("selected (k=%d)" % k for k in a.k) + " |")
        print("|---|---|---|---|" + "---|" * len(a.k))
        sel_count = {k: np.zeros(len(members), dtype=int) for k in a.k}
        stats = {}
        for k in a.k:
            agree = 0
            regret = []
            optimism = []
            total = 0
            for S in itertools.combinations(range(len(seqs)), k):
                C = [i for i in range(len(seqs)) if i not in S]
                sub = M[:, list(S)].mean(axis=1)
                comp = M[:, C].mean(axis=1)
                w = int(np.argmin(sub))
                sel_count[k][w] += 1
                total += 1
                agree += int(w == int(np.argmin(comp)))
                regret.append(comp[w] - comp.min())
                optimism.append(comp[w] - sub[w])
            stats[k] = (agree, total, float(np.mean(regret)), float(np.max(regret)),
                        float(np.mean(optimism)))
        for i, (lab, run) in enumerate(members):
            mark = " **(best on all 7)**" if i == full_best else ""
            print("| %s%s | `%s` | %d | %.4f | " % (lab, mark, run, seeds.get(run, 0), full[i])
                  + " | ".join("%d/%d" % (sel_count[k][i], sel_count[k].sum()) for k in a.k) + " |")
        print()
        for k in a.k:
            agree, total, mr, xr, opt = stats[k]
            print("- k = %d: the configuration selected on the subset is also the best on the held-out "
                  "%d sequences in %d of %d splits; mean held-out regret %.4f Abs Rel (max %.4f); "
                  "mean selection optimism %+.4f." % (k, len(seqs) - k, agree, total, mr, xr, opt))
        spread = full.max() - full.min()
        print("- Range of the family on all seven sequences: %.4f Abs Rel.\n" % spread)


if __name__ == "__main__":
    main()
