"""Paired comparison of any runs against a reference, per test sequence.

For each dataset: the reference and every run averaged within sequence and over seeds, the
difference run - reference (negative = the run is better), its 95% bootstrap interval over
sequences, the number of sequences where the run wins and the exact Wilcoxon p.

    python tools/paired.py --ref ii-only-ms --runs ii-chroma ii-chroma-only ii-pure-ms
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hamlyn_table import load, per_unit, wilcoxon  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/cviu/per_frame.csv")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["scared", "hamlyn", "c3vd"])
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for ds in a.datasets:
        acc = load(a.csv, ds, set(a.runs) | {a.ref})
        seeds = {}
        for (m, s, _, _) in acc:
            seeds.setdefault(m, set()).add(s)
        by, unit, _ = per_unit(acc, a.block)
        if a.ref not in by:
            print("%s: reference %s not evaluated" % (ds, a.ref))
            continue
        print("%s (unit: %s)   %s = %.4f, %d seed(s)"
              % (ds, unit, a.ref, np.mean(list(by[a.ref].values())), len(seeds[a.ref])))
        for r in a.runs:
            if r not in by:
                print("  %-22s not evaluated" % r)
                continue
            u = sorted(set(by[r]) & set(by[a.ref]))
            d = np.array([by[r][x] - by[a.ref][x] for x in u])
            m = d[rng.integers(0, len(d), size=(a.boot, len(d)))].mean(axis=1)
            lo, hi = np.percentile(m, 2.5), np.percentile(m, 97.5)
            p = wilcoxon(d)
            print("  %-22s %.4f  diff %+.4f [%+.4f, %+.4f]%s  wins %d/%d%s  seeds %d"
                  % (r, np.mean([by[r][x] for x in u]), d.mean(), lo, hi,
                     " *" if lo > 0 or hi < 0 else "  ", int((d < 0).sum()), len(d),
                     ("  p=%.3f" % p) if p == p else "", len(seeds[r])))


if __name__ == "__main__":
    main()
