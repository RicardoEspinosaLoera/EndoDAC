"""Regenerate the Hamlyn block of the paired-comparison table from results/cviu/per_frame.csv.

Reads the per-frame metrics directly, so it does not need the `stats` stage and can be run while
part of the grid is still being re-evaluated. The unit of analysis follows the same rule as
`stats`: the video sequence when at least three are available, otherwise contiguous blocks of
`--block` frames of the single sequence.

    python tools/hamlyn_table.py                 # MonoIIF = C1, the default reference
    python tools/hamlyn_table.py --ref E8
"""
import argparse
import collections
import csv
import itertools

import numpy as np

METHODS = [("HADepth", "HADepth"), ("MonoPCC", "MonoPCC"), ("EndoDAC_MICCAI", "EndoDAC"),
           ("AF_SfMLearner", "AF-SfMLearner"), ("Monodepth2", "Monodepth2")]


def load(path, dataset, wanted):
    """(method, seed, sequence) -> mean abs_rel, dropping frames with no valid ground truth."""
    acc = collections.defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] != dataset or r["method"] not in wanted:
                continue
            try:
                v = float(r["abs_rel"])
            except ValueError:
                continue
            if v != v:
                continue
            acc[(r["method"], r["seed"], r["sequence"], int(r["frame"]) if r["frame"].isdigit() else 0)].append(v)
    return acc


def per_unit(acc, block):
    """-> {method: {unit: value}}, averaging frames within a unit and then across seeds."""
    frames = collections.defaultdict(list)          # (method, seed, sequence) -> [(frame, v)]
    for (m, s, seq, fr), vs in acc.items():
        frames[(m, s, seq)].append((fr, float(np.mean(vs))))
    seqs = sorted({k[2] for k in frames})
    use_blocks = len(seqs) < 3
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for (m, s, seq), fv in frames.items():
        fv.sort()
        for i, (_, v) in enumerate(fv):
            unit = "{}#{:04d}".format(seq, i // block) if use_blocks else seq
            by[m][unit].append(v)
    out = {m: {u: float(np.mean(v)) for u, v in d.items()} for m, d in by.items()}
    return out, ("block of %d frames" % block) if use_blocks else "sequence", len(seqs)


def wilcoxon(d):
    d = d[d != 0]
    n = len(d)
    if n == 0 or n > 18:
        return float("nan")
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    obs = r[d > 0].sum()
    dist = np.array([sum(r[i] for i in range(n) if s[i]) for s in itertools.product([0, 1], repeat=n)])
    return min(1.0, 2.0 * min((dist >= obs).sum(), (dist <= obs).sum()) / len(dist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/cviu/per_frame.csv")
    ap.add_argument("--ref", default="C1", help="run standing for MonoIIF")
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()

    wanted = {a.ref} | {m for m, _ in METHODS}
    by, unit, nseq = per_unit(load(a.csv, "hamlyn", wanted), a.block)
    missing = [m for m in wanted if m not in by]
    if missing:
        print("%% missing from the csv, skipped: " + ", ".join(sorted(missing)))
    rng = np.random.default_rng(0)

    def boot(x):
        i = rng.integers(0, len(x), size=(a.boot, len(x)))
        m = x[i].mean(axis=1)
        return np.percentile(m, 2.5), np.percentile(m, 97.5)

    ref = by[a.ref]
    rows = []
    v = np.array([ref[u] for u in sorted(ref)])
    lo, hi = boot(v)
    rows.append((v.mean(), "MonoIIF (proposed)", "%.4f" % v.mean(), "$[%.4f, %.4f]$" % (lo, hi), None, None, None))
    for m, lab in METHODS:
        if m not in by:
            continue
        units = sorted(set(by[m]) & set(ref))
        x = np.array([by[m][u] for u in units])
        d = np.array([by[m][u] - ref[u] for u in units])
        lo, hi = boot(x)
        dlo, dhi = boot(d)
        p = wilcoxon(d)
        mean = "%+.4f" % d.mean()
        if dlo > 0 or dhi < 0:
            mean = r"\mathbf{%s}" % mean
        rows.append((x.mean(), lab, "%.4f" % x.mean(), "$[%.4f, %.4f]$" % (lo, hi),
                     "$%s$ $[%+.4f, %+.4f]$" % (mean, dlo, dhi),
                     "%d/%d" % (int((d > 0).sum()), len(d)),
                     ("%.3f" % p) if p == p else "---"))
    rows.sort(key=lambda t: t[0])

    n = len(next(iter(by.values())))
    print("%% Hamlyn: %d sequence(s), unit = %s, n = %d, reference = %s" % (nseq, unit, n, a.ref))
    print(r"\multirow{%d}{*}{Hamlyn}" % len(rows))
    for i, (_, lab, val, ci, pair, kn, p) in enumerate(rows):
        val = (r"\textbf{%s}" % val) if i == 0 else ((r"\underline{%s}" % val) if i == 1 else val)
        if pair is None:
            print(" & %-18s & %-37s & %-39s & %-5s & %-5s \\\\" % (lab, val + " " + ci, "---", "---", "---"))
        else:
            print(" & %-18s & %-37s & %-39s & %-5s & %-5s \\\\" % (lab, val + " " + ci, pair, kn, p))


if __name__ == "__main__":
    main()
