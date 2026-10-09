"""Two-fold comparison of the runs trained on Hamlyn: II loss alone against the photometric loss.

Every Hamlyn sequence is scored by the two runs of the fold that did NOT train on it (fold A is
tested on rectified06 + 14, fold B on rectified08 + 11). Reports Abs Rel per held-out sequence
(mean over seeds), the difference II - photometric, and a bootstrap interval over blocks of
consecutive frames, since four sequences are too few to resample.

    python tools/hamlyn_cv.py
"""
import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hamlyn_table import load  # noqa: E402

HELD_OUT = {"A": ["rectified06", "rectified14"], "B": ["rectified08", "rectified11"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/cviu/per_frame.csv")
    ap.add_argument("--a", default="ii", help="run suffix compared ... ")
    ap.add_argument("--b", default="photo", help="... against this one (the reference)")
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    runs = {"ham-f{}-{}".format(f, k) for f in HELD_OUT for k in (args.a, args.b)}
    acc = load(args.csv, "hamlyn", runs)
    # (run, sequence) -> {frame: mean over seeds}
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    seeds = collections.defaultdict(set)
    for (m, s, seq, fr), vs in acc.items():
        per[(m, seq)][fr].append(float(np.mean(vs)))
        seeds[m].add(s)
    rng = np.random.default_rng(0)
    seq_diffs, blocks = [], []
    print("held-out sequence   {:>8} {:>8} {:>9}   frames  seeds".format(args.b, args.a, "diff"))
    for fold, seqs in HELD_OUT.items():
        ra, rb = "ham-f{}-{}".format(fold, args.a), "ham-f{}-{}".format(fold, args.b)
        for seq in seqs:
            fa, fb = per.get((ra, seq)), per.get((rb, seq))
            if not fa or not fb:
                print("{:<18} not evaluated ({} / {})".format(seq, ra, rb))
                continue
            common = sorted(set(fa) & set(fb))
            va = np.array([np.mean(fa[f]) for f in common])
            vb = np.array([np.mean(fb[f]) for f in common])
            d = va - vb
            seq_diffs.append(d.mean())
            blocks += [d[i:i + args.block].mean() for i in range(0, len(d), args.block)]
            print("{:<18} {:8.4f} {:8.4f} {:+9.4f}   {:6d}  {}/{}".format(
                seq + " (f" + fold + ")", vb.mean(), va.mean(), d.mean(), len(common),
                len(seeds[ra]), len(seeds[rb])))
    if not seq_diffs:
        return
    seq_diffs, blocks = np.array(seq_diffs), np.array(blocks)
    m = blocks[rng.integers(0, len(blocks), size=(args.boot, len(blocks)))].mean(axis=1)
    print("mean over sequences: {:+.4f}   {} wins {}/{} sequences".format(
        seq_diffs.mean(), args.a, int((seq_diffs < 0).sum()), len(seq_diffs)))
    print("blocks of {} frames: mean {:+.4f}, 95% interval [{:+.4f}, {:+.4f}], n = {} blocks "
          "(blocks of one video are not independent: read the interval as optimistic)".format(
              args.block, blocks.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(blocks)))


if __name__ == "__main__":
    main()
