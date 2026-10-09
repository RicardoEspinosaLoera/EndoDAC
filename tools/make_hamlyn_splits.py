"""Training splits for Hamlyn, two folds over the sequences on disk.

Writes splits/hamlyn_fA and splits/hamlyn_fB, each with train_files.txt and val_files.txt in the
format of the SCARED split ("<sequence> <frame> l"). Fold A trains on the sequences of --fold_a
and is meant to be tested on the others; fold B is the reverse. Only frames whose two temporal
neighbours exist are listed, and the last --val_frac of every sequence is held out for the
validation log (contiguous, so it shares no neighbour with the training frames).

    python tools/make_hamlyn_splits.py --data_path /workspace/HAMLYN
"""
import argparse
import glob
import os

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sequence_dir(data_path, seq):
    d = os.path.join(data_path, seq)
    if not os.path.isdir(os.path.join(d, "image01")):
        d = os.path.join(d, seq)
    return d


def frames(data_path, seq):
    files = sorted(glob.glob(os.path.join(sequence_dir(data_path, seq), "image01", "*.jpg")))
    bad = [f for f in files if len(os.path.basename(f)) != 14]
    if bad:
        raise SystemExit("{}: file names are not 10 digits (e.g. {})".format(seq, os.path.basename(bad[0])))
    idx = sorted(int(os.path.basename(f)[:-4]) for f in files)
    have = set(idx)
    usable = [i for i in idx if i - 1 in have and i + 1 in have]
    size = Image.open(files[0]).size if files else None
    return usable, len(idx), size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--fold_a", nargs="+", default=["rectified08", "rectified11"])
    ap.add_argument("--fold_b", nargs="+", default=["rectified06", "rectified14"])
    ap.add_argument("--val_frac", type=float, default=0.05)
    a = ap.parse_args()
    for fold, seqs in (("A", a.fold_a), ("B", a.fold_b)):
        train, val = [], []
        for seq in seqs:
            usable, total, size = frames(a.data_path, seq)
            if not usable:
                raise SystemExit("{}: no frames under {}".format(seq, sequence_dir(a.data_path, seq)))
            n_val = max(1, int(round(len(usable) * a.val_frac)))
            # drop one frame at the boundary so no validation neighbour is a training frame
            tr, va = usable[:-n_val - 1], usable[-n_val:]
            train += ["{} {} l".format(seq, i) for i in tr]
            val += ["{} {} l".format(seq, i) for i in va]
            print("fold {}  {}: {} frames on disk, {} with both neighbours, image {}x{} -> {} train, {} val"
                  .format(fold, seq, total, len(usable), size[0], size[1], len(tr), len(va)))
        out = os.path.join(ROOT, "splits", "hamlyn_f" + fold)
        os.makedirs(out, exist_ok=True)
        for name, lines in (("train", train), ("val", val)):
            with open(os.path.join(out, name + "_files.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        print("fold {}: {} train, {} val -> {}".format(fold, len(train), len(val), out))


if __name__ == "__main__":
    main()
