#!/usr/bin/env bash
# Round-2 cross-domain component table: inference only, no training.
# Run on the server from the repository root (where cviu_revision.py and ./logs are):
#
#     bash cviu_r2/v6_text/run_crossdomain_server.sh            # GPU
#     bash cviu_r2/v6_text/run_crossdomain_server.sh --cpu      # no GPU (slow)
#
# What it does
#   1. evaluates every trained seed of the eight runs of the table on SCARED and C3VD
#      (only what is missing from per_frame.csv, e.g. seeds 1 and 2 of cal0-local);
#   2. re-evaluates these runs on Hamlyn with every rectified sequence present in the Hamlyn
#      folder (rectified06/08/11/14), so that the unit of analysis is the sequence;
#   3. rebuilds the LaTeX table.
set -euo pipefail

PY=${PYTHON:-python}
RUNS="E3 E4 da-bas-1 da-lam-050 R1 cal0-local MonoII-none MonoII"
OUT=results/cviu
EXTRA="$*"

test -f cviu_revision.py || { echo "run this from the repository root"; exit 1; }

# everything printed below is also written to a log file next to the results
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/crossdomain_$STAMP.log"
mkdir -p "$OUT"
exec > >(tee "$LOG") 2>&1
echo "log: $LOG"

# --- 0. what is on disk -----------------------------------------------------------------
HAM=$($PY - <<'EOF'
import yaml
print(yaml.safe_load(open("cviu_config.yaml"))["data"]["hamlyn"])
EOF
)
echo "Hamlyn folder: $HAM"
ls -d "$HAM"/rectified* 2>/dev/null || ls "$HAM" | head -20
echo
echo "Seeds with weights on disk:"
for r in $RUNS; do
    printf "  %-12s" "$r"
    ls -d logs/cviu_${r}_s*/models 2>/dev/null | sed -E 's/.*_s([0-9]+)\/models/\1/' | tr '\n' ' '
    echo
done
echo

# --- 1. backup --------------------------------------------------------------------------
cp "$OUT/per_frame.csv" "$OUT/per_frame.backup_$STAMP.csv"
echo "backup: $OUT/per_frame.backup_$STAMP.csv"

# --- 2. inference -----------------------------------------------------------------------
# missing (run, seed) pairs on SCARED and C3VD; existing rows are left untouched
$PY cviu_revision.py predict --only $RUNS --datasets scared c3vd $EXTRA
# Hamlyn: redo these runs so that all of them cover the same sequences
$PY cviu_revision.py predict --only $RUNS --datasets hamlyn --force $EXTRA

# --- 3. table ---------------------------------------------------------------------------
$PY -B cviu_r2/v6_text/make_table.py . cviu_r2/v6_text/components_crossdomain_v6.tex
echo
cat cviu_r2/v6_text/components_crossdomain_v6.tex

# --- 4. what to bring back ----------------------------------------------------------------
tar czf cviu_r2_crossdomain_$STAMP.tgz "$OUT/per_frame.csv" "$LOG" cviu_r2/v6_text/components_crossdomain_v6.tex
echo
echo "bring back: cviu_r2_crossdomain_$STAMP.tgz"
