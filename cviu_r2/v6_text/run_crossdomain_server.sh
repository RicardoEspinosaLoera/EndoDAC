#!/usr/bin/env bash
# Round-2 cross-domain component table: inference only, no training.
# Run on the server from the repository root (where cviu_revision.py and ./logs are):
#
#     bash cviu_r2/v6_text/run_crossdomain_server.sh --bg       # in the background (survives logout)
#     bash cviu_r2/v6_text/run_crossdomain_server.sh            # in the foreground
#     ... --cpu                                                 # no GPU (slow); goes after --bg
#
# The run can be stopped and relaunched: work already written to per_frame.csv is not redone.
#
# What it does
#   1. evaluates every trained seed of the runs of the table on SCARED and C3VD
#      (only what is missing from per_frame.csv, e.g. seeds 1 and 2 of cal0-local);
#   2. re-evaluates these runs on Hamlyn with every rectified sequence present in the Hamlyn
#      folder (rectified06/08/11/14), so that the unit of analysis is the sequence; runs whose
#      seeds all have multi-sequence Hamlyn rows already are skipped;
#   3. rebuilds the LaTeX table.
set -euo pipefail

PY=${PYTHON:-python}
RUNS="E3 E4 E5 E7 da-bas-1 da-lam-050 da-lam-010 R1 cal0-local MonoII-none MonoII"
OUT=results/cviu

test -f cviu_revision.py || { echo "run this from the repository root"; exit 1; }
mkdir -p "$OUT"

if [ "${1:-}" = "--bg" ]; then
    shift
    nohup setsid bash "$0" "$@" > "$OUT/crossdomain_launcher.out" 2>&1 < /dev/null &
    echo "started in the background, pid $!"
    echo "follow it with:   tail -f \$(ls -t $OUT/crossdomain_2*.log | head -1)"
    echo "stop it with:     pkill -f run_crossdomain_server.sh; pkill -f 'cviu_revision.py predict'"
    exit 0
fi
EXTRA="$*"

# everything printed below is also written to a log file next to the results
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/crossdomain_$STAMP.log"
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

# Hamlyn: a run is redone unless every seed with weights already has rows from several sequences
TODO=$($PY - $RUNS <<'EOF'
import csv, glob, os, sys
runs = sys.argv[1:]
seqs = {}
with open("results/cviu/per_frame.csv", newline="") as f:
    for r in csv.DictReader(f):
        if r["dataset"] == "hamlyn" and r["method"] in runs:
            seqs.setdefault((r["method"], r["seed"]), set()).add(r["sequence"])
todo = []
for run in runs:
    seeds = [os.path.basename(os.path.dirname(d)).rsplit("_s", 1)[-1]
             for d in glob.glob(os.path.join("logs", "cviu_%s_s*" % run, "models"))]
    if not seeds or any(len(seqs.get((run, s), ())) < 2 for s in seeds):
        todo.append(run)
print(" ".join(todo))
EOF
)
echo "Hamlyn, runs still to evaluate on all sequences: ${TODO:-none}"
if [ -n "$TODO" ]; then
    $PY cviu_revision.py predict --only $TODO --datasets hamlyn --force $EXTRA
fi

# --- 3. table ---------------------------------------------------------------------------
$PY -B cviu_r2/v6_text/make_table.py . cviu_r2/v6_text/components_crossdomain_v6.tex
echo
cat cviu_r2/v6_text/components_crossdomain_v6.tex

# --- 4. what to bring back ----------------------------------------------------------------
tar czf cviu_r2_crossdomain_$STAMP.tgz "$OUT/per_frame.csv" "$LOG" cviu_r2/v6_text/components_crossdomain_v6.tex
echo
echo "bring back: cviu_r2_crossdomain_$STAMP.tgz"
echo "FINISHED"
