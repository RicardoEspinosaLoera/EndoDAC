#!/bin/bash
# State of every run of the CVIU grid (see CVIU_REVISION_PLAN.md). Run from the repo root.
#
#   done   its cviu_done.txt / train_complete.txt marker is there and no process is alive
#   RUN    one train_end_to_end.py alive with that --model_name, with its last progress line
#   DUPn   n processes share that --model_name: they overwrite each other's checkpoints, kill
#          all but the one with the largest ELAPSED (ps -eo pid,ppid,etime,args | grep <name>)
#   STOP   a log but no process and no marker: it crashed, or it ended after its launcher was
#          gone (the next launcher recovers the marker from the log)
#
# A queued run has no log yet, so it does not appear at all.
cd "$(dirname "$0")" || exit 1

# One line per run actually training: its dataloader workers inherit the same command line, so
# count only the processes whose parent is not itself a train_end_to_end.py (i.e. the trainers).
alive=$(ps -eo pid,ppid,args 2>/dev/null | grep -a train_end_to_end.py | grep -v grep \
  | awk '{ name="?"
           for (i = 1; i <= NF; i++) if ($i == "--model_name") { name = $(i + 1); break }
           mine[$1] = 1; par[$1] = $2; run[$1] = name }
         END { for (p in mine) if (!(par[p] in mine)) print run[p] }' | sort | uniq -c)

d=0; r=0; s=0; dup=0
for f in results/cviu/train_logs/cviu_*.log; do
  [ -e "$f" ] || continue
  n=$(basename "$f" .log)
  k=$(printf '%s\n' "$alive" | awk -v n="$n" '$2 == n {print $1}')
  last=$(tail -c 20000 "$f" | tr '\r' '\n' | grep -a 'time left' | tail -n 1)
  # the process decides first: a marker can be there while the run is still going
  if [ -n "$k" ] && [ "$k" -gt 1 ]; then
    st="DUP$k"; dup=$((dup + 1)); r=$((r + 1))
  elif [ -n "$k" ]; then
    st="RUN "; r=$((r + 1))
  elif [ -e "logs/$n/cviu_done.txt" ] || [ -e "logs/$n/train_complete.txt" ]; then
    st="done"; d=$((d + 1)); last=""
  else
    st="STOP"; s=$((s + 1))
  fi
  printf '%-24s %s %s\n' "$n" "$st" "$last"
done

echo "-- terminadas:$d  corriendo:$r  paradas:$s"
[ "$dup" -gt 0 ] && echo "-- ATENCION: $dup corrida(s) duplicada(s), mata las sobrantes antes de seguir"
ps -eo pid,etime,args 2>/dev/null | grep -a cviu_revision.py | grep -v grep \
  | cut -c1-110 | sed 's/^/-- lanzador /'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
  | sed 's/^/-- gpu /'
