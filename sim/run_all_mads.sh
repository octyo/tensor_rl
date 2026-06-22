#!/bin/bash
# Overnight full battery on the mads-feedback branch (CP = target-network agent).
# Backs up the existing results, runs every (layout x size) config --spawn both in
# parallel, then regenerates all summary figures in full/medium/small tiers.
set -u
cd "$(dirname "$0")"                      # -> sim/
PY=../code/venv/bin/python
LOG=/tmp/mads_run
mkdir -p "$LOG"
# single-threaded per process; parallelism comes from xargs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "[start] $(date)"

# 1. back up the current (nlms) results, start a fresh data dir
if [ -d data ] && [ ! -d data_nlms_backup ]; then
    mv data data_nlms_backup && echo "[backup] data -> data_nlms_backup"
fi
mkdir -p data

# 2. build the job list (big sizes first for better load balancing)
declare -A EMAP=( [8]=2000 [12]=4000 [16]=6000 [24]=14000 [32]=26000 [48]=58000 )
JOBS="$LOG/jobs.txt"; : > "$JOBS"
for S in 48 32 24 16 12 8; do
    E=${EMAP[$S]}
    MS=$(( 12 * S ))
    EV=$(( E / 50 ));  (( EV < 100 )) && EV=100
    SN=$(( E / 200 )); (( SN < 50 )) && SN=50
    for L in empty narrow chicane islands swirl; do
        echo "$PY obstacle_lab.py --layout $L --size $S --spawn both --episodes $E --max-steps $MS --eval-every $EV --snap-every $SN > $LOG/${L}_${S}.log 2>&1" >> "$JOBS"
    done
done
echo "[jobs] $(wc -l < "$JOBS") configs queued"

# 3. run the battery, 6 configs at a time
xargs -P 6 -I CMD bash -c CMD < "$JOBS"
echo "[battery done] $(date)"

# 4. regenerate every summary / figure from the fresh metrics
$PY summary.py
$PY accuracy_curves.py
$PY params_overview.py
# full (6 sizes)
$PY perf_overview.py
$PY efficiency_grid.py
$PY tradeoff_grid.py
$PY map_overview.py --sizes 8 12 16 24 32 48
# medium (8 12 24 48)
$PY perf_overview.py   --sizes 8 12 24 48 --fig perf_overview_medium.png   --table perf_table_medium.md
$PY efficiency_grid.py --sizes 8 12 24 48 --fig efficiency_grid_medium.png
$PY tradeoff_grid.py   --sizes 8 12 24 48 --fig tradeoff_grid_medium.png
$PY map_overview.py    --sizes 8 12 24 48 --tag _medium
# small (8 24 48), bigger cells + fonts
$PY perf_overview.py   --sizes 8 24 48 --fig perf_overview_small.png   --table perf_table_small.md --cell 4.2 --fs 1.7
$PY efficiency_grid.py --sizes 8 24 48 --fig efficiency_grid_small.png --cell 4.2 --fs 1.7
$PY tradeoff_grid.py   --sizes 8 24 48 --fig tradeoff_grid_small.png --cell 4.2 --fs 1.7
$PY map_overview.py    --sizes 8 24 48 --tag _small --cell 4.0

echo "[ALL DONE] $(date)"
