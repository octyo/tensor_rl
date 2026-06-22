#!/bin/bash
# Render the key report line charts in Manim and copy them into sim/data/_summary
# as *_manim.png. Run from repo root:  bash code/visualizations/render_manim_plots.sh
set -e
cd "$(dirname "$0")"                       # -> code/visualizations
M=media/images/report_line_plots
S=../../sim/data/_summary
declare -a MAP=( "ParamsReduction:params_reduction" "AccuracyAvg:accuracy_avg" "EfficiencyAvg:efficiency_avg" "TrainingAvg:training_avg" )
for pair in "${MAP[@]}"; do
    scene="${pair%%:*}"; name="${pair##*:}"
    .venv/bin/manim -s -qh report_line_plots.py "$scene"
    cp "$M/${scene}_ManimCE_v0.20.1.png" "$S/${name}_manim.png"
    echo "[OK] $S/${name}_manim.png"
done
