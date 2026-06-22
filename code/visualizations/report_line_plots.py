"""Manim versions of the key report line charts (read from sim/data metrics.json).

Renders static PNGs (white background) to match the matplotlib report figures:
  ParamsReduction  -> params_reduction_manim.png
  AccuracyAvg      -> accuracy_avg_manim.png      (fixed | random panels)
  EfficiencyAvg    -> efficiency_avg_manim.png     (fixed | random panels)

Render (from repo root), e.g.:
    uv run --project code/visualizations manim -s -qh \
        code/visualizations/report_line_plots.py ParamsReduction
Then copy media/images/report_line_plots/<Scene>.png -> sim/data/_summary/<name>_manim.png
(the helper script render_manim_plots.sh does all three + the copy).
"""

import glob
import json
import os
import re

import numpy as np
from manim import *

config.background_color = WHITE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "sim", "data"))
A = 4
SIZES = [8, 12, 16, 24, 32, 48]
XLAB = [f"{s}x{s}" for s in SIZES]
METHODS = ["Tabular", "CP rank=3", "CP rank=6", "CP rank=12"]
MCOLOR = {"Tabular": BLACK, "CP rank=3": BLUE_D,
          "CP rank=6": GREEN_D, "CP rank=12": PURPLE}
SPAWNS = ["fixed", "random"]
PAT = re.compile(r"minigrid_(\d+)x\d+_(\d+)_([a-z]+)_(fixed|random)$")
DARK = BLACK


def n_params(method, size):
    if method == "Tabular":
        return size * size * A
    return int(method.split("=")[1]) * (2 * size + A)


def load_runs():
    runs = {}
    for d in glob.glob(os.path.join(DATA, "minigrid_*")):
        m = PAT.search(os.path.basename(d))
        mj = os.path.join(d, "metrics.json")
        if not m or not os.path.exists(mj):
            continue
        runs[(m.group(3), int(m.group(1)), m.group(4))] = json.load(open(mj))
    return runs


def avg_over_layouts(runs, spawn, metric):
    layouts = sorted({l for (l, _s, sp) in runs if sp == spawn})
    out = {}
    for meth in METHODS:
        ys = []
        for s in SIZES:
            vals = []
            for lay in layouts:
                m = runs.get((lay, s, spawn))
                if not m or meth not in m.get("final_acc", {}):
                    continue
                acc = m["final_acc"][meth]
                vals.append(acc if metric == "acc" else acc / n_params(meth, s) * 1000)
            ys.append(float(np.mean(vals)) if vals else np.nan)
        out[meth] = ys
    return out


# ── shared drawing helpers ───────────────────────────────────────────────────

def _panel(series, y_range, y_title, sub, x_len=5.2, y_len=4.4):
    """One line-chart panel as a VGroup; series = {name: list_of_y (len 6)}."""
    n = len(SIZES)
    axes = Axes(x_range=[0, n - 1, 1], y_range=y_range, x_length=x_len, y_length=y_len,
                tips=False,
                axis_config={"color": DARK, "stroke_width": 2,
                             "include_numbers": False},
                y_axis_config={"include_numbers": True, "font_size": 20,
                               "decimal_number_config": {"num_decimal_places":
                                   0 if y_range[1] > 5 else 1}})
    axes.get_axes()[1].numbers.set_color(DARK)
    lines = VGroup()
    for name, ys in series.items():
        pts = [axes.c2p(i, y) for i, y in enumerate(ys)]
        col = MCOLOR[name]
        poly = VMobject(color=col, stroke_width=3.5)
        poly.set_points_as_corners(pts)
        dots = VGroup(*[Dot(p, radius=0.055, color=col) for p in pts])
        lines.add(poly, dots)
    # x tick labels
    xt = VGroup()
    for i, lab in enumerate(XLAB):
        t = Text(lab, font_size=18, color=DARK).next_to(axes.c2p(i, y_range[0]), DOWN, buff=0.18)
        xt.add(t)
    yl = Text(y_title, font_size=20, color=DARK).rotate(PI / 2).next_to(axes, LEFT, buff=0.15)
    st = Text(sub, font_size=24, color=DARK, weight=BOLD).next_to(axes, UP, buff=0.2)
    return VGroup(axes, lines, xt, yl, st)


def _legend():
    items = VGroup()
    for name in METHODS:
        bar = Line(ORIGIN, RIGHT * 0.5, color=MCOLOR[name], stroke_width=5)
        lab = Text(name, font_size=20, color=DARK).next_to(bar, RIGHT, buff=0.12)
        items.add(VGroup(bar, lab))
    items.arrange(RIGHT, buff=0.5)
    return items


class ParamsReduction(Scene):
    def construct(self):
        series = {"Tabular": [n_params("Tabular", s) for s in SIZES]}
        for r in (3, 6, 12):
            series[f"CP rank={r}"] = [n_params(f"CP rank={r}", s) for s in SIZES]
        panel = _panel(series, [0, 9500, 2000], "number of parameters",
                       "", x_len=8.5, y_len=5.0)
        title = Text("Parameter count by different map sizes", font_size=30,
                     color=DARK, weight=BOLD)
        leg = _legend()
        title.to_edge(UP, buff=0.3)
        panel.next_to(title, DOWN, buff=0.35)
        leg.next_to(panel, DOWN, buff=0.3)
        self.add(title, panel, leg)


class _AvgBase(Scene):
    metric = "acc"
    y_range = [0, 1.05, 0.2]
    y_title = "final accuracy (mean over layouts)"
    title_txt = "Final accuracy vs grid size (averaged over layouts)"

    def construct(self):
        runs = load_runs()
        panels = VGroup()
        for spawn in SPAWNS:
            series = avg_over_layouts(runs, spawn, self.metric)
            panels.add(_panel(series, self.y_range, self.y_title, f"{spawn} start"))
        panels.arrange(RIGHT, buff=1.2)
        title = Text(self.title_txt, font_size=30, color=DARK, weight=BOLD).to_edge(UP, buff=0.3)
        leg = _legend()
        panels.next_to(title, DOWN, buff=0.45)
        leg.next_to(panels, DOWN, buff=0.35)
        self.add(title, panels, leg)


class AccuracyAvg(_AvgBase):
    metric = "acc"
    y_range = [0, 1.05, 0.2]
    y_title = "final accuracy"
    title_txt = "Final accuracy vs grid size (averaged over layouts)"


class EfficiencyAvg(_AvgBase):
    metric = "eff"
    y_range = [0, 16, 4]
    y_title = "accuracy / 1k params"
    title_txt = "Parameter efficiency vs grid size (averaged over layouts)"
