"""
Manim figure: a tabular Q-table for the report.

Rows are states s_1 ... s_S, columns are the three MiniGrid actions
(turn left / turn right / move forward).  Cells are a heat-map of the Q-value,
and the greedy action argmax_a Q(s, a) in each row is outlined in gold.

Render a static image (saves the last frame as a PNG):
    manim -s -qh q_table.py QTable

Output lands in media/images/q_table/.
"""

from manim import *
import numpy as np


ACTIONS = ["turn left", "turn right", "move forward"]
N_STATES = 10

# Heat-map ramp (low -> high Q-value).
_RAMP = color_gradient([BLUE_E, TEAL_D, GREEN_D, YELLOW_E, GOLD_B], 256)


def _value_color(t):
    return _RAMP[int(np.clip(t, 0, 1) * 255)]


def _make_q_values(seed=7):
    """A plausible, structured Q-table (values grow toward a goal)."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0.2, 0.9, N_STATES)[:, None]          # closer to goal -> higher
    pref = rng.uniform(-0.15, 0.25, (N_STATES, len(ACTIONS)))  # per-action spread
    q = base + pref
    return np.round(np.clip(q, 0, 1) * 10 - 1, 1)            # spread into a readable range


class QTable(Scene):
    def construct(self):
        q = _make_q_values()
        lo, hi = q.min(), q.max()

        cw, ch = 1.7, 0.56
        rows, cols = N_STATES, len(ACTIONS)

        # ── grid of heat-mapped cells with their values ──────────────────────
        grid = VGroup()
        cells = {}
        for i in range(rows):
            for j in range(cols):
                t = (q[i, j] - lo) / (hi - lo)
                cell = Rectangle(
                    width=cw, height=ch,
                    fill_color=_value_color(t), fill_opacity=0.95,
                    stroke_color=GREY_D, stroke_width=1.0,
                )
                cell.move_to([j * cw, -i * ch, 0])
                txt = Text(f"{q[i, j]:.1f}", font_size=20,
                           color=BLACK if t > 0.5 else WHITE)
                txt.move_to(cell)
                cells[(i, j)] = cell
                grid.add(cell, txt)

        # ── greedy action per state: gold outline ────────────────────────────
        greedy = VGroup()
        for i in range(rows):
            j = int(np.argmax(q[i]))
            box = cells[(i, j)].copy().set_fill(opacity=0)
            box.set_stroke(GOLD_A, width=3.5)
            greedy.add(box)

        # ── column headers (actions) ─────────────────────────────────────────
        headers = VGroup()
        for j, name in enumerate(ACTIONS):
            h = Text(name, font_size=22)
            h.move_to([j * cw, ch * 0.5 + 0.45, 0])
            headers.add(h)

        # ── row labels (states) ──────────────────────────────────────────────
        row_labels = VGroup()
        for i in range(rows):
            lbl = MathTex(f"s_{{{i + 1}}}", font_size=26)
            lbl.move_to([-cw * 0.5 - 0.45, -i * ch, 0])
            row_labels.add(lbl)

        table = VGroup(grid, greedy, headers, row_labels)

        # ── braces with S / A labels ─────────────────────────────────────────
        rb = Brace(VGroup(*[cells[(i, 0)] for i in range(rows)]), LEFT, buff=0.85)
        rb_lbl = rb.get_tex("S").scale(1.0)
        cb = Brace(VGroup(*[cells[(0, j)] for j in range(cols)]), UP, buff=0.95)
        cb_lbl = cb.get_tex("A").scale(1.0)
        braces = VGroup(rb, rb_lbl, cb, cb_lbl)

        # ── caption ──────────────────────────────────────────────────────────
        full = VGroup(table, braces)
        caption = VGroup(
            MathTex(r"Q(s, a)\;\in\;\mathbb{R}^{S \times A}", font_size=40),
            Text("greedy policy:  argmaxₐ Q(s, a)", font_size=22, color=GOLD_A),
        ).arrange(DOWN, buff=0.18)
        caption.next_to(full, DOWN, buff=0.55)

        everything = VGroup(full, caption)
        everything.scale_to_fit_height(config.frame_height - 0.8).move_to(ORIGIN)

        self.add(everything)
