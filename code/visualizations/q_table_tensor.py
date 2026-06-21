"""
Manim figure: the two Q representations side by side.

    Q in R^{S x A}   <=>   Q in R^{X x Y x A}

Left: the flat tabular Q-table (state index s on rows, actions on columns).
Right: the same Q as a 3-way tensor with the state split into its (x, y) grid
coordinates.  Both are coloured from one shared Q array, so the equivalence is
literal: the table is just the tensor reshaped.

Render a static image (saves the last frame as a PNG):
    manim -s -qh q_table_tensor.py QTableTensor

Output lands in media/images/q_table_tensor/.
"""

from manim import *
import numpy as np

from q_table import build_q_table
from q_tensor import build_q_tensor, _make_q_tensor, X, Y


def _shared_q():
    """One Q-tensor (X x Y x A) and its flattened S x A table view."""
    q_t = _make_q_tensor()
    q_2d = np.zeros((X * Y, q_t.shape[2]))
    for i in range(X * Y):
        x, y = i % X, i // X          # matches q_table's s_i : (x, y) labelling
        q_2d[i] = q_t[x, y]
    return q_t, q_2d


class QTableTensor(Scene):
    def construct(self):
        q_t, q_2d = _shared_q()

        # Build both representations from the shared Q and match their heights.
        H = 5.0
        table = build_q_table(q_2d).scale_to_fit_height(H)
        tensor = build_q_tensor(q_t).scale_to_fit_height(H)

        left = VGroup(
            table,
            MathTex(r"Q \in \mathbb{R}^{S \times A}", font_size=38),
        ).arrange(DOWN, buff=0.4)
        right = VGroup(
            tensor,
            MathTex(r"Q \in \mathbb{R}^{X \times Y \times A}", font_size=38),
        ).arrange(DOWN, buff=0.4)

        middle = VGroup(
            MathTex(r"\Longleftrightarrow", font_size=72),
            Text("same Q,\nreshaped", font_size=22, color=GREY_B,
                 line_spacing=0.7).set_opacity(0.9),
        ).arrange(DOWN, buff=0.25)

        row = VGroup(left, middle, right).arrange(RIGHT, buff=0.7)
        row.scale_to_fit_width(config.frame_width - 0.6)
        if row.height > config.frame_height - 0.5:
            row.scale_to_fit_height(config.frame_height - 0.5)
        row.move_to(ORIGIN)

        self.add(row)
