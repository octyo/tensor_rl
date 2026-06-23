"""
Manim figure: the flat matrix Q-table approximated by a CP decomposition.

    Q(s, a)  ≈  a_1 ∘ b_1 ∘ c_1  +  ...  +  a_r ∘ b_r ∘ c_r

The left-hand side is the real heat-mapped matrix Q-table (build_q_table from
q_table.py).  The right-hand side keeps the CP terms in 3D: each rank-one term
is the outer product of three factor vectors drawn as thin 3D bars
(rank_one_triad from cp_decomposition.py) — the table is a matrix, but the CP we
actually use represents it as a 3-way tensor.  The two sides are joined by ≈ to
stress that a low-rank CP only approximates the full table.

Render a static image (saves the last frame as a PNG):
    manim -s -qh -o q_matrix_cp.png q_matrix_cp.py QMatrixCP

Output lands in media/images/q_matrix_cp/.
"""

from manim import *

# Reuse the shared building blocks so all report figures match visually.
from cp_decomposition import rank_one_triad
from q_table import build_q_table


class QMatrixCP(Scene):
    def construct(self):
        # ── LHS: the real heat-mapped matrix Q-table ─────────────────────────
        q_table = build_q_table()
        lhs = VGroup(q_table)

        approx = MathTex(r"\approx").scale(1.6)

        # ── RHS: a sum of rank-one CP terms, kept in 3D ──────────────────────
        t1 = rank_one_triad(1)
        t2 = rank_one_triad(2)
        tr = rank_one_triad("r")

        plus = MathTex("+").scale(1.5)
        dots = MathTex(r"+\;\cdots\;+").scale(1.3)
        rhs = VGroup(t1, plus, t2, dots, tr).arrange(RIGHT, buff=0.55, aligned_edge=DOWN)

        equation = VGroup(lhs, approx, rhs).arrange(RIGHT, buff=0.7)
        equation.scale_to_fit_width(config.frame_width - 1.0).move_to(ORIGIN)

        self.add(equation)
