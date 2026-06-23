"""
Manim figure: flat matrix Q-table (grid actions) approximated by a CP decomposition.

Same as q_matrix_cp.py, but the action columns are the four grid moves
(up / down / right / left) instead of the MiniGrid turn actions.

    Q(s, a)  ≈  a_1 ∘ b_1 ∘ c_1  +  ...  +  a_r ∘ b_r ∘ c_r

The left-hand side is the real heat-mapped matrix Q-table (build_q_table from
q_table.py); the right-hand side keeps the CP terms in 3D.  Joined by ≈ to stress
that a low-rank CP only approximates the full table.

Render a static image (saves the last frame as a PNG):
    manim -s -qh -o q_matrix_cp_dirs.png q_matrix_cp_dirs.py QMatrixCPDirs

Output lands in media/images/q_matrix_cp_dirs/.
"""

from manim import *

# Swap in the four grid-move actions before building the table.  q_table reads
# its module-level ACTIONS at call time, so this re-sizes the table to S×4.
import q_table
q_table.ACTIONS = ["up", "down", "right", "left"]

from cp_decomposition import rank_one_triad
from q_table import build_q_table


class QMatrixCPDirs(Scene):
    def construct(self):
        # ── LHS: the real heat-mapped matrix Q-table (grid actions) ──────────
        q_table_mob = build_q_table()
        lhs = VGroup(q_table_mob)

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
