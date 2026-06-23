"""
Manim figure: the "lego" Q-tensor approximated by a CP decomposition.

    Q(x, y, a)  ≈  a_1 ∘ b_1 ∘ c_1  +  ...  +  a_r ∘ b_r ∘ c_r

The left-hand side is the value-coloured 3-way Q-tensor — a stack of voxels that
reads like a wall of lego bricks (from q_tensor.py); the right-hand side is a
sum of rank-one terms, each the outer product of three factor vectors drawn as
thin 3D bars (from cp_decomposition.py).  The two sides are joined by ≈ to stress
that a low-rank CP only approximates the full table.

Render a static image (saves the last frame as a PNG):
    manim -s -qh -o q_lego_cp.png q_lego_cp.py QLegoCP

Output lands in media/images/q_lego_cp/.
"""

from manim import *

# Reuse the shared building blocks so all report figures match visually.
from cp_decomposition import rank_one_triad
from q_tensor import build_q_tensor


class QLegoCP(Scene):
    def construct(self):
        # ── LHS: the value-coloured Q-tensor with its caption ────────────────
        q_tensor = build_q_tensor()
        q_caption = VGroup(
            Tex("Q-table").scale(0.8),
            MathTex(r"Q(x, y, a)").scale(0.8),
        ).arrange(DOWN, buff=0.12).next_to(q_tensor, DOWN, buff=0.45)
        lhs = VGroup(q_tensor, q_caption)

        approx = MathTex(r"\approx").scale(1.6)

        # ── RHS: a sum of rank-one CP terms ──────────────────────────────────
        t1 = rank_one_triad(1)
        t2 = rank_one_triad(2)
        tr = rank_one_triad("r")

        plus = MathTex("+").scale(1.5)
        dots = MathTex(r"+\;\cdots\;+").scale(1.3)
        rhs = VGroup(t1, plus, t2, dots, tr).arrange(RIGHT, buff=0.55, aligned_edge=DOWN)

        equation = VGroup(lhs, approx, rhs).arrange(RIGHT, buff=0.6)
        equation.scale_to_fit_width(config.frame_width - 1.0).move_to(ORIGIN)

        self.add(equation)
