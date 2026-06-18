"""
Manim figure: CP (CANDECOMP/PARAFAC) decomposition of a 3-way tensor.

    A  =  a_1 ∘ b_1 ∘ c_1  +  a_2 ∘ b_2 ∘ c_2  +  ...  +  a_r ∘ b_r ∘ c_r

A 3-way tensor written as a sum of r rank-one tensors.  Each rank-one term is
the outer product of three factor vectors, drawn as three thin 3D bars (slabs)
meeting at a corner: a_i (mode 1), b_i (mode 2), c_i (mode 3).

Everything is hand-projected with a fixed isometric camera (no ThreeDScene), so
the cube and every bar share one consistent 3D perspective and the equation
still lays out cleanly left-to-right.

Render a static image (saves the last frame as a PNG):
    manim -s -qh cp_decomposition.py CPDecomposition

Output lands in media/images/cp_decomposition/.
"""

from manim import *
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fixed isometric projection  (world 3D  ->  screen 2D)
#   world axes:  +x width,  +y depth (into screen),  +z height (up)
# ─────────────────────────────────────────────────────────────────────────────

_VIEW = np.array([1.0, -1.0, 0.55])           # direction from scene toward camera
_VIEW /= np.linalg.norm(_VIEW)
_RIGHT = np.cross([0, 0, 1.0], _VIEW)
_RIGHT /= np.linalg.norm(_RIGHT)
_UP = np.cross(_VIEW, _RIGHT)                  # already unit length
_LIGHT = np.array([0.3, -0.6, 0.7])           # fake light direction
_LIGHT /= np.linalg.norm(_LIGHT)


def P(p):
    """Project a world point onto the (x, y, 0) screen plane."""
    p = np.asarray(p, dtype=float)
    return np.array([p @ _RIGHT, p @ _UP, 0.0])


def _depth(p):
    return float(np.asarray(p, dtype=float) @ _VIEW)


# Factor colours: one per tensor mode.
C_A = BLUE_B      # mode 1
C_B = GREEN_B     # mode 2
C_C = RED_B       # mode 3


# ─────────────────────────────────────────────────────────────────────────────
# Isometric box: a shaded 3D cuboid from a corner + three edge vectors
# ─────────────────────────────────────────────────────────────────────────────

def iso_box(corner, e1, e2, e3, fill, stroke=None, base_op=0.95, sw=1.5):
    o = np.array(corner, float)
    edges = [np.array(e1, float), np.array(e2, float), np.array(e3, float)]
    center = o + 0.5 * sum(edges)

    faces = []
    for k in range(3):
        u, v, w = edges[(k + 1) % 3], edges[(k + 2) % 3], edges[k]
        for side in (0, 1):
            base = o + side * w
            corners = [base, base + u, base + u + v, base + v]
            n = np.cross(u, v)
            if np.dot(n, (base + 0.5 * (u + v)) - center) < 0:
                n = -n
            faces.append((corners, n))

    visible = [(c, n) for c, n in faces if np.dot(n, _VIEW) > 0]
    visible.sort(key=lambda cn: np.mean([_depth(p) for p in cn[0]]))

    g = VGroup()
    for corners, n in visible:
        nn = n / np.linalg.norm(n)
        shade = base_op * (0.55 + 0.45 * max(0.0, float(np.dot(nn, _LIGHT))))
        g.add(Polygon(
            *[P(c) for c in corners],
            fill_color=fill, fill_opacity=shade,
            stroke_color=stroke if stroke is not None else fill, stroke_width=sw,
        ))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def tensor_cube(s=1.9):
    """A shaded isometric cube standing in for the 3-way tensor."""
    return iso_box([0, 0, 0], [s, 0, 0], [0, s, 0], [0, 0, s],
                   fill=BLUE_D, stroke=BLUE_A, sw=2.0)


def factor_bar(origin, axis, length, color, w=0.18):
    """A thin 3D slab of `length` along `axis` (a unit vector), centred on it."""
    axis = np.asarray(axis, float)
    # The two cross-section directions are the world axes orthogonal to `axis`.
    perp = [np.array(e, float) for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1])
            if abs(np.dot(e, axis)) < 0.5]
    p1, p2 = perp[0], perp[1]
    o = np.asarray(origin, float) - 0.5 * w * (p1 + p2)
    return iso_box(o, axis * length, p1 * w, p2 * w, fill=color, stroke=WHITE, sw=1.0)


def rank_one_triad(idx, la=1.7, lb=1.5, lc=1.3, gap=0.42):
    """One rank-one term: bars a_i (up), b_i (width), c_i (depth) from a corner.

    `gap` pushes each bar a little off the shared corner so the ends don't
    overlap in a muddle.
    """
    sub = str(idx)
    o = np.array([0.0, 0.0, 0.0])
    za, xb, yc = np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])

    # `a` runs straight up, so its gap reads larger on screen than the
    # foreshortened b/c gaps; shrink it so all three look equally spaced.
    gap_a = gap * 0.6

    a = factor_bar(o + za * gap_a, za, la, C_A)   # mode 1 -> up
    b = factor_bar(o + xb * gap, xb, lb, C_B)     # mode 2 -> width
    c = factor_bar(o + yc * gap, yc, lc, C_C)     # mode 3 -> depth

    a_lab = MathTex(f"a_{{{sub}}}", color=C_A).scale(0.85).next_to(P(o + za * (gap_a + la)), UP, buff=0.14)
    b_lab = MathTex(f"b_{{{sub}}}", color=C_B).scale(0.85).next_to(P(o + xb * (gap + lb)), DR, buff=0.10)
    c_lab = MathTex(f"c_{{{sub}}}", color=C_C).scale(0.85).next_to(P(o + yc * (gap + lc)), UR, buff=0.06)

    # Draw depth bar first, then width, then the upright bar on top.
    return VGroup(c, b, a, a_lab, b_lab, c_lab)


# ─────────────────────────────────────────────────────────────────────────────
# Scene
# ─────────────────────────────────────────────────────────────────────────────

class CPDecomposition(Scene):
    def construct(self):
        cube = tensor_cube()
        cube_caption = VGroup(
            Tex("3-way tensor").scale(0.8),
            MathTex(r"\mathcal{A}").scale(1.1),
        ).arrange(DOWN, buff=0.12).next_to(cube, DOWN, buff=0.45)
        lhs = VGroup(cube, cube_caption)

        eq = MathTex("=").scale(1.6)

        t1 = rank_one_triad(1)
        t2 = rank_one_triad(2)
        tr = rank_one_triad("r")

        plus = MathTex("+").scale(1.5)
        dots = MathTex(r"+\;\cdots\;+").scale(1.3)

        rhs = VGroup(t1, plus, t2, dots, tr).arrange(RIGHT, buff=0.55, aligned_edge=DOWN)

        equation = VGroup(lhs, eq, rhs).arrange(RIGHT, buff=0.6)
        equation.scale_to_fit_width(config.frame_width - 1.0).move_to(ORIGIN)

        self.add(equation)


# ─────────────────────────────────────────────────────────────────────────────
# Animated version: the tensor decomposes into its rank-one terms
#   manim -qh cp_decomposition.py CPDecompositionAnim
# ─────────────────────────────────────────────────────────────────────────────

class CPDecompositionAnim(Scene):
    def construct(self):
        # Build the final layout first, so every piece has its resting place.
        cube = tensor_cube()
        cube_caption = VGroup(
            Tex("3-way tensor").scale(0.8),
            MathTex(r"\mathcal{A}").scale(1.1),
        ).arrange(DOWN, buff=0.12).next_to(cube, DOWN, buff=0.45)
        lhs = VGroup(cube, cube_caption)

        eq = MathTex("=").scale(1.6)

        t1, t2, tr = rank_one_triad(1), rank_one_triad(2), rank_one_triad("r")
        plus = MathTex("+").scale(1.5)
        dots = MathTex(r"+\;\cdots\;+").scale(1.3)
        rhs = VGroup(t1, plus, t2, dots, tr).arrange(RIGHT, buff=0.55, aligned_edge=DOWN)

        equation = VGroup(lhs, eq, rhs).arrange(RIGHT, buff=0.6)
        equation.scale_to_fit_width(config.frame_width - 1.0).move_to(ORIGIN)

        terms = [t1, t2, tr]
        bars = [VGroup(*t[:3]) for t in terms]      # the three slabs
        labels = [VGroup(*t[3:]) for t in terms]    # a_i, b_i, c_i

        # --- 1. The tensor arrives big and centre stage, with a little pulse.
        intro = lhs.copy().scale(1.45).move_to(ORIGIN)
        self.play(FadeIn(intro, scale=0.6), run_time=1.0)
        self.play(intro.animate.scale(1.06), rate_func=there_and_back, run_time=0.8)

        # --- 2. It slides to the left and the equals sign is written.
        self.play(Transform(intro, lhs), run_time=1.0)
        self.play(Write(eq), run_time=0.6)

        # --- 3. Each term peels off as a copy of the cube and bursts into bars.
        connectors = [None, plus, dots]
        for term_bars, term_labels, connector in zip(bars, labels, connectors):
            if connector is not None:
                self.play(FadeIn(connector, scale=0.5), run_time=0.4)
            self.play(
                TransformFromCopy(intro[0], term_bars, lag_ratio=0.12),
                run_time=1.3,
            )
            self.play(
                LaggedStart(*[Write(m) for m in term_labels], lag_ratio=0.2),
                run_time=0.7,
            )

        # --- 4. A final shimmer across the right-hand side.
        self.play(
            LaggedStart(*[Indicate(b, scale_factor=1.12, color=None) for b in bars],
                        lag_ratio=0.25),
            run_time=1.6,
        )
        self.wait(1.0)
