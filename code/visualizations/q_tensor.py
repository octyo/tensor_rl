"""
Manim figure: the Q-function as a 3-way tensor for the report.

Instead of flattening the state into one axis S (giving a 2-D table S x A, see
q_table.py), the state's grid coordinates (x, y) become their own tensor modes.
The Q-function is then a 3-way tensor

    Q(x, y, a)  in  R^{X x Y x A}

drawn as a grid of voxels, coloured by Q-value (same red->green ramp as the
table) and with the greedy action argmax_a Q(x, y, a) outlined in blue.

Render a static image (saves the last frame as a PNG):
    manim -s -qh q_tensor.py QTensor

Output lands in media/images/q_tensor/.
"""

from manim import *
import numpy as np

# Reuse the isometric projection from the CP figure and the palette from the
# 2-D Q-table so all three report figures share one visual language.
from cp_decomposition import P, _depth, iso_box
from q_table import _RAMP, _value_color, _POLICY_COLOR

X, Y, A = 5, 2, 3                       # state-x, state-y, actions
ACTIONS = ["turn left", "turn right", "move forward"]
PITCH = 1.0                             # voxel centre-to-centre spacing
VOX = 0.82                              # voxel side length (gap = PITCH - VOX)
_VOX_OPACITY = 0.92

# Push the voxel block out into the positive octant so the axis frame (the y
# axis especially) sits clear in the lower-left instead of under the voxels.
SHIFT = np.array([1.3, 0.0, 1.0])


def _make_q_tensor(seed=7):
    """Structured Q-tensor: value grows toward the (high-x, high-y) goal."""
    rng = np.random.default_rng(seed)
    q = np.zeros((X, Y, A))
    for x in range(X):
        for y in range(Y):
            base = 0.2 + 0.7 * (0.6 * x / (X - 1) + 0.4 * y / (Y - 1))
            q[x, y] = np.clip(base + rng.uniform(-0.12, 0.22, A), 0, 1)
    return np.round(q * 10 - 1, 1)


def _axis_arrow(origin, vec, color=GREY_B):
    return Arrow(P(origin), P(np.asarray(origin, float) + np.asarray(vec, float)),
                 buff=0.0, color=color, stroke_width=4,
                 max_tip_length_to_length_ratio=0.12)


def build_q_tensor(q=None):
    """Build the 3-way Q-tensor mobject: VGroup(floor, block, axes).

    `q` is an X x Y x A array; if omitted a structured demo tensor is generated.
    """
    if q is None:
        q = _make_q_tensor()
    lo, hi = q.min(), q.max()
    greedy = {(x, y): int(np.argmax(q[x, y])) for x in range(X) for y in range(Y)}

    # ── voxels, drawn back-to-front for correct occlusion ────────────────
    block = VGroup()
    voxels = []
    for x in range(X):
        for y in range(Y):
            for a in range(A):
                centre = np.array([x * PITCH, y * PITCH, a * PITCH]) + SHIFT
                corner = centre - VOX / 2
                t = (q[x, y, a] - lo) / (hi - lo)
                is_greedy = greedy[(x, y)] == a
                vox = iso_box(
                    corner,
                    [VOX, 0, 0], [0, VOX, 0], [0, 0, VOX],
                    fill=_value_color(t),
                    stroke=_POLICY_COLOR if is_greedy else GREY_E,
                    base_op=_VOX_OPACITY,
                    sw=3.0 if is_greedy else 1.0,
                )
                voxels.append((_depth(centre), vox))
    for _, vox in sorted(voxels, key=lambda dv: dv[0]):
        block.add(vox)

    # ── axis frame from the origin, reaching just past the shifted block ──
    m = 0.85                                            # origin margin
    o = np.array([-VOX / 2 - m, -VOX / 2 - m, -VOX / 2 - m])

    # ── dashed (x, y) coordinate grid on the floor beneath the tensor ────
    zf = o[2]
    xs = [(i - 0.5) * PITCH + SHIFT[0] for i in range(X + 1)]
    ys = [(j - 0.5) * PITCH + SHIFT[1] for j in range(Y + 1)]
    floor = VGroup()
    for xe in xs:
        floor.add(DashedLine(P([xe, ys[0], zf]), P([xe, ys[-1], zf]),
                             color=GREY_B, stroke_width=2, dash_length=0.11))
    for ye in ys:
        floor.add(DashedLine(P([xs[0], ye, zf]), P([xs[-1], ye, zf]),
                             color=GREY_B, stroke_width=2, dash_length=0.11))
    for x in range(X):
        for y in range(Y):
            c = P([x * PITCH + SHIFT[0], y * PITCH + SHIFT[1], zf])
            floor.add(MathTex(rf"({x},{y})", color=GREY_A).scale(0.5).move_to(c))
    end_x = (X - 1) * PITCH + VOX / 2 + SHIFT[0] + 0.55
    end_y = (Y - 1) * PITCH + VOX / 2 + SHIFT[1] + 0.55
    end_a = (A - 1) * PITCH + VOX / 2 + SHIFT[2] + 0.55
    ax_x = _axis_arrow(o, [end_x - o[0], 0, 0])
    ax_y = _axis_arrow(o, [0, end_y - o[1], 0])
    ax_a = _axis_arrow(o, [0, 0, end_a - o[2]])
    lx = MathTex("x", color=GREY_A).scale(1.0).next_to(P([end_x, o[1], o[2]]), DR, buff=0.1)
    ly = MathTex("y", color=GREY_A).scale(1.0).next_to(P([o[0], end_y, o[2]]), UL, buff=0.1)
    la = MathTex("a", color=GREY_A).scale(1.0).next_to(P([o[0], o[1], end_a]), UP, buff=0.1)
    axes = VGroup(ax_x, ax_y, ax_a, lx, ly, la)

    return VGroup(floor, block, axes)                 # floor behind, axes on top


class QTensor(Scene):
    def construct(self):
        tensor = build_q_tensor()

        # ── caption ──────────────────────────────────────────────────────────
        title = MathTex(
            r"Q(x, y, a)\;\in\;\mathbb{R}^{X \times Y \times A}", font_size=42,
        )
        subtitle = Text(
            "state (x, y) split into two tensor modes;   "
            "blue = greedy action  argmaxₐ Q(x, y, a)",
            font_size=22, color=GREY_B,
        )
        legend = VGroup(*[
            Text(f"a = {i}:  {name}", font_size=20, color=GREY_A)
            for i, name in enumerate(ACTIONS)
        ]).arrange(RIGHT, buff=0.6)
        caption = VGroup(title, subtitle, legend).arrange(DOWN, buff=0.22)

        # ── assemble and fit to frame ────────────────────────────────────────
        scene = VGroup(tensor, caption).arrange(DOWN, buff=0.55)
        scene.scale_to_fit_height(config.frame_height - 0.7)
        if scene.width > config.frame_width - 0.8:
            scene.scale_to_fit_width(config.frame_width - 0.8)
        scene.move_to(ORIGIN)

        self.add(scene)
