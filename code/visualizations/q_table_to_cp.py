"""
Manim animation: swapping a Q-table for a CP decomposition.

Render with:
    pip install manim
    manim -pql q_table_to_cp.py QTableToCP        # quick low-quality preview
    manim -pqh q_table_to_cp.py QTableToCP        # high-quality

Scenes in this file
-------------------
QTableToCP   — main scene (recommended entry point):
    Part 1  — classic Q-table: what it stores and the memory cost
    Part 2  — CP swap: factor_s (S×R) and factor_a (A×R) replace the table
    Part 3  — lookup: Q(s,a) = factor_s[s] · factor_a[a]  (dot product)
    Part 4  — multi-dim CP: N state dimensions each get their own factor matrix
"""

from manim import *
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cell_center(i, j, cell_w, cell_h, rows, cols):
    return np.array([
        j * cell_w - (cols - 1) * cell_w / 2,
        -i * cell_h + (rows - 1) * cell_h / 2,
        0,
    ])


def matrix_block(rows, cols, cell_w=0.50, cell_h=0.40,
                 fill=BLUE_D, opacity=0.35, stroke=WHITE, sw=1.0):
    """VGroup of individual Rectangle cells (one per entry)."""
    g = VGroup()
    for i in range(rows):
        for j in range(cols):
            rect = Rectangle(
                width=cell_w, height=cell_h,
                fill_color=fill, fill_opacity=opacity,
                stroke_color=stroke, stroke_width=sw,
            )
            rect.move_to(_cell_center(i, j, cell_w, cell_h, rows, cols))
            g.add(rect)
    return g


def row_highlight(grid_block, row_idx, cols, color=YELLOW, opacity=0.75):
    """Return copies of all cells in row_idx of grid_block, filled with color."""
    cells = VGroup()
    for j in range(cols):
        c = grid_block[row_idx * cols + j].copy()
        c.set_fill(color, opacity=opacity)
        c.set_stroke(color, width=2)
        cells.add(c)
    return cells


def labeled_matrix(rows, cols, row_label, col_label,
                   fill=BLUE_D, cell_w=0.50, cell_h=0.40,
                   label_fs=26, brace_buff=0.15):
    """Returns (VGroup_total, grid, row_brace_grp, col_brace_grp)."""
    grid = matrix_block(rows, cols, cell_w=cell_w, cell_h=cell_h, fill=fill)

    rb = Brace(grid, direction=LEFT, buff=brace_buff)
    rb_lbl = MathTex(row_label, font_size=label_fs)
    rb.put_at_tip(rb_lbl, buff=0.1)
    row_brace = VGroup(rb, rb_lbl)

    cb = Brace(grid, direction=UP, buff=brace_buff)
    cb_lbl = MathTex(col_label, font_size=label_fs)
    cb.put_at_tip(cb_lbl, buff=0.1)
    col_brace = VGroup(cb, cb_lbl)

    total = VGroup(grid, row_brace, col_brace)
    return total, grid, row_brace, col_brace


# ─────────────────────────────────────────────────────────────────────────────
# Main scene
# ─────────────────────────────────────────────────────────────────────────────

class QTableToCP(Scene):
    # Display dimensions (kept small so grids fit on screen)
    S = 6    # states
    A = 4    # actions
    R = 2    # CP rank

    # ── convenience ──────────────────────────────────────────────────────────

    def _section_title(self, text):
        t = Text(text, font_size=34, weight=BOLD)
        t.to_edge(UP, buff=0.3)
        return t

    def _formula_below(self, mob, tex, fs=28, buff=0.35, color=WHITE):
        f = MathTex(tex, font_size=fs, color=color)
        f.next_to(mob, DOWN, buff=buff)
        return f

    # ── Part 1 — classic Q-table ─────────────────────────────────────────────

    def _part1_q_table(self):
        title = self._section_title("Part 1 — Classic Q-Table")
        self.play(Write(title))

        total, grid, rb, cb = labeled_matrix(
            self.S, self.A, "S", "A", fill=BLUE_D,
        )
        total.move_to(ORIGIN + DOWN * 0.3)

        q_type = MathTex(
            r"Q \;\in\; \mathbb{R}^{S \times A}",
            font_size=30,
        ).next_to(total, DOWN, buff=0.45)

        self.play(FadeIn(grid), run_time=0.8)
        self.play(FadeIn(rb), FadeIn(cb))
        self.play(Write(q_type))
        self.wait(0.8)

        # Highlight one cell and annotate Q(s, a)
        mid_row, mid_col = self.S // 2, self.A // 2
        cell = grid[mid_row * self.A + mid_col]
        hl = cell.copy().set_fill(YELLOW, opacity=0.9).set_stroke(YELLOW, width=2.5)
        hl_lbl = MathTex(r"Q(s,a)", font_size=18, color=BLACK).move_to(hl)
        self.play(FadeIn(hl), Write(hl_lbl))
        self.wait(0.6)

        # Memory cost callout
        cost_box = SurroundingRectangle(q_type, color=RED, buff=0.15, corner_radius=0.08)
        cost_lbl = Text(
            f"Memory: S × A  =  {self.S} × {self.A}  =  {self.S * self.A} values",
            font_size=22, color=RED,
        ).next_to(cost_box, DOWN, buff=0.2)
        self.play(Create(cost_box), Write(cost_lbl))
        self.wait(1.5)

        # Scale-up warning
        scale_warn = MathTex(
            r"\text{Scales as } \mathcal{O}(S \cdot A) \;\longrightarrow\; \text{huge for large } S",
            font_size=22, color=ORANGE,
        ).next_to(cost_lbl, DOWN, buff=0.2)
        self.play(Write(scale_warn))
        self.wait(2)

        self.play(
            FadeOut(title), FadeOut(total), FadeOut(q_type),
            FadeOut(cost_box), FadeOut(cost_lbl), FadeOut(scale_warn),
            FadeOut(hl), FadeOut(hl_lbl),
        )

    # ── Part 2 — CP decomposition swap ───────────────────────────────────────

    def _part2_cp_swap(self):
        title = self._section_title("Part 2 — CP Decomposition")
        self.play(Write(title))

        # ── Left: original Q-table (small, faded) ────────────────────────────
        q_total, q_grid, q_rb, q_cb = labeled_matrix(
            self.S, self.A, "S", "A", fill=BLUE_D,
        )
        q_total.scale(0.8).to_edge(LEFT, buff=1.2).shift(DOWN * 0.2)
        q_lbl = MathTex(r"Q", font_size=28).next_to(q_total, DOWN, buff=0.3)

        self.play(FadeIn(q_total, shift=LEFT * 0.3), Write(q_lbl))
        self.wait(0.5)

        # ── Arrow and "replace with" ──────────────────────────────────────────
        swap_arrow = Arrow(
            q_total.get_right() + RIGHT * 0.15,
            q_total.get_right() + RIGHT * 1.4,
            buff=0, color=WHITE, stroke_width=3,
        )
        swap_lbl = Text("swap", font_size=20, color=GREY_A).next_to(swap_arrow, UP, buff=0.08)
        self.play(GrowArrow(swap_arrow), Write(swap_lbl))

        # ── Right: factor_s (S×R) and factor_a (A×R) ─────────────────────────
        fs_total, fs_grid, fs_rb, fs_cb = labeled_matrix(
            self.S, self.R, "S", "R", fill=GREEN_D, cell_w=0.55, cell_h=0.40,
        )
        fa_total, fa_grid, fa_rb, fa_cb = labeled_matrix(
            self.A, self.R, "A", "R", fill=ORANGE, cell_w=0.55, cell_h=0.40,
        )

        factors_group = VGroup(fs_total, fa_total).arrange(RIGHT, buff=0.9)
        factors_group.move_to(RIGHT * 2.2 + DOWN * 0.2)

        fs_lbl = MathTex(r"\mathbf{F}_s", font_size=26, color=GREEN_D).next_to(fs_total, DOWN, buff=0.3)
        fa_lbl = MathTex(r"\mathbf{F}_a", font_size=26, color=ORANGE).next_to(fa_total, DOWN, buff=0.3)

        times = MathTex(r"\times", font_size=28).move_to(
            (fs_total.get_right() + fa_total.get_left()) / 2
        )

        self.play(FadeIn(fs_total, shift=UP * 0.2), Write(fs_lbl))
        self.play(Write(times))
        self.play(FadeIn(fa_total, shift=UP * 0.2), Write(fa_lbl))
        self.wait(0.5)

        # ── Reconstruction formula ─────────────────────────────────────────────
        formula = MathTex(
            r"Q \;=\; \mathbf{F}_s \,\mathbf{F}_a^\top",
            font_size=28,
        ).next_to(factors_group, DOWN, buff=0.5)
        self.play(Write(formula))
        self.wait(0.8)

        # ── Memory comparison ─────────────────────────────────────────────────
        mem_before = MathTex(
            rf"S \times A = {self.S} \times {self.A} = {self.S * self.A}",
            font_size=22, color=RED,
        )
        mem_after = MathTex(
            rf"(S + A) \times R = ({self.S} + {self.A}) \times {self.R} = {(self.S + self.A) * self.R}",
            font_size=22, color=GREEN,
        )
        mem_group = VGroup(mem_before, mem_after).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        mem_group.next_to(formula, DOWN, buff=0.35)

        mem_title = Text("Memory:", font_size=21, color=GREY_A).next_to(mem_group, LEFT, buff=0.2)
        self.play(Write(mem_title), Write(mem_before))
        self.play(Write(mem_after))
        self.wait(2)

        # ── Rank annotation ───────────────────────────────────────────────────
        rank_note = MathTex(
            r"R \ll \min(S, A) \;\Rightarrow\; \text{compact representation}",
            font_size=22, color=YELLOW,
        ).next_to(mem_group, DOWN, buff=0.25)
        self.play(Write(rank_note))
        self.wait(2)

        self.play(
            FadeOut(title), FadeOut(q_total), FadeOut(q_lbl),
            FadeOut(swap_arrow), FadeOut(swap_lbl),
            FadeOut(fs_total), FadeOut(fs_lbl),
            FadeOut(fa_total), FadeOut(fa_lbl),
            FadeOut(times), FadeOut(formula),
            FadeOut(mem_title), FadeOut(mem_before), FadeOut(mem_after),
            FadeOut(rank_note),
        )

    # ── Part 3 — lookup: dot-product Q(s, a) ─────────────────────────────────

    def _part3_lookup(self):
        title = self._section_title("Part 3 — Computing Q(s, a)")
        self.play(Write(title))

        # Factor matrices
        fs_total, fs_grid, _, _ = labeled_matrix(
            self.S, self.R, "S", "R", fill=GREEN_D, cell_w=0.56, cell_h=0.40,
        )
        fa_total, fa_grid, _, _ = labeled_matrix(
            self.A, self.R, "A", "R", fill=ORANGE, cell_w=0.56, cell_h=0.40,
        )

        fs_lbl = MathTex(r"\mathbf{F}_s", font_size=26, color=GREEN_D)
        fa_lbl = MathTex(r"\mathbf{F}_a", font_size=26, color=ORANGE)

        fs_group = VGroup(fs_total, fs_lbl).arrange(DOWN, buff=0.2)
        fa_group = VGroup(fa_total, fa_lbl).arrange(DOWN, buff=0.2)

        matrices = VGroup(fs_group, fa_group).arrange(RIGHT, buff=1.5)
        matrices.move_to(UP * 0.4)

        self.play(FadeIn(matrices))
        self.wait(0.5)

        # ── Highlight row s in F_s ─────────────────────────────────────────────
        s_idx = 1   # the state we query
        a_idx = 2   # the action we query

        fs_row_hl = row_highlight(fs_grid, s_idx, self.R, color=GREEN, opacity=0.85)
        fs_row_hl.set_stroke(GREEN, width=2.5)
        fs_arrow = Arrow(
            fs_group.get_left() + LEFT * 0.6 + DOWN * (s_idx * 0.40 - (self.S - 1) * 0.20),
            fs_group.get_left() + LEFT * 0.05 + DOWN * (s_idx * 0.40 - (self.S - 1) * 0.20),
            buff=0, color=GREEN, stroke_width=2.5,
        )
        fs_arrow_lbl = MathTex(r"s", font_size=22, color=GREEN).next_to(fs_arrow, LEFT, buff=0.08)

        self.play(FadeIn(fs_row_hl), GrowArrow(fs_arrow), Write(fs_arrow_lbl))

        # ── Highlight row a in F_a ─────────────────────────────────────────────
        fa_row_hl = row_highlight(fa_grid, a_idx, self.R, color=ORANGE, opacity=0.85)
        fa_row_hl.set_stroke(ORANGE, width=2.5)
        fa_arrow = Arrow(
            fa_group.get_left() + LEFT * 0.6 + DOWN * (a_idx * 0.40 - (self.A - 1) * 0.20),
            fa_group.get_left() + LEFT * 0.05 + DOWN * (a_idx * 0.40 - (self.A - 1) * 0.20),
            buff=0, color=ORANGE, stroke_width=2.5,
        )
        fa_arrow_lbl = MathTex(r"a", font_size=22, color=ORANGE).next_to(fa_arrow, LEFT, buff=0.08)

        self.play(FadeIn(fa_row_hl), GrowArrow(fa_arrow), Write(fa_arrow_lbl))
        self.wait(0.6)

        # ── Dot-product formula ────────────────────────────────────────────────
        dot_formula = MathTex(
            r"Q(s, a) \;=\; \mathbf{F}_s[s] \;\cdot\; \mathbf{F}_a[a]",
            r"\;=\; \sum_{r=1}^{R} F_s[s, r] \cdot F_a[a, r]",
            font_size=28,
            tex_to_color_map={r"\mathbf{F}_s": GREEN, r"\mathbf{F}_a": ORANGE},
        )
        dot_formula.next_to(matrices, DOWN, buff=0.7)

        self.play(Write(dot_formula[0]))
        self.wait(0.5)
        self.play(Write(dot_formula[1]))
        self.wait(1.5)

        # ── Rank-R sum visualisation ───────────────────────────────────────────
        rank_boxes = VGroup()
        r_labels = VGroup()
        dot_labels = VGroup()
        for r in range(self.R):
            s_cell = fs_grid[s_idx * self.R + r].copy().set_fill(GREEN, 0.7)
            a_cell = fa_grid[a_idx * self.R + r].copy().set_fill(ORANGE, 0.7)

            s_cell.scale(1.1).move_to(DOWN * 2.4 + LEFT * 1.8 + RIGHT * r * 2.2)
            dot = MathTex(r"\cdot", font_size=28).next_to(s_cell, RIGHT, buff=0.15)
            a_cell.scale(1.1).next_to(dot, RIGHT, buff=0.15)

            rank_boxes.add(VGroup(s_cell, dot, a_cell))

            r_lbl = MathTex(rf"r={r+1}", font_size=18, color=YELLOW)
            r_lbl.next_to(VGroup(s_cell, a_cell), DOWN, buff=0.1)
            r_labels.add(r_lbl)

        plus_signs = VGroup()
        for k in range(self.R - 1):
            p = MathTex(r"+", font_size=26).move_to(
                (rank_boxes[k].get_right() + rank_boxes[k + 1].get_left()) / 2
            )
            plus_signs.add(p)

        self.play(
            *[FadeIn(b, shift=DOWN * 0.2) for b in rank_boxes],
            *[Write(l) for l in r_labels],
            *[Write(p) for p in plus_signs],
        )
        self.wait(2)

        self.play(
            FadeOut(title), FadeOut(matrices),
            FadeOut(fs_row_hl), FadeOut(fs_arrow), FadeOut(fs_arrow_lbl),
            FadeOut(fa_row_hl), FadeOut(fa_arrow), FadeOut(fa_arrow_lbl),
            FadeOut(dot_formula), FadeOut(rank_boxes),
            FadeOut(r_labels), FadeOut(plus_signs),
        )

    # ── Part 4 — multi-dimensional CP ─────────────────────────────────────────

    def _part4_multidim(self):
        title = self._section_title("Part 4 — Multi-Dimensional States (e.g. FrozenLake row × col)")
        self.play(Write(title))

        # Explain the 2D state case
        state_desc = MathTex(
            r"\text{State} = (i_1, i_2) \;\in\; D_1 \times D_2",
            font_size=28,
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(state_desc))
        self.wait(0.8)

        # Q-tensor label
        tensor_lbl = MathTex(
            r"Q \;\in\; \mathbb{R}^{D_1 \times D_2 \times A}",
            font_size=26,
        ).next_to(state_desc, DOWN, buff=0.35)
        self.play(Write(tensor_lbl))
        self.wait(0.6)

        # CP formula
        cp_formula = MathTex(
            r"Q[i_1, i_2, a] \;=\; \sum_{r=1}^{R}",
            r"\underbrace{F_1[i_1, r]}_{\text{row factor}}",
            r"\cdot",
            r"\underbrace{F_2[i_2, r]}_{\text{col factor}}",
            r"\cdot",
            r"\underbrace{F_a[a, r]}_{\text{action factor}}",
            font_size=24,
        )
        cp_formula[1].set_color(BLUE)
        cp_formula[3].set_color(GREEN)
        cp_formula[5].set_color(ORANGE)
        cp_formula.next_to(tensor_lbl, DOWN, buff=0.5)
        self.play(Write(cp_formula))
        self.wait(1)

        # Factor matrices side by side
        D1, D2 = 4, 4
        f1_total, f1_grid, _, _ = labeled_matrix(D1, self.R, "D_1", "R", fill=BLUE_D, cell_w=0.5, cell_h=0.38)
        f2_total, f2_grid, _, _ = labeled_matrix(D2, self.R, "D_2", "R", fill=GREEN_D, cell_w=0.5, cell_h=0.38)
        fa_total, fa_grid, _, _ = labeled_matrix(self.A, self.R, "A", "R", fill=ORANGE, cell_w=0.5, cell_h=0.38)

        f1_lbl = MathTex(r"F_1", font_size=22, color=BLUE).next_to(f1_total, DOWN, buff=0.15)
        f2_lbl = MathTex(r"F_2", font_size=22, color=GREEN).next_to(f2_total, DOWN, buff=0.15)
        fa_lbl = MathTex(r"F_a", font_size=22, color=ORANGE).next_to(fa_total, DOWN, buff=0.15)

        hadamard_1 = MathTex(r"\odot", font_size=28)
        hadamard_2 = MathTex(r"\odot", font_size=28)

        factors_row = VGroup(
            VGroup(f1_total, f1_lbl),
            hadamard_1,
            VGroup(f2_total, f2_lbl),
            hadamard_2,
            VGroup(fa_total, fa_lbl),
        ).arrange(RIGHT, buff=0.5)
        factors_row.move_to(DOWN * 1.0)

        self.play(FadeIn(factors_row, shift=DOWN * 0.3))
        self.wait(0.8)

        # Highlight one row in each factor and show Hadamard product
        s1_idx, s2_idx, a_idx = 1, 2, 1

        hl1 = row_highlight(f1_grid, s1_idx, self.R, color=BLUE, opacity=0.85)
        hl2 = row_highlight(f2_grid, s2_idx, self.R, color=GREEN, opacity=0.85)
        hl_a = row_highlight(fa_grid, a_idx, self.R, color=ORANGE, opacity=0.85)
        self.play(FadeIn(hl1), FadeIn(hl2), FadeIn(hl_a))
        self.wait(0.5)

        hadamard_note = MathTex(
            r"\mathbf{h} = F_1[i_1]\odot F_2[i_2] \in \mathbb{R}^R \quad Q(i_1,i_2,a) = \mathbf{h} \cdot F_a[a]",
            font_size=22, color=YELLOW,
        ).next_to(factors_row, DOWN, buff=0.45)
        self.play(Write(hadamard_note))
        self.wait(1.5)

        # Memory comparison
        mem_tensor = MathTex(
            r"\text{Full tensor: } D_1 \cdot D_2 \cdot A",
            font_size=22, color=RED,
        )
        mem_cp = MathTex(
            r"\text{CP factors: } (D_1 + D_2 + A) \cdot R",
            font_size=22, color=GREEN,
        )
        VGroup(mem_tensor, mem_cp).arrange(DOWN, buff=0.2, aligned_edge=LEFT).next_to(hadamard_note, DOWN, buff=0.3)
        self.play(Write(mem_tensor), Write(mem_cp))
        self.wait(2)

        # Closing remark
        closing = Text(
            "Replacing Q-tables with CP factors\n"
            "keeps the RL loop identical — only the storage and update change.",
            font_size=22, color=GREY_A, line_spacing=1.4,
        ).next_to(mem_cp, DOWN, buff=0.35)
        self.play(Write(closing))
        self.wait(3)

        self.play(*[FadeOut(m) for m in self.mobjects])

    # ── Entry point ───────────────────────────────────────────────────────────

    def construct(self):
        self._part1_q_table()
        self._part2_cp_swap()
        self._part3_lookup()
        self._part4_multidim()
