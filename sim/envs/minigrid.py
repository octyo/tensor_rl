"""MiniGrid-based environments for the CP-vs-tabular study.

Built on top of gymnasium-MiniGrid (`minigrid`): every layout is a real
`MiniGridEnv` subclass that builds its map with `Grid`/`Wall`/`Goal` and is
renderable like any MiniGrid env. The user's own `SimpleEnv` (locked door + key)
lives here too.

The CP study, however, needs a *dense*, directionless, fully-known MDP so it can
(a) compute exact Q* by value iteration, (b) treat the Q-table as a clean
`(row x col x action)` tensor, and (c) get the smooth -1/step reward surface that
makes the low-rank story work. So for training/analysis we *derive* a dense
GridWorld from each MiniGrid layout by extracting its interior walls and goal
(`extract_layout`). The MiniGrid env is the source of truth for the map; the
derived `*Grid` env provides the dense dynamics. The two share the exact same
wall geometry.

  MiniGrid env (renderable)        derived dense study env (LAYOUTS)
  ------------------------         ---------------------------------
  OpenEnv                          OpenGrid
  ChicaneEnv                       ChicaneGrid
  SymmetricEnv                     SymmetricGrid
  IslandsEnv                       IslandsGrid
  SimpleEnv (door+key)             - (kept as a real MiniGrid env)
"""

import math
from collections import deque

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

from envs.gridworld import GridWorld, NUM_ACTIONS, _DELTA
from envs.obstacle_gridworld import ObstacleGridWorld, chicane_walls

__all__ = ["OpenEnv", "ChicaneEnv", "SymmetricEnv", "IslandsEnv", "SwirlEnv",
           "SimpleEnv", "OpenGrid", "ChicaneGrid", "SymmetricGrid", "IslandsGrid",
           "SwirlGrid", "LAYOUTS", "MINIGRID_ENVS", "extract_layout",
           "chicane_walls", "symmetric_walls", "island_walls", "swirl_walls"]


# ── wall-geometry generators (interior (row,col) coords, 0-indexed) ──────────

def _bres(a, b):
    (r0, c0), (r1, c1) = a, b
    cells = set(); dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1; sc = 1 if c1 > c0 else -1
    err = dr - dc; r, c = r0, c0
    while True:
        cells.add((r, c))
        if (r, c) == (r1, c1):
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc
    return cells


_SYM_UPPER = [(0.00, 0.30), (0.12, 0.285), (0.26, 0.305), (0.40, 0.29), (0.54, 0.30),
              (0.66, 0.315), (0.72, 0.37), (0.725, 0.45)]


def symmetric_walls(size):
    pts = [(round(y * (size - 1)), round(x * (size - 1))) for (x, y) in _SYM_UPPER]
    walls = set()
    for i in range(len(pts) - 1):
        walls |= _bres(pts[i], pts[i + 1])
    walls |= {(size - 1 - r, size - 1 - c) for (r, c) in walls}   # 180-degree rotation
    walls.discard((0, 0)); walls.discard((size - 1, size - 1))
    return frozenset((r, c) for (r, c) in walls if 0 <= r < size and 0 <= c < size)


def _connected_to_goal(walls, size):
    free = {(r, c) for r in range(size) for c in range(size) if (r, c) not in walls}
    goal = (size - 1, size - 1)
    if goal not in free:
        return False
    seen = {goal}; q = deque(seen)
    while q:
        r, c = q.popleft()
        for dr, dc in _DELTA.values():
            nb = (r + dr, c + dc)
            if nb in free and nb not in seen:
                seen.add(nb); q.append(nb)
    return len(seen) == len(free)


def island_walls(size, n_islands=None, seed=1, max_block=2):
    if n_islands is None:
        n_islands = max(8, round(size * size * 0.07))
    import random as _random
    forbidden = set()
    for s in [(0, 0), (size - 1, size - 1)]:
        forbidden.add(s)
        for dr, dc in _DELTA.values():
            forbidden.add((s[0] + dr, s[1] + dc))
    for attempt in range(seed, seed + 2000):
        rng = _random.Random(attempt)
        walls = set()
        for _ in range(n_islands):
            r = rng.randint(0, size - 1); c = rng.randint(0, size - 1)
            bh = rng.randint(1, max_block); bw = rng.randint(1, max_block)
            block = {(r + i, c + j) for i in range(bh) for j in range(bw)
                     if r + i < size and c + j < size}
            if block & forbidden:
                continue
            walls |= block
        walls -= forbidden
        if walls and _connected_to_goal(walls, size):
            return frozenset(walls)
    raise RuntimeError("no connected island layout found")


def _arc(cx, cy, rad, a0, a1, size, n=90):
    """Rasterise a circular arc (normalised center/radius) to (row,col) cells."""
    pts = []
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / n)
        x = cx + rad * math.cos(a)
        y = cy + rad * math.sin(a)
        pts.append((round(y * (size - 1)), round(x * (size - 1))))
    cells = set()
    for i in range(len(pts) - 1):
        cells |= _bres(pts[i], pts[i + 1])
    return cells


def swirl_walls(size):
    """3-arm swirl, mirror-symmetric about the main (TL->BR) diagonal.

    A big arc centred on the diagonal (self-symmetric) + a right-edge blade; the
    transpose then adds the bottom-edge blade and guarantees exact symmetry.
    """
    big = _arc(0.62, 0.62, 0.45, 170, 280, size)     # centre on diagonal
    blade = _arc(0.78, 0.20, 0.27, 42, 143, size)    # right-edge blade
    w = big | blade
    w |= {(c, r) for (r, c) in w}                     # main-diagonal symmetry
    w.discard((0, 0)); w.discard((size - 1, size - 1))
    return frozenset((r, c) for (r, c) in w if 0 <= r < size and 0 <= c < size)


# ── real MiniGrid env layouts ────────────────────────────────────────────────
# grid_size = inner_size + 2 (the +2 is MiniGrid's surrounding wall ring); the
# interior cell (col+1, row+1) corresponds to dense-grid cell (row, col).

def _mission():
    return "reach the goal"


class _LayoutEnv(MiniGridEnv):
    """Empty-by-default MiniGrid; subclasses supply interior walls via wall_fn."""

    @staticmethod
    def wall_fn(inner):
        return frozenset()

    def __init__(self, inner_size=20, max_steps=None, **kwargs):
        self.inner = inner_size
        if max_steps is None:
            max_steps = 4 * (inner_size + 2) ** 2
        super().__init__(mission_space=MissionSpace(mission_func=_mission),
                         grid_size=inner_size + 2, see_through_walls=True,
                         max_steps=max_steps, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for (r, c) in self.wall_fn(self.inner):
            self.grid.set(c + 1, r + 1, Wall())
        self.put_obj(Goal(), width - 2, height - 2)
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.mission = _mission()


class OpenEnv(_LayoutEnv):
    pass


class ChicaneEnv(_LayoutEnv):
    @staticmethod
    def wall_fn(inner):
        return chicane_walls(inner, inner)


class SymmetricEnv(_LayoutEnv):
    @staticmethod
    def wall_fn(inner):
        return symmetric_walls(inner)


class IslandsEnv(_LayoutEnv):
    @staticmethod
    def wall_fn(inner):
        return island_walls(inner, seed=1)


class SwirlEnv(_LayoutEnv):
    @staticmethod
    def wall_fn(inner):
        return swirl_walls(inner)


class SimpleEnv(MiniGridEnv):
    """User's env: vertical separation wall with a locked door + key, goal corner.
    Moved here from code/Reinforcement_Learning/boilerplate_test.py."""

    def __init__(self, size=10, agent_start_pos=(1, 1), agent_start_dir=0,
                 max_steps: int | None = None, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        if max_steps is None:
            max_steps = 4 * size ** 2
        super().__init__(mission_space=MissionSpace(mission_func=lambda: "grand mission"),
                         grid_size=size, see_through_walls=True,
                         max_steps=max_steps, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for i in range(0, height):
            self.grid.set(5, i, Wall())
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, Key(COLOR_NAMES[0]))
        self.put_obj(Goal(), width - 2, height - 2)
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.mission = "grand mission"


MINIGRID_ENVS = {"open": OpenEnv, "chicane": ChicaneEnv, "symmetric": SymmetricEnv,
                 "islands": IslandsEnv, "swirl": SwirlEnv, "simple": SimpleEnv}


# ── derive a dense GridWorld from a MiniGrid layout ──────────────────────────

def extract_layout(mg_env):
    """Read interior walls + goal from a (reset) MiniGrid env.

    Returns (inner_size, walls, goal) in dense-grid (row,col) coords. Doors are
    treated as passable gaps (the dense MDP has no key/inventory).
    """
    mg_env.reset()
    g = mg_env.grid
    W, H = g.width, g.height
    walls = set(); goal = None
    for x in range(W):
        for y in range(H):
            obj = g.get(x, y)
            if obj is None:
                continue
            if obj.type == "wall" and 1 <= x < W - 1 and 1 <= y < H - 1:
                walls.add((y - 1, x - 1))
            elif obj.type == "goal":
                goal = (y - 1, x - 1)
    return W - 2, frozenset(walls), goal


class _StudyGrid(ObstacleGridWorld):
    """Dense (row,col)x4 MDP whose walls are extracted from a MiniGrid layout."""

    minigrid_cls = OpenEnv

    def __init__(self, size=20, random_start=False):
        self.minigrid = self.minigrid_cls(inner_size=size)
        inner, walls, _goal = extract_layout(self.minigrid)
        super().__init__(inner, inner, walls=walls, random_start=random_start)


class OpenGrid(_StudyGrid):
    minigrid_cls = OpenEnv


class ChicaneGrid(_StudyGrid):
    minigrid_cls = ChicaneEnv


class SymmetricGrid(_StudyGrid):
    minigrid_cls = SymmetricEnv


class IslandsGrid(_StudyGrid):
    minigrid_cls = IslandsEnv


class SwirlGrid(_StudyGrid):
    minigrid_cls = SwirlEnv


LAYOUTS = {"open": OpenGrid, "chicane": ChicaneGrid, "symmetric": SymmetricGrid,
           "islands": IslandsGrid, "swirl": SwirlGrid}
