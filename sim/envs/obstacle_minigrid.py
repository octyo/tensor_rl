"""Swappable grid-world layouts as standalone env classes.

Each class is an ObstacleGridWorld (reward -1/step, goal bottom-right, 4 actions)
with a fixed wall layout, parametrised by `size` and `random_start`. Use the
LAYOUTS registry to pick one by name. These drop straight into value_iteration_q,
the agents, and all the plotting/experiment code.

    OpenGrid       no walls (smooth, low-rank value surface)
    ChicaneGrid    one forced down->across->up detour
    SymmetricGrid  180-degree-symmetric hooked walls; right edge sealed
    IslandsGrid    scattered random islands (kept fully connected)
"""

from collections import deque

from envs.gridworld import GridWorld, NUM_ACTIONS, _DELTA
from envs.obstacle_gridworld import ObstacleGridWorld, chicane_walls

__all__ = ["OpenGrid", "ChicaneGrid", "SymmetricGrid", "IslandsGrid", "LAYOUTS",
           "chicane_walls", "symmetric_walls", "island_walls"]


# ── wall generators ──────────────────────────────────────────────────────────

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


# upper hooked wall (normalised x,y in [0,1]); the lower wall is its 180-deg rotation
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
    """Scatter small rectangular islands; regenerate until fully connected."""
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


# ── env classes ──────────────────────────────────────────────────────────────

class OpenGrid(ObstacleGridWorld):
    def __init__(self, size=20, random_start=False):
        super().__init__(size, size, walls=frozenset(), random_start=random_start)


class ChicaneGrid(ObstacleGridWorld):
    def __init__(self, size=20, random_start=False):
        super().__init__(size, size, walls=chicane_walls(size, size),
                         random_start=random_start)


class SymmetricGrid(ObstacleGridWorld):
    def __init__(self, size=20, random_start=False):
        super().__init__(size, size, walls=symmetric_walls(size),
                         random_start=random_start)


class IslandsGrid(ObstacleGridWorld):
    def __init__(self, size=20, random_start=False, seed=1):
        super().__init__(size, size, walls=island_walls(size, seed=seed),
                         random_start=random_start)


LAYOUTS = {"open": OpenGrid, "chicane": ChicaneGrid,
           "symmetric": SymmetricGrid, "islands": IslandsGrid}
