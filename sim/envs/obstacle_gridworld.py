"""GridWorld with internal walls (a maze) that force detours away from the goal.

Same contract as envs/gridworld.GridWorld (reward -1/step, goal at bottom-right,
4 actions) so it drops straight into value_iteration_q, the agents, and all the
plotting/experiment code. The only additions are impassable `walls` cells and an
optional random start.

The default layout is a "narrow" detour that forces the path to *double back*: two sealing
walls create (a) a left zone connected to a vertical corridor only at the very
bottom, and (b) that corridor connected to the goal column only via a gap partway
up. So the shortest route runs down the left edge to the bottom-left corner, along
the bottom, then *climbs back up* the corridor toward the middle, and finally down
the far-right column to the goal — i.e. it must move away from (above) the goal for
a stretch. This breaks the near-additive (low-rank) structure of the open grid,
which is the whole point of the obstacle test.

A serpentine maze is also provided for comparison.
"""

from envs.gridworld import GridWorld, NUM_ACTIONS, _DELTA


def narrow_walls(rows: int, cols: int) -> frozenset:
    """Two sealing walls that force a down -> across -> back-up -> down detour.

    left_col is walled top-to-(bottom-1) so the left zone reaches the corridor
    only at the bottom row; strip_col is walled everywhere except one mid-height
    gap, so the corridor reaches the goal column only by climbing back up to it.
    """
    left_col = cols - 4
    strip_col = cols - 2
    mid_gap = rows // 3
    walls = {(r, left_col) for r in range(0, rows - 1)}
    walls |= {(r, strip_col) for r in range(0, rows) if r != mid_gap}
    walls.discard((0, 0))
    walls.discard((rows - 1, cols - 1))
    return frozenset(walls)


def serpentine_walls(rows: int, cols: int, gap: int = 2) -> frozenset:
    """Horizontal barriers at ~1/4, 1/2, 3/4 height with alternating L/R gaps."""
    walls = set()
    wall_rows = [rows // 4, rows // 2, (3 * rows) // 4]
    for i, wr in enumerate(wall_rows):
        walled_cols = range(0, cols - gap) if i % 2 == 0 else range(gap, cols)
        for c in walled_cols:
            walls.add((wr, c))
    walls.discard((0, 0))                 # never wall the start
    walls.discard((rows - 1, cols - 1))   # never wall the goal
    return frozenset(walls)


class ObstacleGridWorld(GridWorld):
    """Deterministic grid world with impassable wall cells.

    walls         : iterable of (row, col) wall cells; default = serpentine maze.
    random_start  : if True, reset() drops on a uniformly random non-wall, non-goal
                    cell (otherwise always (0, 0)).
    Moving into a wall (or off-grid) leaves the agent in place, just like a
    boundary collision in the base GridWorld.
    """

    def __init__(self, rows: int = 12, cols: int = 12, walls=None,
                 random_start: bool = False):
        super().__init__(rows, cols)
        self.walls = frozenset(walls) if walls is not None else narrow_walls(rows, cols)
        self.random_start = random_start
        self._free = [s for s in self._all_cells()
                      if s not in self.walls and s != self.goal]

    def _all_cells(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def _move(self, state, action):
        dr, dc = _DELTA[action]
        r, c = state
        nr = max(0, min(self.rows - 1, r + dr))
        nc = max(0, min(self.cols - 1, c + dc))
        if (nr, nc) in self.walls:        # blocked -> stay put
            nr, nc = r, c
        return nr, nc

    def reset(self) -> tuple:
        if self.random_start:
            import random
            self._state = random.choice(self._free)
        else:
            self._state = (0, 0)
        return self._state

    def step(self, action: int):
        self._state = self._move(self._state, action)
        done = self._state == self.goal
        return self._state, (0.0 if done else -1.0), done

    def transition(self, state: tuple, action: int):
        ns = self._move(state, action)
        return ns, (0.0 if ns == self.goal else -1.0)

    def all_states(self):
        return [s for s in self._all_cells() if s not in self.walls]

    def nonterminal_states(self):
        return [s for s in self.all_states() if s != self.goal]
