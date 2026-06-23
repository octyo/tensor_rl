# ── Action constants ──────────────────────────────────────────────────────────
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
NUM_ACTIONS = 4


_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

class GridWorld:
    """Deterministic grid world.

    Goal  = bottom-right corner (rows-1, cols-1), terminal with reward 0.
    Every non-terminal step gives reward -1.
    Wall collisions are absorbed (agent stays in place).
    """

    def __init__(self, rows: int = 1000, cols: int = 1000):
        self.rows = rows
        self.cols = cols
        self.goal = (rows - 1, cols - 1)
        self._state: tuple = (0, 0)

    def reset(self) -> tuple:
        self._state = (0, 0)
        return self._state

    def step(self, action: int) -> tuple[tuple, float, bool]:
        dr, dc = _DELTA[action]
        r, c = self._state
        r = max(0, min(self.rows - 1, r + dr))
        c = max(0, min(self.cols - 1, c + dc))
        self._state = (r, c)
        done = self._state == self.goal
        reward = 0.0 if done else -1.0
        return self._state, reward, done

    def all_states(self) -> list[tuple]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def nonterminal_states(self) -> list[tuple]:
        return [s for s in self.all_states() if s != self.goal]

    def transition(self, state: tuple, action: int) -> tuple[tuple, float]:
        """Deterministic transition for DP (does not modify internal state)."""
        dr, dc = _DELTA[action]
        r, c = state
        r = max(0, min(self.rows - 1, r + dr))
        c = max(0, min(self.cols - 1, c + dc))
        ns = (r, c)
        reward = 0.0 if ns == self.goal else -1.0
        return ns, reward