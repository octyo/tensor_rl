import numpy as np
import random

# ---------------------------------------------------------------------------
# Base for multi-dimensional tensor Q-agents
# ---------------------------------------------------------------------------

class _TensorTabularBase:
    """Abstract base: epsilon schedule + state index conversion for N-D states.

    Subclasses must implement _q_row(idx), _q_val(idx, action), _update_params(idx, action, delta).
    State is always passed as a flat integer; _to_idx converts it to a multi-D tuple.
    """

    def __init__(self, state_dims, action_space_size, rank, lr, gamma,
                 epsilon_start, epsilon_end, epsilon_decay_steps):
        self.state_dims = tuple(state_dims)
        self.action_space_size = action_space_size
        self.rank = rank
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps = 0

    def epsilon(self) -> float:
        progress = min(1.0, self.steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def _to_idx(self, state: int) -> tuple:
        return np.unravel_index(int(state), self.state_dims)

    def select_action(self, state: int, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon():
            return random.randint(0, self.action_space_size - 1)
        return int(np.argmax(self._q_row(self._to_idx(state))))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool,
               next_action: int = None):
        idx = self._to_idx(state)
        next_idx = self._to_idx(next_state)

        current_q = self._q_val(idx, action)
        if done:
            best_next = 0.0
        elif next_action is not None:  # SARSA target
            best_next = self._q_val(next_idx, next_action)
        else:  # Q-learning target
            best_next = float(np.max(self._q_row(next_idx)))

        delta = (reward + self.gamma * best_next) - current_q
        # Clip TD error tightly — multiplicative factor updates amplify large deltas
        delta = float(np.clip(delta, -1.0, 1.0))
        self._update_params(idx, action, delta)
        self.steps += 1

    def _q_row(self, idx): raise NotImplementedError
    def _q_val(self, idx, action): raise NotImplementedError
    def _update_params(self, idx, action, delta): raise NotImplementedError


# ---------------------------------------------------------------------------
# CP multi-dimensional Q-agent
# Q[i1,...,iN,a] = sum_r f1[i1,r] * f2[i2,r] * ... * fN[iN,r] * fa[a,r]
# ---------------------------------------------------------------------------

class CPMultiTabularQAgent(_TensorTabularBase):
    """CP-decomposed Q-tensor for N-dimensional state spaces.

    Works for N=1 (flat) and N>1 (e.g. FrozenLake as row×col).
    For N=1 this is identical to TensorizedTabularQAgent.
    """

    def __init__(self, state_dims, action_space_size, rank=4, lr=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000):
        super().__init__(state_dims, action_space_size, rank, lr, gamma,
                         epsilon_start, epsilon_end, epsilon_decay_steps)
        scale = 1.0 / np.sqrt(np.prod(state_dims))
        self.factors = [np.random.uniform(-scale, scale, (d, rank)) for d in state_dims]
        self.factor_a = np.random.uniform(-scale, scale, (action_space_size, rank))

    def _hadamard(self, idx) -> np.ndarray:
        """Element-wise product of factor rows across all state modes → shape (rank,)."""
        h = np.ones(self.rank)
        for n, i in enumerate(idx):
            h *= self.factors[n][i]
        return h

    def _q_row(self, idx) -> np.ndarray:
        return self._hadamard(idx) @ self.factor_a.T  # (A,)

    def _q_val(self, idx, action) -> float:
        return float(self._hadamard(idx) @ self.factor_a[action])

    _MAX_FACTOR_NORM = 3.0  # clip factor rows to prevent blowup in bilinear updates

    def _update_params(self, idx, action, delta: float):
        f = [self.factors[n][idx[n]].copy() for n in range(len(self.state_dims))]
        fa = self.factor_a[action].copy()

        for n in range(len(self.state_dims)):
            grad = fa.copy()
            for m, fm in enumerate(f):
                if m != n:
                    grad *= fm
            self.factors[n][idx[n]] += self.lr * delta * grad
            # clip row norm
            row = self.factors[n][idx[n]]
            norm = np.linalg.norm(row)
            if norm > self._MAX_FACTOR_NORM:
                self.factors[n][idx[n]] *= self._MAX_FACTOR_NORM / norm

        h = np.ones(self.rank)
        for fm in f:
            h *= fm
        self.factor_a[action] += self.lr * delta * h
        row = self.factor_a[action]
        norm = np.linalg.norm(row)
        if norm > self._MAX_FACTOR_NORM:
            self.factor_a[action] *= self._MAX_FACTOR_NORM / norm


class SarsaCPAgent(CPMultiTabularQAgent):
    """CP Q-tensor with on-policy SARSA updates."""

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool,
               next_action: int = None):
        if next_action is None and not done:
            next_action = self.select_action(next_state)
        super().update(state, action, reward, next_state, done, next_action=next_action)


# ---------------------------------------------------------------------------
# Tucker multi-dimensional Q-agent
# Q[i1,...,iN,a] = G ×1 U1[i1] ×2 U2[i2] ... ×N UN[iN] ×(N+1) Ua[a]
# G has shape (rank,)*(N+1); factors[n]: (d_n, rank); factor_a: (A, rank)
# ---------------------------------------------------------------------------

class TuckerMultiTabularQAgent(_TensorTabularBase):
    """Tucker-decomposed Q-tensor for N-dimensional state spaces.

    Core tensor G of shape (rank,)*(N+1) is contracted with one factor vector
    per mode (N state modes + 1 action mode). Unlike CP, the core captures
    cross-mode interactions, making Tucker strictly more expressive.
    """

    def __init__(self, state_dims, action_space_size, rank=4, lr=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000):
        super().__init__(state_dims, action_space_size, rank, lr, gamma,
                         epsilon_start, epsilon_end, epsilon_decay_steps)
        N = len(state_dims)
        scale = 1.0 / np.sqrt(np.prod(state_dims))
        self.G = np.random.uniform(-scale, scale, (rank,) * (N + 1))
        self.factors = [np.random.uniform(-scale, scale, (d, rank)) for d in state_dims]
        self.factor_a = np.random.uniform(-scale, scale, (action_space_size, rank))

    def _state_feature(self, idx) -> np.ndarray:
        """Contract G with all state factor vectors; action mode axis is preserved → (rank,)."""
        result = self.G
        N = len(self.state_dims)
        for n in range(N - 1, -1, -1):
            result = np.tensordot(result, self.factors[n][idx[n]], axes=([n], [0]))
        return result  # shape (rank,)

    def _q_row(self, idx) -> np.ndarray:
        return self._state_feature(idx) @ self.factor_a.T  # (A,)

    def _q_val(self, idx, action) -> float:
        return float(self._state_feature(idx) @ self.factor_a[action])

    @staticmethod
    def _contract_except(G, vs, skip_n: int) -> np.ndarray:
        """Contract G (shape (rank,)*(M)) with each v in vs except vs[skip_n].
        Contracts in reverse axis order so axis numbering stays valid.
        Returns array of shape (rank,) corresponding to the skipped axis.
        """
        result = G
        for m in sorted(range(len(vs)), reverse=True):
            if m == skip_n:
                continue
            result = np.tensordot(result, vs[m], axes=([m], [0]))
        return result

    _MAX_FACTOR_NORM = 3.0

    def _clip_row(self, vec):
        norm = np.linalg.norm(vec)
        if norm > self._MAX_FACTOR_NORM:
            vec *= self._MAX_FACTOR_NORM / norm

    def _update_params(self, idx, action, delta: float):
        N = len(self.state_dims)
        vs = [self.factors[n][idx[n]].copy() for n in range(N)] + [self.factor_a[action].copy()]
        G_old = self.G.copy()

        grad_G = vs[0].copy()
        for v in vs[1:]:
            grad_G = np.multiply.outer(grad_G, v)
        self.G += self.lr * delta * grad_G
        # Clip G element-wise
        np.clip(self.G, -self._MAX_FACTOR_NORM, self._MAX_FACTOR_NORM, out=self.G)

        for n in range(N):
            grad = self._contract_except(G_old, vs, skip_n=n)
            self.factors[n][idx[n]] += self.lr * delta * grad
            self._clip_row(self.factors[n][idx[n]])

        grad_a = self._contract_except(G_old, vs, skip_n=N)
        self.factor_a[action] += self.lr * delta * grad_a
        self._clip_row(self.factor_a[action])


class SarsaTuckerAgent(TuckerMultiTabularQAgent):
    """Tucker Q-tensor with on-policy SARSA updates."""

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool,
               next_action: int = None):
        if next_action is None and not done:
            next_action = self.select_action(next_state)
        super().update(state, action, reward, next_state, done, next_action=next_action)


# ---------------------------------------------------------------------------
# TT (Tensor Train / MPS) multi-dimensional Q-agent
# Cores: C0(d1,r), cores[k](r,d_{k+2},r) for k=0..N-2, Ca(r,A)
# Q[i1,...,iN,a] = C0[i1] @ cores[0][:,i2,:] @ ... @ cores[N-2][:,iN,:] @ Ca[:,a]
# ---------------------------------------------------------------------------

class TTMultiTabularQAgent(_TensorTabularBase):
    """Tensor-Train Q-tensor for N-dimensional state spaces.

    For N=1: reduces to a rank-r matrix factorisation (equivalent to CP/1D).
    For N≥2: each mode gets its own TT-core, creating a genuine MPS structure
    that is more expressive than CP (not equivalent).
    """

    def __init__(self, state_dims, action_space_size, rank=4, lr=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000):
        super().__init__(state_dims, action_space_size, rank, lr, gamma,
                         epsilon_start, epsilon_end, epsilon_decay_steps)
        N = len(state_dims)
        scale = 1.0 / np.sqrt(np.prod(state_dims))
        # Left boundary core: (d0, rank)
        self.C0 = np.random.uniform(-scale, scale, (state_dims[0], rank))
        # Middle cores for state dims 1..N-1: each (rank, d_n, rank)
        self.cores = [np.random.uniform(-scale, scale, (rank, state_dims[n], rank))
                      for n in range(1, N)]
        # Action core (right boundary): (rank, A)
        self.Ca = np.random.uniform(-scale, scale, (rank, action_space_size))

    def _forward(self, idx: tuple, action: int):
        """Forward pass: returns (Q_val, L) where L[n] is the left env after core n."""
        N = len(self.state_dims)
        L = [self.C0[idx[0]].copy()]  # L[0] shape (rank,)
        for n in range(1, N):
            L.append(L[-1] @ self.cores[n - 1][:, idx[n], :])  # (rank,)
        Q_val = float(L[-1] @ self.Ca[:, action])
        return Q_val, L

    def _right_envs(self, idx: tuple, action: int) -> list:
        """Right environments R: R[n] is the product of all cores to the right of state core n.
        R[N-1] = Ca[:,action]; R[n] = cores[n][:,idx[n+1],:] @ R[n+1] for n<N-1.
        """
        N = len(self.state_dims)
        R = [None] * N
        R[N - 1] = self.Ca[:, action].copy()  # shape (rank,)
        for n in range(N - 2, -1, -1):
            R[n] = self.cores[n][:, idx[n + 1], :] @ R[n + 1]
        return R

    def _q_row(self, idx: tuple) -> np.ndarray:
        N = len(self.state_dims)
        L = [self.C0[idx[0]].copy()]
        for n in range(1, N):
            L.append(L[-1] @ self.cores[n - 1][:, idx[n], :])
        return L[-1] @ self.Ca  # (A,)

    def _q_val(self, idx: tuple, action: int) -> float:
        _, L = self._forward(idx, action)
        return float(L[-1] @ self.Ca[:, action])

    _MAX_NORM = 3.0

    def _clip(self, arr):
        np.clip(arr, -self._MAX_NORM, self._MAX_NORM, out=arr)

    def _update_params(self, idx: tuple, action: int, delta: float):
        N = len(self.state_dims)
        _, L = self._forward(idx, action)
        R = self._right_envs(idx, action)

        self.C0[idx[0]] += self.lr * delta * R[0]
        self._clip(self.C0[idx[0]])

        for n in range(1, N):
            self.cores[n - 1][:, idx[n], :] += self.lr * delta * np.outer(L[n - 1], R[n])
            self._clip(self.cores[n - 1][:, idx[n], :])

        self.Ca[:, action] += self.lr * delta * L[-1]
        self._clip(self.Ca[:, action])


class SarsaTTAgent(TTMultiTabularQAgent):
    """TT Q-tensor with on-policy SARSA updates."""

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool,
               next_action: int = None):
        if next_action is None and not done:
            next_action = self.select_action(next_state)
        super().update(state, action, reward, next_state, done, next_action=next_action)


class TabularQAgent:
    def __init__(self, state_space_size, action_space_size, lr=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.action_space_size = action_space_size
        self.steps = 0

    def epsilon(self) -> float:
        progress = min(1.0, self.steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: int, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon():
            return random.randint(0, self.action_space_size - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        best_next = np.max(self.q_table[next_state]) if not done else 0.0
        target = reward + self.gamma * best_next
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        self.steps += 1


class SarsaAgent(TabularQAgent):
    """On-policy TD control — update uses the action actually taken in next_state."""

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool,
               next_action: int = None):
        if next_action is None:
            next_action = self.select_action(next_state)
        next_q = self.q_table[next_state, next_action] if not done else 0.0
        target = reward + self.gamma * next_q
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        self.steps += 1


class TensorizedTabularQAgent(TabularQAgent):
    """CP-decomposed Q-table: Q(s, a) = factor_s[s] · factor_a[a].
    The full Q matrix is factor_s @ factor_a.T of shape (S, A).
    TD updates apply a rank-1 gradient step on the relevant rows of each factor.
    """

    def __init__(self, state_space_size, action_space_size, rank=4, lr=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000):
        # Skip TabularQAgent.__init__ to avoid allocating the full Q-table
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.rank = rank
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps = 0

        scale = 1.0 / np.sqrt(state_space_size)
        self.factor_s = np.random.uniform(-scale, scale, (state_space_size, rank))
        self.factor_a = np.random.uniform(-scale, scale, (action_space_size, rank))

    @property
    def q_table(self):
        """Reconstruct full Q matrix on demand (for compatibility with base class)."""
        return self.factor_s @ self.factor_a.T

    def _q(self, state: int, action: int) -> float:
        return float(self.factor_s[state] @ self.factor_a[action])

    def select_action(self, state: int, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon():
            return random.randint(0, self.action_space_size - 1)
        q_row = self.factor_s[state] @ self.factor_a.T  # shape (A,)
        return int(np.argmax(q_row))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        current_q = self._q(state, action)
        best_next = float(np.max(self.factor_s[next_state] @ self.factor_a.T)) if not done else 0.0
        target = reward + self.gamma * best_next
        delta = target - current_q

        # Capture pre-update factors for the symmetric gradient step
        fs = self.factor_s[state].copy()
        fa = self.factor_a[action].copy()

        self.factor_s[state] += self.lr * delta * fa
        self.factor_a[action] += self.lr * delta * fs

        self.steps += 1


class SarsaTensorAgent(TensorizedTabularQAgent):
    """CP-decomposed Q-table with SARSA (on-policy) TD target."""

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool,
               next_action: int = None):
        if next_action is None:
            next_action = self.select_action(next_state)

        current_q = self._q(state, action)
        next_q = self._q(next_state, next_action) if not done else 0.0
        target = reward + self.gamma * next_q
        delta = target - current_q

        fs = self.factor_s[state].copy()
        fa = self.factor_a[action].copy()

        self.factor_s[state] += self.lr * delta * fa
        self.factor_a[action] += self.lr * delta * fs

        self.steps += 1
