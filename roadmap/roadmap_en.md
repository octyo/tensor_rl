# Roadmap: Implement a CP Agent and Compare with a Tabular Baseline

**Goal in one sentence:** Replace the `Q`-table in tabular Q-learning with a **low-rank
CP tensor**, and empirically demonstrate *when* this yields the same policy with far fewer
parameters — and when it does not.

You implement everything yourselves (no ready-made tensor libraries for the agent itself —
`numpy` for array arithmetic is fine). The roadmap provides the formulas and the few
pitfalls that would otherwise cost you a week.

---

## 1. Which Algorithm: CP Q-Learning (Online, Tabular-Style)

Choose **this single** algorithm for the core project. It is the simplest CP agent and
the one *most directly comparable* to a tabular baseline: same training loop, same
update — the only difference is how `Q` is stored.
(SARSA, Expected SARSA, fitted-Q, etc. are good *extensions* once this works — not
starting points.)

The update is ordinary semi-gradient Q-learning. For a transition
`(s, a, r, s')` with discount factor `γ`:

$$\delta = r + \gamma \max_{a'} Q[s',a'] - Q[s,a] \qquad\text{(TD error)},$$

after which `Q[s,a]` is nudged toward its target. The entire research question is: *what
happens to learning when `Q` is forced to be low-rank?*

### The Concrete Algorithm (Complete)

Notation: a state is a factor index `s = (s_1,…,s_m)` and an action
`a = (a_1,…,a_k)`; the full tensor index for an `(s,a)` pair is the concatenation
`(s_1,…,s_m, a_1,…,a_k)`. `entry(A, idx)` and `entry_grad(A, idx)` are
the helper functions from Section 3. The only difference from tabular Q-learning is that
`Q` *is* the CP factor matrices `A`, and that the update is the normalised step.

```
ALGORITHM 1 — CP Q-Learning (ε-greedy, online)

Input:  environment; rank R; learning rate lr (≈ 0.3–0.5); discount γ;
        ε-schedule (ε_start, ε_min, decay); number of episodes T; ε_0 = 1e-8
Output: factor matrices A_1,…,A_N representing Q

1:  for each axis d = 1..N:                       # N = #state factors + #action factors
2:      A_d  ←  N(0, σ²) matrix of shape (axis_size_d × R),  σ = R^(−1/(2N))
3:  ε ← ε_start
4:  for episode = 1..T:
5:      s ← environment.reset()
6:      repeat until terminal:
7:          # ── ε-greedy action selection ──
8:          with probability ε:   a ← random action
9:          else:                 a ← argmax_{a'}  entry(A, (s, a'))   # over all actions a'
10:         (s', r, terminal) ← environment.step(a)
11:         # ── TD target ──
12:         if terminal:  y ← r
13:         else:         y ← r + γ · max_{a'} entry(A, (s', a'))
14:         # ── normalised (NLMS) semi-gradient update of cell (s,a) ──
15:         idx   ← (s_1,…,s_m, a_1,…,a_k)
16:         δ     ← y − entry(A, idx)
17:         g     ← entry_grad(A, idx)                 # list of N gradient rows
18:         norm2 ← Σ_d ‖g_d‖²  +  ε_0
19:         for d = 1..N:   A_d[idx_d]  ←  A_d[idx_d] + lr · δ · g_d / norm2
20:         s ← s'
21:     ε ← max(ε_min, ε · decay)                      # exploration decays
```

Lines 9 and 13 are the only places the action space is iterated over — which is why the
method is cheap for small action spaces (a single factor). The **tabular baseline is the
exact same algorithm**, just with `Q` as a dense array and lines 14–19 replaced by
`Q[s,a] ← Q[s,a] + lr·δ` (small `lr`). This makes the two directly comparable.

---

## 2. Background You Need (Brief)

`Q` is a lookup table: give it a state and an action, get a value. In a
**factored** problem it is a multi-dimensional grid
`Q[s_1,…,s_m, a_1,…,a_k]` — one cell per combination. The number of cells
**multiplies** across axes and therefore explodes.

A **CP decomposition** (CANDECOMP/PARAFAC) approximates the entire grid as a
**sum of a few simple building blocks**. Each building block gives each axis its own
small "scorecard" (one value per index), and the value in a cell is the scores for that
cell's coordinates **multiplied together**. The number of building blocks is called the
**rank** `R`.

For a tensor with `N` axes and rank `R`, represented by one **factor matrix**
`A_d` per axis (shape `(axis_size, R)`), a cell is:

$$Q[i_1,\dots,i_N] = \sum_{r=1}^{R} \prod_{d=1}^{N} A_d[i_d,\, r].$$

The parameter count goes from `∏(axis_sizes)` (table) to `R·Σ(axis_sizes)`
(CP) — from *multiplicative* to *additive*. That is the whole point.

---

## 3. How to Implement the CP Agent

### (a) Representation
Store one factor matrix `A_d` per axis: one for each state factor and one for each
action factor. `R` is a hyperparameter you vary later.

### (b) Read One Cell (`entry`)
```python
def entry(factors, index):          # index = (i_1, ..., i_N)
    prod = np.ones(R)
    for d in range(N):
        prod = prod * factors[d][index[d]]   # element-wise product over the rank axis
    return float(prod.sum())
```
Note: you **never** build the full tensor — you take one row from each factor matrix,
multiply them element-wise, and sum. This is exactly why the method suits RL, where you
only touch one `Q[s,a]` per step.

### (c) Gradient w.r.t. the Factor Rows (`entry_grad`)
The gradient of the cell w.r.t. row `A_d[i_d, :]` is **the product of all the other
axes' selected rows** ("leave-one-out"):
$$\frac{\partial Q[i]}{\partial A_d[i_d,:]} = \prod_{d' \neq d} A_{d'}[i_{d'},:].$$
```python
def entry_grad(factors, index):
    rows = [factors[d][index[d]] for d in range(N)]
    grads = []
    for d in range(N):
        loo = np.ones(R)
        for d2 in range(N):
            if d2 != d:
                loo = loo * rows[d2]
        grads.append(loo)           # gradient w.r.t. row index[d] in factor d
    return grads
```

### (d) The Update — and the Pitfall That Matters Most
The naive semi-gradient update `A_d[i_d,:] += lr·δ·loo` **diverges** on
tensors with many axes. The reason: the actual change in `Q[s,a]` scales with
`‖gradient‖²`, which grows with (number of axes · rank) — so the step overshoots.

Use the **normalised (NLMS) step** instead: divide the entire update by
`‖gradient‖² + ε`:
```python
def td_update(factors, sa_index, target, lr, eps=1e-8):
    delta = target - entry(factors, sa_index)
    grads = entry_grad(factors, sa_index)
    norm2 = sum(float((g*g).sum()) for g in grads) + eps
    for d in range(N):
        factors[d][sa_index[d]] += lr * delta * grads[d] / norm2
    return delta
```
This makes the change in `Q[s,a]` ≈ `lr·δ`, regardless of the number of axes — exactly
like a tabular update. Consequence: **`lr` is large here, around 0.3–0.5**, *not* the
small values dense Q-learning uses. Write this on the board; it is the most common reason
the project stalls.

> Why it works: the change in the cell is
> `ΔQ ≈ Σ_d grad_d · Δrow_d = lr·δ·(Σ_d ‖grad_d‖²)/norm2 ≈ lr·δ`.

### (e) Initialisation (Axis-Aware)
Initialise each factor entry as `N(0, σ²)` with
$$\sigma = R^{-1/(2N)}.$$
This gives the reconstructed cells unit variance, so gradients neither vanish
nor explode as the number of axes grows. A fixed small initialisation quietly
destroys learning on many-axis tensors.

### (f) Action Selection (argmax)
For a small action space (a single action factor): compute `Q[s,·]` for all actions
and take the argmax. This is perfectly fine for the core project. (The "factored argmax"
is only needed for large multi-agent action spaces — leave that as an extension.)
Add `ε`-greedy exploration as in tabular Q-learning.

---

## 4. The Tabular Baseline

Identical training loop, identical shared hyperparameters (`γ`, `ε`-schedule). `Q` is a
dense `(num_states, num_actions)` array, and the update is the standard
`Q[s,a] += lr·δ` with a normal (small) `lr`. **Get this working first** — it is your
warm-up and your reference point.

---

## 5. Environments (Choose Them So the Contrast Is Clear)

Keep them small enough that you can compute the exact optimal `Q*` via
**value iteration** — this unlocks the best metric (Section 6).

- **A factored grid world** (e.g. `Q[row, col, action]`): smooth, weakly
  coupled → **low-rank** `Q*`. CP *should* win here (match the table with fewer
  parameters). The positive result.
- **(Optional, for the honest contrast) a harder, coupled environment** where `Q*` is
  high-rank — CP *should* not be able to compress it and should need almost full rank.
  This sharpens the conclusion: *CP helps precisely when `Q*` is low-rank.*

You must be able to set the environment's state (for value iteration) and know the
dynamics and reward. A deterministic grid world with `-1` per step and a terminal goal
state is more than sufficient.

---

## 6. How to Compare — Metrics

Run everything over **≥5 random seeds** and report mean ± a band. Four
metrics, one figure each:

1. **Parameter count** — CP `R·Σ(sizes)` vs. table `∏(sizes)`. The entire point.
2. **Learning curve** — return (total reward per episode) vs. episodes, smoothed and
   averaged over seeds. Does CP reach the same level?
3. **Value accuracy** (the metric most projects *cannot* produce, but you can,
   because you have `Q*`): the relative error `‖Q̂ − Q*‖ / ‖Q*‖` between the learned
   `Q` and the true `Q*`, plotted over training. It tells you whether CP learned the
   *correct values* — not just a reasonable policy. Use the Frobenius norm (square root
   of the sum of squared cell differences).
4. **Performance vs. parameters (the main figure)** — sweep CP rank `R = 1, 2, 4, 8, …`
   and plot final return (and final value error) against parameter count, with the table
   as a single reference point. The story emerges as a frontier: CP matches the table as
   soon as `R` reaches the intrinsic rank of the value function — at a fraction of the
   parameters.

> **Extra insight if you have time:** Find the smallest rank at which a *direct*
> CP fit of `Q*` (not learning, just least-squares fitting of the tensor)
> hits e.g. 5 % error. That is the **intrinsic rank** — the "floor" that learning
> can reach at best. Plot the learned value error on top of the floor: if the learned
> error lies *above* the floor, the bottleneck is *learning*, not the rank.

---

## 7. Concrete Experiments and Expected Results

- **Exp. A — rank sweep on the grid.** CP with too low a rank fails; as soon as
  `R ≥` the intrinsic rank, CP matches (often slightly beats) the table with far fewer
  parameters. *Expected headline:* "CP rank 4 matches the table with ~3× fewer
  parameters."
- **Exp. B — value error curves.** CP converges toward `Q*`; compare the plateau
  across ranks.
- **Exp. C (optional) — the hard environment.** CP cannot compress a high-rank `Q*`
  → no parameter savings → conclusion: low-rank structure is a property of the
  *environment*, not a free lunch.
- **Exp. D (optional) — ablation showing NLMS is necessary.** Run naive SGD against
  the normalised step; the naive version diverges. Validates the design and produces a
  good figure.

---

## 8. Milestones (The Actual Roadmap)

| | Milestone | Deliverable |
|---|---|---|
| **M1** | Tabular Q-learning + grid environment + `Q*` via value iteration | baseline learning curve; a working `Q*` |
| **M2** | CP representation: `entry`, `entry_grad` | **numerical gradient check** passes (see pitfalls) |
| **M3** | CP Q-learning update with NLMS | CP agent learns the grid |
| **M4** | Rank sweep + the four comparison figures | performance-vs-parameters figure |
| **M5** | Report + one extension (hard environment, or SARSA-CP) | report |

Realistic order: M1 and M2 can run in parallel; M3 is the actual research step; M4
produces the results; M5 interprets them.

---

## 9. What the Report Must Show

A successful report contains: (1) the performance-vs-parameters figure with the table
as a reference point, (2) value error curves against `Q*`, (3) a clear conclusion about
*when* CP pays off (low-rank environment) and *when it does not* (high-rank environment),
and (4) a discussion of whether the remaining error is due to rank or learning. These are
concrete, falsifiable claims — exactly what a good empirical project delivers.
