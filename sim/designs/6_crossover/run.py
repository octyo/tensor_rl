"""Design 6: Middle-ground dimensionality — 8D navigation, (3,)*8.

6561 states: complex enough for CP's structural prior to matter,
small enough that MLP can still explore adequately in 500 episodes.
"""

import sys, os, time, json, math

DESIGN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DESIGN_DIR, '..', '..'))

import numpy as np
import torch
import torch.nn as nn
from envs.ndim_nav import make_ndim_env
from agents.double_dqn_agent import DoubleDQNAgent
from models.q_networks import QNetwork
from models.tensor_layers import TTEmbedding, CPEmbedding

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Config ---
DIMS = (3,) * 8
OBS_DIM = sum(DIMS)       # 24
N_DIMS = len(DIMS)
ACTION_DIM = 2 * N_DIMS   # 16
N_EPISODES = 250
SEED = 42
MAX_STEPS = 200
EPS_DECAY = N_EPISODES * 100
LR = 1e-3
GAMMA = 0.99
TAU = 0.05
BUFFER_SIZE = 50000
BATCH_SIZE = 64
REPLAY_MIN = 500


# --- Models ---

def make_baseline():
    return QNetwork(OBS_DIM, ACTION_DIM, hidden_sizes=(128, 128), network_type='standard')

def make_cp_flat(rank):
    return QNetwork(OBS_DIM, ACTION_DIM, hidden_sizes=(128, 128), network_type='cp', rank=rank)


class NDimCPQFunction(nn.Module):
    """Pure CP Q-function: Q(s,a) = Σᵣ (∏ₙ uₙᵣ(sₙ)) · zᵣ(a)"""
    def __init__(self, dims, action_dim, rank):
        super().__init__()
        self.dims = dims
        self.action_dim = action_dim
        self.rank = rank
        self.spatial_factors = nn.ParameterList([
            nn.Parameter(torch.empty(d, rank)) for d in dims
        ])
        self.action_factor = nn.Parameter(torch.empty(action_dim, rank))
        self.bias = nn.Parameter(torch.zeros(action_dim))
        scale = 1.0 / math.sqrt(sum(dims))
        for f in self.spatial_factors:
            nn.init.uniform_(f, -scale, scale)
        nn.init.uniform_(self.action_factor, -scale, scale)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        offset = 0
        h = None
        for n, d in enumerate(self.dims):
            proj = x[:, offset:offset+d] @ self.spatial_factors[n]
            offset += d
            h = proj if h is None else h * proj
        return torch.einsum('br, ar -> ba', h, self.action_factor) + self.bias


class NDimStructuredEmbeddingNet(nn.Module):
    """Tensor embedding (TT/CP) on structured input reshaped to (N_DIMS, dim_size), then MLP."""
    def __init__(self, dims, action_dim, embedding_type, rank, hidden_size=128):
        super().__init__()
        self.n_dims = len(dims)
        self.dim_size = dims[0]
        self.action_dim = action_dim
        structured_shape = (self.n_dims, self.dim_size)
        if embedding_type == 'tt':
            self.embedding = TTEmbedding(structured_shape, out_features=hidden_size, rank=rank)
        elif embedding_type == 'cp':
            self.embedding = CPEmbedding(structured_shape, out_features=hidden_size, rank=rank)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        structured = x.reshape(x.shape[0], self.n_dims, self.dim_size)
        h = torch.relu(self.embedding(structured))
        h = torch.relu(self.hidden(h))
        return self.output(h)


# --- Training ---

def train_one(model, label):
    env = make_ndim_env(dims=DIMS, max_steps=MAX_STEPS)
    env.action_space.seed(SEED)
    agent = DoubleDQNAgent(
        model, action_dim=ACTION_DIM, epsilon_decay_steps=EPS_DECAY,
        tau=TAU, lr=LR, gamma=GAMMA, buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE, replay_min_size=REPLAY_MIN,
    )

    data_dir = os.path.join(DESIGN_DIR, 'data')
    logs_dir = os.path.join(DESIGN_DIR, 'logs')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    json_path = os.path.join(data_dir, f'{label}_seed{SEED}.json')
    if os.path.exists(json_path):
        print(f"    [{label}] — cached", flush=True)
        with open(json_path) as f:
            return json.load(f)

    log_path = os.path.join(logs_dir, f'{label}_seed{SEED}.log')
    lf = open(log_path, 'w')

    rewards = []
    t0 = time.time()
    milestone = max(1, N_EPISODES // 10)

    for ep in range(1, N_EPISODES + 1):
        state, _ = env.reset(seed=SEED + ep)
        done, ep_rew = False, 0.0
        while not done:
            action = agent.select_action(state)
            ns, r, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.replay_buffer.push(state, action, r, ns, float(done))
            agent.update()
            state = ns
            ep_rew += r
        rewards.append(ep_rew)

        if ep % milestone == 0:
            window = rewards[max(0, ep - milestone):]
            elapsed = time.time() - t0
            line = (f"[ep {ep}/{N_EPISODES}]  {ep/N_EPISODES*100:.0f}%  "
                    f"solve={np.mean([r > 0.5 for r in window]):.0%}  "
                    f"reward={np.mean(window):.3f}  "
                    f"eps={agent.epsilon():.2f}  elapsed={elapsed:.1f}s")
            print(f"      {line}", flush=True)
            lf.write(line + '\n')
            lf.flush()

    lf.close()
    env.close()
    elapsed = time.time() - t0
    final = rewards[-50:]

    bucket = max(1, N_EPISODES // 10)
    quantile_means = []
    for i in range(10):
        chunk = rewards[i * bucket: (i + 1) * bucket]
        quantile_means.append(float(np.mean(chunk)) if chunk else 0.0)

    result = {
        'config': label,
        'n_params': sum(p.numel() for p in model.parameters()),
        'all_rewards': [float(r) for r in rewards],
        'quantile_means': quantile_means,
        'solve_rate_final': float(np.mean([r > 0.5 for r in final])),
        'reward_mean_final': float(np.mean(final)),
        'elapsed_s': elapsed,
    }
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    return result


# --- Main ---

def main():
    configs = [
        ('baseline',          lambda: make_baseline()),
        ('cp_flat_r4',        lambda: make_cp_flat(4)),
        ('cp_flat_r8',        lambda: make_cp_flat(8)),
        ('cp_flat_r16',       lambda: make_cp_flat(16)),
        ('cp_qfunc_r4',      lambda: NDimCPQFunction(DIMS, ACTION_DIM, rank=4)),
        ('cp_qfunc_r8',      lambda: NDimCPQFunction(DIMS, ACTION_DIM, rank=8)),
        ('cp_qfunc_r16',     lambda: NDimCPQFunction(DIMS, ACTION_DIM, rank=16)),
        ('embed_tt_r4',      lambda: NDimStructuredEmbeddingNet(DIMS, ACTION_DIM, 'tt', 4)),
        ('embed_tt_r8',      lambda: NDimStructuredEmbeddingNet(DIMS, ACTION_DIM, 'tt', 8)),
        ('embed_tt_r16',     lambda: NDimStructuredEmbeddingNet(DIMS, ACTION_DIM, 'tt', 16)),
        ('embed_cp_r8',      lambda: NDimStructuredEmbeddingNet(DIMS, ACTION_DIM, 'cp', 8)),
    ]

    n = len(configs)
    print(f"\n{'='*70}")
    print(f"Design 6: 8D Navigation — Middle-Ground Complexity")
    print(f"  Env dims     : {DIMS}  (obs_dim={OBS_DIM}, actions={ACTION_DIM}, states={3**8})")
    print(f"  Episodes     : {N_EPISODES}")
    print(f"  Seed         : {SEED}")
    print(f"  Configs      : {n}")
    print(f"{'='*70}\n", flush=True)

    all_results = {}
    t0 = time.time()
    for i, (label, factory) in enumerate(configs, 1):
        print(f"\n  [{i}/{n}] {label}", flush=True)
        model = factory()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    params={n_params}", flush=True)
        result = train_one(model, label)
        all_results[label] = result
        elapsed_total = time.time() - t0
        eta = elapsed_total / i * (n - i)
        print(f"    -> solve={result['solve_rate_final']:.0%}  reward={result['reward_mean_final']:.3f}  "
              f"time={result['elapsed_s']:.1f}s  ETA={eta/60:.1f}min", flush=True)

    # --- Figures ---
    fig_dir = os.path.join(DESIGN_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Learning curves
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, res in all_results.items():
        rews = res['all_rewards']
        window = 20
        if len(rews) >= window:
            smooth = np.convolve(rews, np.ones(window)/window, mode='valid')
            ax.plot(smooth, label=f"{label} ({res['n_params']}p)", linewidth=1.2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (rolling mean, window=20)')
    ax.set_title(f'Design 6: 8D Navigation (3^8={3**8} states) — Learning Curves')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'learning_curves_8d.png'), dpi=150)
    plt.close(fig)

    # Param efficiency
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(all_results.keys())
    params = [all_results[l]['n_params'] for l in labels]
    solves = [all_results[l]['solve_rate_final'] for l in labels]
    ax.scatter(params, solves, s=60, zorder=5)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (params[i], solves[i]), fontsize=6, textcoords='offset points', xytext=(4, 4))
    ax.set_xlabel('Parameters (log)')
    ax.set_ylabel('Solve Rate (last 50 eps)')
    ax.set_title(f'Design 6: 8D Nav (3^8={3**8} states) — Parameter Efficiency')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'param_efficiency_8d.png'), dpi=150)
    plt.close(fig)

    print(f"\n  Figures saved to {fig_dir}/")

    # Output summary
    lines = [f'Design 6: 8D Navigation (3^8={3**8} states)', '=' * 60, '']
    for label, res in all_results.items():
        lines.append(f"  {label:25s}  params={res['n_params']:>7d}  "
                     f"solve={res['solve_rate_final']:.0%}  reward={res['reward_mean_final']:.3f}")
    lines.append('')
    with open(os.path.join(DESIGN_DIR, 'output.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[Saved → {os.path.join(DESIGN_DIR, 'output.txt')}]")
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
