"""Design 5: High-dimensional test — 10D navigation environment.

Tests whether tensor methods scale to higher dimensions where the structured
representation should provide a stronger inductive bias.

Reduced sweep: 1 seed, fewer ranks (low+high), CP and Tucker only, longer training.
"""

import sys, os, time, json

DESIGN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DESIGN_DIR, '..', '..'))

import numpy as np
import torch
from envs.ndim_nav import make_ndim_env
from agents.double_dqn_agent import DoubleDQNAgent
from models.q_networks import QNetwork
from models.tensor_layers import CPEmbedding, TuckerEmbedding, TTEmbedding

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Config ---
DIMS = (3,) * 10
OBS_DIM = sum(DIMS)       # 30
N_DIMS = len(DIMS)
ACTION_DIM = 2 * N_DIMS   # 20
N_EPISODES = 500
SEED = 42
MAX_STEPS = 300
EPS_DECAY = N_EPISODES * 40
LR = 1e-3
GAMMA = 0.99
TAU = 0.05
BUFFER_SIZE = 20000
BATCH_SIZE = 64
REPLAY_MIN = 500


# --- Models ---

def make_baseline():
    return QNetwork(OBS_DIM, ACTION_DIM, hidden_sizes=(128, 128), network_type='standard')

def make_cp_flat(rank):
    return QNetwork(OBS_DIM, ACTION_DIM, hidden_sizes=(128, 128), network_type='cp', rank=rank)

def make_tucker_flat(rank):
    return QNetwork(OBS_DIM, ACTION_DIM, hidden_sizes=(128, 128), network_type='tucker', rank=rank)


class NDimCPQFunction(torch.nn.Module):
    """Pure CP Q-function for N-dimensional navigation.
    Q(s,a) = Σᵣ (∏ₙ uₙᵣ(sₙ)) · zᵣ(a)
    Each sₙ is a one-hot vector of size dims[n].
    """
    def __init__(self, dims, action_dim, rank):
        super().__init__()
        self.dims = dims
        self.action_dim = action_dim
        self.rank = rank
        self.spatial_factors = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(d, rank)) for d in dims
        ])
        self.action_factor = torch.nn.Parameter(torch.empty(action_dim, rank))
        self.bias = torch.nn.Parameter(torch.zeros(action_dim))
        self._reset()

    def _reset(self):
        import math
        scale = 1.0 / math.sqrt(sum(self.dims))
        for f in self.spatial_factors:
            torch.nn.init.uniform_(f, -scale, scale)
        torch.nn.init.uniform_(self.action_factor, -scale, scale)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        offset = 0
        h = None
        for n, d in enumerate(self.dims):
            x_n = x[:, offset:offset+d]  # (batch, d)
            offset += d
            proj = x_n @ self.spatial_factors[n]  # (batch, rank)
            h = proj if h is None else h * proj
        q = torch.einsum('br, ar -> ba', h, self.action_factor)
        return q + self.bias


class NDimStructuredEmbeddingNet(torch.nn.Module):
    """Tensor embedding (TT/CP/Tucker) on structured 10D input, then MLP.

    The flat obs is reshaped to (batch, N_DIMS, dim_size)
    so the embedding treats each spatial dimension as a mode.
    """
    def __init__(self, dims, action_dim, embedding_type, rank, hidden_size=128):
        super().__init__()
        self.dims = dims
        self.n_dims = len(dims)
        self.dim_size = dims[0]
        self.action_dim = action_dim
        # Structured shape: (n_dims, dim_size) e.g. (10, 5)
        structured_shape = (self.n_dims, self.dim_size)
        self.embedding_type = embedding_type
        if embedding_type == 'tt':
            self.embedding = TTEmbedding(structured_shape, out_features=hidden_size, rank=rank)
        elif embedding_type == 'cp':
            self.embedding = CPEmbedding(structured_shape, out_features=hidden_size, rank=rank)
        elif embedding_type == 'tucker':
            self.embedding = TuckerEmbedding(structured_shape, out_features=hidden_size, ranks=rank)
        self.hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        # Reshape flat one-hot (batch, 50) -> (batch, 10, 5)
        structured = x.reshape(batch, self.n_dims, self.dim_size)
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
        ('cp_flat_r2',        lambda: make_cp_flat(2)),
        ('cp_flat_r8',        lambda: make_cp_flat(8)),
        ('cp_flat_r16',       lambda: make_cp_flat(16)),
        ('tucker_flat_r2',    lambda: make_tucker_flat(2)),
        ('tucker_flat_r8',    lambda: make_tucker_flat(8)),
        ('tucker_flat_r16',   lambda: make_tucker_flat(16)),
        ('cp_qfunc_r4',      lambda: NDimCPQFunction(DIMS, ACTION_DIM, rank=4)),
        ('cp_qfunc_r8',      lambda: NDimCPQFunction(DIMS, ACTION_DIM, rank=8)),
        ('cp_qfunc_r32',     lambda: NDimCPQFunction(DIMS, ACTION_DIM, rank=32)),
        ('embed_tt_r4',      lambda: NDimStructuredEmbeddingNet(DIMS, ACTION_DIM, 'tt', 4)),
        ('embed_tt_r8',      lambda: NDimStructuredEmbeddingNet(DIMS, ACTION_DIM, 'tt', 8)),
        ('embed_tt_r16',     lambda: NDimStructuredEmbeddingNet(DIMS, ACTION_DIM, 'tt', 16)),
        ('embed_cp_r4',      lambda: NDimStructuredEmbeddingNet(DIMS, ACTION_DIM, 'cp', 4)),
        ('embed_cp_r8',      lambda: NDimStructuredEmbeddingNet(DIMS, ACTION_DIM, 'cp', 8)),
        ('embed_cp_r16',     lambda: NDimStructuredEmbeddingNet(DIMS, ACTION_DIM, 'cp', 16)),
    ]

    n = len(configs)
    print(f"\n{'='*70}")
    print(f"Design 5: High-Dimensional (10D) Navigation")
    print(f"  Env dims     : {DIMS}  (obs_dim={OBS_DIM}, actions={ACTION_DIM})")
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
    ax.set_title('Design 5: 10D Navigation — Learning Curves')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'learning_curves_10d.png'), dpi=150)
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
    ax.set_title('Design 5: 10D Nav — Parameter Efficiency')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'param_efficiency_10d.png'), dpi=150)
    plt.close(fig)

    print(f"\n  Figures saved to {fig_dir}/")

    # Output summary
    lines = ['Design 5: High-Dimensional (10D) Navigation', '=' * 60, '']
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
