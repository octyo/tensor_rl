"""Design 2: Weight Decomposition — Sub-case A (flat) and Sub-case B (tensor input)."""

import sys, os

DESIGN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DESIGN_DIR, '..', '..'))
sys.path.insert(0, DESIGN_DIR)

from model import TensorizedQNetwork
from models.q_networks import QNetwork
from designs.shared.config import ENVS, HIDDEN_SIZES
from designs.shared.training import run_design, train_flat, train_structured, load_baseline_data
from designs.shared.plotting import generate_design_figures

RANKS = [2, 4, 8, 16]


def main():
    from envs.env_utils import make_env, make_env_structured
    env = make_env(ENVS[0])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    # --- Sub-case A: flat input with CPLinear/TuckerLinear ---
    configs_a = []
    for net_type in ['cp', 'tucker']:
        for r in RANKS:
            label = f'2A_{net_type}_r{r}'
            def factory(_nt=net_type, _r=r):
                return QNetwork(state_dim, action_dim, hidden_sizes=HIDDEN_SIZES,
                                network_type=_nt, rank=_r)
            configs_a.append((label, factory))

    results_a = run_design(configs_a, train_flat, DESIGN_DIR,
                           'Design 2A: Weight Decomposition (Flat Input)')

    # --- Sub-case B: structured input with TensorizedLinear ---
    configs_b = []
    for r in RANKS:
        label = f'2B_tensor_r{r}'
        def factory(md, _r=r):
            return TensorizedQNetwork(md, action_dim, rank=_r)
        configs_b.append((label, factory))

    results_b = run_design(configs_b, train_structured, DESIGN_DIR,
                           'Design 2B: Tensorized Layer (Structured Input)')

    # Merge results for figures
    merged = {}
    for env_id in ENVS:
        merged[env_id] = {}
        if env_id in results_a:
            merged[env_id].update(results_a[env_id])
        if env_id in results_b:
            merged[env_id].update(results_b[env_id])

    generate_design_figures(merged, DESIGN_DIR, baseline_loader=load_baseline_data)

    lines = ['Design 2: Weight Decomposition', '=' * 60, '']
    for env_id, env_results in merged.items():
        for label, agg in env_results.items():
            lines.append(f"{env_id}  {label}  params={agg['n_params']}  "
                         f"solve={agg['solve_mean']:.0%}  reward={agg['reward_mean']:.3f}")
    lines.append('')
    with open(os.path.join(DESIGN_DIR, 'output.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[Saved → {os.path.join(DESIGN_DIR, 'output.txt')}]")


if __name__ == '__main__':
    main()
