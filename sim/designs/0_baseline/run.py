"""Design 0: Standard Double DQN baseline on all environments."""

import sys, os

DESIGN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DESIGN_DIR, '..', '..'))

from models.q_networks import QNetwork
from designs.shared.config import HIDDEN_SIZES
from designs.shared.training import run_design, train_flat
from designs.shared.plotting import generate_design_figures


def main():
    from envs.env_utils import make_env
    env = make_env('MiniGrid-Empty-5x5-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    def factory():
        return QNetwork(state_dim, action_dim, hidden_sizes=HIDDEN_SIZES, network_type='standard')

    configs = [('baseline_128x128', factory)]

    results = run_design(configs, train_flat, DESIGN_DIR, 'Design 0: Standard Double DQN Baseline')

    generate_design_figures(results, DESIGN_DIR, baseline_loader=None)

    lines = ['Design 0: Standard Double DQN Baseline', '=' * 60, '']
    for env_id, env_results in results.items():
        for label, agg in env_results.items():
            lines.append(f"{env_id}  {label}  params={agg['n_params']}  "
                         f"solve={agg['solve_mean']:.0%}  reward={agg['reward_mean']:.3f}")
    lines.append('')
    with open(os.path.join(DESIGN_DIR, 'output.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[Saved → {os.path.join(DESIGN_DIR, 'output.txt')}]")


if __name__ == '__main__':
    main()
