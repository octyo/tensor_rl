"""Design 4: True MPS Tensor Network — FeatureMapMPSQNetwork only.

Learned per-site embeddings (local feature maps) followed by MPS core contraction.
No StructuredMPSQNetwork (that's a multilinear map without feature maps, not true MPS).
"""

import sys, os

DESIGN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DESIGN_DIR, '..', '..'))

from models.mps_network import FeatureMapMPSQNetwork
from designs.shared.config import ENVS
from designs.shared.training import run_design, train_flat, load_baseline_data
from designs.shared.plotting import generate_design_figures

BOND_DIMS = [1, 2, 4, 8, 16]
PHYS_DIMS = [4, 8, 16]


def main():
    from envs.env_utils import make_env
    env = make_env(ENVS[0])
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    configs = []
    for bd in BOND_DIMS:
        for pd in PHYS_DIMS:
            label = f'fmps_r{bd}_d{pd}'
            def factory(_bd=bd, _pd=pd):
                return FeatureMapMPSQNetwork(input_dim, action_dim, rank=_bd, phys_dim=_pd)
            configs.append((label, factory))

    results = run_design(configs, train_flat, DESIGN_DIR,
                         'Design 4: Feature Map MPS Q-Network')

    generate_design_figures(results, DESIGN_DIR, baseline_loader=load_baseline_data)

    lines = ['Design 4: True MPS Tensor Network (FeatureMap)', '=' * 60, '']
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
