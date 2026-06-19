"""Design 4: True MPS Tensor Network — no dense layers in computation path."""

import sys, os

DESIGN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DESIGN_DIR, '..', '..'))

from models.mps_network import StructuredMPSQNetwork, FeatureMapMPSQNetwork
from designs.shared.config import ENVS
from designs.shared.training import run_design, train_structured, train_flat, load_baseline_data
from designs.shared.plotting import generate_design_figures

BOND_DIMS = [1, 2, 4, 8, 16]
PHYS_DIMS = [4, 8, 16]


def main():
    from envs.env_utils import make_env_structured, make_env
    env, mode_dims = make_env_structured(ENVS[0])
    action_dim = env.action_space.n
    env.close()

    env_flat = make_env(ENVS[0])
    input_dim = env_flat.observation_space.shape[0]
    env_flat.close()

    # --- Sub-sweep A: StructuredMPSQNetwork (structured input) ---
    configs_a = []
    for bd in BOND_DIMS:
        label = f'smps_r{bd}'
        def factory(md, _bd=bd):
            return StructuredMPSQNetwork(md, action_dim, rank=_bd)
        configs_a.append((label, factory))

    results_a = run_design(configs_a, train_structured, DESIGN_DIR,
                           'Design 4A: Structured MPS Q-Network')

    # --- Sub-sweep B: FeatureMapMPSQNetwork (flat input) ---
    configs_b = []
    for bd in [2, 4, 8, 16]:
        for pd in PHYS_DIMS:
            label = f'fmps_r{bd}_d{pd}'
            def factory(_bd=bd, _pd=pd):
                return FeatureMapMPSQNetwork(input_dim, action_dim, rank=_bd, phys_dim=_pd)
            configs_b.append((label, factory))

    results_b = run_design(configs_b, train_flat, DESIGN_DIR,
                           'Design 4B: Feature Map MPS Q-Network')

    # Merge for figures
    merged = {}
    for env_id in ENVS:
        merged[env_id] = {}
        if env_id in results_a:
            merged[env_id].update(results_a[env_id])
        if env_id in results_b:
            merged[env_id].update(results_b[env_id])

    generate_design_figures(merged, DESIGN_DIR, baseline_loader=load_baseline_data)

    lines = ['Design 4: True MPS Tensor Network', '=' * 60, '']
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
