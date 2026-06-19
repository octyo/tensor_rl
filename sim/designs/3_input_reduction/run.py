"""Design 3: Input Space Reduction via Tensor Embedding + standard MLP."""

import sys, os

DESIGN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DESIGN_DIR, '..', '..'))

from models.q_networks import StructuredQNetwork
from designs.shared.config import ENVS
from designs.shared.training import run_design, train_structured, load_baseline_data
from designs.shared.plotting import generate_design_figures

EMBEDDING_TYPES = ['tt', 'cp', 'tucker']
RANKS = [2, 4, 8, 16]


def main():
    from envs.env_utils import make_env_structured
    env, mode_dims = make_env_structured(ENVS[0])
    action_dim = env.action_space.n
    env.close()

    configs = []
    for emb in EMBEDDING_TYPES:
        for r in RANKS:
            label = f'{emb}_r{r}'
            def factory(md, _emb=emb, _r=r):
                return StructuredQNetwork(md, action_dim, hidden_size=128,
                                          embedding_type=_emb, rank=_r)
            configs.append((label, factory))

    results = run_design(configs, train_structured, DESIGN_DIR,
                         'Design 3: Input Space Reduction via Tensor Embedding')

    generate_design_figures(results, DESIGN_DIR, baseline_loader=load_baseline_data)

    lines = ['Design 3: Input Space Reduction', '=' * 60, '']
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
