"""
Case 3: Structured Input Embedding (Input Space Reduction)
==========================================================
Instead of flattening the (7,7,3) MiniGrid observation, a tensor-network
embedding layer maps the raw 3D observation to a hidden vector while
preserving the spatial structure. Uses TTEmbedding / CPEmbedding /
TuckerEmbedding as the first layer.

Sweep: embedding_type in [tt, cp, tucker]  x  rank in [2, 4, 8, 16]
Baseline: standard DQN on flattened obs
Seeds: 3  |  Episodes: 200  |  Algo: DQN
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import (ENV_ID, N_EPISODES, SEEDS, ALGO,
                   train_flat, train_structured,
                   run_seeds, get_flat_dims, get_structured_dims,
                   print_table, build_output, save_output)
from models.q_networks import QNetwork, StructuredQNetwork

METHODS  = ['tt', 'cp', 'tucker']
RANKS    = [2, 4, 8, 16]
OUT_FILE = os.path.join(os.path.dirname(__file__), 'case3_structured_output.txt')


def main():
    state_dim, action_dim = get_flat_dims()
    mode_dims, _          = get_structured_dims()   # (7, 7, 3)

    ref_net    = QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                          network_type='standard', rank=4)
    std_params = sum(p.numel() for p in ref_net.parameters())

    rows = []
    t0   = time.time()

    # Baseline: standard flat DQN
    print('\n  Config: standard (flat baseline)')

    def std_factory():
        return QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                        network_type='standard', rank=4)

    res = run_seeds(train_flat, std_factory, label='standard')
    rows.append(['standard', res['n_params'], '1.00',
                 f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                 f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])

    # Structured embedding sweep
    for method in METHODS:
        for rank in RANKS:
            label = f'{method}_r{rank}'
            print(f'\n  Config: {label}  mode_dims={mode_dims}')

            def factory(md=mode_dims, m=method, r=rank):
                return StructuredQNetwork(md, action_dim,
                                         hidden_size=128,
                                         embedding_type=m,
                                         rank=r)

            try:
                res = run_seeds(train_structured, factory, label=label)
                vs  = f"{res['n_params'] / std_params:.3f}"
                rows.append([label, res['n_params'], vs,
                             f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                             f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
            except Exception as e:
                print(f"    ERROR: {e}")
                rows.append([label, 'ERR', '-', '-', str(e)[:40]])

    headers = ['Config', 'Params', 'vs_std', 'Solve%', 'Reward']
    title   = (f"Case 3: Structured Input Embedding  "
               f"(env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})")
    print_table(rows, headers, title=title)
    print(f"\n  Standard baseline: {std_params} params")
    print(f"  mode_dims={mode_dims}  action_dim={action_dim}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

    notes = [f"Standard baseline: {std_params} params (flat obs)",
             f"Structured input: mode_dims={mode_dims}",
             f"env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS}, algo={ALGO}",
             f"Total runtime: {(time.time()-t0)/60:.1f} min"]
    lines = build_output(title, f"env={ENV_ID}", rows, headers, notes)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()
