"""
Case 7 Rerun (DoorKey-5x5): MPS Efficiency with 1000 episodes and 7 seeds
==========================================================================
Reruns case 7 DoorKey-5x5 with more data to get reliable results.

Sub-experiment A: MPS rank sweep r1-r8
Sub-experiment B: Param-matched MPS vs CP vs standard_tiny

Seeds: 7  |  Episodes: 1000  |  Algo: double_dqn
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import (ALGO, train_flat, run_seeds, get_flat_dims,
                   print_table, build_output, save_output)
from models.q_networks import QNetwork

ENV_ID     = 'MiniGrid-DoorKey-5x5-v0'
SEEDS      = [42, 43, 44, 45, 46, 47, 48]
N_EPISODES = 1000
MPS_RANKS  = [1, 2, 3, 4, 5, 6, 7, 8]
OUT_FILE   = os.path.join(os.path.dirname(__file__),
                          'case7_doorkey5x5_1000eps_output.txt')

MATCHED_CONFIGS = [
    ('mps_r1',  'mps',      1, (128, 128), '~500'),
    ('std_h3',  'standard', 4, (3,),       '~500'),
    ('mps_r2',  'mps',      2, (128, 128), '~900'),
    ('cp_r1',   'cp',       1, (128, 128), '~900'),
    ('std_h6',  'standard', 4, (6,),       '~900'),
    ('mps_r4',  'mps',      4, (128, 128), '~2100'),
    ('cp_r3',   'cp',       3, (128, 128), '~2100'),
    ('std_h14', 'standard', 4, (14,),      '~2100'),
    ('mps_r8',  'mps',      8, (128, 128), '~6200'),
    ('cp_r8',   'cp',       8, (128, 128), '~6200'),
    ('std_h40', 'standard', 4, (40,),      '~6200'),
]


def make_net(state_dim, action_dim, net_type, rank=4, hidden_sizes=(128, 128)):
    return QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes,
                    network_type=net_type, rank=rank, tensorize_layers='all')


def main():
    import utils
    utils.ENV_ID = ENV_ID

    state_dim, action_dim = get_flat_dims()
    ref_net    = make_net(state_dim, action_dim, 'standard')
    std_params = sum(p.numel() for p in ref_net.parameters())

    rows_a, rows_b = [], []
    conv_rows_a, conv_rows_b = [], []
    t0        = time.time()
    n_configs = len(MPS_RANKS) + len(MATCHED_CONFIGS)
    config_idx = 0

    print(f'\n  Environment: {ENV_ID}', flush=True)
    print(f'  state_dim={state_dim}  action_dim={action_dim}  '
          f'episodes={N_EPISODES}  seeds={SEEDS}', flush=True)

    # --- Sub-experiment A: MPS rank sweep ---
    print(f'\n{"="*64}', flush=True)
    print(f'A) MPS rank sweep r1-r8  '
          f'({len(MPS_RANKS)} configs x {len(SEEDS)} seeds x {N_EPISODES} eps)',
          flush=True)
    print('='*64, flush=True)

    for rank in MPS_RANKS:
        config_idx += 1
        label = f'mps_r{rank}'
        print(f'\n  [config {config_idx}/{n_configs}] {label}', flush=True)

        def factory(r=rank):
            return make_net(state_dim, action_dim, 'mps', rank=r)

        res = run_seeds(train_flat, factory, label=label,
                        n_episodes=N_EPISODES, seeds=SEEDS, algo=ALGO,
                        config_idx=config_idx, config_total=n_configs,
                        experiment_t0=t0)
        vs = f"{res['n_params'] / std_params:.4f}"
        rows_a.append([label, res['n_params'], vs,
                        f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                        f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
        conv_rows_a.append([label] + [f'{q:.3f}' for q in res['avg_quantiles']])

    headers = ['Config', 'Params', 'vs_std', 'Solve%', 'Reward']
    print_table(rows_a, headers, title='A) MPS rank sweep r1-r8')

    # --- Sub-experiment B: param-matched comparison ---
    print(f'\n{"="*64}', flush=True)
    print(f'B) Param-matched: MPS vs CP vs standard_tiny  '
          f'({len(MATCHED_CONFIGS)} configs)', flush=True)
    print('='*64, flush=True)

    current_bucket = None
    for label, net_type, rank, hidden_sizes, bucket in MATCHED_CONFIGS:
        config_idx += 1
        if bucket != current_bucket:
            current_bucket = bucket
            print(f'\n  --- Param bucket {bucket} ---', flush=True)

        probe = make_net(state_dim, action_dim, net_type, rank=rank,
                         hidden_sizes=hidden_sizes)
        actual_params = sum(p.numel() for p in probe.parameters())
        del probe

        print(f'\n  [config {config_idx}/{n_configs}] {label}  '
              f'params={actual_params}  bucket={bucket}', flush=True)

        def factory(nt=net_type, r=rank, hs=hidden_sizes):
            return make_net(state_dim, action_dim, nt, rank=r, hidden_sizes=hs)

        try:
            res = run_seeds(train_flat, factory, label=label,
                            n_episodes=N_EPISODES, seeds=SEEDS, algo=ALGO,
                            config_idx=config_idx, config_total=n_configs,
                            experiment_t0=t0)
            vs = f"{res['n_params'] / std_params:.4f}"
            rows_b.append([label, res['n_params'], bucket, vs,
                            f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                            f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
            conv_rows_b.append([label] + [f'{q:.3f}' for q in res['avg_quantiles']])
        except Exception as e:
            import traceback
            print(f'    ERROR: {e}', flush=True)
            traceback.print_exc()
            rows_b.append([label, actual_params, bucket, '-', 'ERR', str(e)[:30]])
            conv_rows_b.append([label] + ['-'] * 10)

    headers_b = ['Config', 'Params', 'Bucket', 'vs_std', 'Solve%', 'Reward']
    print_table(rows_b, headers_b,
                title='B) Param-matched: MPS vs CP vs standard_tiny')

    elapsed = (time.time() - t0) / 60
    print(f"\n  Standard h128x128 reference: {std_params} params", flush=True)
    print(f"  Total time: {elapsed:.1f} min", flush=True)

    # --- Save ---
    pct_headers  = [f'{(i+1)*10}%' for i in range(10)]
    conv_headers = ['Config'] + pct_headers
    title        = (f"Case 7 rerun (DoorKey-5x5): MPS Efficiency  "
                    f"(env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})")
    notes = [
        f"Standard h128x128 reference: {std_params} params",
        f"Task: find key -> pick up -> open locked door -> reach goal",
        f"Sub-exp A: MPS r1-r8 rank sweep ({len(SEEDS)} seeds, {N_EPISODES} eps)",
        f"Sub-exp B: param-matched MPS vs CP vs standard_tiny at 4 param budgets",
        f"Param buckets: ~500, ~900, ~2100, ~6200",
        f"Total runtime: {elapsed:.1f} min",
    ]

    lines_a = build_output(
        title + " — A: MPS rank sweep", f"env={ENV_ID}",
        rows_a, headers,
        convergence_rows=conv_rows_a, convergence_headers=conv_headers,
    )
    lines_b = build_output(
        title + " — B: Param-matched comparison", f"env={ENV_ID}",
        rows_b, headers_b, notes=notes,
        convergence_rows=conv_rows_b, convergence_headers=conv_headers,
    )
    save_output(lines_a + ['', ''] + lines_b, OUT_FILE)


if __name__ == '__main__':
    main()
