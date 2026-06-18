"""
Case 8: True MPS Tensor Network vs Tensorized Layers
=====================================================
Compares the true MPS tensor network (StructuredMPSQNetwork) against:
  - Standard DQN (flat obs, nn.Linear layers)
  - MPSLinear DQN (flat obs, MPS-parameterized nn.Linear)
  - StructuredQNetwork with TTEmbedding (structured obs, TT first layer + linear hidden)

The true MPS model operates on the (7,7,3) structured observation and
contracts it directly through 3 MPS cores to produce Q-values. There are
no activation functions and no weight matrices — the entire model is a
single tensor network contraction.

Key difference from MPSLinear: MPSLinear reconstructs a full weight matrix
from TT cores then does y=Wx+b. StructuredMPSQNetwork contracts the input
tensor directly through the cores — no W is ever materialized.

Key difference from TTEmbedding: TTEmbedding is only the first layer,
followed by standard Linear+ReLU layers. StructuredMPSQNetwork is the
ENTIRE model — input to Q-values in one contraction.

Sweep: rank in [2, 4, 8, 16, 32]
Seeds: 7  |  Episodes: 200  |  Algo: double_dqn
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import (ENV_ID, ALGO,
                   train_flat, train_structured,
                   run_seeds, get_flat_dims, get_structured_dims,
                   print_table, build_output, save_output)
from models.q_networks import QNetwork, StructuredQNetwork
from models.mps_network import StructuredMPSQNetwork

RANKS      = [2, 4, 8, 16, 32]
SEEDS      = [42, 43, 44, 45, 46, 47, 48]
N_EPISODES = 200
OUT_FILE   = os.path.join(os.path.dirname(__file__), 'case8_true_mps_output.txt')


def main():
    state_dim, action_dim = get_flat_dims()
    mode_dims, _          = get_structured_dims()   # (7, 7, 3)

    rows      = []
    conv_rows = []
    t0        = time.time()

    configs = []
    # 1. Standard flat baseline
    configs.append(('standard_flat', 'flat', None, None))
    # 2. MPSLinear flat (tensorized layers, NOT true TN)
    for r in RANKS:
        configs.append((f'mpslinear_r{r}', 'mps_flat', r, None))
    # 3. TTEmbedding structured (TN first layer + linear hidden)
    for r in RANKS:
        configs.append((f'tt_embed_r{r}', 'tt_struct', r, None))
    # 4. True MPS (entire model is one TN contraction)
    for r in RANKS:
        configs.append((f'true_mps_r{r}', 'true_mps', r, None))

    n_configs = len(configs)

    print(f'\n  Environment: {ENV_ID}', flush=True)
    print(f'  Structured obs: mode_dims={mode_dims}', flush=True)
    print(f'  Episodes: {N_EPISODES}  Seeds: {SEEDS}', flush=True)

    for config_idx, (label, kind, rank, _) in enumerate(configs, start=1):
        print(f'\n  [config {config_idx}/{n_configs}] {label}', flush=True)

        if kind == 'flat':
            def factory():
                return QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                                network_type='standard', rank=4)
            train_fn = train_flat
        elif kind == 'mps_flat':
            def factory(r=rank):
                return QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                                network_type='mps', rank=r, tensorize_layers='all')
            train_fn = train_flat
        elif kind == 'tt_struct':
            def factory(md=mode_dims, r=rank):
                return StructuredQNetwork(md, action_dim, hidden_size=128,
                                         embedding_type='tt', rank=r)
            train_fn = train_structured
        elif kind == 'true_mps':
            def factory(md=mode_dims, r=rank):
                return StructuredMPSQNetwork(md, action_dim, rank=r)
            train_fn = train_structured
        else:
            raise ValueError(kind)

        try:
            res = run_seeds(train_fn, factory, label=label,
                            n_episodes=N_EPISODES, seeds=SEEDS, algo=ALGO,
                            config_idx=config_idx, config_total=n_configs,
                            experiment_t0=t0)
            rows.append([label, res['n_params'],
                         f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                         f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
            conv_rows.append([label] + [f'{q:.3f}' for q in res['avg_quantiles']])
        except Exception as e:
            import traceback
            print(f'    ERROR: {e}', flush=True)
            traceback.print_exc()
            rows.append([label, 'ERR', '-', str(e)[:40]])
            conv_rows.append([label] + ['-'] * 10)

    pct_headers  = [f'{(i+1)*10}%' for i in range(10)]
    conv_headers = ['Config'] + pct_headers
    headers = ['Config', 'Params', 'Solve%', 'Reward']
    title   = (f"Case 8: True MPS vs Tensorized Layers  "
               f"(env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})")

    print_table(rows, headers, title=title)
    elapsed = (time.time() - t0) / 60
    print(f"\n  mode_dims={mode_dims}  action_dim={action_dim}", flush=True)
    print(f"  Total time: {elapsed:.1f} min", flush=True)

    notes = [
        f"mode_dims={mode_dims}  action_dim={action_dim}",
        f"true_mps: StructuredMPSQNetwork — entire model is one MPS contraction, no activations",
        f"mpslinear: QNetwork with MPSLinear layers — MPS parameterizes W, then y=Wx+b with ReLU",
        f"tt_embed: StructuredQNetwork — TTEmbedding first layer + Linear hidden + ReLU",
        f"standard_flat: QNetwork with nn.Linear — baseline",
        f"Total runtime: {elapsed:.1f} min",
    ]
    lines = build_output(title, f"env={ENV_ID}", rows, headers, notes,
                         convergence_rows=conv_rows, convergence_headers=conv_headers)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()
