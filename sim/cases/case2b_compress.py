"""
Case 2b: Post-Hoc Compression
==============================
Trains a standard DQN to convergence, then decomposes each nn.Linear layer
using CP / Tucker / MPS at varying ranks. Reports performance before and after
compression, and the resulting parameter count reduction.

Sweep: method in [cp, tucker, mps]  x  rank in [2, 4, 8, 16]
Seeds: 3  |  Train episodes: 200  |  Algo: DQN
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import (ENV_ID, N_EPISODES, SEEDS, ALGO,
                   train_flat, run_seeds, get_flat_dims,
                   print_table, build_output, save_output)
from models.q_networks import QNetwork
from models.compression import compress_network
from envs.env_utils import make_env
from agents.dqn_agent import DQNAgent

METHODS  = ['cp', 'tucker', 'mps']
RANKS    = [2, 4, 8, 16]
OUT_FILE = os.path.join(os.path.dirname(__file__), 'case2b_compress_output.txt')


def eval_net(q_net, n_eval=50, seed=9000):
    """Greedy evaluation of q_net over n_eval episodes."""
    env   = make_env(ENV_ID)
    agent = DQNAgent(q_net, epsilon_start=0.0, epsilon_end=0.0, epsilon_decay_steps=1)
    rewards = []
    for ep in range(n_eval):
        state, _ = env.reset(seed=seed + ep)
        done, ep_rew = False, 0.0
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_rew += r
        rewards.append(ep_rew)
    env.close()
    final = rewards
    return {
        'reward_mean': float(np.mean(final)),
        'reward_std':  float(np.std(final)),
        'solve_rate':  float(np.mean([r > 0.5 for r in final])),
    }


def run_seed(seed):
    """Train standard DQN, return (q_net, pre_eval)."""
    state_dim, action_dim = get_flat_dims()
    q_net = QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                     network_type='standard', rank=4)
    m = train_flat(lambda: q_net, n_episodes=N_EPISODES, seed=seed, algo=ALGO)
    pre = eval_net(q_net, seed=seed + 10000)
    return m['q_net'], pre


def main():
    state_dim, action_dim = get_flat_dims()
    ref_net    = QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                          network_type='standard', rank=4)
    std_params = sum(p.numel() for p in ref_net.parameters())

    rows = []
    t0   = time.time()

    # Train one standard net per seed (shared baseline)
    print('\n  Training standard networks for all seeds...')
    trained_nets = []
    pre_rewards  = []
    for seed in SEEDS:
        print(f'    seed={seed} ...', end=' ', flush=True)
        q_net, pre = run_seed(seed)
        trained_nets.append(q_net)
        pre_rewards.append(pre)
        print(f"reward={pre['reward_mean']:.3f}  solve={pre['solve_rate']:.0%}")

    pre_reward_mean = float(np.mean([p['reward_mean'] for p in pre_rewards]))
    pre_reward_std  = float(np.std([p['reward_mean'] for p in pre_rewards]))
    pre_solve_mean  = float(np.mean([p['solve_rate'] for p in pre_rewards]))
    rows.append(['standard', std_params, '1.00', '-',
                 f"{pre_solve_mean:.0%}",
                 f"{pre_reward_mean:.3f} +/- {pre_reward_std:.3f}"])

    # Compress and evaluate
    for method in METHODS:
        for rank in RANKS:
            label = f'{method}_r{rank}'
            print(f'\n  Compressing: {label}')
            post_rewards, post_solves, post_params = [], [], []

            for i, (q_net, seed) in enumerate(zip(trained_nets, SEEDS)):
                try:
                    compressed = compress_network(q_net, method=method, rank=rank)
                    n_params   = sum(p.numel() for p in compressed.parameters())
                    post       = eval_net(compressed, seed=seed + 20000)
                    post_rewards.append(post['reward_mean'])
                    post_solves.append(post['solve_rate'])
                    post_params.append(n_params)
                    print(f"    seed={seed}: "
                          f"params {std_params}→{n_params}  "
                          f"reward={post['reward_mean']:.3f}  "
                          f"solve={post['solve_rate']:.0%}")
                except Exception as e:
                    print(f"    seed={seed}: ERROR {e}")

            if post_rewards:
                n_p  = int(np.mean(post_params))
                vs   = f"{n_p / std_params:.3f}"
                rows.append([label, n_p, vs,
                              f"{n_p / std_params:.0%} of std",
                              f"{np.mean(post_solves):.0%} +/- {np.std(post_solves):.0%}",
                              f"{np.mean(post_rewards):.3f} +/- {np.std(post_rewards):.3f}"])

    headers = ['Config', 'Params', 'vs_std', 'Compression', 'Solve%', 'Reward']
    title   = (f"Case 2b: Post-Hoc Compression  "
               f"(env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})")
    print_table(rows, headers, title=title)
    print(f"\n  Pre-compression baseline: "
          f"{pre_reward_mean:.3f} +/- {pre_reward_std:.3f} reward, "
          f"{pre_solve_mean:.0%} solve")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

    notes = [f"Standard pre-compress: {std_params} params, "
             f"reward={pre_reward_mean:.3f}, solve={pre_solve_mean:.0%}",
             f"Convergence note: post-hoc compression — no training curve (eval only)",
             f"Total runtime: {(time.time()-t0)/60:.1f} min"]
    lines = build_output(title, f"env={ENV_ID}", rows, headers, notes)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()
