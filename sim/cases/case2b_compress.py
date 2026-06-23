"""
Case 2b: Post-Hoc Compression
==============================
Trains a standard DQN to convergence, then decomposes each nn.Linear layer
using CP / Tucker / MPS at varying ranks. Reports performance before compression,
immediately after compression, and after a short fine-tuning phase.

Pipeline per seed:
  1. Train standard DQN for TRAIN_EPISODES (longer run for proper convergence)
  2. Eval pre-compression (greedy)
  3. Compress with tensorly (SVD init)
  4. Eval post-compression (greedy, no training)
  5. Fine-tune compressed model for FINETUNE_EPISODES
  6. Eval post-finetune (greedy)

Sweep: method in [cp, tucker, mps]  x  rank in [2, 4, 8, 16]
Seeds: 3  |  Train episodes: 300  |  Finetune episodes: 50  |  Algo: double_dqn
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import types
import numpy as np
from utils import (ENV_ID, SEEDS, ALGO,
                   train_flat, get_flat_dims,
                   print_table, build_output, save_output)
from models.q_networks import QNetwork
from models.compression import compress_network, finetune
from envs.env_utils import make_env
from agents.dqn_agent import DQNAgent

METHODS          = ['cp', 'tucker', 'mps']
RANKS            = [2, 4, 8, 16]
TRAIN_EPISODES   = 300    # longer baseline training for proper convergence
FINETUNE_EPISODES = 50    # fine-tune episodes after compression
OUT_FILE         = os.path.join(os.path.dirname(__file__), 'case2b_compress_output.txt')


def eval_net(q_net, n_eval=50, seed=9000):
    """Greedy evaluation of q_net over n_eval episodes."""
    env    = make_env(ENV_ID)
    agent  = DQNAgent(q_net, epsilon_start=0.0, epsilon_end=0.0, epsilon_decay_steps=1)
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
    return {
        'reward_mean': float(np.mean(rewards)),
        'reward_std':  float(np.std(rewards)),
        'solve_rate':  float(np.mean([r > 0.5 for r in rewards])),
    }


def train_standard(seed):
    """Train standard DQN for TRAIN_EPISODES, return (q_net, pre_eval)."""
    state_dim, action_dim = get_flat_dims()
    q_net = QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                     network_type='standard', rank=4)
    train_flat(lambda: q_net, n_episodes=TRAIN_EPISODES, seed=seed, algo=ALGO)
    pre = eval_net(q_net, seed=seed + 10000)
    return q_net, pre


def main():
    state_dim, action_dim = get_flat_dims()
    ref_net    = QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                          network_type='standard', rank=4)
    std_params = sum(p.numel() for p in ref_net.parameters())

    rows = []
    t0   = time.time()

    n_compress_configs = len(METHODS) * len(RANKS)
    n_total_configs    = 1 + n_compress_configs   # baseline + sweep
    config_idx         = 0

    # --- Train one standard net per seed ---
    config_idx += 1
    print(f'\n  [config {config_idx}/{n_total_configs}] Training standard networks '
          f'({TRAIN_EPISODES} eps × {len(SEEDS)} seeds)', flush=True)

    trained_nets = []
    pre_rewards  = []
    for s_idx, seed in enumerate(SEEDS):
        print(f'\n    [standard] seed {s_idx+1}/{len(SEEDS)} (={seed})  '
              f'({TRAIN_EPISODES} eps)', flush=True)
        q_net, pre = train_standard(seed)
        trained_nets.append(q_net)
        pre_rewards.append(pre)
        elapsed = time.time() - t0
        print(f'      -> pre-compress: reward={pre["reward_mean"]:.3f}  '
              f'solve={pre["solve_rate"]:.0%}  elapsed={elapsed:.1f}s', flush=True)

    pre_reward_mean = float(np.mean([p['reward_mean'] for p in pre_rewards]))
    pre_reward_std  = float(np.std([p['reward_mean'] for p in pre_rewards]))
    pre_solve_mean  = float(np.mean([p['solve_rate'] for p in pre_rewards]))
    rows.append(['standard', std_params, '1.00', '-', '-',
                 f"{pre_solve_mean:.0%}",
                 f"{pre_reward_mean:.3f} +/- {pre_reward_std:.3f}"])

    # --- Compress, eval, finetune, eval ---
    for method in METHODS:
        for rank in RANKS:
            config_idx += 1
            label = f'{method}_r{rank}'
            frac_done = (config_idx - 1) / n_total_configs
            eta_str = ''
            if frac_done > 0:
                eta = (time.time() - t0) / frac_done * (1 - frac_done)
                eta_str = f'  ETA={eta/60:.1f}min'
            print(f'\n  [config {config_idx}/{n_total_configs}] {label}{eta_str}',
                  flush=True)

            post_c_rewards, post_c_solves = [], []
            post_ft_rewards, post_ft_solves, post_params = [], [], []

            for i, (q_net, seed) in enumerate(zip(trained_nets, SEEDS)):
                print(f'    seed {i+1}/{len(SEEDS)} (={seed}): compressing...', flush=True)
                try:
                    compressed = compress_network(q_net, method=method, rank=rank)
                    n_params   = sum(p.numel() for p in compressed.parameters())

                    # Post-compression eval (no training)
                    post_c = eval_net(compressed, seed=seed + 20000)
                    post_c_rewards.append(post_c['reward_mean'])
                    post_c_solves.append(post_c['solve_rate'])
                    print(f'      post-compress:  reward={post_c["reward_mean"]:.3f}  '
                          f'solve={post_c["solve_rate"]:.0%}  params={n_params}', flush=True)

                    # Fine-tune
                    print(f'      fine-tuning ({FINETUNE_EPISODES} eps)...', flush=True)
                    ft_env = make_env(ENV_ID)
                    ft_args = types.SimpleNamespace()
                    finetune(compressed, ft_env, ft_args, n_episodes=FINETUNE_EPISODES)
                    ft_env.close()

                    # Post-finetune eval
                    post_ft = eval_net(compressed, seed=seed + 30000)
                    post_ft_rewards.append(post_ft['reward_mean'])
                    post_ft_solves.append(post_ft['solve_rate'])
                    post_params.append(n_params)
                    print(f'      post-finetune:  reward={post_ft["reward_mean"]:.3f}  '
                          f'solve={post_ft["solve_rate"]:.0%}', flush=True)

                except Exception as e:
                    import traceback
                    print(f'    seed={seed}: ERROR {e}', flush=True)
                    traceback.print_exc()

            if post_ft_rewards:
                n_p = int(np.mean(post_params))
                vs  = f"{n_p / std_params:.3f}"
                rows.append([
                    label, n_p, vs,
                    f"{n_p / std_params:.0%} of std",
                    f"{np.mean(post_c_solves):.0%} +/- {np.std(post_c_solves):.0%}",
                    f"{np.mean(post_ft_solves):.0%} +/- {np.std(post_ft_solves):.0%}",
                    f"{np.mean(post_ft_rewards):.3f} +/- {np.std(post_ft_rewards):.3f}",
                ])

    headers = ['Config', 'Params', 'vs_std', 'Compression',
               'Solve%(no-ft)', 'Solve%(ft)', 'Reward(ft)']
    title   = (f"Case 2b: Post-Hoc Compression  "
               f"(env={ENV_ID}, train_eps={TRAIN_EPISODES}, "
               f"finetune_eps={FINETUNE_EPISODES}, seeds={SEEDS})")
    print_table(rows, headers, title=title)
    elapsed = (time.time() - t0) / 60
    print(f"\n  Pre-compression baseline: "
          f"{pre_reward_mean:.3f} +/- {pre_reward_std:.3f} reward, "
          f"{pre_solve_mean:.0%} solve")
    print(f"  Total time: {elapsed:.1f} min")

    notes = [
        f"Standard pre-compress: {std_params} params, "
        f"reward={pre_reward_mean:.3f}, solve={pre_solve_mean:.0%}",
        f"Training: {TRAIN_EPISODES} eps before compression  |  "
        f"Fine-tune: {FINETUNE_EPISODES} eps after compression",
        f"Solve%(no-ft): greedy eval immediately after compression, no fine-tuning",
        f"Solve%(ft) / Reward(ft): greedy eval after {FINETUNE_EPISODES}-episode fine-tune",
        f"Total runtime: {elapsed:.1f} min",
    ]
    lines = build_output(title, f"env={ENV_ID}", rows, headers, notes)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()
