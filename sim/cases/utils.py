"""Shared experiment utilities for Phase 2 DQN case studies."""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gymnasium as gym
from gymnasium.envs.registration import register as _gym_register

try:
    _gym_register(
        id='FrozenLakeSimple-v0',
        entry_point='gymnasium.envs.toy_text:FrozenLakeEnv',
        kwargs={'desc': ['SFFF', 'FFFF', 'FFFF', 'FFFG'], 'is_slippery': False},
        max_episode_steps=200,
    )
except Exception:
    pass

from envs.env_utils import make_env, make_env_structured, make_env_cnn

ENV_ID    = 'MiniGrid-Empty-5x5-v0'
N_EPISODES = 100
SEEDS      = [42, 43, 44]
ALGO       = 'double_dqn'


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def _training_loop(q_net, env, n_episodes, seed, algo='dqn', label='', verbose=True):
    import time
    from agents.dqn_agent import DQNAgent
    from agents.double_dqn_agent import DoubleDQNAgent

    env.action_space.seed(seed)
    action_dim  = env.action_space.n
    eps_decay   = n_episodes * 40

    if algo == 'dqn':
        agent = DQNAgent(q_net, epsilon_decay_steps=eps_decay)
    elif algo == 'double_dqn':
        agent = DoubleDQNAgent(q_net, action_dim=action_dim,
                               epsilon_decay_steps=eps_decay, tau=0.05)
    else:
        from agents.dueling_dqn_agent import DuelingDQNAgent
        agent = DuelingDQNAgent(q_net, action_dim=action_dim,
                                epsilon_decay_steps=eps_decay, tau=0.05)

    rewards   = []
    t_start   = time.time()
    milestone = max(1, n_episodes // 10)   # print every 10%

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed + ep)
        done, ep_rew = False, 0.0
        while not done:
            action = agent.select_action(state)
            ns, r, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.replay_buffer.push(state, action, r, ns, float(done))
            agent.update()
            state = ns
            ep_rew += r
        rewards.append(ep_rew)

        if verbose and ep % milestone == 0:
            window = rewards[max(0, ep - milestone):]
            pct    = ep / n_episodes * 100
            elapsed = time.time() - t_start
            print(f"      {pct:5.0f}%  ep={ep:4d}/{n_episodes}"
                  f"  solve={np.mean([r > 0.5 for r in window]):.0%}"
                  f"  reward={np.mean(window):.3f}"
                  f"  elapsed={elapsed:.1f}s", flush=True)

    final   = rewards[-50:]
    elapsed = time.time() - t_start

    # Convergence profile: mean reward in each 10% bucket of training
    bucket = max(1, n_episodes // 10)
    quantile_means = []
    for i in range(10):
        chunk = rewards[i * bucket: (i + 1) * bucket]
        quantile_means.append(float(np.mean(chunk)) if chunk else 0.0)

    return {
        'n_params':        sum(p.numel() for p in q_net.parameters()),
        'reward_mean':     float(np.mean(final)),
        'reward_std':      float(np.std(final)),
        'solve_rate':      float(np.mean([r > 0.5 for r in final])),
        'q_net':           q_net,
        'all_rewards':     rewards,
        'quantile_means':  quantile_means,
        'elapsed_s':       elapsed,
    }


def train_flat(q_net_factory, n_episodes=N_EPISODES, seed=42, algo=ALGO):
    """Train on flat (flattened MiniGrid) obs."""
    env   = make_env(ENV_ID)
    q_net = q_net_factory()
    m     = _training_loop(q_net, env, n_episodes, seed, algo)
    env.close()
    return m


def train_structured(q_net_factory, n_episodes=N_EPISODES, seed=42, algo=ALGO):
    """Train on structured (7,7,3) MiniGrid obs."""
    env, mode_dims = make_env_structured(ENV_ID)
    q_net = q_net_factory(mode_dims)
    m     = _training_loop(q_net, env, n_episodes, seed, algo)
    env.close()
    return m


def train_cnn(q_net_factory, n_episodes=N_EPISODES, seed=42, algo=ALGO):
    """Train on channels-first (3,7,7) MiniGrid obs."""
    env, obs_shape = make_env_cnn(ENV_ID)
    q_net = q_net_factory(obs_shape)
    m     = _training_loop(q_net, env, n_episodes, seed, algo)
    env.close()
    return m


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_seeds(train_fn, factory, n_episodes=N_EPISODES, seeds=SEEDS,
              algo=ALGO, label='', config_idx=None, config_total=None,
              experiment_t0=None):
    import time as _time
    results = []
    for s_idx, seed in enumerate(seeds):
        seed_tag = f"seed {s_idx+1}/{len(seeds)} (={seed})"
        cfg_tag  = (f"  [config {config_idx}/{config_total}]"
                    if config_idx is not None else "")
        print(f"\n    [{label}]{cfg_tag}  {seed_tag}  ({n_episodes} eps)", flush=True)
        m = train_fn(factory, n_episodes=n_episodes, seed=seed, algo=algo)
        results.append(m)
        eta_str = ""
        if experiment_t0 is not None and config_idx is not None and config_total is not None:
            elapsed_total = _time.time() - experiment_t0
            frac_done = ((config_idx - 1) * len(seeds) + (s_idx + 1)) / (config_total * len(seeds))
            if frac_done > 0:
                eta = elapsed_total / frac_done * (1 - frac_done)
                eta_str = f"  ETA={eta/60:.1f}min"
        print(f"      -> reward={m['reward_mean']:.3f}  "
              f"solve={m['solve_rate']:.0%}  "
              f"params={m['n_params']}  "
              f"time={m['elapsed_s']:.1f}s{eta_str}", flush=True)
    # Per-seed quantile means averaged across seeds
    n_quantiles = len(results[0]['quantile_means'])
    avg_quantiles = [float(np.mean([r['quantile_means'][i] for r in results]))
                     for i in range(n_quantiles)]
    return {
        'n_params':       results[0]['n_params'],
        'reward_mean':    float(np.mean([r['reward_mean'] for r in results])),
        'reward_std':     float(np.std([r['reward_mean'] for r in results])),
        'solve_mean':     float(np.mean([r['solve_rate'] for r in results])),
        'solve_std':      float(np.std([r['solve_rate'] for r in results])),
        'avg_quantiles':  avg_quantiles,
        'total_elapsed_s': float(sum(r['elapsed_s'] for r in results)),
        '_runs':          results,
    }


# ---------------------------------------------------------------------------
# Env dimension helpers
# ---------------------------------------------------------------------------

def get_flat_dims():
    env = make_env(ENV_ID)
    s, a = env.observation_space.shape[0], env.action_space.n
    env.close()
    return s, a


def get_structured_dims():
    env, mode_dims = make_env_structured(ENV_ID)
    a = env.action_space.n
    env.close()
    return mode_dims, a


def get_cnn_dims():
    env, obs_shape = make_env_cnn(ENV_ID)
    a = env.action_space.n
    env.close()
    return obs_shape, a


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_table(rows, headers, title=''):
    SEP = '=' * 76
    print(f'\n{SEP}')
    if title:
        print(title)
        print('-' * 76)

    widths = [max(len(str(r[i])) for r in [headers] + rows) + 2
              for i in range(len(headers))]

    def fmt(row):
        return '  '.join(f'{str(v):<{w}}' for v, w in zip(row, widths))

    print(fmt(headers))
    print('-' * 76)
    for row in rows:
        print(fmt(row))
    print(SEP)


def build_output(title, env_info, rows, headers, notes=None,
                 convergence_rows=None, convergence_headers=None):
    """Build a list of lines for saving to an output file.

    convergence_rows / convergence_headers: optional second table showing
    per-config training curves (10% quantile buckets).
    """
    lines = [title, env_info, '']
    lines += [f'Environment : {ENV_ID}',
              f'Agent       : {ALGO}',
              f'Episodes    : {N_EPISODES} per seed   Seeds: {SEEDS}',
              '']
    SEP = '-' * 76

    widths = [max(len(str(r[i])) for r in [headers] + rows) + 2
              for i in range(len(headers))]

    def fmt(row, ws):
        return '  '.join(f'{str(v):<{w}}' for v, w in zip(row, ws))

    lines += [fmt(headers, widths), SEP]
    for row in rows:
        lines.append(fmt(row, widths))
    lines.append(SEP)

    if convergence_rows and convergence_headers:
        lines += ['', 'Training convergence (mean reward in each 10% episode bucket):']
        cw = [max(len(str(r[i])) for r in [convergence_headers] + convergence_rows) + 2
              for i in range(len(convergence_headers))]
        lines += [fmt(convergence_headers, cw), SEP]
        for row in convergence_rows:
            lines.append(fmt(row, cw))
        lines.append(SEP)

    if notes:
        lines += [''] + notes
    return lines


def save_output(lines, path):
    with open(path, 'w') as f:
        f.write('\n'.join(str(l) for l in lines) + '\n')
    print(f'\n[Saved → {path}]')
