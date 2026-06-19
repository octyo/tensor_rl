"""Training loop and sweep driver for sim/designs/."""

import sys, os, time, json, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from envs.env_utils import make_env, make_env_structured
from agents.double_dqn_agent import DoubleDQNAgent
from designs.shared.config import (
    ENVS, N_EPISODES, SEEDS, ALGO, EPS_DECAY,
    LR, GAMMA, BUFFER_SIZE, BATCH_SIZE, REPLAY_MIN, TAU,
)


def training_loop(q_net, env, n_episodes, seed, label='', log_file=None):
    env.action_space.seed(seed)
    action_dim = env.action_space.n
    agent = DoubleDQNAgent(
        q_net, action_dim=action_dim,
        epsilon_decay_steps=EPS_DECAY, tau=TAU,
        lr=LR, gamma=GAMMA, buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE, replay_min_size=REPLAY_MIN,
    )

    rewards = []
    t_start = time.time()
    milestone = max(1, n_episodes // 10)

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

        if ep % milestone == 0:
            window = rewards[max(0, ep - milestone):]
            pct = ep / n_episodes * 100
            elapsed = time.time() - t_start
            line = (f"[ep {ep}/{n_episodes}]  {pct:.0f}%  "
                    f"solve={np.mean([r > 0.5 for r in window]):.0%}  "
                    f"reward={np.mean(window):.3f}  "
                    f"eps={agent.epsilon():.2f}  "
                    f"elapsed={elapsed:.1f}s")
            print(f"      {line}", flush=True)
            if log_file:
                log_file.write(line + '\n')
                log_file.flush()

    final = rewards[-50:]
    elapsed = time.time() - t_start

    bucket = max(1, n_episodes // 10)
    quantile_means = []
    for i in range(10):
        chunk = rewards[i * bucket: (i + 1) * bucket]
        quantile_means.append(float(np.mean(chunk)) if chunk else 0.0)

    return {
        'n_params': sum(p.numel() for p in q_net.parameters()),
        'reward_mean': float(np.mean(final)),
        'reward_std': float(np.std(final)),
        'solve_rate': float(np.mean([r > 0.5 for r in final])),
        'all_rewards': [float(r) for r in rewards],
        'quantile_means': quantile_means,
        'elapsed_s': elapsed,
    }


def train_flat(factory, env_id, n_episodes=N_EPISODES, seed=42, label='', log_file=None):
    env = make_env(env_id)
    q_net = factory()
    m = training_loop(q_net, env, n_episodes, seed, label=label, log_file=log_file)
    env.close()
    return m


def train_structured(factory, env_id, n_episodes=N_EPISODES, seed=42, label='', log_file=None):
    env, mode_dims = make_env_structured(env_id)
    q_net = factory(mode_dims)
    m = training_loop(q_net, env, n_episodes, seed, label=label, log_file=log_file)
    env.close()
    return m


def env_slug(env_id):
    return env_id.replace('MiniGrid-', '').replace('-v0', '').lower()


def _json_path(data_dir, config_label, env_id, seed):
    return os.path.join(data_dir, f"{config_label}_{env_slug(env_id)}_seed{seed}.json")


def _seed_done(data_dir, config_label, env_id, seed):
    return os.path.exists(_json_path(data_dir, config_label, env_id, seed))


def save_seed_result(result, config_label, env_id, seed, algo, data_dir):
    path = _json_path(data_dir, config_label, env_id, seed)
    obj = {
        'config': config_label,
        'env': env_id,
        'seed': seed,
        'algo': algo,
        'n_params': result['n_params'],
        'all_rewards': result['all_rewards'],
        'quantile_means': result['quantile_means'],
        'solve_rate_final': result['solve_rate'],
        'reward_mean_final': result['reward_mean'],
        'elapsed_s': result['elapsed_s'],
    }
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_baseline_data(env_id):
    baseline_dir = os.path.join(os.path.dirname(__file__), '..', '0_baseline', 'data')
    pattern = os.path.join(baseline_dir, f"*_{env_slug(env_id)}_seed*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No baseline data for {env_id} in {baseline_dir}")
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    return {
        'n_params': results[0]['n_params'],
        'reward_mean': float(np.mean([r['reward_mean_final'] for r in results])),
        'solve_mean': float(np.mean([r['solve_rate_final'] for r in results])),
        'per_seed_rewards': [r['all_rewards'] for r in results],
        'avg_quantiles': [float(np.mean([r['quantile_means'][i] for r in results]))
                          for i in range(len(results[0]['quantile_means']))],
    }


def aggregate_seeds(seed_results):
    return {
        'n_params': seed_results[0]['n_params'],
        'reward_mean': float(np.mean([r['reward_mean'] for r in seed_results])),
        'reward_std': float(np.std([r['reward_mean'] for r in seed_results])),
        'solve_mean': float(np.mean([r['solve_rate'] for r in seed_results])),
        'solve_std': float(np.std([r['solve_rate'] for r in seed_results])),
        'per_seed_rewards': [r['all_rewards'] for r in seed_results],
        'avg_quantiles': [float(np.mean([r['quantile_means'][i] for r in seed_results]))
                          for i in range(len(seed_results[0]['quantile_means']))],
        'total_elapsed_s': float(sum(r['elapsed_s'] for r in seed_results)),
    }


def run_design(configs, train_fn, design_dir, design_name):
    """Run a full design sweep.

    configs: list of (label, factory) tuples.
        factory is either:
          - callable() -> nn.Module  (for flat input)
          - callable(mode_dims) -> nn.Module  (for structured input)
    train_fn: either train_flat or train_structured
    """
    data_dir = os.path.join(design_dir, 'data')
    logs_dir = os.path.join(design_dir, 'logs')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    n_configs = len(configs)
    total_runs = n_configs * len(ENVS) * len(SEEDS)
    print(f"\n{'='*70}")
    print(f"{design_name}")
    print(f"  Environments : {len(ENVS)}")
    print(f"  Configs      : {n_configs}")
    print(f"  Seeds        : {len(SEEDS)}")
    print(f"  Total runs   : {total_runs}")
    print(f"{'='*70}\n", flush=True)

    results = {}
    t0 = time.time()
    runs_done = 0

    for env_id in ENVS:
        results[env_id] = {}
        for ci, (label, factory) in enumerate(configs, 1):
            seed_results = []
            for si, seed in enumerate(SEEDS):
                if _seed_done(data_dir, label, env_id, seed):
                    path = _json_path(data_dir, label, env_id, seed)
                    with open(path) as f:
                        cached = json.load(f)
                    seed_results.append({
                        'n_params': cached['n_params'],
                        'reward_mean': cached['reward_mean_final'],
                        'reward_std': 0.0,
                        'solve_rate': cached['solve_rate_final'],
                        'all_rewards': cached['all_rewards'],
                        'quantile_means': cached['quantile_means'],
                        'elapsed_s': cached['elapsed_s'],
                    })
                    runs_done += 1
                    print(f"    [{label}] {env_slug(env_id)} seed {seed} — cached", flush=True)
                    continue

                log_path = os.path.join(logs_dir, f"{label}_{env_slug(env_id)}_seed{seed}.log")
                print(f"\n    [{label}] {env_slug(env_id)} seed {si+1}/{len(SEEDS)} (={seed})", flush=True)
                with open(log_path, 'a') as lf:
                    m = train_fn(factory, env_id, N_EPISODES, seed, label=label, log_file=lf)
                seed_results.append(m)
                save_seed_result(m, label, env_id, seed, ALGO, data_dir)
                runs_done += 1

                elapsed_total = time.time() - t0
                frac = runs_done / total_runs
                eta = elapsed_total / frac * (1 - frac) if frac > 0 else 0
                print(f"      -> reward={m['reward_mean']:.3f}  solve={m['solve_rate']:.0%}  "
                      f"params={m['n_params']}  time={m['elapsed_s']:.1f}s  "
                      f"ETA={eta/60:.1f}min", flush=True)

            agg = aggregate_seeds(seed_results)
            results[env_id][label] = agg
            print(f"  [{ci}/{n_configs} configs] {label} on {env_slug(env_id)}: "
                  f"solve={agg['solve_mean']:.0%}  reward={agg['reward_mean']:.3f}", flush=True)

    total_time = time.time() - t0
    print(f"\n{design_name} complete in {total_time/60:.1f} min\n")
    return results
