import argparse
import json
import os
import numpy as np

from envs.env_utils import make_env
from models.q_networks import QNetwork, DuelingQNetwork
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from utils.logging_utils import Logger, calculate_bitsize
from config import (DEFAULT_ENV, DEFAULT_SEED, DEFAULT_EPISODES, DEFAULT_NETWORK,
                    DEFAULT_RANK, HIDDEN_SIZES, TAU)

DEEP_ALGOS = {"dqn", "double_dqn", "dueling_dqn"}
TABULAR_ALGOS = {"tabular_q", "sarsa"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=DEFAULT_ENV)
    parser.add_argument("--algo", type=str, default="dqn",
                        choices=["dqn", "double_dqn", "dueling_dqn", "tabular_q", "sarsa"])
    parser.add_argument("--network", type=str, default=DEFAULT_NETWORK,
                        choices=["standard", "cp", "tucker", "tt"])
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--tensorize_layers", type=str, default="all",
                        help=("Which layers to tensorize. 'all' (default), 'none', or "
                              "comma-separated 0-based indices e.g. '0,2'. "
                              "Indices: 0=first hidden, 1=second hidden, ..., N=output layer. "
                              "For dueling networks, indices refer to trunk hidden layers only."))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--tau", type=float, default=TAU,
                        help="Polyak update rate for Double/Dueling DQN target network")
    parser.add_argument("--seeds", type=str, default=str(DEFAULT_SEED),
                        help="Comma-separated seeds, e.g. '42,43,44'")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def make_agent(algo: str, state_dim: int, action_dim: int, args):
    """Factory: returns (q_net, agent) for the chosen algo."""
    tz = args.tensorize_layers

    if algo == "dqn":
        q_net = QNetwork(state_dim, action_dim, hidden_sizes=HIDDEN_SIZES,
                         network_type=args.network, rank=args.rank, tensorize_layers=tz)
        return q_net, DQNAgent(q_net, epsilon_decay_steps=(args.episodes * 100))

    elif algo == "double_dqn":
        q_net = QNetwork(state_dim, action_dim, hidden_sizes=HIDDEN_SIZES,
                         network_type=args.network, rank=args.rank, tensorize_layers=tz)
        agent = DoubleDQNAgent(q_net, action_dim=action_dim,
                               epsilon_decay_steps=(args.episodes * 100), tau=args.tau)
        return q_net, agent

    elif algo == "dueling_dqn":
        q_net = DuelingQNetwork(state_dim, action_dim, hidden_sizes=HIDDEN_SIZES,
                                network_type=args.network, rank=args.rank, tensorize_layers=tz)
        agent = DuelingDQNAgent(q_net, action_dim=action_dim,
                                epsilon_decay_steps=(args.episodes * 100), tau=args.tau)
        return q_net, agent

    elif algo in TABULAR_ALGOS:
        raise ValueError(
            f"Tabular algo '{algo}' requires a discrete state space. "
            "Instantiate TabularQAgent/SarsaAgent directly with a compatible env."
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")


def evaluate(env, agent, seed, n_episodes=5) -> float:
    """Run n_episodes with no exploration and return mean reward."""
    rewards = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i)
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
    return float(np.mean(rewards))


def run_one_seed(env, args, seed, run_name) -> dict:
    """Train for one seed. Returns the metrics dict."""
    state_dim = env.observation_space.shape[0] if len(env.observation_space.shape) > 0 else 1
    action_dim = env.action_space.n
    env.action_space.seed(seed)

    q_net, agent = make_agent(args.algo, state_dim, action_dim, args)
    bitsize, total_bytes = calculate_bitsize(q_net)
    print(f"  [{args.network}/layers:{args.tensorize_layers}] {bitsize} params (~{total_bytes/1024:.2f} KB)")

    logger = Logger(use_wandb=args.wandb, project="tensor-rl", run_name=run_name)

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0
        steps = 0
        ep_losses, ep_qs, ep_gnorms = [], [], []

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            loss, mean_q, grad_norm = agent.update()

            if loss > 0:  # only count steps where an actual update happened
                ep_losses.append(loss)
                ep_qs.append(mean_q)
                ep_gnorms.append(grad_norm)

            state = next_state
            ep_reward += reward
            steps += 1

        eval_rew = evaluate(env, agent, seed=seed + 10000 + episode)

        logger.log(
            episode, ep_reward, steps,
            bitsize=(bitsize, total_bytes),
            loss=float(np.mean(ep_losses)) if ep_losses else 0.0,
            mean_q=float(np.mean(ep_qs)) if ep_qs else 0.0,
            grad_norm=float(np.mean(ep_gnorms)) if ep_gnorms else 0.0,
            eval_reward=eval_rew,
        )

        if episode % 50 == 0:
            eps = agent.epsilon() if hasattr(agent, 'epsilon') else 0.0
            ep_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            print(f"  Ep {episode}/{args.episodes} | train={ep_reward:.2f} | eval={eval_rew:.2f} "
                  f"| loss={ep_loss:.4f} | eps={eps:.2f}")

    logger.finish()
    return logger.metrics


def aggregate_seeds(all_metrics: list, run_base: str):
    """Average per-episode metrics across seeds and save as *_agg.json."""
    keys = ["rewards", "eval_reward", "loss", "mean_q", "grad_norm"]
    agg = {"episodes": all_metrics[0]["episodes"]}

    for k in keys:
        arrays = np.array([m[k] for m in all_metrics if k in m])
        agg[f"{k}_mean"] = arrays.mean(axis=0).tolist()
        agg[f"{k}_std"] = arrays.std(axis=0).tolist()

    os.makedirs("sim/data", exist_ok=True)
    path = os.path.join("sim/data", f"{run_base}_agg.json")
    with open(path, "w") as f:
        json.dump(agg, f, indent=4)
    print(f"Aggregated results saved to {path}")


def build_run_name(args, seed: int) -> str:
    tz_spec = args.tensorize_layers
    # Encode non-default tensorize specs in the run name (replace commas to avoid shell issues)
    tz = f"_tz{tz_spec.replace(',', '-')}" if tz_spec != "all" else ""
    return f"{args.env}_{args.algo}_{args.network}_rank{args.rank}{tz}_seed{seed}"


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"Setting up [{args.algo}] on [{args.env}] | network=[{args.network}] "
          f"rank={args.rank} tensorize_layers={args.tensorize_layers} | seeds={seeds}")

    if args.algo not in DEEP_ALGOS:
        raise ValueError("Tabular algos not supported via train.py — use agents directly.")

    env = make_env(args.env)
    all_metrics = []

    for seed in seeds:
        run_name = build_run_name(args, seed)
        print(f"\n=== Seed {seed} | run: {run_name} ===")
        metrics = run_one_seed(env, args, seed, run_name)
        all_metrics.append(metrics)

    if len(seeds) > 1:
        run_base = build_run_name(args, seeds[0]).rsplit("_seed", 1)[0]
        aggregate_seeds(all_metrics, run_base)

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
