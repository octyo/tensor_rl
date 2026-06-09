import argparse
import json
import os
import numpy as np

import gymnasium as gym
from envs.env_utils import make_env, make_env_structured
from models.q_networks import QNetwork, DuelingQNetwork, StructuredQNetwork
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from utils.logging_utils import Logger, calculate_bitsize
from config import (DEFAULT_ENV, DEFAULT_SEED, DEFAULT_EPISODES, DEFAULT_NETWORK,
                    DEFAULT_RANK, HIDDEN_SIZES, TAU)

DEEP_ALGOS = {"dqn", "double_dqn", "dueling_dqn"}
TABULAR_ALGOS = {
    "tabular_q", "sarsa",
    "tabular_q_tensor", "sarsa_tensor",   # legacy CP aliases
    "tabular_q_cp", "sarsa_cp",
    "tabular_q_tucker", "sarsa_tucker",
    "tabular_q_tt", "sarsa_tt",
}
SARSA_ALGOS = {"sarsa", "sarsa_tensor", "sarsa_cp", "sarsa_tucker", "sarsa_tt"}
STRUCTURED_NETWORKS = {"tt", "cp"}
TABULAR_ENVS = {"FrozenLake-v1", "CliffWalking-v1"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=DEFAULT_ENV)
    parser.add_argument("--algo", type=str, default="dqn",
                        choices=["dqn", "double_dqn", "dueling_dqn",
                                 "tabular_q", "sarsa",
                                 "tabular_q_tensor", "sarsa_tensor",
                                 "tabular_q_cp", "sarsa_cp",
                                 "tabular_q_tucker", "sarsa_tucker",
                                 "tabular_q_tt", "sarsa_tt"])
    parser.add_argument("--network", type=str, default=DEFAULT_NETWORK,
                        choices=["standard", "cp", "tucker", "tt"])
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--tensorize_layers", type=str, default="all",
                        help=("Which layers to tensorize. 'all' (default), 'none', or "
                              "comma-separated 0-based indices e.g. '0,2'. "
                              "Indices: 0=first hidden, 1=second hidden, ..., N=output layer. "
                              "For dueling networks, indices refer to trunk hidden layers only. "
                              "Ignored when --structured is set."))
    parser.add_argument("--structured", action="store_true",
                        help=("Use StructuredQNetwork with a TTEmbedding or CPEmbedding first "
                              "layer that preserves the (7,7,3) MiniGrid observation as a "
                              "genuine 3-mode tensor instead of flattening it. "
                              "Requires a MiniGrid env and --network tt or cp."))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--tau", type=float, default=TAU,
                        help="Polyak update rate for Double/Dueling DQN target network")
    parser.add_argument("--seeds", type=str, default=str(DEFAULT_SEED),
                        help="Comma-separated seeds, e.g. '42,43,44'")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps per episode for tabular training loop (default: 200)")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def validate_structured_args(args):
    """Raise a clear error if --structured is used with incompatible arguments."""
    if not args.structured:
        return
    if "MiniGrid" not in args.env:
        raise ValueError(
            "--structured requires a MiniGrid environment (e.g. MiniGrid-Empty-5x5-v0). "
            f"Got: {args.env}"
        )
    if args.network not in STRUCTURED_NETWORKS:
        raise ValueError(
            f"--structured only supports --network tt or cp (got '{args.network}'). "
            "Tucker and standard linear are not applicable to the embedding approach."
        )


def make_agent(algo: str, action_dim: int, args, state_dim: int = None, mode_dims: tuple = None):
    """Factory: returns (q_net, agent).

    For the flat path, state_dim must be provided.
    For the structured path (args.structured=True), mode_dims must be provided.
    """
    tz = args.tensorize_layers

    if args.structured:
        # StructuredQNetwork: genuine multi-mode TN embedding, then standard hidden layers.
        # hidden_size is taken from the first element of HIDDEN_SIZES.
        q_net = StructuredQNetwork(
            mode_dims=mode_dims,
            action_dim=action_dim,
            hidden_size=HIDDEN_SIZES[0],
            embedding_type=args.network,
            rank=args.rank,
        )
        if algo == "dqn":
            return q_net, DQNAgent(q_net, epsilon_decay_steps=(args.episodes * 100))
        elif algo == "double_dqn":
            agent = DoubleDQNAgent(q_net, action_dim=action_dim,
                                   epsilon_decay_steps=(args.episodes * 100), tau=args.tau)
            return q_net, agent
        elif algo == "dueling_dqn":
            agent = DuelingDQNAgent(q_net, action_dim=action_dim,
                                    epsilon_decay_steps=(args.episodes * 100), tau=args.tau)
            return q_net, agent

    # --- Flat path ---
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


def run_one_seed(env, args, seed, run_name, mode_dims=None) -> dict:
    """Train for one seed. Returns the metrics dict."""
    action_dim = env.action_space.n
    env.action_space.seed(seed)

    if args.structured:
        q_net, agent = make_agent(args.algo, action_dim, args, mode_dims=mode_dims)
    else:
        obs_shape = env.observation_space.shape
        state_dim = obs_shape[0] if len(obs_shape) > 0 else 1
        q_net, agent = make_agent(args.algo, action_dim, args, state_dim=state_dim)

    bitsize, total_bytes = calculate_bitsize(q_net)
    label = f"structured/{args.network}" if args.structured else f"{args.network}/layers:{args.tensorize_layers}"
    print(f"  [{label}] {bitsize} params (~{total_bytes/1024:.2f} KB)")

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

            if loss > 0:
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


def get_state_dims(env) -> tuple:
    """Return the multi-dimensional state shape for tabular envs.

    For FrozenLake: (nrow, ncol).  For CliffWalking: (4, 12).
    Falls back to (n_states,) for any other Discrete env.
    """
    uw = env.unwrapped
    if hasattr(uw, 'nrow') and hasattr(uw, 'ncol'):
        return (uw.nrow, uw.ncol)
    if hasattr(uw, 'shape') and isinstance(uw.shape, tuple) and len(uw.shape) >= 2:
        return uw.shape
    return (env.observation_space.n,)


def make_tabular_agent(algo: str, env, args):
    """Factory for tabular agents. env must have Discrete observation and action spaces."""
    from agents.tabular_q import (
        TabularQAgent, SarsaAgent,
        TensorizedTabularQAgent, SarsaTensorAgent,
        CPMultiTabularQAgent, SarsaCPAgent,
        TuckerMultiTabularQAgent, SarsaTuckerAgent,
        TTMultiTabularQAgent, SarsaTTAgent,
    )
    S = env.observation_space.n
    A = env.action_space.n
    decay = args.episodes * 10
    common = dict(epsilon_decay_steps=decay)
    state_dims = get_state_dims(env)

    if algo == "tabular_q":
        return TabularQAgent(S, A, **common)
    elif algo == "sarsa":
        return SarsaAgent(S, A, **common)
    elif algo in ("tabular_q_tensor", "tabular_q_cp"):
        return CPMultiTabularQAgent(state_dims, A, rank=args.rank, **common)
    elif algo in ("sarsa_tensor", "sarsa_cp"):
        return SarsaCPAgent(state_dims, A, rank=args.rank, **common)
    elif algo == "tabular_q_tucker":
        return TuckerMultiTabularQAgent(state_dims, A, rank=args.rank, **common)
    elif algo == "sarsa_tucker":
        return SarsaTuckerAgent(state_dims, A, rank=args.rank, **common)
    elif algo == "tabular_q_tt":
        return TTMultiTabularQAgent(state_dims, A, rank=args.rank, **common)
    elif algo == "sarsa_tt":
        return SarsaTTAgent(state_dims, A, rank=args.rank, **common)
    else:
        raise ValueError(f"Unknown tabular algo: {algo}")


def run_tabular(args, seed: int):
    """Tabular training loop — no torch, no replay buffer, no DQN infrastructure.
    Works with any env that has integer (Discrete) observations, e.g. FrozenLake-v1.
    """
    env = gym.make(args.env)
    state_dims = get_state_dims(env)
    agent = make_tabular_agent(args.algo, env, args)
    run_name = build_run_name(args, seed)

    if state_dims != (env.observation_space.n,):
        print(f"  state_dims={state_dims} (multi-D tensor indexing)")

    logger = Logger(use_wandb=args.wandb, project="tensor-rl", run_name=run_name)
    is_sarsa = args.algo in SARSA_ALGOS

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0
        steps = 0
        next_action = None

        while not done and steps < args.max_steps:
            action = next_action if next_action is not None else agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if is_sarsa:
                next_action = agent.select_action(next_state) if not done else None
                agent.update(state, action, reward, next_state, done, next_action=next_action)
            else:
                agent.update(state, action, reward, next_state, done)
                next_action = None

            state = next_state
            ep_reward += reward
            steps += 1

        logger.log(episode, ep_reward, steps)

        if episode % 100 == 0:
            print(f"  Ep {episode}/{args.episodes} | reward={ep_reward:.2f} | eps={agent.epsilon():.3f}")

    logger.finish()
    return logger.metrics


def build_run_name(args, seed: int) -> str:
    if args.algo in TABULAR_ALGOS:
        uses_rank = args.algo not in {"tabular_q", "sarsa"}
        rank_suffix = f"_rank{args.rank}" if uses_rank else ""
        return f"{args.env}_{args.algo}{rank_suffix}_seed{seed}"
    if args.structured:
        return f"{args.env}_{args.algo}_{args.network}_rank{args.rank}_struct_seed{seed}"
    tz_spec = args.tensorize_layers
    tz = f"_tz{tz_spec.replace(',', '-')}" if tz_spec != "all" else ""
    return f"{args.env}_{args.algo}_{args.network}_rank{args.rank}{tz}_seed{seed}"


def main():
    args = parse_args()
    validate_structured_args(args)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    mode = "structured" if args.structured else f"tensorize_layers={args.tensorize_layers}"
    print(f"Setting up [{args.algo}] on [{args.env}] | network=[{args.network}] "
          f"rank={args.rank} {mode} | seeds={seeds}")

    if args.algo in TABULAR_ALGOS:
        for seed in seeds:
            print(f"\n=== Seed {seed} | run: {build_run_name(args, seed)} ===")
            run_tabular(args, seed)
        print("\nTraining finished.")
        return

    if args.structured:
        env, mode_dims = make_env_structured(args.env)
        print(f"  Structured input shape: {mode_dims}")
    else:
        env = make_env(args.env)
        mode_dims = None

    all_metrics = []
    for seed in seeds:
        run_name = build_run_name(args, seed)
        print(f"\n=== Seed {seed} | run: {run_name} ===")
        metrics = run_one_seed(env, args, seed, run_name, mode_dims=mode_dims)
        all_metrics.append(metrics)

    if len(seeds) > 1:
        run_base = build_run_name(args, seeds[0]).rsplit("_seed", 1)[0]
        aggregate_seeds(all_metrics, run_base)

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
