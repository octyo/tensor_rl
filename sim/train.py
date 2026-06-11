import argparse
import json
import os
import numpy as np

import gymnasium as gym
from gymnasium.envs.registration import register as _gym_register

# Simple 4×4 no-hole FrozenLake: straight path from S→G, no traps, is_slippery=False.
# Ideal for fast benchmarking — random exploration finds the goal reliably.
_gym_register(
    id="FrozenLakeSimple-v0",
    entry_point="gymnasium.envs.toy_text:FrozenLakeEnv",
    kwargs={"desc": ["SFFF", "FFFF", "FFFF", "FFFG"], "is_slippery": False},
    max_episode_steps=200,
)

from envs.env_utils import make_env, make_env_structured, make_env_cnn
from models.q_networks import QNetwork, DuelingQNetwork, StructuredQNetwork
from models.cnn_networks import CNNQNetwork
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
STRUCTURED_NETWORKS = {"tt", "cp", "tucker"}
TABULAR_ENVS = {"FrozenLake-v1", "FrozenLakeSimple-v0", "CliffWalking-v1"}


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
                        choices=["standard", "cp", "tucker", "tt", "mps"])
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
    parser.add_argument("--tt_dims", type=str, default=None,
                        help=("Mode shapes for MPS/TT weight decomposition, e.g. '4,4,8'. "
                              "Product must equal in_features for each layer. "
                              "If omitted, shapes are auto-factored. Used with --network mps."))
    parser.add_argument("--cnn_mode", type=str, default=None,
                        choices=["backbone", "tensorized"],
                        help=("Enable CNN input processing (Case 4). "
                              "'backbone': standard Conv2d + tensorized linear head. "
                              "'tensorized': CP/Tucker decomposed conv layers + standard head."))
    parser.add_argument("--compress", type=str, default=None,
                        choices=["cp", "tucker", "tt", "mps"],
                        help=("Post-hoc compression (Case 2b). Train a standard network "
                              "first, then compress each linear layer with the chosen method."))
    parser.add_argument("--finetune_episodes", type=int, default=0,
                        help="Fine-tuning episodes after post-hoc compression (default: 0).")
    return parser.parse_args()


def validate_args(args):
    """Raise a clear error if incompatible flags are combined."""
    if args.structured:
        if "MiniGrid" not in args.env:
            raise ValueError(
                "--structured requires a MiniGrid environment (e.g. MiniGrid-Empty-5x5-v0). "
                f"Got: {args.env}"
            )
        if args.network not in STRUCTURED_NETWORKS:
            raise ValueError(
                f"--structured only supports --network tt, cp, or tucker (got '{args.network}'). "
            )
    if args.cnn_mode is not None and args.structured:
        raise ValueError("--cnn_mode and --structured are mutually exclusive.")
    if args.compress is not None and args.network != "standard":
        raise ValueError(
            "--compress performs post-hoc compression of a trained standard network. "
            f"Use --network standard (got '{args.network}')."
        )
    if args.finetune_episodes > 0 and args.compress is None:
        raise ValueError("--finetune_episodes requires --compress to be set.")
    if args.tt_dims is not None and args.network not in ("mps", "tt"):
        print(f"[warning] --tt_dims is only used with --network mps; ignored for '{args.network}'.")


# Keep old name as alias for backwards compatibility
validate_structured_args = validate_args


def make_agent(algo: str, action_dim: int, args, state_dim: int = None,
               mode_dims: tuple = None, obs_shape: tuple = None):
    """Factory: returns (q_net, agent).

    Routing priority:
        1. args.cnn_mode set  → CNNQNetwork (Case 4)
        2. args.structured    → StructuredQNetwork (Case 3)
        3. args.network=="mps"→ QNetwork with MPSLinear (Case 6)
        4. else               → QNetwork / DuelingQNetwork (Cases 1/2a/5)

    For CNN path:  obs_shape (C, H, W) must be provided.
    For structured path: mode_dims must be provided.
    For flat path: state_dim must be provided.
    """
    tz = args.tensorize_layers
    eps_decay = args.episodes * 100
    tt_dims = (tuple(int(x) for x in args.tt_dims.split(","))
               if getattr(args, 'tt_dims', None) else None)

    def _wrap_agent(q_net):
        if algo == "dqn":
            return DQNAgent(q_net, epsilon_decay_steps=eps_decay)
        elif algo == "double_dqn":
            return DoubleDQNAgent(q_net, action_dim=action_dim,
                                  epsilon_decay_steps=eps_decay, tau=args.tau)
        elif algo == "dueling_dqn":
            return DuelingDQNAgent(q_net, action_dim=action_dim,
                                   epsilon_decay_steps=eps_decay, tau=args.tau)
        else:
            raise ValueError(f"Unknown deep algo: {algo}")

    # --- Case 4: CNN path ---
    if getattr(args, 'cnn_mode', None) is not None:
        q_net = CNNQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            rank=args.rank,
            network_type=args.network,
            cnn_mode=args.cnn_mode,
            hidden_sizes=HIDDEN_SIZES,
            tt_dims=tt_dims,
        )
        return q_net, _wrap_agent(q_net)

    # --- Case 3: Structured embedding path ---
    if args.structured:
        q_net = StructuredQNetwork(
            mode_dims=mode_dims,
            action_dim=action_dim,
            hidden_size=HIDDEN_SIZES[0],
            embedding_type=args.network,
            rank=args.rank,
        )
        return q_net, _wrap_agent(q_net)

    # --- Cases 1/2a/5/6: Flat path ---
    if algo in ("dqn", "double_dqn"):
        q_net = QNetwork(state_dim, action_dim, hidden_sizes=HIDDEN_SIZES,
                         network_type=args.network, rank=args.rank,
                         tensorize_layers=tz, tt_dims=tt_dims)
        return q_net, _wrap_agent(q_net)

    elif algo == "dueling_dqn":
        q_net = DuelingQNetwork(state_dim, action_dim, hidden_sizes=HIDDEN_SIZES,
                                network_type=args.network, rank=args.rank,
                                tensorize_layers=tz, tt_dims=tt_dims)
        return q_net, _wrap_agent(q_net)

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


def run_one_seed(env, args, seed, run_name, mode_dims=None, obs_shape=None) -> dict:
    """Train for one seed. Returns the metrics dict and the trained q_net."""
    action_dim = env.action_space.n
    env.action_space.seed(seed)

    if getattr(args, 'cnn_mode', None) is not None:
        q_net, agent = make_agent(args.algo, action_dim, args, obs_shape=obs_shape)
        label = f"cnn_{args.cnn_mode}/{args.network}"
    elif args.structured:
        q_net, agent = make_agent(args.algo, action_dim, args, mode_dims=mode_dims)
        label = f"structured/{args.network}"
    else:
        flat_shape = env.observation_space.shape
        state_dim = flat_shape[0] if len(flat_shape) > 0 else 1
        q_net, agent = make_agent(args.algo, action_dim, args, state_dim=state_dim)
        label = f"{args.network}/layers:{args.tensorize_layers}"

    bitsize, total_bytes = calculate_bitsize(q_net)
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
    return logger.metrics, q_net


def run_with_compression(env, args, seed, run_name) -> dict:
    """Case 2b: train a standard network, compress it, optionally fine-tune, then evaluate.

    Returns metrics dict from the post-compression evaluation run.
    """
    import copy as _copy
    from models.compression import compress_network, finetune

    # Step 1: train a standard network
    std_args = _copy.copy(args)
    std_args.network = "standard"
    std_args.compress = None
    std_run_name = run_name + "_pretrain"
    print(f"  [compress] Training standard network for {args.episodes} episodes...")
    std_metrics, q_net = run_one_seed(env, std_args, seed, std_run_name)

    # Step 2: compress
    tt_dims = (tuple(int(x) for x in args.tt_dims.split(","))
               if getattr(args, 'tt_dims', None) else None)
    print(f"  [compress] Compressing with method='{args.compress}' rank={args.rank}...")
    compressed_net = compress_network(q_net, method=args.compress,
                                      rank=args.rank, tt_dims=tt_dims)

    # Step 3: optional fine-tune
    if args.finetune_episodes > 0:
        print(f"  [compress] Fine-tuning for {args.finetune_episodes} episodes...")
        finetune(compressed_net, env, args, n_episodes=args.finetune_episodes)

    # Step 4: evaluate compressed net over N episodes and log
    from agents.dqn_agent import DQNAgent
    from utils.logging_utils import Logger, calculate_bitsize

    eval_agent = DQNAgent(compressed_net, epsilon_start=0.0, epsilon_end=0.0,
                          epsilon_decay_steps=1)
    bitsize, total_bytes = calculate_bitsize(compressed_net)
    print(f"  [compress] Compressed: {bitsize} params (~{total_bytes/1024:.2f} KB)")

    logger = Logger(use_wandb=args.wandb, project="tensor-rl", run_name=run_name)
    eval_episodes = min(100, args.episodes)
    for episode in range(1, eval_episodes + 1):
        state, _ = env.reset(seed=seed + episode + 20000)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            action = eval_agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1
        logger.log(episode, ep_reward, steps, bitsize=(bitsize, total_bytes))

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
    if getattr(args, 'compress', None):
        ft = f"_ft{args.finetune_episodes}" if args.finetune_episodes else ""
        return f"{args.env}_{args.algo}_compress{args.compress}_rank{args.rank}{ft}_seed{seed}"
    if getattr(args, 'cnn_mode', None):
        return f"{args.env}_{args.algo}_{args.network}_rank{args.rank}_cnn{args.cnn_mode}_seed{seed}"
    if args.structured:
        return f"{args.env}_{args.algo}_{args.network}_rank{args.rank}_struct_seed{seed}"
    if args.network == "mps":
        td = ("_td" + args.tt_dims.replace(",", "-")) if getattr(args, 'tt_dims', None) else ""
        return f"{args.env}_{args.algo}_mps_rank{args.rank}{td}_seed{seed}"
    tz_spec = args.tensorize_layers
    tz = f"_tz{tz_spec.replace(',', '-')}" if tz_spec != "all" else ""
    return f"{args.env}_{args.algo}_{args.network}_rank{args.rank}{tz}_seed{seed}"


def main():
    args = parse_args()
    validate_args(args)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    cnn_mode = getattr(args, 'cnn_mode', None)
    compress = getattr(args, 'compress', None)
    mode = ("cnn:" + cnn_mode if cnn_mode
            else "structured" if args.structured
            else "compress:" + compress if compress
            else f"tensorize_layers={args.tensorize_layers}")
    print(f"Setting up [{args.algo}] on [{args.env}] | network=[{args.network}] "
          f"rank={args.rank} {mode} | seeds={seeds}")

    if args.algo in TABULAR_ALGOS:
        for seed in seeds:
            print(f"\n=== Seed {seed} | run: {build_run_name(args, seed)} ===")
            run_tabular(args, seed)
        print("\nTraining finished.")
        return

    # Set up environment
    obs_shape = None
    mode_dims = None
    if cnn_mode is not None:
        env, obs_shape = make_env_cnn(args.env)
        print(f"  CNN input shape (C,H,W): {obs_shape}")
    elif args.structured:
        env, mode_dims = make_env_structured(args.env)
        print(f"  Structured input shape: {mode_dims}")
    else:
        env = make_env(args.env)

    all_metrics = []
    for seed in seeds:
        run_name = build_run_name(args, seed)
        print(f"\n=== Seed {seed} | run: {run_name} ===")

        if compress is not None:
            metrics = run_with_compression(env, args, seed, run_name)
        else:
            metrics, _ = run_one_seed(env, args, seed, run_name,
                                      mode_dims=mode_dims, obs_shape=obs_shape)
        all_metrics.append(metrics)

    if len(seeds) > 1:
        run_base = build_run_name(args, seeds[0]).rsplit("_seed", 1)[0]
        aggregate_seeds(all_metrics, run_base)

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
