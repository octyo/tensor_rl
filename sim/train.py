import argparse
import time
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
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--tau", type=float, default=TAU,
                        help="Polyak update rate for Double/Dueling DQN target network")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def make_agent(algo: str, state_dim: int, action_dim: int, args):
    """Factory: returns the appropriate agent given --algo."""
    if algo == "dqn":
        q_net = QNetwork(state_dim, action_dim, hidden_sizes=HIDDEN_SIZES,
                         network_type=args.network, rank=args.rank)
        return q_net, DQNAgent(q_net, epsilon_decay_steps=(args.episodes * 100))

    elif algo == "double_dqn":
        q_net = QNetwork(state_dim, action_dim, hidden_sizes=HIDDEN_SIZES,
                         network_type=args.network, rank=args.rank)
        agent = DoubleDQNAgent(q_net, action_dim=action_dim,
                               epsilon_decay_steps=(args.episodes * 100), tau=args.tau)
        return q_net, agent

    elif algo == "dueling_dqn":
        q_net = DuelingQNetwork(state_dim, action_dim, hidden_sizes=HIDDEN_SIZES,
                                network_type=args.network, rank=args.rank)
        agent = DuelingDQNAgent(q_net, action_dim=action_dim,
                                epsilon_decay_steps=(args.episodes * 100), tau=args.tau)
        return q_net, agent

    elif algo in ("tabular_q", "sarsa"):
        raise ValueError(
            f"Tabular algo '{algo}' requires a discrete state space. "
            "Use a tabular-compatible env (e.g. FrozenLake-v1) and call TabularQAgent/SarsaAgent directly."
        )

    else:
        raise ValueError(f"Unknown algo: {algo}")


def run_deep(env, agent, q_net, args, logger):
    bitsize, total_bytes = calculate_bitsize(q_net)
    print(f"[{args.network}] Architecture size: {bitsize} parameters (~{total_bytes / 1024:.2f} KB)")

    for episode in range(1, args.episodes + 1):
        state, info = env.reset(seed=args.seed + episode)
        done = False
        episodic_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            agent.update()

            state = next_state
            episodic_reward += reward
            steps += 1

        logger.log(episode, episodic_reward, steps, bitsize=(bitsize, total_bytes))

        if episode % 50 == 0:
            eps = agent.epsilon() if hasattr(agent, 'epsilon') else 0.0
            print(f"Episode: {episode}/{args.episodes} | Reward: {episodic_reward:.2f} | Epsilon: {eps:.2f}")

    return bitsize, total_bytes


def main():
    args = parse_args()

    env = make_env(args.env)
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0] if len(env.observation_space.shape) > 0 else 1
    action_dim = env.action_space.n

    print(f"Setting up [{args.algo}] on [{args.env}] - network type: [{args.network}] (rank: {args.rank})")

    run_name = f"{args.env}_{args.algo}_{args.network}_rank{args.rank}_seed{args.seed}"
    logger = Logger(use_wandb=args.wandb, project="tensor-rl", run_name=run_name)

    if args.algo in DEEP_ALGOS:
        q_net, agent = make_agent(args.algo, state_dim, action_dim, args)
        run_deep(env, agent, q_net, args, logger)
    else:
        raise ValueError(f"Tabular algos not yet supported via train.py for this env type.")

    logger.finish()
    print("Training finished.")


if __name__ == "__main__":
    main()
