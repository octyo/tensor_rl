import argparse
import time
import numpy as np

# Assuming local package imports
from envs.env_utils import make_env
from models.q_networks import QNetwork
from agents.dqn_agent import DQNAgent
from utils.logging_utils import Logger, calculate_bitsize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--algo", type=str, default="dqn")
    parser.add_argument("--network", type=str, default="standard", choices=["standard", "cp", "tucker", "tt"])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--dynamic_rank", action="store_true")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Environment Wrapper (Flattens images and shapes)
    env = make_env(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    
    state_dim = env.observation_space.shape[0] if len(env.observation_space.shape) > 0 else 1
    action_dim = env.action_space.n
    
    print(f"Setting up [{args.algo}] on [{args.env}] - network type: [{args.network}] (rank: {args.rank})")
    
    # 2. Target Networks using custom Tensor Layers or Standard Linear Layers
    q_net = QNetwork(state_dim, action_dim, hidden_sizes=(128, 128), 
                     network_type=args.network, rank=args.rank)
    
    # Calculate initial sizes
    bitsize, total_bytes = calculate_bitsize(q_net)
    print(f"[{args.network}] Architecture size: {bitsize} parameters (~{total_bytes / 1024:.2f} KB)")
    
    # 3. DQN Agent logic
    agent = DQNAgent(q_net, epsilon_decay_steps=(args.episodes * 100))
    
    # 4. Logger Setup
    run_name = f"{args.env}_{args.algo}_{args.network}_rank{args.rank}_seed{args.seed}"
    logger = Logger(use_wandb=args.wandb, project="tensor-rl", run_name=run_name)
    
    # 5. Core Training Loop
    total_steps = 0
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        episodic_reward = 0
        steps = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            agent.update()
            
            state = next_state
            episodic_reward += reward
            steps += 1
            total_steps += 1
            
            if done:
                break
        
        logger.log(episode, episodic_reward, steps, bitsize=(bitsize, total_bytes))
        
        if episode % 50 == 0:
            print(f"Episode: {episode}/{args.episodes} | Reward: {episodic_reward:.2f} | Epsilon: {agent.epsilon():.2f}")
            
    logger.finish()
    print("Training finished.")

if __name__ == "__main__":
    main()
