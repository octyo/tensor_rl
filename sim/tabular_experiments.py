#!/usr/bin/env python3
"""
Tabular RL Experiments: CP Tensorized Q-Learning vs. Baselines

Comparing standard Q-Learning and SARSA against CP-decomposed Q-tables on FrozenLake-v1.

Experiment Design:
- Environment: FrozenLake-v1 (4x4 grid, 16 states, 4 actions)
- Baselines: TabularQAgent, SarsaAgent
- Tensor Method: TensorizedTabularQAgent (CP rank-1 + low-rank factorization)
- Rank Sweep: [1, 2, 4, 8, 16]
- Metrics: Sample efficiency, final reward, parameter count
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime
from agents.tabular_q import TabularQAgent, SarsaAgent, TensorizedTabularQAgent


# ============================================================================
# 1. ENVIRONMENT & HYPERPARAMETERS
# ============================================================================

ENV_ID = 'FrozenLake-v1'
STATE_SPACE = 16  # 4x4 grid
ACTION_SPACE = 4

# Training hyperparameters
LEARNING_RATE = 0.5
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 2000

# Experiment settings
RANKS = [1, 2, 4, 8, 16]
N_STEPS = 5000  # Total environment steps per agent
N_SEEDS = 3  # Number of random seeds for averaging
EVAL_INTERVAL = 100  # Steps between evaluation windows


def print_config():
    """Print configuration details."""
    print("="*70)
    print("TABULAR RL EXPERIMENTS: CP Tensorized Q-Learning")
    print("="*70)
    print(f"Environment: {ENV_ID}")
    print(f"State space: {STATE_SPACE}, Action space: {ACTION_SPACE}")
    print(f"Training steps per agent: {N_STEPS}")
    print(f"Number of seeds: {N_SEEDS}")
    print(f"Ranks to test: {RANKS}")
    print("="*70 + "\n")


# ============================================================================
# 2. PARAMETER COUNT ANALYSIS
# ============================================================================

def count_parameters(agent_type, rank=None):
    """Calculate parameter count for each agent type."""
    if agent_type == 'tabular':
        # Full Q-table: (states × actions)
        return STATE_SPACE * ACTION_SPACE
    elif agent_type == 'sarsa':
        # Same as tabular
        return STATE_SPACE * ACTION_SPACE
    elif agent_type == 'tensorized':
        # CP decomposition: factor_s (S × r) + factor_a (A × r)
        return STATE_SPACE * rank + ACTION_SPACE * rank
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def print_parameter_counts():
    """Display parameter counts for all ranks."""
    print("\n" + "="*60)
    print("Parameter Count Comparison (FrozenLake-v1)")
    print("="*60)

    params_full = count_parameters('tabular')
    print(f"{'Agent':<20} {'Rank':<8} {'Parameters':<12} {'Compression':<15}")
    print("-"*60)
    print(f"{'TabularQ':<20} {'-':<8} {params_full:<12} {'Baseline':<15}")
    print(f"{'Sarsa':<20} {'-':<8} {params_full:<12} {'Baseline':<15}")

    for rank in RANKS:
        params_cp = count_parameters('tensorized', rank=rank)
        compression = f"{(1 - params_cp / params_full) * 100:.1f}%"
        print(f"{'TensorizedQ (CP)':<20} {rank:<8} {params_cp:<12} {compression:<15}")

    print("="*60)


# ============================================================================
# 3. RUN AGENT FUNCTION
# ============================================================================

def run_agent(agent, env_id='FrozenLake-v1', n_steps=2000, seed=42):
    """
    Run a single agent for n_steps and return episode rewards + convergence metrics.
    
    Args:
        agent: The RL agent to run
        env_id: Gymnasium environment ID
        n_steps: Total environment steps
        seed: Random seed
    
    Returns:
        ep_rewards: List of episode rewards
        step_log: List of (step, avg_reward) tuples
    """
    env = gym.make(env_id, is_slippery=False)
    state, _ = env.reset(seed=seed)
    
    ep_reward = 0.0
    ep_rewards = []
    step_log = []  # Log every EVAL_INTERVAL steps
    
    for step in range(n_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # For Sarsa, pass the next action taken
        if isinstance(agent, SarsaAgent):
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, done, next_action)
        else:
            agent.update(state, action, reward, next_state, done)
        
        ep_reward += reward
        
        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            state, _ = env.reset()
        else:
            state = next_state
        
        # Log checkpoint
        if (step + 1) % EVAL_INTERVAL == 0:
            avg_reward = np.mean(ep_rewards[-10:]) if len(ep_rewards) >= 10 else np.mean(ep_rewards) if ep_rewards else 0.0
            step_log.append((step + 1, avg_reward))
    
    return ep_rewards, step_log


# ============================================================================
# 4. DEMO: INLINE EXPERIMENT
# ============================================================================

def run_demo():
    """Run a quick demo with selected agents and ranks."""
    print("\n" + "="*70)
    print("RUNNING INLINE DEMO (3 seeds, FrozenLake-v1, 2000 steps per seed)")
    print("="*70)

    demo_ranks = [2, 4, 8]
    demo_agent_types = [
        ('TabularQ', 'tabular', None),
        ('Sarsa', 'sarsa', None),
    ]

    for rank in demo_ranks:
        demo_agent_types.append((f'TensorizedQ (rank={rank})', 'tensorized', rank))

    results = {}

    for agent_name, agent_type, rank in demo_agent_types:
        all_rewards = []
        all_step_logs = []

        for seed in range(N_SEEDS):
            print(f"\n  {agent_name:<25} Seed {seed + 1}/{N_SEEDS}...", end=" ")

            if agent_type == 'tabular':
                agent = TabularQAgent(STATE_SPACE, ACTION_SPACE, epsilon_decay_steps=EPSILON_DECAY_STEPS, seed=seed)
            elif agent_type == 'sarsa':
                agent = SarsaAgent(STATE_SPACE, ACTION_SPACE, epsilon_decay_steps=EPSILON_DECAY_STEPS, seed=seed)
            else:  # tensorized
                agent = TensorizedTabularQAgent(STATE_SPACE, ACTION_SPACE, rank=rank, epsilon_decay_steps=EPSILON_DECAY_STEPS, seed=seed)

            ep_rewards, step_log = run_agent(agent, ENV_ID, N_STEPS, seed=seed)
            all_rewards.append(ep_rewards)
            all_step_logs.append(step_log)
            print(f"[OK] Final avg reward: {np.mean(ep_rewards[-20:]):.3f}")

        results[agent_name] = {
            'all_rewards': all_rewards,
            'step_logs': all_step_logs,
            'final_mean': np.mean([np.mean(r[-20:]) for r in all_rewards]),
            'final_std': np.std([np.mean(r[-20:]) for r in all_rewards]),
        }

    print("\n" + "="*70)
    return results


# ============================================================================
# 5. PRINT RESULTS SUMMARY
# ============================================================================

def print_results_summary(results):
    """Print formatted results table."""
    print("\nDemo Results Summary:")
    print("="*90)
    print(f"{'Agent':<25} {'Final Reward':<20} {'Std Dev':<15} {'Parameters':<15}")
    print("-"*90)

    for agent_name in results.keys():
        result = results[agent_name]
        final_mean = result['final_mean']
        final_std = result['final_std']
        
        # Extract rank if tensorized
        if 'rank=' in agent_name:
            rank = int(agent_name.split('rank=')[1].rstrip(')'))
            params = count_parameters('tensorized', rank=rank)
            param_str = f"{params} ({100*(1-params/64):.0f}% less)"
        else:
            params = 64
            param_str = f"{params} (baseline)"
        
        print(f"{agent_name:<25} {final_mean:<20.4f} {final_std:<15.4f} {param_str:<15}")

    print("="*90)


# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

def plot_results(results):
    """Create side-by-side comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Learning curves (reward over episodes) for all agents
    ax = axes[0, 0]
    for agent_name, result in results.items():
        # Smooth rewards with rolling average
        all_ep_rewards = np.concatenate(result['all_rewards'])
        smoothed = np.convolve(all_ep_rewards, np.ones(10)/10, mode='valid')
        ax.plot(smoothed, label=agent_name, alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Reward (smoothed, window=10)', fontsize=11)
    ax.set_title('Learning Curves: All Agents', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final reward vs. rank (for tensorized agents)
    ax = axes[0, 1]
    tensorized_agents = {k: v for k, v in results.items() if 'Tensorized' in k}
    if tensorized_agents:
        ranks_tested = []
        final_rewards = []
        for agent_name, result in sorted(tensorized_agents.items()):
            rank = int(agent_name.split('rank=')[1].rstrip(')'))
            ranks_tested.append(rank)
            final_rewards.append(result['final_mean'])
        
        ax.plot(ranks_tested, final_rewards, 'o-', linewidth=2, markersize=8, color='steelblue')
        baseline_reward = results['TabularQ']['final_mean']
        ax.axhline(baseline_reward, color='red', linestyle='--', label='TabularQ Baseline', linewidth=2)
        ax.set_xlabel('Rank', fontsize=11)
        ax.set_ylabel('Final Reward (avg last 20 episodes)', fontsize=11)
        ax.set_title('Reward vs. Rank (CP Tensorized)', fontsize=12, fontweight='bold')
        ax.set_xticks(ranks_tested)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Parameter count vs. final reward
    ax = axes[1, 0]
    agent_names = []
    param_counts = []
    final_rewards_all = []
    colors = []
    
    for agent_name, result in results.items():
        agent_names.append(agent_name)
        final_rewards_all.append(result['final_mean'])
        
        if 'rank=' in agent_name:
            rank = int(agent_name.split('rank=')[1].rstrip(')'))
            params = count_parameters('tensorized', rank=rank)
            colors.append('steelblue')
        else:
            params = 64
            colors.append('red' if agent_name == 'TabularQ' else 'orange')
        
        param_counts.append(params)
    
    scatter = ax.scatter(param_counts, final_rewards_all, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    for i, name in enumerate(agent_names):
        ax.annotate(name.replace('TensorizedQ (', 'TQ(').replace(')', ')'), 
                    (param_counts[i], final_rewards_all[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax.set_xlabel('Parameter Count', fontsize=11)
    ax.set_ylabel('Final Reward', fontsize=11)
    ax.set_title('Parameter Efficiency Trade-off', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Compression ratio vs. reward gap
    ax = axes[1, 1]
    baseline_reward = results['TabularQ']['final_mean']
    compression_ratios = []
    reward_gaps = []
    agent_labels = []
    
    for agent_name, result in results.items():
        if 'rank=' not in agent_name:
            continue
        rank = int(agent_name.split('rank=')[1].rstrip(')'))
        params = count_parameters('tensorized', rank=rank)
        compression = 100 * (1 - params / 64)
        reward_gap = baseline_reward - result['final_mean']
        compression_ratios.append(compression)
        reward_gaps.append(reward_gap)
        agent_labels.append(f"rank={rank}")
    
    ax.bar(agent_labels, reward_gaps, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No performance loss')
    ax.set_ylabel('Reward Gap (TabularQ - TensorizedQ)', fontsize=11)
    ax.set_xlabel('Configuration', fontsize=11)
    ax.set_title('Performance Loss vs. Compression', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('tabular_demo_results.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("\n[OK] Plots saved to 'tabular_demo_results.png'")


# ============================================================================
# 7. FULL BATCH SWEEP
# ============================================================================

def run_full_sweep():
    """Run complete experiment with all agents, ranks, and seeds."""
    FULL_RANKS = [1, 2, 4, 8, 16]
    FULL_SEEDS = 5
    
    full_results = {}
    
    print("\n" + "="*70)
    print("FULL EXPERIMENT SWEEP (5 seeds × 3 agents × 5 ranks = 75 runs)")
    print("="*70)
    
    run_count = 0
    total_runs = (2 + len(FULL_RANKS)) * FULL_SEEDS  # 2 baselines + tensorized
    
    for agent_type in ['tabular', 'sarsa']:
        config_name = agent_type.upper()
        full_results[config_name] = {'final_rewards': [], 'step_logs': []}
        
        for seed in range(FULL_SEEDS):
            run_count += 1
            if agent_type == 'tabular':
                agent = TabularQAgent(STATE_SPACE, ACTION_SPACE, epsilon_decay_steps=EPSILON_DECAY_STEPS, seed=seed)
            else:
                agent = SarsaAgent(STATE_SPACE, ACTION_SPACE, epsilon_decay_steps=EPSILON_DECAY_STEPS, seed=seed)

            print(f"[{run_count}/{total_runs}] {config_name:<8} Seed {seed+1}/{FULL_SEEDS}...", end=" ")
            ep_rewards, step_log = run_agent(agent, ENV_ID, N_STEPS, seed=seed)
            full_results[config_name]['final_rewards'].append(ep_rewards)
            full_results[config_name]['step_logs'].append(step_log)
            print(f"[OK] Final reward: {np.mean(ep_rewards[-20:]):.3f}")

    for rank in FULL_RANKS:
        config_name = f'TENSORIZED_CP_RANK{rank}'
        full_results[config_name] = {'final_rewards': [], 'step_logs': []}
        
        for seed in range(FULL_SEEDS):
            run_count += 1
            agent = TensorizedTabularQAgent(STATE_SPACE, ACTION_SPACE, rank=rank, epsilon_decay_steps=EPSILON_DECAY_STEPS, seed=seed)

            print(f"[{run_count}/{total_runs}] TensorizedQ (r={rank}) Seed {seed+1}/{FULL_SEEDS}...", end=" ")
            ep_rewards, step_log = run_agent(agent, ENV_ID, N_STEPS, seed=seed)
            full_results[config_name]['final_rewards'].append(ep_rewards)
            full_results[config_name]['step_logs'].append(step_log)
            print(f"[OK] Final reward: {np.mean(ep_rewards[-20:]):.3f}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'data/tabular_full_sweep_{timestamp}.json'
    os.makedirs('data', exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    for config, data in full_results.items():
        data['final_rewards'] = [r.tolist() if isinstance(r, np.ndarray) else r for r in data['final_rewards']]
        data['step_logs'] = [s for s in data['step_logs']]
    
    with open(filename, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n[OK] Results saved to {filename}")
    return full_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Tabular RL Experiments: CP Tensorized Q-Learning vs. Baselines'
    )
    parser.add_argument('--demo', action='store_true', default=True,
                        help='Run demo with 3 seeds (default: True)')
    parser.add_argument('--full-sweep', action='store_true',
                        help='Run full sweep with 5 seeds (takes ~15 min)')
    parser.add_argument('--no-demo', action='store_true',
                        help='Skip demo and go directly to full sweep')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plotting after demo')
    
    args = parser.parse_args()
    
    print_config()
    print_parameter_counts()
    
    if args.no_demo:
        args.demo = False
    
    if args.demo and not args.full_sweep:
        print("\nRunning demo experiment...")
        results = run_demo()
        print_results_summary(results)
        
        if not args.no_plots:
            print("\nGenerating plots...")
            plot_results(results)
        
        print("\n" + "="*70)
        print("Demo complete!")
        print("To run full 5-seed sweep, use: python tabular_experiments.py --full-sweep")
        print("="*70)
    
    elif args.full_sweep or (args.no_demo and not args.demo):
        print("\nStarting full sweep...")
        full_results = run_full_sweep()
        
        print("\n" + "="*70)
        print("Full sweep complete!")
        print("Results saved to 'data/tabular_full_sweep_*.json'")
        print("="*70)
