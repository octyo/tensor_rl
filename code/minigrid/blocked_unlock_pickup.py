import numpy as np
from collections import defaultdict
from boilerplate_test import SimpleEnv

env = SimpleEnv()

# Use a subset of useful MiniGrid actions: left, right, forward, pickup, toggle
# (drop/done are excluded to reduce useless exploration)
ACTION_MAP = np.array([0, 1, 2, 3, 5], dtype=np.int64)
NUM_ACTIONS = len(ACTION_MAP)

# Simple Tabular Q-Learning (Reinforcement Learning) setup
q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float32))

learning_rate = 0.1
discount_factor = 0.99
episodes = 3000
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_episodes = 2500
step_penalty = -0.01
eval_every = 100
eval_episodes = 10

def get_state(env):
    """
    Compact state representation for this specific map.
    This avoids huge observation-based tables and improves learning stability.
    """
    u = env.unwrapped
    door = u.grid.get(5, 6)  # fixed by SimpleEnv
    carrying = u.carrying

    return (
        int(u.agent_pos[0]),
        int(u.agent_pos[1]),
        int(u.agent_dir),
        int(carrying is not None),
        int(getattr(door, "is_open", False)),
        int(getattr(door, "is_locked", False)),
    )

def epsilon_by_episode(episode_idx):
    frac = min(1.0, episode_idx / epsilon_decay_episodes)
    return epsilon_start + frac * (epsilon_end - epsilon_start)

def argmax_random_tie(values):
    max_v = np.max(values)
    max_idxs = np.flatnonzero(values == max_v)
    return int(np.random.choice(max_idxs))

def run_episode(env, q_table, epsilon=0.0, update_q=False):
    """Run one episode. Returns (episode_return, solved, steps)."""
    env.reset()
    state = get_state(env)

    terminated, truncated = False, False
    episode_return = 0.0
    steps = 0
    solved = False

    while not (terminated or truncated):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action_idx = np.random.randint(NUM_ACTIONS)
        else:
            action_idx = argmax_random_tie(q_table[state])

        action = int(ACTION_MAP[action_idx])

        # Standard Gymnasium step
        _, reward, terminated, truncated, info = env.step(action)
        next_state = get_state(env)

        # Shape rewards slightly to discourage spinning/idle behavior.
        shaped_reward = reward + step_penalty
        episode_return += reward
        steps += 1

        # In this environment, reward > 0 means goal reached.
        if reward > 0:
            solved = True

        if update_q:
            # Q-learning update step
            old_value = q_table[state][action_idx]
            next_max = np.max(q_table[next_state])

            # Q(s,a) = Q(s,a) + alpha * (R + gamma * max Q(s',a') - Q(s,a))
            target = shaped_reward if (terminated or truncated) else (shaped_reward + discount_factor * next_max)
            new_value = old_value + learning_rate * (target - old_value)
            q_table[state][action_idx] = new_value

        # Update current state
        state = next_state

    return episode_return, solved, steps

def evaluate_policy(env, q_table, n_episodes=10):
    returns = []
    solved_count = 0
    steps_list = []

    for _ in range(n_episodes):
        ep_return, solved, steps = run_episode(env, q_table, epsilon=0.0, update_q=False)
        returns.append(ep_return)
        solved_count += int(solved)
        steps_list.append(steps)

    return {
        "mean_return": float(np.mean(returns)),
        "success_rate": solved_count / n_episodes,
        "mean_steps": float(np.mean(steps_list)),
    }

# Training loop following standard Gymnasium syntax
train_returns = []
train_successes = []

for episode in range(1, episodes + 1):
    epsilon = epsilon_by_episode(episode)
    ep_return, solved, steps = run_episode(env, q_table, epsilon=epsilon, update_q=True)
    train_returns.append(ep_return)
    train_successes.append(int(solved))

    if episode % eval_every == 0:
        window = min(eval_every, len(train_returns))
        recent_return = float(np.mean(train_returns[-window:]))
        recent_success = float(np.mean(train_successes[-window:]))
        eval_metrics = evaluate_policy(env, q_table, n_episodes=eval_episodes)

        print(
            f"Episode {episode:4d}/{episodes} | "
            f"train_avg_return(last {window})={recent_return:.3f} | "
            f"train_success(last {window})={recent_success:.2%} | "
            f"epsilon={epsilon:.3f} | "
            f"eval_return={eval_metrics['mean_return']:.3f} | "
            f"eval_success={eval_metrics['success_rate']:.2%} | "
            f"eval_steps={eval_metrics['mean_steps']:.1f}"
        )

final_eval = evaluate_policy(env, q_table, n_episodes=50)
print("\nFinal greedy-policy evaluation over 50 episodes:")
print(
    f"  mean_return={final_eval['mean_return']:.3f}, "
    f"success_rate={final_eval['success_rate']:.2%}, "
    f"mean_steps={final_eval['mean_steps']:.1f}"
)

env.close()

# Optional demonstration runs at the end (tries to open a render window).
try:
    demo_env = SimpleEnv(render_mode="human")
    print("\nDemo runs with greedy policy:")
    for i in range(3):
        ep_return, solved, steps = run_episode(demo_env, q_table, epsilon=0.0, update_q=False)
        print(f"  demo {i+1}: return={ep_return:.3f}, solved={solved}, steps={steps}")
    demo_env.close()
except Exception as e:
    print(f"\nCould not open human render window for demos ({e}).")

print("Simple Q-Learning finished running.")
