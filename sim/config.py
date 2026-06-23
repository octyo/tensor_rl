# Central hyperparameter config — import from here rather than hardcoding defaults.
# Values informed by course ex13 reference implementations (rl_course/ex13/).

# --- Environment ---
DEFAULT_ENV = "MiniGrid-Empty-5x5-v0"
DEFAULT_SEED = 42
DEFAULT_EPISODES = 500

# --- Network architecture ---
DEFAULT_NETWORK = "standard"   # standard | cp | tucker | tt
DEFAULT_RANK = 4
HIDDEN_SIZES = (128, 128)

# --- Shared agent ---
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 100_000
REPLAY_MIN_SIZE = 300          # steps before training starts (course uses 300-500)

# --- Epsilon-greedy exploration ---
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DELAY = 300            # steps at epsilon_start before decay begins (course: 300)
# decay steps are set as episodes * 100 in train.py so they scale with run length

# --- Target network ---
# DQN: hard copy every N steps; Double/Dueling: soft Polyak update every step
TARGET_UPDATE_INTERVAL = 200   # DQN hard update frequency (lower = more stable on short episodes)
TAU = 0.05                     # Polyak rate for Double/Dueling (course uses 0.08)
