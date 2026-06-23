ENVS = [
    'MiniGrid-Empty-5x5-v0',
    'MiniGrid-DoorKey-5x5-v0',
    'MiniGrid-DistShift1-v0',
]

N_EPISODES   = 200
SEEDS        = [42, 43, 44, 45, 46]
ALGO         = 'double_dqn'
EPS_DECAY    = N_EPISODES * 40
HIDDEN_SIZES = (128, 128)
BATCH_SIZE   = 64
BUFFER_SIZE  = 10000
REPLAY_MIN   = 500
LR           = 1e-3
GAMMA        = 0.99
TAU          = 0.05
