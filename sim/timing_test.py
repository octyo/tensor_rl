"""Timing + correctness test for all tabular tensor agents via train.py CLI."""
import subprocess, sys, time

ALGOS = ['tabular_q', 'tabular_q_cp', 'tabular_q_tucker', 'tabular_q_tt',
         'sarsa', 'sarsa_cp', 'sarsa_tucker', 'sarsa_tt']

BASE = ['conda', 'run', '-n', 'tensor', 'python', 'train.py',
        '--env', 'FrozenLake-v1',
        '--episodes', '1000',
        '--max_steps', '200',
        '--rank', '4']

print(f"{'Algo':20s}  {'Time':>6s}  Status")
print('-' * 40)
for algo in ALGOS:
    cmd = BASE + ['--algo', algo]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    ok = '✓' if result.returncode == 0 else '✗ ' + result.stderr[-200:]
    print(f"  {algo:20s}  {elapsed:5.1f}s  {ok}")
