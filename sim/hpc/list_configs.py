#!/usr/bin/env python3
"""Print the (size, layout, spawn, episodes, ...) configs for the LSF array.

The 1-based line number is the $LSB_JOBINDEX. Use the total count as the array
range in run_sweep.lsf  (#BSUB -J minigrid_cp[1-N]).

    python sim/hpc/list_configs.py          # full table + count
"""

SIZES = [8, 12, 16, 24, 32, 48]


def layouts(s):
    base = ["empty", "narrow", "chicane", "islands"]
    return base + (["swirl"] if s >= 12 else [])   # swirl needs >=11


def configs():
    out = []
    for s in SIZES:
        for lay in layouts(s):
            for sp in ["random", "fixed"]:
                E = max(2000, round(25 * s * s / 1000) * 1000)
                out.append((s, lay, sp, E, 12 * s, max(100, E // 50), max(50, E // 200)))
    return out


if __name__ == "__main__":
    cfgs = configs()
    print(f"{'idx':>4}  {'size':>4} {'layout':<10} {'spawn':<7} {'episodes':>8} "
          f"{'maxstep':>7} {'eval':>5} {'snap':>5}")
    for i, (s, lay, sp, E, ms, ev, sn) in enumerate(cfgs, 1):
        print(f"{i:>4}  {s:>4} {lay:<10} {sp:<7} {E:>8} {ms:>7} {ev:>5} {sn:>5}")
    print(f"\nTotal tasks: {len(cfgs)}   ->   #BSUB -J minigrid_cp[1-{len(cfgs)}]")
