"""Timing of one closure-residual evaluation of the integrated cycle, component by component (about 10-20 s).
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/Cycle/profile_cycle.py" """
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import design_point as dp
from cycle_offdesign import OffDesignCycle

res, hs, x = dp.load_design()
cyc = OffDesignCycle(dp, res, hs, x)
c = {"T_gas": cyc.T_gas_des, "m_gas": cyc.m_gas_des, "T_wb": cyc.T_wb_des, "N_mc": 1.0, "N_rc": 1.0, "r_cool": 1.0, "bypass": 0.0, "p1": cyc.p1_des}
cyc._eval(cyc.u0, c)                                     # warm-up (first calls build caches)
cyc.timers.clear()
n = 3
t0 = time.perf_counter()
for _ in range(n):
    cyc._eval(cyc.u0, c)
tot = (time.perf_counter() - t0) / n
print(f"One residual evaluation: {tot:.2f} s (mean of {n})")
for k, v in sorted(cyc.timers.items(), key=lambda kv: -kv[1]):
    print(f"  {k:16s} {v / n:6.2f} s  {v / n / tot * 100:5.1f} %")
