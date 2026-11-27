"""Check of the integrated off-design cycle (step 1: assembly and design-point closure).

A  Residuals at the design unknowns: every component is run at its design inputs; the seven closure residuals show how well the
   component models reproduce the design cycle (they are small but not zero: the models are not the cycle model).
B  Solve at the design operating point (design gas, wet bulb, speeds, cooler flow, p1): the states, flows, powers and heat input
   against the design values.
C  (optional, --extra) three off-design points started from the design solution: warm day (wet bulb 25 C), gas flow 90 %,
   gas temperature -50 K. They show convergence, T1, speeds needed and the map flags, not yet an optimised control.
Output: tables/cycle_design_check.csv.
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/Cycle/check_cycle.py" [--extra]
Run all component shares scripts first (heater, recuperators, cooler). It takes several minutes: each residual evaluation runs
every component model once."""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import design_point as dp
from cycle_offdesign import OffDesignCycle, K

ap = argparse.ArgumentParser()
ap.add_argument("--extra", action="store_true", help="also run three off-design points")
args = ap.parse_args()

t0 = time.time()
res, hs, x = dp.load_design()
cyc = OffDesignCycle(dp, res, hs, x)
st = res["states"]
m0 = hs["m_co2"]
print(f"Cycle assembled in {time.time() - t0:.0f} s. Design: m {m0:.3f} kg/s, x {x:.4f}, net shaft power {m0 * res['w_net'] / 1e6:.3f} MW, "
      f"heater duty {m0 * res['q_in'] / 1e6:.3f} MW")

print("\nA. Closure residuals at the design unknowns (scaled; 1 = 0.1 % of m, 10 kPa, 10 kJ/kg, 1 kPa, 10 kJ/kg, 10 kJ/kg, 1 kPa)")
c = {"T_gas": cyc.T_gas_des, "m_gas": cyc.m_gas_des, "T_wb": cyc.T_wb_des, "N_mc": 1.0, "N_rc": 1.0, "r_cool": 1.0, "bypass": 0.0, "p1": cyc.p1_des}
r0, i0 = cyc._eval(cyc.u0, c)
names = ["turbine flow - m", "p4 - p3", "h8 turbine", "h9 HTR hot out", "p9 HTR hot out", "h10 LTR hot out", "p10 LTR - cooler"]
for n_, v in zip(names, r0):
    print(f"   {n_:20s} {v:+.3e}")

print("\nB. Solve at the design operating point")
sol = cyc.solve()
print(f"   converged {sol['converged']}, max residual {sol['max_residual']:.1e}, {sol['n_eval']} evaluations, {sol['time']:.0f} s")
rows = []
print(f"   {'state':>5} {'T (C)':>9} {'design':>9} {'dT (K)':>8} | {'p (MPa)':>9} {'design':>9}")
for k in range(1, 11):
    a, d = sol["st"][k], st[k]
    Ta = a.get("T", np.nan)
    if np.isnan(Ta):
        Ta = dp.PropsSI("T", "H", a["h"], "P", a["p"], "CO2")
    print(f"   {k:5d} {Ta - K:9.2f} {d['T'] - K:9.2f} {Ta - d['T']:+8.2f} | {a['p'] / 1e6:9.4f} {d['p'] / 1e6:9.4f}")
    rows.append({"state": k, "T model (C)": Ta - K, "T design (C)": d["T"] - K, "p model (MPa)": a["p"] / 1e6, "p design (MPa)": d["p"] / 1e6})
print(f"\n   m {sol['m']:.3f} kg/s (design {m0:.3f}), x {sol['x']:.4f} ({x:.4f})")
print(f"   turbine {sol['W_t'] / 1e6:.3f} MW (design {m0 * res['w_t'] / 1e6:.3f}), MC {sol['W_mc'] / 1e6:.3f} ({m0 * (1 - x) * res['w_mc'] / 1e6:.3f}), "
      f"RC {sol['W_rc'] / 1e6:.3f} ({m0 * x * res['w_rc'] / 1e6:.3f})")
print(f"   net shaft power {sol['W_net'] / 1e6:.3f} MW (design {m0 * res['w_net'] / 1e6:.3f}), heater duty {sol['Q_heater'] / 1e6:.3f} MW ({m0 * res['q_in'] / 1e6:.3f}), "
      f"gas-side duty {sol['Q_gas'] / 1e6:.3f} MW, cycle efficiency {sol['eta_cycle'] * 100:.2f} % ({res['eta'] * 100:.2f})")
print(f"   T7 {sol['T7'] - K:.2f} C ({st[7]['T'] - K:.2f}), stack {sol['T_stack'] - K:.2f} C ({hs['T_out'] - K:.2f}), "
      f"cooler auxiliaries {sol['W_aux'] / 1e3:.1f} kW, net after auxiliaries {sol['W_net_after_aux'] / 1e6:.3f} MW")
print(f"   flags: {sol['flags']}")
pd.DataFrame(rows).round(5).to_csv(dp.TABLES / "cycle_design_check.csv", index=False, encoding="utf-8-sig")

if args.extra:
    print("\nC. Off-design points (started from the design solution; speeds at design, cooler flow at design, no bypass)")
    cases = [("warm day, wet bulb 25 C", dict(T_wb=25.0)), ("gas flow 90 %", dict(m_gas=0.9 * cyc.m_gas_des)),
             ("gas temperature -50 K", dict(T_gas=cyc.T_gas_des - 50.0))]
    for label, kw in cases:
        s = cyc.solve(guess=sol["u"], **kw)
        print(f"  {label}: converged {s['converged']} (res {s['max_residual']:.1e}, {s['n_eval']} evals, {s['time']:.0f} s)")
        print(f"     m {s['m']:.2f} kg/s, x {s['x']:.3f}, p2 {s['p2'] / 1e6:.2f} MPa, T1 {s['T1'] - K:.2f} C, T7 {s['T7'] - K:.1f} C, "
              f"net {s['W_net'] / 1e6:.3f} MW, eta {s['eta_cycle'] * 100:.2f} %, stack {s['T_stack'] - K:.1f} C")
        print(f"     flags {s['flags']}")
print(f"\nSaved: {dp.TABLES / 'cycle_design_check.csv'}; total time {time.time() - t0:.0f} s")
