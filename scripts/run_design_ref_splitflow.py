"""Step 5a: re-optimise the split-flow cycle for the MEAN gas of the measured
reference cycle (Dal Magro 2017, Fig. 3.40).

OBJECTIVE: net ELECTRICAL power of the plant (after fans, pumps, losses), not the
cycle shaft power used for the 800 C design. The previous optimum file is kept
as splitflow_optimum_ref_cycleobj.json.

Design gas: mean temperature of the digitised cycle and the mean flow of the
figure (1.34e5 Nm3/h), converted to kg/s with an ASSUMED normal density.
Same variables, bounds, penalty and pressure limit as run_splitflow_opt_adv.py.
Result is saved to data/results/splitflow_optimum_ref.json (the 800 C optimum
in splitflow_optimum.json is NOT touched).

Run: python scripts/run_design_ref_splitflow.py"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, Bounds

import shutil
import pandas as pd
from sco2.cycle_adv import splitflow_adv
from sco2.heat_source_adv import EAF, GasStream, couple_cascade
from sco2.plant_splitflow import evaluate_plant
from sco2.profile_eaf import make_profile_ref

K = 273.15
P_MAX, P1 = 25e6, 7.7e6
PR_MAX = P_MAX / P1
DT_REC = 10.0
PEN = 1.0
FLOW_NM3H = 1.34e5          # Nm3/h, mean flow read from the figure (dashed line)
RHO_N = 1.3                 # kg/Nm3, ASSUMED normal density of the off-gas

T_DES_C = float(np.mean(make_profile_ref()["T_C"]))
M_DES = FLOW_NM3H * RHO_N / 3600.0
T_DES = T_DES_C + K
print(f"Design gas: {T_DES_C:.0f} C, {M_DES:.1f} kg/s "
      f"({FLOW_NM3H:.3g} Nm3/h x {RHO_N} kg/Nm3)\n", flush=True)

gas = GasStream(EAF["X"], EAF["p"], T0=298.15)
dTp, Tmin = EAF["dT_pinch_heater"], EAF["T_stack_min"]


def obj(z):
    """Minus the plant net electrical power (MW) plus the pinch penalty."""
    try:
        Pl = evaluate_plant(z[0], z[1] * 100 + K, z[2], T_in=T_DES, m_gas=M_DES)
    except Exception:
        return 10.0
    p = Pl["res"]["pinch"]
    viol = (max(0.0, DT_REC - p["HTR"]["min_dT"])
            + max(0.0, DT_REC - p["LTR"]["min_dT"]))
    return -Pl["net_el"] / 1e6 + PEN * viol


with open("data/results/splitflow_optimum.json") as f:
    old = json.load(f)
starts = [[old["y"], (old["T_b"] - K) / 100, old["pr"]],
          [0.35, 3.8, 3.2], [0.42, 3.4, 3.1]]
lb, ub = np.array([0.02, 1.0, 1.5]), np.array([0.6, 5.0, PR_MAX])
steps = [0.05, 0.5, 0.3]
best = None
for k, z0 in enumerate(starts, 1):
    z0 = np.clip(np.array(z0, float), lb, ub)
    verts = [z0]
    for j, st in enumerate(steps):
        v = z0.copy()
        v[j] += st if z0[j] + st <= ub[j] else -st
        verts.append(v)
    o = minimize(obj, z0, method="Nelder-Mead", bounds=Bounds(lb, ub),
                 options={"initial_simplex": np.array(verts), "maxfev": 250,
                          "xatol": 1e-3, "fatol": 1e-5})
    print(f"  start {k}: net power {-o.fun:.4f} MW, z = {np.round(o.x, 3).tolist()}, "
          f"evaluations {o.nfev}", flush=True)
    if best is None or o.fun < best.fun:
        best = o
z = best.x
for n, v, lo, hi in [("y", z[0], 0.02, 0.6), ("T_b", z[1], 1.0, 5.0)]:
    if abs(v - lo) < 1e-3 or abs(v - hi) < 1e-3:
        print(f"WARNING: {n} is at its bound")
if abs(z[2] - PR_MAX) < 1e-3:
    print("NOTE: pr is at the pressure limit (expected)")

y, T_b, pr = float(z[0]), float(z[1] * 100 + K), float(z[2])
Pl = evaluate_plant(y, T_b, pr, T_in=T_DES, m_gas=M_DES)
hs = Pl["hs"]
print(f"\nOptimum: y = {y:.4f}, T_b = {T_b - K:.1f} C, pr = {pr:.4f} "
      f"(p2 = {pr * P1 / 1e6:.2f} MPa)")
print(f"CO2 flow {hs['m_co2']:.2f} kg/s | stack {hs['T_out'] - K:.0f} C | binding: {hs['binding']}")
print(f"Net electrical {Pl['net_el'] / 1e6:.3f} MW | cycle efficiency "
      f"{Pl['res']['eta'] * 100:.1f} % | heat available {hs['Q_max'] / 1e6:.2f} MW | "
      f"heat recovery {Pl['net_el'] / hs['Q_max'] * 100:.1f} % | exergy efficiency "
      f"{Pl['net_el'] / hs['ex_in'] * 100:.1f} %")
print(f"Reference: the 800 C / 25 kg/s design gives 6.197 MW")

Path("data/results").mkdir(parents=True, exist_ok=True)
bak = Path("data/results/splitflow_optimum_ref_cycleobj.json")
if not bak.exists():
    shutil.copy("data/results/splitflow_optimum_ref.json", bak)
    print(f"Previous optimum (cycle-power objective) kept as {bak}")
with open("data/results/splitflow_optimum_ref.json", "w") as f:
    json.dump({"model": "splitflow_adv", "y": y, "T_b": T_b, "pr": pr,
               "p_max_limit_Pa": P_MAX, "dT_recuperator_min": DT_REC,
               "T_in_K": T_DES, "m_gas": M_DES, "flow_Nm3_h": FLOW_NM3H,
               "rho_N": RHO_N}, f, indent=2)
print("\nSaved: data/results/splitflow_optimum_ref.json")

# ---- tables for the report -----------------------------------------------
res = Pl["res"]
st = res["states"]
Path("tables").mkdir(exist_ok=True)
states = pd.DataFrame([{"State": k, "T (C)": round(st[k]["T"] - K, 1),
                        "p (MPa)": round(st[k]["p"] / 1e6, 3),
                        "h (kJ/kg)": round(st[k]["h"] / 1e3, 1)} for k in sorted(st)])
states.to_csv("tables/31_design_ref_states.csv", index=False, encoding="utf-8-sig")
shaft = hs["m_co2"] * res["w_net"]
perf = pd.DataFrame([
    ("Gas inlet temperature", T_DES_C, "C"), ("Gas mass flow", M_DES, "kg/s"),
    ("Branch fraction y", y, "-"), ("Branch heater outlet temperature", T_b - K, "C"),
    ("Compressor pressure ratio", pr, "-"), ("Maximum cycle pressure", pr * P1 / 1e6, "MPa"),
    ("CO2 mass flow (turbine)", hs["m_co2"], "kg/s"),
    ("Gas stack temperature", hs["T_out"] - K, "C"),
    ("Cycle shaft power", shaft / 1e6, "MW"),
    ("Net electrical power", Pl["net_el"] / 1e6, "MW"),
    ("Cycle thermal efficiency", res["eta"] * 100, "%"),
    ("Heat available above the stack limit", hs["Q_max"] / 1e6, "MW"),
    ("Heat recovery (net electrical / heat available)", Pl["net_el"] / hs["Q_max"] * 100, "%"),
    ("System exergy efficiency (net electrical / gas exergy)", Pl["net_el"] / hs["ex_in"] * 100, "%"),
    ("Exergy lost with the stack gas", hs["ex_out"] / 1e6, "MW"),
    ("HTR minimum temperature difference", res["pinch"]["HTR"]["min_dT"], "K"),
    ("LTR minimum temperature difference", res["pinch"]["LTR"]["min_dT"], "K"),
    ("Limiting constraint of the gas coupling", hs["binding"], "")],
    columns=["Parameter", "Value", "Unit"])
perf["Value"] = perf["Value"].map(lambda v: round(float(v), 3) if not isinstance(v, str) else v)
perf.to_csv("tables/32_design_ref_performance.csv", index=False, encoding="utf-8-sig")
print()
print(perf.to_string(index=False))
print("\nSaved: tables/31_design_ref_states.csv, tables/32_design_ref_performance.csv")
