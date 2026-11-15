import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

from sco2.cycle_adv import (recompression_adv, recompression_cascade,
                            splitflow_adv)
from sco2.heat_source_adv import EAF, GasStream, couple_heater, couple_cascade

K = 273.15
P_MAX = 30e6                 # Pa, maximum cycle pressure (design limit)
P1 = 7.7e6                   # Pa, compressor inlet pressure (model default)
PR_MAX = P_MAX / P1          # about 3.896
DT_REC = 10.0                # K, minimum temperature difference in HTR and LTR
PEN = 1.0                    # MW of penalty per K of violation

gas = GasStream(EAF["X"], EAF["p"], T0=298.15)
dTp, Tmin, mg = EAF["dT_pinch_heater"], EAF["T_stack_min"], EAF["m_gas"]
counter = [0]


def violation(res):
    p = res["pinch"]
    return (max(0.0, DT_REC - p["HTR"]["min_dT"])
            + max(0.0, DT_REC - p["LTR"]["min_dT"]))


def tick():
    counter[0] += 1
    if counter[0] % 50 == 0:
        print(f"    evaluations: {counter[0]}", flush=True)


# ---- Baseline: single heater, variables [x, pr] ---------------------------
def run_base(z):
    res = recompression_adv(x=z[0], pr=z[1])
    hs = couple_heater(res, gas, EAF["T_in"], mg, dT_pinch=dTp, T_stack_min=Tmin)
    return res, hs


def obj_base(z):
    tick()
    try:
        res, hs = run_base(z)
    except Exception:
        return 10.0
    return -hs["m_co2"] * res["w_net"] / 1e6 + PEN * violation(res)


# ---- Cascade with recompressor, variables [x, y, T_b/100 (C), pr] ----------
def run_cas(z):
    res = recompression_cascade(x=z[0], y=z[1], T_b=z[2] * 100 + K, pr=z[3],
                                check_pinch=True)
    hs = couple_cascade(res, gas, EAF["T_in"], mg, dTp, dTp, Tmin)
    return res, hs


def obj_cas(z):
    tick()
    try:
        res, hs = run_cas(z)
    except Exception:
        return 10.0
    return -hs["m_co2"] * res["w_net"] / 1e6 + PEN * violation(res)


# ---- Split-flow without recompressor, variables [y, T_b/100 (C), pr] -------
def run_sf(z):
    res = splitflow_adv(y=z[0], T_b=z[1] * 100 + K, pr=z[2], check_pinch=True)
    hs = couple_cascade(res, gas, EAF["T_in"], mg, dTp, dTp, Tmin)
    return res, hs


def obj_sf(z):
    tick()
    try:
        res, hs = run_sf(z)
    except Exception:
        return 10.0
    return -hs["m_co2"] * res["w_net"] / 1e6 + PEN * violation(res)


# ---- Multi-start Nelder-Mead ----------------------------------------------
def multistart(obj, starts, steps, lb, ub, label, maxfev=250):
    lb, ub = np.array(lb, float), np.array(ub, float)
    results = []
    for k, z0 in enumerate(starts, 1):
        z0 = np.array(z0, float)
        verts = [z0]
        for j, st in enumerate(steps):
            v = z0.copy()
            v[j] += st if z0[j] + st <= ub[j] else -st
            verts.append(v)
        counter[0] = 0
        o = minimize(obj, z0, method="Nelder-Mead", bounds=Bounds(lb, ub),
                     options={"initial_simplex": np.array(verts), "maxfev": maxfev,
                              "xatol": 1e-3, "fatol": 1e-5})
        print(f"  {label} start {k}: net power = {-o.fun:.4f} MW, "
              f"z = {np.round(o.x, 3).tolist()}, evaluations = {o.nfev}", flush=True)
        results.append((o.fun, o.x))
    best = min(results, key=lambda t: t[0])
    return best


print(f"Pressure limit: {P_MAX/1e6:.0f} MPa  ->  pr_max = {PR_MAX:.3f}\n", flush=True)

print("1) Baseline (x, pr)...", flush=True)
fb, zb = multistart(obj_base, [[0.30, 3.0], [0.20, 3.6]], [0.05, 0.3],
                    [0.0, 1.5], [0.6, PR_MAX], "baseline", maxfev=150)

print("\n2) Split-flow without recompressor (y, T_b, pr)...", flush=True)
fs, zs = multistart(obj_sf, [[0.29, 3.4, 3.8], [0.20, 3.0, 3.0], [0.35, 4.0, 3.5]],
                    [0.05, 0.5, 0.3], [0.02, 1.0, 1.5], [0.6, 5.0, PR_MAX],
                    "split-flow")

print("\n3) Check: cascade WITH recompressor, x free (x, y, T_b, pr)...", flush=True)
fc, zc = multistart(obj_cas, [[0.05, 0.29, 3.4, 3.6], [0.15, 0.20, 3.0, 3.0]],
                    [0.05, 0.05, 0.5, 0.3], [0.0, 0.02, 1.0, 1.5],
                    [0.6, 0.6, 5.0, PR_MAX], "cascade", maxfev=300)

for n, v, lo, hi in [("y", zs[0], 0.02, 0.6), ("T_b", zs[1], 1.0, 5.0)]:
    if abs(v - lo) < 1e-3 or abs(v - hi) < 1e-3:
        print(f"WARNING: {n} of the split-flow optimum is at its bound")
if abs(zs[2] - PR_MAX) < 1e-3:
    print("NOTE: pr is at the 30 MPa design limit (expected)")

# ---- Final results ---------------------------------------------------------
resb, hsb = run_base(zb)
ress, hss = run_sf(zs)
resc, hsc = run_cas(zc)
Wb, Ws, Wc = (hsb["m_co2"] * resb["w_net"], hss["m_co2"] * ress["w_net"],
              hsc["m_co2"] * resc["w_net"])
bal = ress["q_in"] + ress["q_b"] - ress["w_net"] - ress["q_out"]


def r(v, n=3):
    return round(float(v), n) if isinstance(v, (int, float, np.floating)) else v


def pin(res, name):
    return res["pinch"][name]["min_dT"]


rows = [
    ("Recompression fraction x", resb["x"], resc["x"], 0.0, "-"),
    ("Branch fraction y", "-", resc["y"], ress["y"], "-"),
    ("Branch heater outlet temperature", "-", resc["states"][11]["T"] - K,
     ress["states"][11]["T"] - K, "C"),
    ("Compressor pressure ratio pr", zb[1], zc[3], zs[2], "-"),
    ("Maximum cycle pressure", zb[1] * P1 / 1e6, zc[3] * P1 / 1e6, zs[2] * P1 / 1e6, "MPa"),
    ("CO2 mass flow (turbine)", hsb["m_co2"], hsc["m_co2"], hss["m_co2"], "kg/s"),
    ("Gas stack temperature", hsb["T_out"] - K, hsc["T_out"] - K, hss["T_out"] - K, "C"),
    ("Net power", Wb / 1e6, Wc / 1e6, Ws / 1e6, "MW"),
    ("Cycle thermal efficiency", resb["eta"] * 100, resc["eta"] * 100, ress["eta"] * 100, "%"),
    ("Heat recovery efficiency (net power / available heat)",
     Wb / hsb["Q_max"] * 100, Wc / hsc["Q_max"] * 100, Ws / hss["Q_max"] * 100, "%"),
    ("System exergy efficiency (net power / gas exergy)",
     Wb / hsb["ex_in"] * 100, Wc / hsc["ex_in"] * 100, Ws / hss["ex_in"] * 100, "%"),
    ("Exergy lost with the stack gas", hsb["ex_out"] / 1e6, hsc["ex_out"] / 1e6,
     hss["ex_out"] / 1e6, "MW"),
    ("HTR minimum temperature difference", pin(resb, "HTR"), pin(resc, "HTR"),
     pin(ress, "HTR"), "K"),
    ("LTR minimum temperature difference", pin(resb, "LTR"), pin(resc, "LTR"),
     pin(ress, "LTR"), "K"),
    ("Limiting constraint of the gas coupling", "-", hsc["binding"], hss["binding"], ""),
]
df = pd.DataFrame([{"Parameter": a, "Baseline (x, pr)": r(b),
                    "Cascade with recompressor": r(c),
                    "Split-flow (no recompressor)": r(d), "Unit": u}
                   for a, b, c, d, u in rows])
print()
print(df.to_string(index=False))
print(f"\nCycle energy balance check (split-flow): {bal:.4f} J/kg")
print(f"Split-flow gain over baseline: {(Ws / Wb - 1) * 100:+.1f} %")
print(f"Recompressor effect (cascade vs split-flow): {(Wc / Ws - 1) * 100:+.2f} % "
      f"(x = {resc['x']:.3f})")

Path("tables/adv").mkdir(parents=True, exist_ok=True)
df.to_csv("tables/adv/table8_splitflow.csv", index=False, encoding="utf-8-sig")

Path("data/results").mkdir(parents=True, exist_ok=True)
opt = {"model": "splitflow_adv", "y": float(zs[0]), "T_b": float(zs[1] * 100 + K),
       "pr": float(zs[2]), "p_max_limit_Pa": P_MAX, "dT_recuperator_min": DT_REC}
with open("data/results/splitflow_optimum.json", "w") as f:
    json.dump(opt, f, indent=2)
print("\nTable saved: tables/adv/table8_splitflow.csv")
print("Optimum saved: data/results/splitflow_optimum.json")