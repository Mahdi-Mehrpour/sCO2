import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

from sco2.cycle_adv import recompression_adv, recompression_cascade
from sco2.heat_source_adv import EAF, GasStream, couple_heater, couple_cascade

K = 273.15
DT_REC = 10.0     # K, minimum temperature difference enforced in HTR and LTR
PEN = 1.0         # MW of penalty per K of violation

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


# ---------------- Baseline: single heater, variables [x, pr] ---------------
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


# ---------------- Cascade: variables [x, y, T_b/100 (C), pr] ---------------
def run_cascade(z):
    res = recompression_cascade(x=z[0], y=z[1], T_b=z[2] * 100 + K, pr=z[3],
                                check_pinch=True)
    hs = couple_cascade(res, gas, EAF["T_in"], mg, dTp, dTp, Tmin)
    return res, hs


def obj_cas(z):
    tick()
    try:
        res, hs = run_cascade(z)
    except Exception:
        return 10.0
    return -hs["m_co2"] * res["w_net"] / 1e6 + PEN * violation(res)


# ---------------- Multi-start Nelder-Mead ----------------------------------
def multistart(obj, starts, steps, lb, ub, label, maxfev=300):
    lb, ub = np.array(lb, float), np.array(ub, float)
    results = []
    for k, z0 in enumerate(starts, 1):
        z0 = np.array(z0, float)
        verts = [z0]
        for j, st in enumerate(steps):
            v = z0.copy()
            v[j] += st if z0[j] + st <= ub[j] else -st   # stay inside the bounds
            verts.append(v)
        counter[0] = 0
        o = minimize(obj, z0, method="Nelder-Mead", bounds=Bounds(lb, ub),
                     options={"initial_simplex": np.array(verts), "maxfev": maxfev,
                              "xatol": 1e-3, "fatol": 1e-5})
        print(f"  {label} start {k}: net power = {-o.fun:.4f} MW, "
              f"z = {np.round(o.x, 3).tolist()}, evaluations = {o.nfev}", flush=True)
        results.append((o.fun, o.x))
    best = min(results, key=lambda t: t[0])
    return best, results


print("Optimising the baseline (x, pr)...", flush=True)
(fb, zb), _ = multistart(obj_base, [[0.30, 2.6], [0.20, 3.2]], [0.05, 0.3],
                         [0.0, 1.5], [0.6, 4.0], "baseline", maxfev=150)
print(f"  -> best baseline: {-fb:.4f} MW\n", flush=True)

print("Optimising the cascade (x, y, T_b, pr)...", flush=True)
starts = [[0.045, 0.214, 4.24, 2.6],    # previous optimum
          [0.25, 0.10, 3.0, 3.0],
          [0.10, 0.30, 3.5, 2.2]]
lb = [0.0, 0.02, 1.0, 1.5]
ub = [0.6, 0.6, 5.0, 4.0]
(fc, zc), _ = multistart(obj_cas, starts, [0.05, 0.05, 0.5, 0.3], lb, ub, "cascade")
print(f"  -> best cascade: {-fc:.4f} MW\n", flush=True)

names = ["x", "y", "T_b (C/100)", "pr"]
for n, v, lo, hi in zip(names, zc, lb, ub):
    if abs(v - lo) < 1e-3 or abs(v - hi) < 1e-3:
        print(f"WARNING: {n} is at its bound ({v:.3f}); widen the bound and rerun")

# ---------------- Final results with full detail ---------------------------
resb, hsb = run_base(zb)
res, hs = run_cascade(zc)
Wb, Wc = hsb["m_co2"] * resb["w_net"], hs["m_co2"] * res["w_net"]
st = res["states"]
bal = res["q_in"] + res["q_b"] - res["w_net"] - res["q_out"]


def r(v, n=3):
    return round(float(v), n) if isinstance(v, (int, float, np.floating)) else v


rows = [
    ("Recompression fraction x", resb["x"], res["x"], "-"),
    ("Branch fraction y", "-", res["y"], "-"),
    ("Branch heater outlet temperature", "-", st[11]["T"] - K, "C"),
    ("Compressor pressure ratio pr", zb[1], zc[3], "-"),
    ("Turbine pressure ratio", resb["pr_turbine"], res["pr_turbine"], "-"),
    ("CO2 mass flow (turbine)", hsb["m_co2"], hs["m_co2"], "kg/s"),
    ("Heat to the cycle, main heater", hsb["Q"] / 1e6, hs["Q1"] / 1e6, "MW"),
    ("Heat to the cycle, branch heater", "-", hs["Q2"] / 1e6, "MW"),
    ("Gas temperature between the heaters", "-", hs["T_gas_mid"] - K, "C"),
    ("Gas stack temperature", hsb["T_out"] - K, hs["T_out"] - K, "C"),
    ("Net power", Wb / 1e6, Wc / 1e6, "MW"),
    ("Cycle thermal efficiency", resb["eta"] * 100, res["eta"] * 100, "%"),
    ("Heat recovery efficiency (net power / available heat)",
     Wb / hsb["Q_max"] * 100, Wc / hs["Q_max"] * 100, "%"),
    ("System exergy efficiency (net power / gas exergy)",
     Wb / hsb["ex_in"] * 100, Wc / hs["ex_in"] * 100, "%"),
    ("Exergy lost with the stack gas", hsb["ex_out"] / 1e6, hs["ex_out"] / 1e6, "MW"),
    ("HTR minimum temperature difference", resb["pinch"]["HTR"]["min_dT"],
     res["pinch"]["HTR"]["min_dT"], "K"),
    ("LTR minimum temperature difference", resb["pinch"]["LTR"]["min_dT"],
     res["pinch"]["LTR"]["min_dT"], "K"),
    ("Limiting constraint of the gas coupling", "-", hs["binding"], ""),
    ("Cycle energy balance check", "-", bal, "J/kg"),
]
df = pd.DataFrame([{"Parameter": a, "Baseline (x, pr optimised)": r(b),
                    "Cascade (x, y, T_b, pr optimised)": r(c), "Unit": u}
                   for a, b, c, u in rows])
print(df.to_string(index=False))
print(f"\nNet power gain of the cascade: {(Wc / Wb - 1) * 100:+.1f} %")

Path("tables/adv").mkdir(parents=True, exist_ok=True)
df.to_csv("tables/adv/table7_cascade_optimised.csv", index=False, encoding="utf-8-sig")

# Save the optimum so that the next steps use exactly this design
Path("data/results").mkdir(parents=True, exist_ok=True)
opt = {"x": float(zc[0]), "y": float(zc[1]), "T_b": float(zc[2] * 100 + K),
       "pr": float(zc[3]), "dT_recuperator_min": DT_REC}
with open("data/results/cascade_optimum.json", "w") as f:
    json.dump(opt, f, indent=2)
print("\nTable saved: tables/adv/table7_cascade_optimised.csv")
print("Optimum saved: data/results/cascade_optimum.json")
