"""Quasi-steady sensitivity of the split-flow cycle to the EAF gas condition.

Case A: cycle re-optimised (y, T_b, pr) at every gas temperature (upper bound).
Case B: cycle settings fixed at the 800 C design; only the CO2 flow adapts.
Also: recompressor check (x free) at selected gas temperatures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

from sco2.cycle_adv import splitflow_adv, recompression_cascade
from sco2.heat_source_adv import EAF, GasStream, couple_cascade

K = 273.15
P_MAX = 30e6                 # Pa, design pressure limit (temporary)
P1 = 7.7e6                   # Pa, compressor inlet pressure
PR_MAX = P_MAX / P1
DT_REC = 10.0                # K, minimum temperature difference in HTR and LTR
PEN = 1.0                    # MW of penalty per K of violation
T7 = 823.15                  # K, turbine inlet temperature (fixed, 550 C)

# ---------------- Settings you can change ----------------------------------
T_GAS_C = [600, 700, 800, 900, 1000, 1100]   # gas inlet temperatures (C)
X_CHECK_C = [600, 800, 1100]                 # temperatures for the x check
FLOW_FACTORS = [0.5, 1.0, 1.5]               # gas flow factors (linearity check)
# ---------------------------------------------------------------------------

gas = GasStream(EAF["X"], EAF["p"], T0=298.15)
dTp, Tmin, mg0 = EAF["dT_pinch_heater"], EAF["T_stack_min"], EAF["m_gas"]

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
design_z = [opt["y"], (opt["T_b"] - K) / 100, opt["pr"]]
counter = [0]


def violation(res):
    p = res["pinch"]
    return (max(0.0, DT_REC - p["HTR"]["min_dT"])
            + max(0.0, DT_REC - p["LTR"]["min_dT"]))


def tick():
    counter[0] += 1
    if counter[0] % 100 == 0:
        print(f"    evaluations: {counter[0]}", flush=True)


# ---- split-flow, variables [y, T_b/100 (C), pr] ----------------------------
def run_sf(z, T_in):
    res = splitflow_adv(y=z[0], T_b=z[1] * 100 + K, pr=z[2], T7=T7,
                        check_pinch=True)
    hs = couple_cascade(res, gas, T_in, mg0, dTp, dTp, Tmin)
    return res, hs


def obj_sf(z, T_in):
    tick()
    try:
        res, hs = run_sf(z, T_in)
    except Exception:
        return 10.0
    return -hs["m_co2"] * res["w_net"] / 1e6 + PEN * violation(res)


# ---- cascade with recompressor, variables [x, y, T_b/100, pr] --------------
def run_cas(z, T_in):
    res = recompression_cascade(x=z[0], y=z[1], T_b=z[2] * 100 + K, pr=z[3],
                                T7=T7, check_pinch=True)
    hs = couple_cascade(res, gas, T_in, mg0, dTp, dTp, Tmin)
    return res, hs


def obj_cas(z, T_in):
    tick()
    try:
        res, hs = run_cas(z, T_in)
    except Exception:
        return 10.0
    return -hs["m_co2"] * res["w_net"] / 1e6 + PEN * violation(res)


def multistart(obj, starts, steps, lb, ub, maxfev=250):
    lb, ub = np.array(lb, float), np.array(ub, float)
    out = []
    for z0 in starts:
        z0 = np.clip(np.array(z0, float), lb, ub)
        verts = [z0]
        for j, st in enumerate(steps):
            v = z0.copy()
            v[j] += st if z0[j] + st <= ub[j] else -st
            verts.append(v)
        o = minimize(obj, z0, method="Nelder-Mead", bounds=Bounds(lb, ub),
                     options={"initial_simplex": np.array(verts),
                              "maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-5})
        out.append((o.fun, o.x, o.nfev))
    return out


LB = [0.02, 1.0, 1.5]
UB = [0.6, 5.5, PR_MAX]

# ---------------- Case B reference: fixed design ----------------------------
res_d = splitflow_adv(y=design_z[0], T_b=design_z[1] * 100 + K, pr=design_z[2],
                      T7=T7, check_pinch=True)

# ---------------- Gas temperature sweep -------------------------------------
rows, best_z, prev = [], {}, None
print(f"Pressure limit {P_MAX / 1e6:.0f} MPa (pr_max = {PR_MAX:.3f}), "
      f"turbine inlet {T7 - K:.0f} C\n", flush=True)
for Tc in T_GAS_C:
    T_in = Tc + K
    print(f"Gas inlet {Tc} C: case A (re-optimised)...", flush=True)
    starts = [design_z, prev if prev is not None else [0.20, 3.0, 3.5]]
    outs = multistart(lambda z: obj_sf(z, T_in), starts, [0.05, 0.5, 0.3], LB, UB)
    for k, (fv, zv, nf) in enumerate(outs, 1):
        print(f"  start {k}: net power = {-fv:.4f} MW, "
              f"z = {np.round(zv, 3).tolist()}, evaluations = {nf}", flush=True)
    fa, za, _ = min(outs, key=lambda t: t[0])
    prev = list(za)
    best_z[Tc] = za
    resA, hsA = run_sf(za, T_in)
    hsB = couple_cascade(res_d, gas, T_in, mg0, dTp, dTp, Tmin)
    WA, WB = hsA["m_co2"] * resA["w_net"], hsB["m_co2"] * res_d["w_net"]
    flags = []
    if abs(za[2] - PR_MAX) < 1e-3:
        flags.append("pr at limit")
    if abs(za[1] - UB[1]) < 1e-3 or abs(za[0] - UB[0]) < 1e-3:
        flags.append("y or T_b at upper bound")
    rows.append({
        "Gas inlet temperature (C)": Tc,
        "Net power A, re-optimised (MW)": WA / 1e6,
        "Net power B, fixed design (MW)": WB / 1e6,
        "Power lost by fixed design (%)": (1 - WB / WA) * 100 if WA > 0 else 0.0,
        "Optimal y (A)": za[0],
        "Optimal T_b (A, C)": za[1] * 100,
        "Optimal pr (A)": za[2],
        "CO2 flow A (kg/s)": hsA["m_co2"],
        "CO2 flow B (kg/s)": hsB["m_co2"],
        "Stack temperature A (C)": hsA["T_out"] - K,
        "Stack temperature B (C)": hsB["T_out"] - K,
        "Heat used A (% of available)": hsA["Q"] / hsA["Q_max"] * 100,
        "Heat recovery efficiency A (%)": WA / hsA["Q_max"] * 100,
        "System exergy efficiency A (%)": WA / hsA["ex_in"] * 100,
        "System exergy efficiency B (%)": WB / hsB["ex_in"] * 100,
        "HTR min dT A (K)": resA["pinch"]["HTR"]["min_dT"],
        "LTR min dT A (K)": resA["pinch"]["LTR"]["min_dT"],
        "Limiting constraint A": hsA["binding"],
        "Limiting constraint B": hsB["binding"],
        "Flags": ", ".join(flags),
    })

df = pd.DataFrame(rows)
num = df.select_dtypes("number").columns
df[num] = df[num].round(3)
print()
print(df.T.to_string(header=False))

# ---------------- Gas flow linearity check ----------------------------------
print("\nGas flow check (fixed design, 800 C):")
base = None
for fct in FLOW_FACTORS:
    h = couple_cascade(res_d, gas, 800 + K, mg0 * fct, dTp, dTp, Tmin)
    W = h["m_co2"] * res_d["w_net"] / 1e6
    base = W if fct == 1.0 else base
    print(f"  gas flow x{fct:.2f}: net power = {W:.3f} MW")
print("  (net power is proportional to gas flow in this model)")

# ---------------- Recompressor check (x free) -------------------------------
print("\nRecompressor check (cascade with x free)...", flush=True)
xrows = []
for Tc in X_CHECK_C:
    if Tc not in best_z:
        continue
    T_in = Tc + K
    za = best_z[Tc]
    starts = [[0.05, za[0], za[1], za[2]], [0.20, za[0], za[1], za[2]]]
    outs = multistart(lambda z: obj_cas(z, T_in), starts, [0.05, 0.05, 0.5, 0.3],
                      [0.0] + LB, [0.6] + UB, maxfev=250)
    fc, zc, _ = min(outs, key=lambda t: t[0])
    resS, hsS = run_sf(za, T_in)
    WS = hsS["m_co2"] * resS["w_net"]
    WC = -fc * 1e6          # penalty-free at the optimum if pinch is satisfied
    xrows.append({"Gas inlet temperature (C)": Tc,
                  "Net power split-flow (MW)": WS / 1e6,
                  "Net power cascade, x free (MW)": WC / 1e6,
                  "Optimal x": zc[0],
                  "Gain from recompressor (%)": (WC / WS - 1) * 100})
    print(f"  {Tc} C: split-flow {WS / 1e6:.4f} MW, cascade {WC / 1e6:.4f} MW, "
          f"x = {zc[0]:.3f}", flush=True)
dfx = pd.DataFrame(xrows)
if len(dfx):
    dfx = dfx.round(4)
    print()
    print(dfx.to_string(index=False))

# ---------------- Save ------------------------------------------------------
T_OUT, F_OUT = Path("tables/sensitivity"), Path("figures/sensitivity")
T_OUT.mkdir(parents=True, exist_ok=True)
F_OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(T_OUT / "table9_gas_temperature.csv", index=False, encoding="utf-8-sig")
dfx.to_csv(T_OUT / "table10_recompressor_check.csv", index=False, encoding="utf-8-sig")

T = df["Gas inlet temperature (C)"]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax = axes[0, 0]
ax.plot(T, df["Net power A, re-optimised (MW)"], "o-", label="A: re-optimised")
ax.plot(T, df["Net power B, fixed design (MW)"], "s--", label="B: fixed design")
ax.set_ylabel("Net power (MW)")
ax.legend()
ax = axes[0, 1]
ax.plot(T, df["Stack temperature A (C)"], "o-", label="A")
ax.plot(T, df["Stack temperature B (C)"], "s--", label="B")
ax.axhline(Tmin - K, color="gray", ls=":", label="Minimum stack temperature")
ax.set_ylabel("Gas stack temperature (°C)")
ax.legend()
axes[1, 0].plot(T, df["Optimal y (A)"], "o-")
axes[1, 0].axhline(design_z[0], color="gray", ls="--", label="800 °C design")
axes[1, 0].set_ylabel("Optimal branch fraction y (case A)")
axes[1, 0].legend()
axes[1, 1].plot(T, df["Optimal T_b (A, C)"], "o-")
axes[1, 1].axhline(design_z[1] * 100, color="gray", ls="--", label="800 °C design")
axes[1, 1].set_ylabel("Optimal branch outlet temperature (°C, case A)")
axes[1, 1].legend()
for ax in axes.flat:
    ax.set_xlabel("EAF gas inlet temperature (°C)")
    ax.grid(True)
fig.tight_layout()
fig.savefig(F_OUT / "gas_temperature_sensitivity.png", dpi=150)
print(f"\nSaved: {T_OUT}/ and {F_OUT}/gas_temperature_sensitivity.png")