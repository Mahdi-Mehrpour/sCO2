"""Quasi-steady sensitivity of the plant to the EAF gas temperature.
Case A: y and T_b re-optimised (maximum net electrical power) at each gas
        temperature; pressure ratio at the cap; plant re-sized for the NPV.
Case B: saved design kept; only the CO2 flow adapts to the gas.
Gas flow is not varied: net power is proportional to it in this model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

from sco2.plant_splitflow import evaluate_plant
from sco2.economics_splitflow import build_inputs, fast_economics
from sco2.settings_splitflow import ECON_BASE

K = 273.15
DT_REC = 10.0                 # K, minimum temperature difference in HTR and LTR
PEN = 1.0                     # MW of penalty per K of violation
T_GAS_C = [600, 700, 800, 900, 1000, 1100]
LB, UB = [0.02, 1.0], [0.60, 5.5]          # y, T_b/100 (C)

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
PR = opt["pr"]
z_design = [opt["y"], (opt["T_b"] - K) / 100]
counter = [0]


def run(y, tb100, Tc):
    return evaluate_plant(y, tb100 * 100 + K, PR, T_in=Tc + K)


def violation(P):
    p = P["res"]["pinch"]
    return (max(0.0, DT_REC - p["HTR"]["min_dT"])
            + max(0.0, DT_REC - p["LTR"]["min_dT"]))


def objective(z, Tc):
    counter[0] += 1
    if counter[0] % 100 == 0:
        print(f"    evaluations: {counter[0]}", flush=True)
    try:
        P = run(z[0], z[1], Tc)
    except Exception:
        return 10.0
    return -P["net_el"] / 1e6 + PEN * violation(P)


def optimise(Tc, starts):
    lb, ub = np.array(LB, float), np.array(UB, float)
    out = []
    for z0 in starts:
        z0 = np.clip(np.array(z0, float), lb, ub)
        verts = [z0]
        for j, s in enumerate([0.05, 0.5]):
            v = z0.copy()
            v[j] += s if z0[j] + s <= ub[j] else -s
            verts.append(v)
        o = minimize(objective, z0, args=(Tc,), method="Nelder-Mead",
                     bounds=Bounds(lb, ub),
                     options={"initial_simplex": np.array(verts), "maxfev": 150,
                              "xatol": 1e-3, "fatol": 1e-5})
        print(f"  start: net electrical power = {-o.fun:.4f} MW, "
              f"z = {np.round(o.x, 3).tolist()}, evaluations = {o.nfev}", flush=True)
        out.append((o.fun, o.x))
    return min(out, key=lambda t: t[0])[1]


rows, prev = [], None
for Tc in T_GAS_C:
    print(f"Gas inlet {Tc} C: case A (re-optimised)...", flush=True)
    z = optimise(Tc, [z_design, prev if prev is not None else [0.20, 3.0]])
    prev = list(z)
    PA = run(z[0], z[1], Tc)
    PB = run(z_design[0], z_design[1], Tc)
    hsA, hsB = PA["hs"], PB["hs"]
    try:
        ci, e = build_inputs(PA, ECON_BASE)
        eco = fast_economics(ci, e)
        capital, cap_kw, npv = eco["TCI"] / 1e6, eco["TCI"] / (eco["W_net"] * 1e3), eco["NPV"] / 1e6
    except Exception as exc:
        capital = cap_kw = npv = float("nan")
        print("  economics failed:", exc)
    flags = []
    if PA["par"]["W_turbine"] / 1e6 < 8.0:
        flags.append("turbine below the 8 MW range of the cost correlation")
    if PA["par"]["W_compressor"] / 1e6 < 1.5:
        flags.append("compressor below the 1.5 MW range")
    if abs(z[0] - LB[0]) < 1e-3 or abs(z[0] - UB[0]) < 1e-3 or abs(z[1] - UB[1]) < 1e-3:
        flags.append("y or T_b at a bound")
    rows.append({
        "Gas inlet temperature (C)": Tc,
        "Net electrical power A, re-optimised (MW)": PA["net_el"] / 1e6,
        "Net electrical power B, fixed design (MW)": PB["net_el"] / 1e6,
        "Power lost by the fixed design (%)": (1 - PB["net_el"] / PA["net_el"]) * 100,
        "Optimal y (A)": z[0], "Optimal T_b (A, C)": z[1] * 100,
        "CO2 flow A (kg/s)": hsA["m_co2"], "CO2 flow B (kg/s)": hsB["m_co2"],
        "Stack temperature A (C)": hsA["T_out"] - K,
        "Stack temperature B (C)": hsB["T_out"] - K,
        "Heat used A (% of available)": hsA["Q"] / hsA["Q_max"] * 100,
        "Net electrical efficiency A (%)": PA["eta_el"] * 100,
        "HTR min dT A (K)": PA["res"]["pinch"]["HTR"]["min_dT"],
        "LTR min dT A (K)": PA["res"]["pinch"]["LTR"]["min_dT"],
        "HTR min dT B (K)": PB["res"]["pinch"]["HTR"]["min_dT"],
        "LTR min dT B (K)": PB["res"]["pinch"]["LTR"]["min_dT"],
        "Capital A, plant re-sized (M$)": capital,
        "Capital A (USD/kWe)": cap_kw,
        "NPV A, plant re-sized (M$)": npv,
        "Limiting constraint A": hsA["binding"],
        "Limiting constraint B": hsB["binding"],
        "Flags": "; ".join(flags)})

df = pd.DataFrame(rows)
num = df.select_dtypes("number").columns
df[num] = df[num].round(3)
Path("tables").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
df.to_csv("tables/12_gas_temperature_sensitivity.csv", index=False, encoding="utf-8-sig")
print()
print(df.T.to_string(header=False))

T = df["Gas inlet temperature (C)"]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax = axes[0, 0]
ax.plot(T, df["Net electrical power A, re-optimised (MW)"], "o-", label="A: re-optimised")
ax.plot(T, df["Net electrical power B, fixed design (MW)"], "s--", label="B: fixed design")
ax.set_ylabel("Net electrical power (MW)")
ax.legend()
ax = axes[0, 1]
ax.plot(T, df["Stack temperature A (C)"], "o-", label="A")
ax.plot(T, df["Stack temperature B (C)"], "s--", label="B")
ax.axhline(150, color="gray", ls=":", label="Minimum stack temperature")
ax.set_ylabel("Gas stack temperature (°C)")
ax.legend()
ax = axes[1, 0]
ax.bar(T, df["NPV A, plant re-sized (M$)"], width=60)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("NPV of the re-sized plant (M$)")
ax = axes[1, 1]
ax.plot(T, df["Optimal y (A)"], "o-", color="tab:blue")
ax.set_ylabel("Optimal branch fraction y (case A)", color="tab:blue")
ax2 = ax.twinx()
ax2.plot(T, df["Optimal T_b (A, C)"], "s--", color="tab:red")
ax2.set_ylabel("Optimal branch outlet temperature (°C)", color="tab:red")
for ax in axes.flat:
    ax.set_xlabel("EAF gas inlet temperature (°C)")
    ax.grid(True)
fig.tight_layout()
fig.savefig("figures/gas_temperature_sensitivity.png", dpi=150)
print("\nSaved: tables/12_gas_temperature_sensitivity.csv, figures/gas_temperature_sensitivity.png")