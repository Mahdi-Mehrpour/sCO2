"""Site case: InfraBuild Sydney (Rooty Hill, NSW). Stable-gas surrogate of a
batch process, so gas temperature and flow are varied together.

Cases (each re-optimised for y and T_b at maximum net electrical power, the
plant re-sized, NPV computed):
  fixed flow     : 14 kg/s at 650, 700 and 800 C
  heat-matched   : flow raised at 650 and 700 C so that the heat available
                   (down to the minimum stack temperature) equals the heat of
                   the 800 C / 14 kg/s case (the stack data fix the amount of
                   gas, not its temperature; more leakage air = cooler, larger
                   stream with about the same heat)
Economics: site electricity and water prices, three electricity values, with
and without +30 % unmodelled capital.

Run: python scripts/run_site_rootyhill.py"""
import json
from dataclasses import fields
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

from sco2.plant_splitflow import evaluate_plant
from sco2.economics_splitflow import build_inputs, fast_economics
from sco2.economics_eaf import EconInputs
from sco2.cooling_splitflow import COOLING

K = 273.15
AUD_USD = 0.65                      # ASSUMED exchange rate, check before use
C_E_AUD = {"low": 0.08, "base": 0.15, "high": 0.25}      # AUD/kWh (avoided purchase)
WATER_AUD_PER_M3 = 3.5              # AUD/m3, makeup + blowdown (Sydney Water, 2025-26)
WET_BULB_C = 20.0                   # representative annual design point
EXTRA_CAP = {"no extra capital": 0.0, "+30 % capital": 0.30}
M_BASE = 14.0                       # kg/s, stack-derived flow (+-20 %)
T_CASES = [650.0, 700.0, 800.0]     # C
R, N = 0.08, 20
DT_REC, PEN = 10.0, 1.0
LB, UB = [0.02, 1.0], [0.60, 5.5]   # y, T_b/100 (C)

COOLING["wet_tower"]["T_wb_C"] = WET_BULB_C

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
PR = opt["pr"]
z_design = [opt["y"], (opt["T_b"] - K) / 100]

# name of the water price field in EconInputs (found by inspection)
water_fields = [f.name for f in fields(EconInputs) if "water" in f.name.lower()
                or f.name in ("c_w", "c_water")]
print("EconInputs fields:", [f.name for f in fields(EconInputs)])
if len(water_fields) != 1:
    raise SystemExit(f"Cannot identify the water price field, candidates: {water_fields}. "
                     "Set WATER_FIELD by hand in this script.")
WATER_FIELD = water_fields[0]
print("Water price field used:", WATER_FIELD, "->", WATER_AUD_PER_M3 * AUD_USD, "USD/m3\n")


def run(y, tb100, Tc, m_gas):
    return evaluate_plant(y, tb100 * 100 + K, PR, T_in=Tc + K, m_gas=m_gas)


def violation(P):
    p = P["res"]["pinch"]
    return (max(0.0, DT_REC - p["HTR"]["min_dT"])
            + max(0.0, DT_REC - p["LTR"]["min_dT"]))


def objective(z, Tc, m_gas):
    try:
        P = run(z[0], z[1], Tc, m_gas)
    except Exception:
        return 10.0
    return -P["net_el"] / 1e6 + PEN * violation(P)


def optimise(Tc, m_gas, starts):
    lb, ub = np.array(LB, float), np.array(UB, float)
    best = None
    for z0 in starts:
        z0 = np.clip(np.array(z0, float), lb, ub)
        verts = [z0]
        for j, s in enumerate([0.05, 0.5]):
            v = z0.copy()
            v[j] += s if z0[j] + s <= ub[j] else -s
            verts.append(v)
        o = minimize(objective, z0, args=(Tc, m_gas), method="Nelder-Mead",
                     bounds=Bounds(lb, ub),
                     options={"initial_simplex": np.array(verts), "maxfev": 150,
                              "xatol": 1e-3, "fatol": 1e-5})
        print(f"    start: {-o.fun:.4f} MW, z = {np.round(o.x, 3).tolist()}, "
              f"evaluations = {o.nfev}", flush=True)
        if best is None or o.fun < best[0]:
            best = (o.fun, o.x)
    return best[1]


# ---- heat to match: 800 C, 14 kg/s --------------------------------------
P_ref = run(*z_design, 800.0, M_BASE)
Q_target = P_ref["hs"]["Q_max"]
print(f"Reference heat available (800 C, {M_BASE} kg/s): {Q_target/1e6:.2f} MW\n")

cases = []
for Tc in T_CASES:
    cases.append((Tc, M_BASE, "fixed flow"))
for Tc in T_CASES[:-1]:
    Q1 = run(*z_design, Tc, M_BASE)["hs"]["Q_max"]
    cases.append((Tc, M_BASE * Q_target / Q1, "heat-matched flow"))

rows = []
for Tc, mg, label in cases:
    print(f"{Tc:.0f} C, {mg:.1f} kg/s ({label})", flush=True)
    z = optimise(Tc, mg, [z_design, [0.30, 3.5]])
    P = run(z[0], z[1], Tc, mg)
    flags = []
    if P["par"]["W_turbine"] / 1e6 < 8.0:
        flags.append("turbine below the 8 MW correlation range")
    if P["par"]["W_compressor"] / 1e6 < 1.5:
        flags.append("compressor below the 1.5 MW range")
    if abs(z[0] - LB[0]) < 1e-3 or abs(z[0] - UB[0]) < 1e-3 or abs(z[1] - UB[1]) < 1e-3:
        flags.append("y or T_b at a bound")
    row = {"Case": label, "Gas inlet (C)": Tc, "Gas flow (kg/s)": mg,
           "Heat available (MW)": P["hs"]["Q_max"] / 1e6,
           "Heat used (% of available)": P["hs"]["Q"] / P["hs"]["Q_max"] * 100,
           "Stack temperature (C)": P["hs"]["T_out"] - K,
           "T1 (C)": P["T1"] - K,
           "Optimal y": z[0], "Optimal T_b (C)": z[1] * 100,
           "Net electrical power (MW)": P["net_el"] / 1e6,
           "Net electrical efficiency (%)": P["eta_el"] * 100,
           "Annual energy (GWh/yr)": P["energy_GWh"],
           "Limiting constraint": P["hs"]["binding"]}
    for ce_name, ce_aud in C_E_AUD.items():
        for cap_name, cap in EXTRA_CAP.items():
            ov = {"r": R, "N": N, "c_e": ce_aud * AUD_USD,
                  WATER_FIELD: WATER_AUD_PER_M3 * AUD_USD}
            ci, e = build_inputs(P, ov)
            eco = fast_economics(ci, e, extra_capital=cap)
            row[f"NPV, {ce_aud:.2f} AUD/kWh, {cap_name} (M$)"] = eco["NPV"] / 1e6
            if ce_name == "base":
                row[f"Capital, {cap_name} (M$)"] = eco["TCI"] / 1e6
                row[f"Capital, {cap_name} (AUD/kWe)"] = (
                    eco["TCI"] / (eco["W_net"] * 1e3) / AUD_USD)
                row[f"LCOE, {cap_name} (USD/MWh)"] = eco["LCOE"] if "LCOE" in eco else float("nan")
    row["Flags"] = "; ".join(flags)
    rows.append(row)

df = pd.DataFrame(rows)
num = df.select_dtypes("number").columns
df[num] = df[num].round(3)
Path("tables").mkdir(exist_ok=True)
df.to_csv("tables/17_site_rootyhill.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print()
print(df.T.to_string(header=False))
print(f"\nExchange rate used: {AUD_USD} USD/AUD (assumed). Wet bulb {WET_BULB_C} C.")
print("ORC study benchmark (NSW, 2019 dollars): about 10,000-11,000 AUD/kWe at 1.6-1.8 MWe.")
print("Saved: tables/17_site_rootyhill.csv")