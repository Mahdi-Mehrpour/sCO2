"""Step 2 of the off-design model: what does the controller do at different
gas conditions? Runs the 'follow the gas' controller (CO2 flow, y, gas bypass,
idle) for a set of gas temperatures and flows, with y fixed at the design
value, and then with y optimised for a few of the cases.

Run: python scripts/run_controller_check.py     (a few minutes)"""
import json
import time
from pathlib import Path
import pandas as pd

from sco2.plant_splitflow import evaluate_plant, get_gas
from sco2.offdesign_splitflow import design_hardware
from sco2.controller_splitflow import operate, CONTROL

K = 273.15

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"])
gas = get_gas()
hw = design_hardware(P, gas)
print(f"Design: gas {hw['T_gas_des'] - K:.0f} C, {hw['m_gas_des']:.0f} kg/s, "
      f"CO2 {hw['m_des']:.1f} kg/s, net electrical {hw['net_el_des'] / 1e6:.3f} MW")
print(f"Controller placeholders: minimum CO2 flow {CONTROL['m_min_frac'] * 100:.0f} % | "
      f"run threshold {CONTROL['net_el_min_run_W'] / 1e6:.2f} MW | "
      f"idle consumption {CONTROL['idle_parasitic_frac'] * hw['P_par_des'] / 1e6:.2f} MW\n")

# (gas temperature in C, gas flow in kg/s), from the design outward
Td, mg0 = hw["T_gas_des"] - K, hw["m_gas_des"]
CASES = [(Td, mg0), (800, mg0), (900, mg0), (1000, mg0), (1100, mg0),
         (600, mg0), (550, mg0), (500, mg0), (450, mg0),
         (Td, 40), (Td, 30), (Td, 20), (Td, 12),
         (550, 25), (450, 20)]
OPTIMISE = [(600, mg0), (550, mg0), (Td, 30), (550, 25)]


def run_case(Tc, mg, y_mode, x0):
    t0 = time.time()
    r = operate(hw, gas, Tc + K, mg, y_mode=y_mode, x0=x0)
    row = {"Gas (C)": round(Tc, 1), "Gas flow (kg/s)": round(mg, 1), "y mode": y_mode,
           "Mode": r["mode"],
           "CO2 flow / design (%)": round(r["m_co2"] / hw["m_des"] * 100, 1),
           "y": round(r["y"], 3),
           "Gas bypass (kg/s)": round(r["m_bypass"], 2),
           "T7 (C)": round(r["T7"] - K, 1) if "T7" in r else None,
           "p max (MPa)": round(r["p2"] / 1e6, 2) if "p2" in r else None,
           "T_b (C)": round(r["T_b"] - K, 1) if "T_b" in r else None,
           "Stack (C)": round(r["T_stack"] - K, 1) if "T_stack" in r else None,
           "Heat used (%)": round(r["heat_used_pct"], 1) if "heat_used_pct" in r else None,
           "Net electrical (MW)": round(r["net_el"] / 1e6, 3),
           "Net if run (MW)": round(r.get("net_el_if_run", float("nan")) / 1e6, 3),
           "Evaluations": r["n_eval"], "Time (s)": round(time.time() - t0, 1),
           "Flags": "; ".join(r["flags"])}
    print(f"  {Tc:.1f} C, {mg:.1f} kg/s, y {y_mode}: {r['mode']}, net {row['Net electrical (MW)']} MW, "
          f"{row['Time (s)']} s", flush=True)
    return row, r.get("x")


rows, x = [], None
print("y fixed at the design value:")
for Tc, mg in CASES:
    row, xn = run_case(Tc, mg, "fixed", x)
    rows.append(row)
    x = xn if xn is not None else x
print("\ny optimised (a few cases):")
for Tc, mg in OPTIMISE:
    row, _ = run_case(Tc, mg, "optimise", None)
    rows.append(row)

df = pd.DataFrame(rows)
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 60)
print()
print(df.drop(columns=["Evaluations", "Flags"]).to_string(index=False))
print("\nFlags:")
for _, r_ in df.iterrows():
    if r_["Flags"]:
        print(f"  {r_['Gas (C)']} C, {r_['Gas flow (kg/s)']} kg/s, y {r_['y mode']}: {r_['Flags']}")

Path("tables").mkdir(exist_ok=True)
df.to_csv("tables/20_controller_check.csv", index=False, encoding="utf-8-sig")
print("\nSaved: tables/20_controller_check.csv")