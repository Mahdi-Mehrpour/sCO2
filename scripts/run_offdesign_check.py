"""Step 1 of the off-design model (with the fast property table): (A) reproduce the design point with the
frozen hardware, (B) show what happens when only the CO2 flow is changed at
the design gas (gas temperature and flow unchanged, all gas through the
heaters, y fixed, sliding pressure).

Run: python scripts/run_offdesign_check.py"""
import json
import time
from pathlib import Path
import pandas as pd

from sco2.plant_splitflow import evaluate_plant, get_gas
from sco2.offdesign_splitflow import design_hardware, offdesign_point
from sco2.co2_tables import get_table

K = 273.15

# ---------------- property table: build time and accuracy -------------------
t0 = time.time()
tbl = get_table()
print(f"CO2 property table built in {time.time() - t0:.1f} s")
for name, v in tbl.verify().items():
    print(f"  {name:22s} max dT {v['max dT (K)']:.4f} K | 99 % dT {v['99 % dT (K)']:.4f} K"
          f" | max dh {v['max dh (kJ/kg)']:.4f} kJ/kg")
print("  (aim: working range max dT below 0.1 K; the near-critical corner may be larger)\n")

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"])
gas = get_gas()
hw = design_hardware(P, gas)
res, hs = P["res"], P["hs"]
st = res["states"]

print("Frozen hardware (from the design point):")
for k in ("UA_HTR", "UA_LTR", "UA_MH", "UA_BH"):
    print(f"  {k:7s} {hw[k] / 1e3:9.1f} kW/K")
print(f"  Turbine constant Kt = {hw['Kt']:.4e}, design CO2 flow {hw['m_des']:.2f} kg/s\n")

# ---------------- A: design point ------------------------------------------
t0 = time.time()
r = offdesign_point(hw, gas, hw["T_gas_des"], hw["m_gas_des"], hw["m_des"],
                    hw["y_des"])
print(f"Design point solved in {time.time() - t0:.1f} s, residual {r['residual']:.1e}\n")
m = hw["m_des"]
checks = [
    ("Net electrical power (MW)", P["net_el"] / 1e6, r["net_el"] / 1e6),
    ("Turbine inlet temperature (C)", st[7]["T"] - K, r["T7"] - K),
    ("Maximum pressure (MPa)", st[2]["p"] / 1e6, r["p2"] / 1e6),
    ("Branch heater outlet temperature (C)", st[11]["T"] - K, r["T_b"] - K),
    ("Gas stack temperature (C)", hs["T_out"] - K, r["T_stack"] - K),
    ("Main heater duty (MW)", m * res["q_in"] / 1e6, r["Q_mh"] / 1e6),
    ("Branch heater duty (MW)", m * res["q_b"] / 1e6, r["Q_bh"] / 1e6),
    ("Net shaft power (MW)", m * res["w_net"] / 1e6, r["net_shaft"] / 1e6),
]
rows = []
for name, a, b in checks:
    rel = abs(b - a) / max(abs(a), 1e-9)
    rows.append({"Quantity": name, "Design model": round(a, 4),
                 "Off-design model": round(b, 4),
                 "Difference (%)": round(rel * 100, 4),
                 "Result": "PASS" if rel < 1e-3 else "CHECK"})
dfa = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("A) Does the off-design model reproduce the design point?")
print(dfa.to_string(index=False))

# ---------------- B: CO2 flow sweep at the design gas ----------------------
print("\nB) CO2 flow changed alone (design gas, y fixed, sliding pressure)")
rows, x = [], None
for frac in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
    t0 = time.time()
    rb = offdesign_point(hw, gas, hw["T_gas_des"], hw["m_gas_des"],
                         frac * hw["m_des"], hw["y_des"], x0=x)
    x = rb["x"]
    rows.append({"CO2 flow / design": frac,
                 "CO2 flow (kg/s)": round(rb["m_co2"], 2),
                 "T7 (C)": round(rb["T7"] - K, 1),
                 "p max (MPa)": round(rb["p2"] / 1e6, 2),
                 "T_b (C)": round(rb["T_b"] - K, 1),
                 "Stack (C)": round(rb["T_stack"] - K, 1),
                 "Heat used (%)": round(rb["heat_used_pct"], 1),
                 "HTR min dT (K)": round(rb["min_dT"]["HTR"], 1),
                 "LTR min dT (K)": round(rb["min_dT"]["LTR"], 1),
                 "Net electrical (MW)": round(rb["net_el"] / 1e6, 3),
                 "Time (s)": round(time.time() - t0, 1),
                 "Flags": "; ".join(rb["flags"])})
dfb = pd.DataFrame(rows)
print(dfb.to_string(index=False))

Path("tables").mkdir(exist_ok=True)
dfa.to_csv("tables/18_offdesign_check.csv", index=False, encoding="utf-8-sig")
dfb.to_csv("tables/19_offdesign_co2_flow_sweep.csv", index=False, encoding="utf-8-sig")
print("\nSaved: tables/18_offdesign_check.csv, tables/19_offdesign_co2_flow_sweep.csv")