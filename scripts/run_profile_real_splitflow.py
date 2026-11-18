"""Step 4: the plant on the REAL EAF gas-temperature profile (digitised figure).

Uses the cached power map (data/results/power_map.npz, built by
run_profile_splitflow.py). Flow is not in the figure, so it is an assumption:
constant, or following the temperature deviation with a lag (a_flow).

Run: python scripts/run_profile_real_splitflow.py"""
import json
from pathlib import Path
import pandas as pd

from sco2.plant_splitflow import evaluate_plant, get_gas
from sco2.offdesign_splitflow import design_hardware
from sco2.controller_splitflow import CONTROL
from sco2.profile_eaf import make_profile_real, make_profile_steady
from sco2.fluctuation_splitflow import (load_map, PowerMap, simulate, summarise,
                                        make_figures, hours_per_year, RESTART)

K = 273.15
MAP_FILE = "data/results/power_map.npz"
with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"])
gas = get_gas()
hw = design_hardware(P, gas)
arr = load_map(MAP_FILE, hw)
if arr is None:
    raise SystemExit("Power map missing or outdated: run scripts/run_profile_splitflow.py first.")
pm = PowerMap(arr, hw)
M0 = hw["m_gas_des"]
print(f"Design net {hw['net_el_des'] / 1e6:.3f} MW | gas flow {M0:.1f} kg/s | "
      f"idle power {pm.P_idle / 1e6:.3f} MW\n")

# ---------------- 1. scenarios --------------------------------------------
SC = [("Steady 600 C (paper mean)", make_profile_steady(600.0, M0)),
      ("Real T, flow const", make_profile_real(m_mean=M0, a_flow=0.0)),
      ("Real T, flow a=0.3", make_profile_real(m_mean=M0, a_flow=0.3)),
      ("Real T, flow a=0.6", make_profile_real(m_mean=M0, a_flow=0.6)),
      ("Real T +209 C (mean 800), const", make_profile_real(m_mean=M0, mean_C=800.0)),
      ("Real T mean 800, a=0.3", make_profile_real(m_mean=M0, a_flow=0.3, mean_C=800.0))]
results = []
for label, prof in SC:
    sim = simulate(pm, prof)
    results.append((label, prof, sim, summarise(pm, prof, sim, hw, label)))
df = pd.DataFrame([r[3] for r in results])
num = df.select_dtypes("number").columns
df[num] = df[num].round(3)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)
print(f"Placeholders: T7 run limit {CONTROL['T7_min_run_K'] - K:.0f} C | min CO2 flow "
      f"{CONTROL['m_min_frac'] * 100:.0f} % | idle {CONTROL['idle_parasitic_frac'] * 100:.0f} % "
      f"of parasitic load | restart {RESTART['t_restart_min']:.0f} min | flow lag 3 min\n")
print(df.set_index("Scenario").T.to_string())

# ---------------- 2. sensitivity of the baseline ---------------------------
base = make_profile_real(m_mean=M0, a_flow=0.0)
rows = []
for T7min in (250, 300, 350, 400, 450):
    pm2 = PowerMap(arr, hw, dict(CONTROL, T7_min_run_K=T7min + K))
    for tr in (3.0, 10.0, 30.0):
        s = simulate(pm2, base, t_restart_min=tr)
        sm = summarise(pm2, base, s, hw, "")
        rows.append({"T7 run limit (C)": T7min, "Restart (min)": tr,
                     "Mean power (MW)": sm["Mean net electrical power (MW)"],
                     "Annual energy (GWh/yr)": sm["Annual energy (GWh/yr)"],
                     "Time running (%)": sm["Time running (%)"],
                     "Time idle (%)": sm["Time idle (%)"],
                     "Time in restart (%)": sm["Time in restart (%)"]})
sens = pd.DataFrame(rows).round(3)
print("\nSensitivity (real T, constant flow):")
print(sens.to_string(index=False))

Path("tables").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
df.to_csv("tables/23_real_profile_scenarios.csv", index=False, encoding="utf-8-sig")
sens.to_csv("tables/24_real_profile_sensitivity.csv", index=False, encoding="utf-8-sig")
make_figures(results, hw, "figures/real_profile_timeseries.png",
             "figures/real_profile_summary.png",
             show_labels=("Real T, flow const", "Real T, flow a=0.3",
                          "Real T mean 800, a=0.3"))
print("\nSaved: tables/23_real_profile_scenarios.csv, tables/24_real_profile_sensitivity.csv, "
      "figures/real_profile_timeseries.png, figures/real_profile_summary.png")