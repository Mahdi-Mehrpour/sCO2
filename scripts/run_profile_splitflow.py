"""Step 3: the plant on a fluctuating EAF gas (quasi-steady, 'follow the gas').

1. Builds the power map (controller on a grid of gas temperature and flow,
   y fixed): about 10 min the first time, then reused from
   data/results/power_map.npz (use --rebuild to force).
2. Checks the interpolation against the controller at three off-grid points.
3. Simulates one 60-minute cycle for several scenarios and writes the tables
   and figures.

Run: python scripts/run_profile_splitflow.py [--rebuild]"""
import json
import sys
import time
from pathlib import Path
import pandas as pd

from sco2.plant_splitflow import evaluate_plant, get_gas
from sco2.offdesign_splitflow import design_hardware
from sco2.controller_splitflow import operate, CONTROL
from sco2.profile_eaf import PROFILE, make_profile
from sco2.fluctuation_splitflow import (build_map, save_map, load_map, PowerMap,
                                        simulate, summarise, make_figures,
                                        hours_per_year, T_GRID_C, M_GRID,
                                        RESTART, FIELDS)

K = 273.15
MAP_FILE = "data/results/power_map.npz"

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"])
gas = get_gas()
hw = design_hardware(P, gas)
print(f"Design: net electrical {hw['net_el_des'] / 1e6:.3f} MW, "
      f"{hours_per_year():.0f} h/yr -> {hw['net_el_des'] / 1e6 * hours_per_year() / 1e3:.1f} GWh/yr steady\n")

# ---------------- 1. power map -------------------------------------------
arr = None if "--rebuild" in sys.argv else load_map(MAP_FILE, hw)
if arr is None:
    print(f"Building the power map ({len(T_GRID_C)} x {len(M_GRID)} points)...")
    t0 = time.time()
    arr = build_map(hw, gas, y_mode="fixed")
    save_map(MAP_FILE, arr, hw)
    print(f"Map built in {(time.time() - t0) / 60:.1f} min, saved to {MAP_FILE}\n")
else:
    print(f"Power map loaded from {MAP_FILE}\n")
pm = PowerMap(arr, hw)

rows = []
for i, Tc in enumerate(T_GRID_C):
    for j, mg in enumerate(M_GRID):
        rows.append({"Gas temperature (C)": Tc, "Gas flow (kg/s)": mg,
                     **{k: round(float(arr[k][i, j]), 3) for k in FIELDS}})
Path("tables").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("tables/21_power_map.csv", index=False, encoding="utf-8-sig")

# ---------------- 2. interpolation check ---------------------------------
print("Interpolation check (map vs the controller at off-grid points):")
for Tc, mg in [(750, 22), (950, 28), (650, 14)]:
    r = operate(hw, gas, Tc + K, mg, y_mode="fixed")
    o = pm.at(Tc, mg)
    direct = r["net_el"] / 1e6
    mapped = float(o["net_if_run"][0]) / 1e6 if bool(o["run"][0]) else pm.P_idle / 1e6
    print(f"  {Tc} C, {mg} kg/s: controller {direct:.3f} MW | map {mapped:.3f} MW "
          f"| difference {abs(mapped - direct) / max(abs(direct), 1e-9) * 100:.2f} %")
print()

# ---------------- 3. scenarios -------------------------------------------
SCENARIOS = [("Steady design gas", 0.0, False),
             ("Weak (amp 1)", 1.0, False),
             ("Amp 2", 2.0, False),
             ("Amp 3", 3.0, False),
             ("Weak + gap", 1.0, True),
             ("Amp 3 + gap", 3.0, True)]
results = []
for label, amp, gap_on in SCENARIOS:
    prof = make_profile(PROFILE, amp=amp, T_mean_C=hw["T_gas_des"] - K,
                        m_mean=hw["m_gas_des"], gap=gap_on)
    sim = simulate(pm, prof)
    results.append((label, prof, sim, summarise(pm, prof, sim, hw, label)))

df = pd.DataFrame([r[3] for r in results])
num = df.select_dtypes("number").columns
df[num] = df[num].round(3)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)
print(f"Placeholders: cycle {PROFILE['cycle_min']:.0f} min | amplitudes at amp 1: "
      f"T {PROFILE['a_T'] * 100:.0f} %, flow {PROFILE['a_m'] * 100:.0f} % | flow lags "
      f"{PROFILE['lag_min']:.0f} min | gap {PROFILE['gap_min']:.0f} min at "
      f"{PROFILE['gap_T_C']:.0f} C, {PROFILE['gap_m_frac'] * 100:.0f} % flow | restart "
      f"{RESTART['t_restart_min']:.0f} min\n")
print(df.set_index("Scenario").T.to_string())
df.to_csv("tables/22_fluctuation_scenarios.csv", index=False, encoding="utf-8-sig")

make_figures(results, hw, "figures/fluctuation_timeseries.png",
             "figures/fluctuation_summary.png")
print("\nSaved: tables/21_power_map.csv, tables/22_fluctuation_scenarios.csv, "
      "figures/fluctuation_timeseries.png, figures/fluctuation_summary.png")