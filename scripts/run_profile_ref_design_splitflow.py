"""Step 5b: the plant DESIGNED for the measured gas, run on the measured cycle.

Hardware is designed at the mean gas of the reference cycle (685 C, about
48 kg/s) times a size factor on the gas flow (1.0 = mean heat; 1.2 and 1.5 =
bigger hardware, more of the peaks absorbed). The simulated gas always has the
measured mean flow. One power map per size (about 10-15 min the first time).

Run: python scripts/run_profile_ref_design_splitflow.py [--size 1.0] [--rebuild]"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sco2.plant_splitflow import evaluate_plant, get_gas
from sco2.offdesign_splitflow import design_hardware
from sco2.controller_splitflow import CONTROL, operate
from sco2.profile_eaf import make_profile_ref, make_profile_steady
from sco2.report_figures import make_report_figures
from sco2.fluctuation_splitflow import (build_map, save_map, load_map, PowerMap,
                                        simulate, summarise, make_figures,
                                        hours_per_year, RESTART)

K = 273.15
size = float(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 1.0
tag = f"size{size:.1f}"
MAP_FILE = f"data/results/power_map_ref_{tag}.npz"

with open("data/results/splitflow_optimum_ref.json") as f:
    opt = json.load(f)
M_MEAN = opt["m_gas"]
P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"], T_in=opt["T_in_K"],
                   m_gas=M_MEAN * size)
gas = get_gas()
hw = design_hardware(P, gas)
print(f"Size {size:.1f}: hardware for {opt['T_in_K'] - K:.0f} C and {M_MEAN * size:.1f} kg/s | "
      f"design net {hw['net_el_des'] / 1e6:.3f} MW | CO2 flow {hw['m_des']:.1f} kg/s | "
      f"steady {hw['net_el_des'] / 1e6 * hours_per_year() / 1e3:.1f} GWh/yr\n")

# Grid: the design gas is a node, because the controller has a kink there (the
# capacity limit); a grid that straddles the kink underestimates the power.
T_GRID = sorted(set(list(range(250, 1051, 50)) + [round(hw["T_gas_des"] - K, 2)]))
M_GRID = sorted(set([12, 20, 28, 36, 44, 56, 66, 78, 90, 100]
                    + [round(hw["m_gas_des"], 2)]))
MAP_FILE = f"data/results/power_map_ref_{tag}_v2.npz"

arr = None if "--rebuild" in sys.argv else load_map(MAP_FILE, hw, T_GRID, M_GRID)
if arr is None:
    print(f"Building the power map ({len(T_GRID)} x {len(M_GRID)} points)...", flush=True)
    t0 = time.time()
    n_pts = len(T_GRID) * len(M_GRID)

    def log(s):
        k = int(s.split("map point")[1].split("/")[0])
        if k % 10 == 0 or k == n_pts:
            print(s, flush=True)
    arr = build_map(hw, gas, y_mode="fixed", log=log, T_grid=T_GRID, m_grid=M_GRID)
    save_map(MAP_FILE, arr, hw, T_GRID, M_GRID)
    print(f"Map built in {(time.time() - t0) / 60:.1f} min, saved to {MAP_FILE}\n")
else:
    print(f"Power map loaded from {MAP_FILE}\n")
pm = PowerMap(arr, hw, T_grid=T_GRID, m_grid=M_GRID)

print("Map checks (controller vs map):")
o = pm.at(hw["T_gas_des"] - K, hw["m_gas_des"])
print(f"  design gas: map {float(o['net_if_run'][0]) / 1e6:.3f} MW | design {hw['net_el_des'] / 1e6:.3f} MW")
print(f"  design gas cycle efficiency: map {float(o['eta_cyc'][0]) * 100:.2f} % | design "
      f"{P['res']['eta'] * 100:.2f} %")
for Tc, mg in [(640, 30), (760, 70), (880, 40)]:
    r = operate(hw, gas, Tc + K, mg, y_mode="fixed")
    o = pm.at(Tc, mg)
    mp = float(o["net_if_run"][0]) / 1e6 if bool(o["run"][0]) else pm.P_idle / 1e6
    print(f"  {Tc} C, {mg} kg/s: controller {r['net_el'] / 1e6:.3f} MW | map {mp:.3f} MW "
          f"| difference {abs(mp - r['net_el'] / 1e6) / max(abs(r['net_el'] / 1e6), 1e-9) * 100:.2f} %")
print()

d = make_profile_ref(M_MEAN, flow="data")
SC = [("Steady at mean gas", make_profile_steady(d["T_C"].mean(), M_MEAN, 40.0)),
      ("Ref cycle, measured flow", d),
      ("Ref cycle, const flow", make_profile_ref(M_MEAN, flow="const"))]
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
      f"of parasitic load | restart {RESTART['t_restart_min']:.0f} min\n")
print(df.set_index("Scenario").T.to_string())

rows = []
for T7min in (250, 300, 350, 400, 450):
    pm2 = PowerMap(arr, hw, dict(CONTROL, T7_min_run_K=T7min + K), T_GRID, M_GRID)
    for tr in (3.0, 10.0, 30.0):
        sm = summarise(pm2, d, simulate(pm2, d, t_restart_min=tr), hw, "")
        rows.append({"T7 run limit (C)": T7min, "Restart (min)": tr,
                     "Mean power (MW)": sm["Mean net electrical power (MW)"],
                     "Annual energy (GWh/yr)": sm["Annual energy (GWh/yr)"],
                     "Time running (%)": sm["Time running (%)"],
                     "Time idle (%)": sm["Time idle (%)"],
                     "Time in restart (%)": sm["Time in restart (%)"]})
sens = pd.DataFrame(rows).round(3)
print("\nSensitivity (measured cycle, measured flow):")
print(sens.to_string(index=False))

Path("tables").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
df.to_csv(f"tables/27_ref_design_{tag}.csv", index=False, encoding="utf-8-sig")
sens.to_csv(f"tables/28_ref_design_sensitivity_{tag}.csv", index=False, encoding="utf-8-sig")
make_figures(results, hw, f"figures/ref_design_timeseries_{tag}.png",
             f"figures/ref_design_summary_{tag}.png",
             show_labels=("Ref cycle, measured flow", "Ref cycle, const flow"))
sim_m = results[1][2]
cols = {"Time (min)": sim_m["t_min"], "Gas temperature (C)": sim_m["T_C"],
        "Gas flow (kg/s)": sim_m["m_gas"], "State": sim_m["state"],
        "Net electrical power (MW)": sim_m["P"] / 1e6,
        "Turbine CO2 flow (kg/s)": sim_m["m_co2"],
        "Cycle thermal efficiency (%)": sim_m["eta_cyc"] * 100,
        "Net electrical efficiency (%)": sim_m["eta_el"] * 100,
        "Turbine inlet T7 (C)": sim_m["T7"] - K}
pd.DataFrame(cols).round(3).to_csv(f"tables/30_ref_timeseries_{tag}.csv", index=False,
                                   encoding="utf-8-sig")
stems = make_report_figures(sim_m, hw, f"figures/report_{tag}_",
                            f"hardware for {size:.1f} x mean gas flow")
print("Report figures:", ", ".join(x + ".png" for x in stems))
m = results[1][3]
print(f"\nSIZE {size:.1f}: design {hw['net_el_des'] / 1e6:.3f} MW | measured cycle "
      f"{m['Mean net electrical power (MW)']:.3f} MW mean, "
      f"{m['Annual energy (GWh/yr)']:.1f} GWh/yr, heat used "
      f"{m['Heat used (% of available)']:.1f} % of available")
print(f"Saved: tables/27_ref_design_{tag}.csv, tables/28_ref_design_sensitivity_{tag}.csv, "
      f"figures/ref_design_*_{tag}.png")
