"""Step 2 of the exergy work: time-resolved exergy account of the measured
reference cycle (Dal Magro 2017, Fig. 3.40) on the designed plant.

For every time step: gas exergy in, net electrical power and where the rest
goes (bypassed gas, stack gas, heat exchangers, turbomachines, cooler,
electromechanical and auxiliary losses, idle and restart). Running steps use
the controller directly (no map interpolation); the run / idle / restart
state comes from the same simulation as the other results.

Needs the power map of the same design (run_profile_ref_design_splitflow.py).
Run: python scripts/run_exergy_profile_splitflow.py [--size 1.0]"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sco2.plant_splitflow import evaluate_plant, get_gas
from sco2.offdesign_splitflow import design_hardware
from sco2.controller_splitflow import operate
from sco2.heat_source_adv import EAF
from sco2.profile_eaf import make_profile_ref
from sco2.fluctuation_splitflow import (load_map, PowerMap, simulate, hours_per_year)
from sco2.exergy_splitflow import exergy_plant, chemical_exergy_CO
from sco2.exergy_offdesign_splitflow import exergy_point, GROUPS
from sco2.report_figures import make_exergy_figures

K = 273.15
size = float(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 1.0
tag = f"size{size:.1f}"
MAP_FILE = f"data/results/power_map_ref_{tag}_v2.npz"

with open("data/results/splitflow_optimum_ref.json") as f:
    opt = json.load(f)
M_MEAN = opt["m_gas"]
P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"], T_in=opt["T_in_K"], m_gas=M_MEAN * size)
gas = get_gas()
hw = design_hardware(P, gas)

# same grid rule as run_profile_ref_design_splitflow.py (needed to read the map)
T_GRID = sorted(set(list(range(250, 1051, 50)) + [round(hw["T_gas_des"] - K, 2)]))
M_GRID = sorted(set([12, 20, 28, 36, 44, 56, 66, 78, 90, 100] + [round(hw["m_gas_des"], 2)]))
arr = load_map(MAP_FILE, hw, T_GRID, M_GRID)
if arr is None:
    raise SystemExit(f"Power map {MAP_FILE} missing or outdated: run "
                     f"scripts/run_profile_ref_design_splitflow.py --size {size:.1f} first.")
pm = PowerMap(arr, hw, T_grid=T_GRID, m_grid=M_GRID)
prof = make_profile_ref(M_MEAN, flow="data")
sim = simulate(pm, prof)
print(f"Size {size:.1f}: design net {hw['net_el_des'] / 1e6:.3f} MW | "
      f"{len(prof['t_min'])} time steps, {np.mean(sim['state'] == 'run') * 100:.1f} % running\n",
      flush=True)

rows, x, n_disagree = [], None, 0
P_idle = pm.P_idle
for k, t in enumerate(prof["t_min"]):
    T_g, m_g = float(prof["T_C"][k]), float(prof["m"][k])
    ex_in = m_g * float(gas.ex(T_g + K))
    ex_chem = chemical_exergy_CO(gas, EAF["X"], m_g)
    state = sim["state"][k]
    rec = {"Time (min)": t, "Gas temperature (C)": T_g, "Gas flow (kg/s)": m_g,
           "State": state, "ex_in": ex_in, "ex_chem_CO": ex_chem}
    pt = None
    if state == "run":
        r = operate(hw, gas, T_g + K, m_g, y_mode="fixed", x0=x)
        pt = r["point"]
        if pt is not None:
            x = r["x"]
            if r["mode"] != "run":
                n_disagree += 1
    if pt is not None and state == "run":
        e = exergy_point(pt, gas, T_g + K, m_g)
        g = dict(e["groups"])
        rec.update({"W": e["W"], "fuel": e["fuel"], "balance": e["balance"],
                    "T7 (C)": pt["T7"] - K, "CO2 flow (kg/s)": pt["m_co2"],
                    "Bypass (kg/s)": pt["m_bypass"], "W_map": sim["P"][k], **g,
                    **{"D " + c: v for c, v in e["components"].items()}})
    else:                                       # idle or restart: nothing is recovered
        rec.update({"W": P_idle, "fuel": 0.0, "balance": 0.0, "T7 (C)": np.nan,
                    "CO2 flow (kg/s)": 0.0, "Bypass (kg/s)": m_g, "W_map": sim["P"][k],
                    **{gname: 0.0 for gname in GROUPS},
                    "Net electrical power": P_idle, "Idle and restart": ex_in - P_idle})
    rows.append(rec)
    if (k + 1) % 20 == 0:
        print(f"  step {k + 1}/{len(prof['t_min'])}", flush=True)
df = pd.DataFrame(rows).fillna(0.0)

# ---------------- checks ---------------------------------------------------
print(f"\nBalance closure: largest error {df['balance'].abs().max() / 1e6:.2e} MW "
      f"(should be 0)")
print(f"Mean net power: direct controller {df['W'].mean() / 1e6:.4f} MW | map "
      f"{df['W_map'].mean() / 1e6:.4f} MW (difference "
      f"{abs(df['W'].mean() - df['W_map'].mean()) / abs(df['W_map'].mean()) * 100:.2f} %)")
if n_disagree:
    print(f"Note: {n_disagree} running steps are at the run threshold (controller says idle).")

# ---------------- tables ---------------------------------------------------
Path("tables").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
mw = lambda c: df[c] / 1e6
ts = pd.DataFrame({"Time (min)": df["Time (min)"], "Gas temperature (C)": df["Gas temperature (C)"],
                   "Gas flow (kg/s)": df["Gas flow (kg/s)"], "State": df["State"],
                   "Gas exergy in (MW)": mw("ex_in")})
for gname in GROUPS:
    ts[gname + " (MW)"] = mw(gname)
for c in [c for c in df.columns if c.startswith("D ")]:
    ts[c[2:] + " destruction (MW)"] = mw(c)
ts.round(4).to_csv(f"tables/34_exergy_timeseries_{tag}.csv", index=False, encoding="utf-8-sig")

mean_in = df["ex_in"].mean()
avg = pd.DataFrame([{"Category": gname, "Mean (MW)": df[gname].mean() / 1e6,
                     "Share of gas exergy (%)": df[gname].mean() / mean_in * 100}
                    for gname in GROUPS])
comp_cols = [c for c in df.columns if c.startswith("D ")]
detail = pd.DataFrame([{"Category": "  of which " + c[2:], "Mean (MW)": df[c].mean() / 1e6,
                        "Share of gas exergy (%)": df[c].mean() / mean_in * 100}
                       for c in comp_cols])
total = pd.DataFrame([{"Category": "Total (= gas exergy in)",
                       "Mean (MW)": sum(df[g].mean() for g in GROUPS) / 1e6,
                       "Share of gas exergy (%)": sum(df[g].mean() for g in GROUPS) / mean_in * 100}])
avg = pd.concat([avg, detail, total], ignore_index=True).round(3)
avg.to_csv(f"tables/35_exergy_cycle_average_{tag}.csv", index=False, encoding="utf-8-sig")

W_mean = df["W"].mean()
ex_d = exergy_plant(P, gas, M_MEAN * size)
eff = pd.DataFrame([
    ("System exergy efficiency (net electrical / gas exergy)", ex_d["eta_system"] * 100,
     W_mean / mean_in * 100),
    ("Fuel-product exergy efficiency (net / exergy taken from the gas)",
     ex_d["eta_fuel_product"] * 100, W_mean / df["fuel"].mean() * 100),
    ("Total exergy efficiency incl. CO chemical exergy", ex_d["eta_total"] * 100,
     W_mean / (mean_in + df["ex_chem_CO"].mean()) * 100)],
    columns=["Definition", "Design point (%)", "Measured cycle, mean (%)"]).round(2)
eff.to_csv(f"tables/36_exergy_efficiencies_cycle_{tag}.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200)
print(f"\nMean gas exergy in: {mean_in / 1e6:.3f} MW | net electrical {W_mean / 1e6:.3f} MW | "
      f"per year at {hours_per_year():.0f} h: gas exergy {mean_in / 1e6 * hours_per_year() / 1e3:.1f} GWh, "
      f"electricity {W_mean / 1e6 * hours_per_year() / 1e3:.1f} GWh\n")
print(avg.to_string(index=False))
print()
print(eff.to_string(index=False))

make_exergy_figures(df, f"figures/exergy_{tag}_", f"hardware for {size:.1f} x mean gas flow")
print(f"\nSaved: tables/34_exergy_timeseries_{tag}.csv, 35_exergy_cycle_average_{tag}.csv, "
      f"36_exergy_efficiencies_cycle_{tag}.csv; figures/exergy_{tag}_1_loop.png, "
      f"exergy_{tag}_2_average.png")
