"""Cost model v2: component costs, total cost rate and sensitivities.

Objective 2 of the planned optimisation is the TOTAL COST RATE (USD per hour of
operation: capital recovery + O&M + makeup water). No revenue, no electricity
price. The simulated annual energy (table 27, measured cycle) is used only for
the LCOE-type output. Sizes F follow run_profile_ref_design_splitflow.py.

Run: python scripts/run_cost_splitflow.py"""
import json
import warnings
from pathlib import Path

import pandas as pd

from sco2.cost_splitflow import COST, cost_plant
from sco2.heat_source_adv import EAF
from sco2.offdesign_splitflow import design_hardware
from sco2.plant_splitflow import evaluate_plant, get_gas

SIZES = [1.0, 1.2, 1.5]
M0 = EAF["m_gas"]

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
gas = get_gas()


def annual_energy_MWh(size):
    """Simulated annual energy of the measured cycle (table 27), in MWh/yr.
    Works whether the table has the metrics as rows or as columns."""
    path = f"tables/27_ref_design_size{size}.csv"
    try:
        t = pd.read_csv(path, index_col=0)
        key = lambda x: str(x).startswith("Annual energy")
        meas = lambda x: "measured" in str(x)
        if any(key(i) for i in t.index):                       # metrics as rows
            row = [i for i in t.index if key(i)][0]
            col = [c for c in t.columns if meas(c)][0]
            return float(t.loc[row, col]) * 1e3
        col = [c for c in t.columns if key(c)][0]              # metrics as columns
        row = [i for i in t.index if meas(i)][0]
        return float(t.loc[row, col]) * 1e3
    except Exception as exc:
        warnings.warn(f"annual energy of size {size} not read from {path}: {exc!r}; "
                      f"first lines of the file: {open(path).read()[:300]!r}")
        return None


plants = {}
for F in SIZES:
    P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"], m_gas=F * M0)
    plants[F] = (P, design_hardware(P, gas))

# ------------------------------------------------------------------ table 37
P, hw = plants[1.0]
base = cost_plant(P, hw, annual_energy_MWh=annual_energy_MWh(1.0))
zt = sum(base["Z_dot"].values())
t37 = pd.DataFrame([{"Component": k, "Size": s, "Purchased cost (k$)": c / 1e3,
                     "Share (%)": c / base["PEC"] * 100,
                     "Cost rate (USD/h)": base["Z_dot"][k], "Basis": src}
                    for k, (s, c, src) in base["rows"].items()])
t37.loc[len(t37)] = ["Total purchased equipment", "", base["PEC"] / 1e3, 100.0, zt, ""]
print("Size 1.0 (design gas, 685 C, 48.4 kg/s)\n")
print(t37.round(1).to_string(index=False))
for w in base["warnings"]:
    print("WARNING:", w)

# ------------------------------------------------------------------ table 38
rows = []
for F in SIZES:
    P, hw = plants[F]
    E = annual_energy_MWh(F)
    c = cost_plant(P, hw, annual_energy_MWh=E)
    rows.append({"Size": F, "Design net power (MW)": c["net_el_design_MW"],
                 "Purchased equipment (M$)": c["PEC"] / 1e6,
                 "Installed capital (M$)": c["TCI"] / 1e6,
                 "Capital (USD/kWe design)": c["capital_per_kWe"],
                 "Annual cost (M$/yr)": c["annual_cost"] / 1e6,
                 "Total cost rate (USD/h)": c["cost_rate"],
                 "Simulated annual energy (GWh/yr)": (E / 1e3) if E else float("nan"),
                 "Cost per simulated energy (USD/MWh)": c.get("LCOE_sim", float("nan"))})
t38 = pd.DataFrame(rows)
print("\nSizes (hardware designed for F x mean gas flow)")
print(t38.round(2).to_string(index=False))

# ------------------------------------------------------------------ table 39
E1 = annual_energy_MWh(1.0)
CASES = [("Base case", {}),
         ("Gas-side U 30 W/m2K", {"U_gas": 30.0}),
         ("Gas-side U 60 W/m2K", {"U_gas": 60.0}),
         ("Heater fouling margin 0 %", {"fouling_margin": 0.0}),
         ("Heater fouling margin 40 %", {"fouling_margin": 0.4}),
         ("Recuperators linear [WDS16]", {"recuperator_basis": "WDS16"}),
         ("Indirect costs 67 % (DLR study)", {"f_indirect": 0.67}),
         ("Turbine cost x1.5", {"mult": {"Turbine": 1.5}}),
         ("Heaters cost x2", {"mult": {"Main heater": 2.0, "Branch heater": 2.0}}),
         ("Cooling tower exponent 0.6", {"tower_exponent": 0.6}),
         ("Discount rate 5 %", {"r": 0.05}),
         ("Discount rate 10 %", {"r": 0.10}),
         ("O&M 3 % of capital", {"om_fraction": 0.03}),
         ("Water price x2", {"water_price": 2 * COST["water_price"]})]
P, hw = plants[1.0]
rows, ref = [], None
for name, over in CASES:
    c = cost_plant(P, hw, p=over, annual_energy_MWh=E1)
    ref = ref or c["cost_rate"]
    rows.append({"Scenario": name, "Installed capital (M$)": c["TCI"] / 1e6,
                 "Total cost rate (USD/h)": c["cost_rate"],
                 "Change vs base (%)": (c["cost_rate"] / ref - 1) * 100,
                 "Cost per simulated energy (USD/MWh)": c.get("LCOE_sim", float("nan"))})
t39 = pd.DataFrame(rows)
print("\nSensitivity of the total cost rate (size 1.0)")
print(t39.round(2).to_string(index=False))

Path("tables").mkdir(exist_ok=True)
t37.to_csv("tables/37_cost_components_size1.0.csv", index=False, encoding="utf-8-sig")
t38.to_csv("tables/38_cost_sizes.csv", index=False, encoding="utf-8-sig")
t39.to_csv("tables/39_cost_sensitivity_size1.0.csv", index=False, encoding="utf-8-sig")
print("\nSaved: tables/37_cost_components_size1.0.csv, 38_cost_sizes.csv, 39_cost_sensitivity_size1.0.csv")
