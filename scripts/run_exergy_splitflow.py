"""Exergy analysis of the design (states, components, efficiencies, figures).

Design: the net-electrical optimum for the mean gas of the measured reference
cycle (data/results/splitflow_optimum_ref.json: 685 C, 48.4 kg/s).

Run: python scripts/run_exergy_splitflow.py"""
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sco2.cycle_adv import splitflow_adv
from sco2.heat_source_adv import EAF
from sco2.parasitics_splitflow import PARASITIC
from sco2.plant_splitflow import evaluate_plant, get_gas, DRIVE
from sco2.exergy_splitflow import exergy_plant
from sco2.plotting_splitflow import plot_ts_splitflow

K = 273.15
T_OUT, F_OUT = Path("tables"), Path("figures")
T_OUT.mkdir(exist_ok=True)
F_OUT.mkdir(exist_ok=True)

with open("data/results/splitflow_optimum_ref.json") as f:
    opt = json.load(f)
M_GAS, T_GAS = opt["m_gas"], opt["T_in_K"]
D0 = {k: v.default for k, v in inspect.signature(splitflow_adv).parameters.items()}

P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"], T_in=T_GAS, m_gas=M_GAS)
res, hs = P["res"], P["hs"]
st, y, m = res["states"], res["y"], hs["m_co2"]
ex = exergy_plant(P, get_gas(), M_GAS)
fe, D, eff = ex["flow_exergy"], ex["destruction_W"], ex["base_eff"]
ex_in, stack, net_el = ex["ex_in"], ex["stack_W"], ex["net_el"]
total_d = sum(D.values())
losses = total_d + stack
print(f"Design: gas {T_GAS - K:.1f} C, {M_GAS:.1f} kg/s | y = {y:.4f}, T_b = {st[11]['T'] - K:.1f} C, "
      f"stack {hs['T_out'] - K:.1f} C\n")


def r(v, n=4):
    return round(float(v), n)


# ---------------- 02: states -----------------------------------------------
desc = {1: "Main compressor inlet", 2: "Main compressor outlet (split point)",
        3: "LTR cold outlet", 5: "Mixer outlet (HTR cold inlet)",
        6: "HTR cold outlet", 7: "Turbine inlet (main heater outlet)",
        8: "Turbine outlet", 9: "HTR hot outlet", 10: "LTR hot outlet (cooler inlet)",
        11: "Branch heater outlet"}
frac = {3: 1 - y, 11: y}
t_states = pd.DataFrame([{
    "State": k, "Description": desc[k],
    "Mass fraction (-)": r(frac.get(k, 1.0)), "Flow (kg/s)": r(frac.get(k, 1.0) * m, 2),
    "T (C)": r(s["T"] - K, 2), "P (MPa)": r(s["p"] / 1e6),
    "h (kJ/kg)": r(s["h"] / 1e3, 3), "s (kJ/kg.K)": r(s["s"] / 1e3, 5),
    "ex (kJ/kg)": r(fe[k] / 1e3, 3)} for k, s in sorted(st.items())])

# ---------------- 03: components and exergy ----------------------------------
gb, gen, mot = PARASITIC["eta_gearbox"], PARASITIC["eta_generator"], PARASITIC["eta_motor"]
cycle_rows = [
    ("Main heater", "All gas heat goes to the CO2", 100.0),
    ("Branch heater", "All gas heat goes to the CO2", 100.0),
    ("HTR", "Effectiveness (enthalpy-based)", D0["eps_htr"] * 100),
    ("LTR", "Effectiveness (enthalpy-based)", D0["eps_ltr"] * 100),
    ("Mixer", "Adiabatic mixing (energy balance)", 100.0),
    ("Turbine", "Isentropic efficiency", D0["eta_t"] * 100),
    ("Main compressor", "Isentropic efficiency", D0["eta_mc"] * 100),
    ("Cooler", "Heat rejected to the environment (exergy loss)", None)]
parasitic_rows = [
    ("Gearboxes", "Efficiency of each gearbox", gb * 100, gb * 100),
    ("Generator", "Efficiency", gen * 100, gen * 100),
    ("Compressor motor", "Efficiency", mot * 100, mot * 100),
    ("Cooling tower fan and pump", "Fan and pump power (work counted as lost)", None, None),
    ("Off-gas fan", "Fan power (work counted as lost)", None, None)]


def row(name, metric, e_eff, x_eff):
    d = D[name]
    return {"Component": name, "Energy performance metric": metric,
            "Energy efficiency (%)": r(e_eff, 2) if e_eff is not None else "n/a",
            "Exergy efficiency (%)": r(x_eff, 2) if x_eff is not None else "n/a",
            "Exergy destruction (MW)": r(d / 1e6),
            "Share of gas exergy (%)": r(d / ex_in * 100, 2),
            "Share of total losses (%)": r(d / losses * 100, 2)}


rows = []
for name, metric, e_eff in cycle_rows:
    xe = eff[name]
    rows.append(row(name, metric, e_eff, xe * 100 if xe is not None else None))
for name, metric, e_eff, x_eff in parasitic_rows:
    if D[name] > 0:
        rows.append(row(name, metric, e_eff, x_eff))
rows.append({"Component": "Stack gas (exergy leaving with the exhaust)",
             "Energy performance metric": "-", "Energy efficiency (%)": "n/a",
             "Exergy efficiency (%)": "n/a",
             "Exergy destruction (MW)": r(stack / 1e6),
             "Share of gas exergy (%)": r(stack / ex_in * 100, 2),
             "Share of total losses (%)": r(stack / losses * 100, 2)})
rows.append({"Component": "Total destruction and losses (incl. stack)",
             "Energy performance metric": "-", "Energy efficiency (%)": "",
             "Exergy efficiency (%)": "", "Exergy destruction (MW)": r(losses / 1e6),
             "Share of gas exergy (%)": r(losses / ex_in * 100, 2),
             "Share of total losses (%)": 100.0})
rows.append({"Component": "Net electrical power (product)",
             "Energy performance metric": "-", "Energy efficiency (%)": "",
             "Exergy efficiency (%)": "", "Exergy destruction (MW)": r(net_el / 1e6),
             "Share of gas exergy (%)": r(net_el / ex_in * 100, 2),
             "Share of total losses (%)": ""})
t_comp = pd.DataFrame(rows)

# ---------------- 33: exergy efficiencies of the plant -----------------------
t_eff = pd.DataFrame([
    ("Physical exergy of the gas (fuel)", ex_in / 1e6, "MW", "Gas at the heater inlet, dead state 25 C, 101.3 kPa"),
    ("Exergy leaving with the stack gas", stack / 1e6, "MW", "Loss"),
    ("Exergy taken from the gas", (ex_in - stack) / 1e6, "MW", "Fuel minus stack"),
    ("Exergy delivered to the CO2", ex["ex_to_co2_W"] / 1e6, "MW", "Main plus branch heater"),
    ("Net electrical power (product)", net_el / 1e6, "MW", ""),
    ("System exergy efficiency", ex["eta_system"] * 100, "%", "Net electrical / gas exergy; proportional to net power for a given gas"),
    ("Fuel-product exergy efficiency", ex["eta_fuel_product"] * 100, "%", "Net electrical / (gas exergy - stack exergy)"),
    ("Cycle exergy efficiency", ex["eta_cycle"] * 100, "%", "Net electrical / exergy delivered to the CO2"),
    ("Chemical exergy of the CO (not recovered)", ex["ex_chem_CO_W"] / 1e6, "MW", "Szargut standard exergy, dilution included"),
    ("Total exergy efficiency incl. CO chemical exergy", ex["eta_total"] * 100, "%", "Net electrical / (physical + chemical)"),
], columns=["Quantity", "Value", "Unit", "Definition"])
t_eff["Value"] = t_eff["Value"].round(3)

t_states.to_csv(T_OUT / "02_states.csv", index=False, encoding="utf-8-sig")
t_comp.to_csv(T_OUT / "03_components_exergy.csv", index=False, encoding="utf-8-sig")
t_eff.to_csv(T_OUT / "33_exergy_efficiencies.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 230)
pd.set_option("display.max_colwidth", 80)
print(t_states.to_string(index=False))
print()
print(t_comp.to_string(index=False))
print()
print(t_eff.to_string(index=False))
print(f"\nExergy balance check      : {ex['balance_W'] / 1e6:.6f} MW (should be 0)")
print(f"Parasitic losses (5 rows) : "
      f"{sum(D[k] for k in ('Gearboxes', 'Generator', 'Compressor motor', 'Cooling tower fan and pump', 'Off-gas fan')) / 1e6:.4f} MW")

# ---------------- figures ---------------------------------------------------
plot_ts_splitflow(res, save_path=F_OUT / "ts_diagram.png")
plt.close("all")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, name in zip(axes[0], ("HTR", "LTR")):
    p = res["pinch"][name]
    ax.plot(p["Q"] * m / 1e6, p["T_hot"] - K, "r-", label="Hot side")
    ax.plot(p["Q"] * m / 1e6, p["T_cold"] - K, "b-", label="Cold side")
    ax.set_title(f"{name} (min dT = {p['min_dT']:.1f} K)")
for ax, (title, Q, Tg, Tc) in zip(axes[1], [
        ("Main heater", hs["Q1"], hs["T_gas1"], hs["T_co2_1"]),
        ("Branch heater", hs["Q2"], hs["T_gas2"], hs["T_co2_2"])]):
    ax.plot(hs["f"] * Q / 1e6, Tg - K, "r-", label="EAF gas")
    ax.plot(hs["f"] * Q / 1e6, Tc - K, "b-", label="CO2")
    ax.set_title(title)
for ax in axes.flat:
    ax.set_xlabel("Heat transferred (MW)")
    ax.set_ylabel("Temperature (C)")
    ax.grid(True)
    ax.legend()
fig.tight_layout()
fig.savefig(F_OUT / "heat_exchangers_QT.png", dpi=150)
plt.close(fig)

parasitic_names = {"Gearboxes", "Generator", "Compressor motor",
                   "Cooling tower fan and pump", "Off-gas fan"}
items = [(k, v) for k, v in D.items() if v > 0] + [("Stack gas loss", stack)]
items.sort(key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["tab:gray" if k in parasitic_names else
          ("tab:orange" if k == "Stack gas loss" else "tab:blue") for k, _ in items]
ax.barh([k for k, _ in items], [v / 1e6 for _, v in items], color=colors)
ax.set_xlabel(f"Exergy destruction or loss (MW); net electrical power = {net_el / 1e6:.2f} MW "
              f"of {ex_in / 1e6:.2f} MW gas exergy")
ax.grid(True, axis="x")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in
                   ("tab:blue", "tab:gray", "tab:orange")],
          labels=["Cycle components and cooler", "Parasitic loads", "Stack gas"],
          loc="lower right")
fig.tight_layout()
fig.savefig(F_OUT / "exergy_destruction.png", dpi=150)
plt.close(fig)

print(f"\nSaved: {T_OUT}/02_states.csv, 03_components_exergy.csv, 33_exergy_efficiencies.csv; "
      f"{F_OUT}/ts_diagram.png, heat_exchangers_QT.png, exergy_destruction.png")
