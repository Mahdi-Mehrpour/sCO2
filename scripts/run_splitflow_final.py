import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sco2.cycle_adv import splitflow_adv
from sco2.heat_source_adv import EAF, GasStream, couple_cascade
from sco2.exergy_splitflow import exergy_splitflow
from sco2.plotting_splitflow import plot_ts_splitflow

K = 273.15
T_OUT, F_OUT = Path("tables/splitflow"), Path("figures/splitflow")
T_OUT.mkdir(parents=True, exist_ok=True)
F_OUT.mkdir(parents=True, exist_ok=True)

# ---------------- Inputs: defaults of splitflow_adv + saved optimum ---------
with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
P = {k: v.default for k, v in inspect.signature(splitflow_adv).parameters.items()}
P.update({"y": opt["y"], "T_b": opt["T_b"], "pr": opt["pr"]})

gas = GasStream(EAF["X"], EAF["p"], T0=298.15)
res = splitflow_adv(**P)
hs = couple_cascade(res, gas, EAF["T_in"], EAF["m_gas"], EAF["dT_pinch_heater"],
                    EAF["dT_pinch_heater"], EAF["T_stack_min"])
ex = exergy_splitflow(res, hs, gas, EAF["m_gas"])
fe, dW, eff = ex["flow_exergy"], ex["destruction_W"], ex["exergy_eff"]
st, y, m = res["states"], res["y"], hs["m_co2"]
ex_in, W = hs["ex_in"], ex["W"]
total_d = sum(dW.values())


def r(v, n=3):
    return round(float(v), n) if isinstance(v, (int, float, np.floating)) else v


# ---------------- Table 1: states ------------------------------------------
desc = {1: "Main compressor inlet", 2: "Main compressor outlet (split point)",
        3: "LTR cold outlet", 5: "Mixer outlet (HTR cold inlet)",
        6: "HTR cold outlet", 7: "Turbine inlet (main heater outlet)",
        8: "Turbine outlet", 9: "HTR hot outlet", 10: "LTR hot outlet (cooler inlet)",
        11: "Branch heater outlet"}
frac = {3: 1 - y, 11: y}
t1 = pd.DataFrame([{
    "State": k, "Description": desc[k],
    "Mass fraction (-)": r(frac.get(k, 1.0), 4),
    "Flow (kg/s)": r(frac.get(k, 1.0) * m, 2),
    "T (C)": r(s["T"] - K, 2), "P (MPa)": r(s["p"] / 1e6, 4),
    "h (kJ/kg)": r(s["h"] / 1e3), "s (kJ/kg.K)": r(s["s"] / 1e3, 5),
    "ex (kJ/kg)": r(fe[k] / 1e3)} for k, s in sorted(st.items())])

# ---------------- Table 2: components --------------------------------------
comp = [
    ("Main heater", "All gas heat goes to the CO2", 100.0),
    ("Branch heater", "All gas heat goes to the CO2", 100.0),
    ("HTR", "Effectiveness (enthalpy-based)", P["eps_htr"] * 100),
    ("LTR", "Effectiveness (enthalpy-based)", P["eps_ltr"] * 100),
    ("Mixer", "Adiabatic mixing (energy balance)", 100.0),
    ("Turbine", "Isentropic efficiency", P["eta_t"] * 100),
    ("Main compressor", "Isentropic efficiency", P["eta_mc"] * 100),
    ("Cooler", "Heat rejected to the environment", None),
]
loss_total = total_d + ex["stack_W"]
rows = []
for name, metric, e_eff in comp:
    xe = eff[name]
    rows.append({
        "Component": name if name != "Cooler" else "Cooler (exergy loss to environment)",
        "Energy performance metric": metric,
        "Energy efficiency (%)": r(e_eff, 2) if e_eff is not None else "n/a",
        "Exergy efficiency (%)": r(xe * 100, 2) if xe is not None else "n/a",
        "Exergy destruction (MW)": r(dW[name] / 1e6, 4),
        "Share of gas exergy (%)": r(dW[name] / ex_in * 100, 2),
        "Share of total losses (%)": r(dW[name] / loss_total * 100, 2)})
rows.append({"Component": "Stack gas (exergy leaving with the exhaust)",
             "Energy performance metric": "-", "Energy efficiency (%)": "n/a",
             "Exergy efficiency (%)": "n/a",
             "Exergy destruction (MW)": r(ex["stack_W"] / 1e6, 4),
             "Share of gas exergy (%)": r(ex["stack_W"] / ex_in * 100, 2),
             "Share of total losses (%)": r(ex["stack_W"] / loss_total * 100, 2)})
t2 = pd.DataFrame(rows)

# ---------------- Table 3: assumptions -------------------------------------
A = []


def add(cat, text, value="", unit=""):
    A.append({"Category": cat, "Assumption": text, "Value": value, "Unit": unit})


add("General", "Steady-state, steady-flow operation")
add("General", "Kinetic and potential energy changes are neglected")
add("General", "No leakage, no mechanical, generator or motor losses; all components adiabatic")
add("General", "Cycle quantities are per 1 kg of CO2 through the turbine; absolute values use the CO2 flow from the heat-source coupling")
add("Heat source", "Electric arc furnace off-gas, constant conditions (values are assumptions, not measured data)")
add("Heat source", "Gas inlet temperature", EAF["T_in"] - K, "C")
add("Heat source", "Gas mass flow", EAF["m_gas"], "kg/s")
add("Heat source", "Gas pressure (no pressure drop on the gas side)", EAF["p"] / 1e3, "kPa")
add("Heat source", "Gas composition (mole fractions)", ", ".join(f"{k} {v}" for k, v in EAF["X"].items()))
add("Heat source", "Ideal-gas mixture, NASA polynomials (Cantera gri30); dust and chemical exergy of CO neglected")
add("Heat source", "Minimum gas outlet (stack) temperature", EAF["T_stack_min"] - K, "C")
add("Heat source", "Minimum gas-to-CO2 temperature difference in each heater", EAF["dT_pinch_heater"], "K")
add("Heat source", "Gas passes the main heater first, then the branch heater")
add("Fluid", "CO2 with real-gas properties (CoolProp, Helmholtz EOS)", P["fluid"])
add("Operating", "Main compressor inlet temperature (state 1)", r(P["T1"] - K, 2), "C")
add("Operating", "Main compressor inlet pressure (state 1)", P["p1"] / 1e6, "MPa")
add("Operating", "Compressor pressure ratio (optimised, at the pressure limit)", r(P["pr"], 4), "-")
add("Operating", "Maximum cycle pressure (design limit)", opt["p_max_limit_Pa"] / 1e6, "MPa")
add("Operating", "Turbine inlet temperature (state 7)", r(P["T7"] - K, 2), "C")
add("Operating", "Branch fraction y (optimised, flat optimum)", r(P["y"], 4), "-")
add("Operating", "Branch heater outlet temperature (state 11, optimised, flat optimum)", r(P["T_b"] - K, 2), "C")
add("Operating", "No recompressor: x = 0 in the optimum of the cascade layout with recompression (800 C gas)")
add("Components", "Main compressor isentropic efficiency", P["eta_mc"], "-")
add("Components", "Turbine isentropic efficiency", P["eta_t"], "-")
add("Components", "LTR and HTR effectiveness (enthalpy-based, at inlet pressures)", P["eps_ltr"], "-")
add("Components", "Minimum temperature difference enforced in HTR and LTR during optimisation", opt["dT_recuperator_min"], "K")
add("Components", "Recuperator pinch check: counterflow profile with segments, pressure linear on each side", P["n_seg"], "segments")
add("Components", "Branch rejoins at the mixer without throttling (branch heater drop = LTR cold side drop)")
for label, key in [("Main heater", "dp_heater"), ("Cooler", "dp_cooler"),
                   ("LTR cold side", "dp_ltr_cold"), ("LTR hot side", "dp_ltr_hot"),
                   ("HTR cold side", "dp_htr_cold"), ("HTR hot side", "dp_htr_hot")]:
    add("Pressure drops", f"{label} (fraction of inlet pressure)", P[key] * 100, "%")
add("Exergy", "Dead state temperature and pressure", "25 C, 101.325 kPa")
add("Exergy", "Only physical (thermo-mechanical) exergy is considered")
add("Exergy", "Cooler: heat is rejected at T0, so the exergy of the cooled stream is lost completely")
add("Numerics", "Convergence tolerance on h of state 5", 1e-3, "J/kg")
t3 = pd.DataFrame(A)

# ---------------- Table 4: performance -------------------------------------
bal_en = res["q_in"] + res["q_b"] - res["w_net"] - res["q_out"]
perf = [
    ("Gas inlet temperature", EAF["T_in"] - K, "C"),
    ("Gas mass flow", EAF["m_gas"], "kg/s"),
    ("Gas stack temperature", hs["T_out"] - K, "C"),
    ("Gas temperature between the heaters", hs["T_gas_mid"] - K, "C"),
    ("Heat available down to the minimum stack temperature", hs["Q_max"] / 1e6, "MW"),
    ("Heat delivered, main heater", hs["Q1"] / 1e6, "MW"),
    ("Heat delivered, branch heater", hs["Q2"] / 1e6, "MW"),
    ("Share of available heat used", hs["Q"] / hs["Q_max"] * 100, "%"),
    ("CO2 mass flow (turbine)", m, "kg/s"),
    ("Net power", W / 1e6, "MW"),
    ("Cycle thermal efficiency (net power / heat delivered)", res["eta"] * 100, "%"),
    ("Heat recovery efficiency (net power / available heat)", W / hs["Q_max"] * 100, "%"),
    ("Exergy of gas at inlet", ex_in / 1e6, "MW"),
    ("System exergy efficiency (net power / gas exergy)", W / ex_in * 100, "%"),
    ("Cycle exergy efficiency (net power / exergy gained by the CO2)", W / ex["ex_to_co2_W"] * 100, "%"),
    ("Exergy destruction, heaters", (dW["Main heater"] + dW["Branch heater"]) / 1e6, "MW"),
    ("Exergy destruction, cycle components (HTR, LTR, mixer, turbine, compressor)",
     (dW["HTR"] + dW["LTR"] + dW["Mixer"] + dW["Turbine"] + dW["Main compressor"]) / 1e6, "MW"),
    ("Exergy loss in the cooler", dW["Cooler"] / 1e6, "MW"),
    ("Total exergy destruction (heaters, cycle, cooler)", total_d / 1e6, "MW"),
    ("Exergy lost with the stack gas", ex["stack_W"] / 1e6, "MW"),
    ("Total exergy losses (destruction + stack)", loss_total / 1e6, "MW"),
    ("Compressor pressure ratio", P["pr"], "-"),
    ("Turbine pressure ratio", res["pr_turbine"], "-"),
    ("Main heater minimum temperature difference", hs["min_dT1"], "K"),
    ("Branch heater minimum temperature difference", hs["min_dT2"], "K"),
    ("HTR minimum temperature difference", res["pinch"]["HTR"]["min_dT"], "K"),
    ("LTR minimum temperature difference", res["pinch"]["LTR"]["min_dT"], "K"),
    ("HTR UA per kg/s of turbine flow", res["pinch"]["HTR"]["UA"] / 1e3, "kJ/kg.K"),
    ("LTR UA per kg/s of turbine flow", res["pinch"]["LTR"]["UA"] / 1e3, "kJ/kg.K"),
    ("Energy balance check (q_in + q_b - w_net - q_out)", bal_en, "J/kg"),
    ("Exergy balance check (system)", ex["balance_W"] / 1e6, "MW"),
]
t4 = pd.DataFrame([{"Parameter": a, "Value": r(b, 4), "Unit": c} for a, b, c in perf])
t4.loc[len(t4)] = ["Limiting constraint of the gas coupling", hs["binding"], ""]

# ---------------- Save tables ----------------------------------------------
tables = {"1_states": t1, "2_components": t2, "3_assumptions": t3,
          "4_performance": t4}
for name, df in tables.items():
    df.to_csv(T_OUT / f"table{name}.csv", index=False, encoding="utf-8-sig")
with pd.ExcelWriter(T_OUT / "splitflow_tables.xlsx") as writer:
    for name, df in tables.items():
        sheet = name.split("_", 1)[1][:31]
        df.to_excel(writer, sheet_name=sheet, index=False)
        ws = writer.sheets[sheet]
        for i, col in enumerate(df.columns):
            width = max(len(str(col)), *(len(str(v)) for v in df[col])) + 2
            ws.column_dimensions[chr(65 + i)].width = min(width, 70)

for name, df in tables.items():
    if name != "3_assumptions":
        print(f"\n=== Table {name} ===")
        print(df.to_string(index=False))

# ---------------- Figures ---------------------------------------------------
plot_ts_splitflow(res, save_path=F_OUT / "ts_diagram.png")

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
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True)
    ax.legend()
fig.tight_layout()
fig.savefig(F_OUT / "heat_exchangers_QT.png", dpi=150)

items = {k: v for k, v in dW.items()}
items["Stack gas loss"] = ex["stack_W"]
items = sorted(items.items(), key=lambda kv: kv[1], reverse=True)
plt.figure(figsize=(9, 5))
plt.bar([k for k, _ in items], [v / 1e6 for _, v in items])
plt.ylabel("Exergy destruction / loss (MW)")
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(F_OUT / "exergy_destruction.png", dpi=150)

print(f"\nSaved tables in {T_OUT}/ and figures in {F_OUT}/")