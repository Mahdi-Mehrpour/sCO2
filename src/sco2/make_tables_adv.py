"""Build the result tables of the advanced cycle (CSV + one Excel file).
Output folder: tables/adv/"""
import inspect
from pathlib import Path

import pandas as pd

from sco2.cycle_adv import recompression_adv
from sco2.exergy_adv import exergy_analysis_adv

OUT = Path("tables/adv")
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Inputs: defaults are read from the functions themselves.
# To run a different case, add overrides here, e.g. {"x": 0.25, "pr": 2.8}
# ----------------------------------------------------------------------
OVERRIDES = {}

P = {k: v.default for k, v in inspect.signature(recompression_adv).parameters.items()}
P.update(OVERRIDES)
E = {k: v.default for k, v in inspect.signature(exergy_analysis_adv).parameters.items()
     if k != "res"}

res = recompression_adv(**P)
ex = exergy_analysis_adv(res, fluid=P["fluid"], T0=E["T0"], p0=E["p0"])
fe = ex["flow_exergy"]
st = res["states"]
x = res["x"]
dest = ex["destruction"]
ex_heat = ex["ex_heat"]
K = 273.15

# ----------------------------------------------------------------------
# Table 1: thermodynamic properties of each state
# ----------------------------------------------------------------------
desc = {1: "Main compressor inlet", 2: "Main compressor outlet",
        3: "LTR cold outlet", 4: "Recompressor outlet",
        5: "Mixer outlet (HTR cold inlet)", 6: "HTR cold outlet",
        7: "Turbine inlet (heater outlet)", 8: "Turbine outlet",
        9: "HTR hot outlet (LTR hot inlet)", 10: "LTR hot outlet (split point)"}
frac = {1: 1 - x, 2: 1 - x, 3: 1 - x, 4: x}

t1 = pd.DataFrame([{
    "State": k,
    "Description": desc[k],
    "Mass fraction (-)": round(frac.get(k, 1.0), 4),
    "T (K)": round(s["T"], 2),
    "T (C)": round(s["T"] - K, 2),
    "P (MPa)": round(s["p"] / 1e6, 4),
    "h (kJ/kg)": round(s["h"] / 1e3, 3),
    "s (kJ/kg.K)": round(s["s"] / 1e3, 5),
    "ex (kJ/kg)": round(fe[k] / 1e3, 3),
} for k, s in st.items()])

# ----------------------------------------------------------------------
# Table 2: component performance
# Exergy efficiency = exergy gained by the product side divided by the
# exergy supplied (fuel). All flows are weighted by the mass fractions.
# ----------------------------------------------------------------------
NA = "n/a"
comp = [
    # name, energy metric, energy eff (%), exergy eff (%)
    ("Main compressor", "Isentropic efficiency", P["eta_mc"] * 100,
     (fe[2] - fe[1]) / res["w_mc"] * 100),
    ("Recompressor", "Isentropic efficiency", P["eta_rc"] * 100,
     (fe[4] - fe[10]) / res["w_rc"] * 100),
    ("Turbine", "Isentropic efficiency", P["eta_t"] * 100,
     res["w_t"] / (fe[7] - fe[8]) * 100),
    ("HTR", "Effectiveness (enthalpy-based)", P["eps_htr"] * 100,
     (fe[6] - fe[5]) / (fe[8] - fe[9]) * 100),
    ("LTR", "Effectiveness (enthalpy-based)", P["eps_ltr"] * 100,
     (1 - x) * (fe[3] - fe[2]) / (fe[9] - fe[10]) * 100),
    ("Mixer", "Adiabatic mixing (energy balance)", 100.0,
     fe[5] / ((1 - x) * fe[3] + x * fe[4]) * 100),
    ("Heater", "All source heat goes to the CO2 (no heat loss)", 100.0,
     (fe[7] - fe[6]) / ex_heat * 100),
    ("Cooler", "Heat rejected to the environment", NA, NA),
]
total_d = ex["total"]
rows = []
for name, metric, e_eff, x_eff in comp:
    d = dest[name]
    rows.append({
        "Component": name if name != "Cooler" else "Cooler (exergy loss to environment)",
        "Energy performance metric": metric,
        "Energy efficiency (%)": round(e_eff, 2) if e_eff != NA else NA,
        "Exergy efficiency (%)": round(x_eff, 2) if x_eff != NA else NA,
        "Exergy destruction (kJ/kg)": round(d / 1e3, 3),
        "Share of heat exergy (%)": round(d / ex_heat * 100, 2),
        "Share of total destruction (%)": round(d / total_d * 100, 2),
    })
t2 = pd.DataFrame(rows)

# ----------------------------------------------------------------------
# Table 3: assumptions
# ----------------------------------------------------------------------
A = []


def add(cat, text, value="", unit=""):
    A.append({"Category": cat, "Assumption": text, "Value": value, "Unit": unit})


add("General", "Steady-state, steady-flow operation")
add("General", "Kinetic and potential energy changes are neglected")
add("General", "All flows are per 1 kg of CO2 through the turbine")
add("General", "No leakage, no mechanical, generator or motor losses")
add("General", "All components are adiabatic (no heat loss to surroundings)")
add("Fluid", "Working fluid with real-gas properties (CoolProp, Helmholtz EOS)", P["fluid"])
add("Operating", "Main compressor inlet temperature (state 1)", round(P["T1"] - K, 2), "C")
add("Operating", "Main compressor inlet pressure (state 1)", P["p1"] / 1e6, "MPa")
add("Operating", "Main compressor pressure ratio p2/p1", P["pr"], "-")
add("Operating", "Turbine inlet temperature (state 7)", round(P["T7"] - K, 2), "C")
add("Operating", "Recompression fraction x", P["x"], "-")
add("Components", "Main compressor isentropic efficiency", P["eta_mc"], "-")
add("Components", "Recompressor isentropic efficiency", P["eta_rc"], "-")
add("Components", "Turbine isentropic efficiency", P["eta_t"], "-")
add("Components", "LTR effectiveness (enthalpy-based, at inlet pressures)", P["eps_ltr"], "-")
add("Components", "HTR effectiveness (enthalpy-based, at inlet pressures)", P["eps_htr"], "-")
add("Components", "Mixing is adiabatic, at equal pressure (p4 = p3)")
add("Components", "Recuperator pinch check: counterflow profile split into segments, pressure linear along each side", P["n_seg"], "segments")
if P["dT_pinch"] is None:
    add("Components", "Recuperator heat duty set by effectiveness only; pinch is reported, not enforced")
else:
    add("Components", "Recuperator heat duty limited so that the minimum temperature difference is at least", P["dT_pinch"], "K")
for label, key in [("Heater", "dp_heater"), ("Cooler", "dp_cooler"),
                   ("LTR cold side", "dp_ltr_cold"), ("LTR hot side", "dp_ltr_hot"),
                   ("HTR cold side", "dp_htr_cold"), ("HTR hot side", "dp_htr_hot")]:
    add("Pressure drops", f"{label} (fraction of inlet pressure)", P[key] * 100, "%")
add("Exergy", "Dead state temperature T0", E["T0"] - K, "C")
add("Exergy", "Dead state pressure p0", E["p0"] / 1e3, "kPa")
add("Exergy", "Only physical (thermo-mechanical) exergy is considered; no chemical exergy")
add("Exergy", "Heat source at constant temperature T_source = turbine inlet temperature",
    round(ex["T_source"] - K, 2), "C")
add("Exergy", "Cooler: heat is rejected at T0, so the exergy of the cooled stream is lost completely")
add("Numerics", "Convergence tolerance on h of state 5 (recuperator loop)", 1e-3, "J/kg")
t3 = pd.DataFrame(A)

# ----------------------------------------------------------------------
# Table 4: cycle performance
# ----------------------------------------------------------------------
bal_en = res["q_in"] - res["w_net"] - res["q_out"]
bal_ex = ex_heat - res["w_net"] - total_d
d_no_cooler = total_d - dest["Cooler"]
perf = [
    ("Net specific work", res["w_net"] / 1e3, "kJ/kg"),
    ("Heat input (heater)", res["q_in"] / 1e3, "kJ/kg"),
    ("Heat rejected (cooler)", res["q_out"] / 1e3, "kJ/kg"),
    ("Exergy of heat input", ex_heat / 1e3, "kJ/kg"),
    ("Total energy (thermal) efficiency", res["eta"] * 100, "%"),
    ("Total exergy efficiency", ex["eta_ex"] * 100, "%"),
    ("Total exergy destruction (incl. cooler loss)", total_d / 1e3, "kJ/kg"),
    ("Total exergy destruction (excl. cooler loss)", d_no_cooler / 1e3, "kJ/kg"),
    ("Total destruction / heat exergy (incl. cooler)", total_d / ex_heat * 100, "%"),
    ("Compressor pressure ratio", P["pr"], "-"),
    ("Turbine pressure ratio", res["pr_turbine"], "-"),
    ("HTR minimum temperature difference (pinch)", res["pinch"]["HTR"]["min_dT"], "K"),
    ("LTR minimum temperature difference (pinch)", res["pinch"]["LTR"]["min_dT"], "K"),
    ("HTR UA per kg/s of turbine flow", res["pinch"]["HTR"]["UA"] / 1e3, "kJ/kg.K"),
    ("LTR UA per kg/s of turbine flow", res["pinch"]["LTR"]["UA"] / 1e3, "kJ/kg.K"),
    ("Energy balance check (q_in - w_net - q_out)", bal_en, "J/kg"),
    ("Exergy balance check (ex_heat - w_net - destruction)", bal_ex, "J/kg"),
]
t4 = pd.DataFrame([{"Parameter": a, "Value": round(b, 4), "Unit": c} for a, b, c in perf])

# ----------------------------------------------------------------------
# Save: CSV (utf-8-sig so Excel opens it correctly) + one Excel workbook
# ----------------------------------------------------------------------
tables = {"1_states": t1, "2_components": t2, "3_assumptions": t3,
          "4_cycle_performance": t4}
for name, df in tables.items():
    df.to_csv(OUT / f"table{name}.csv", index=False, encoding="utf-8-sig")

with pd.ExcelWriter(OUT / "cycle_adv_tables.xlsx") as writer:
    for name, df in tables.items():
        sheet = name.split("_", 1)[1][:31]
        df.to_excel(writer, sheet_name=sheet, index=False)
        ws = writer.sheets[sheet]
        for i, col in enumerate(df.columns):
            width = max(len(str(col)), *(len(str(v)) for v in df[col])) + 2
            ws.column_dimensions[chr(65 + i)].width = min(width, 70)

for name, df in tables.items():
    print(f"\n=== Table {name} ===")
    print(df.to_string(index=False))
print(f"\nSaved in {OUT}/")