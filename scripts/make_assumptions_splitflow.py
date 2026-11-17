"""Builds tables/01_assumptions.csv: every assumption of the model, read from
the code, with a note on where each value comes from."""
import inspect
import json
from pathlib import Path
import pandas as pd

from sco2.cycle_adv import splitflow_adv
from sco2.heat_source_adv import EAF
from sco2.parasitics_splitflow import PARASITIC
from sco2.cooling_splitflow import COOLING
from sco2.operation_splitflow import OPERATION
from sco2.plant_splitflow import evaluate_plant, DRIVE
from sco2.economics_eaf import run
from sco2.economics_splitflow import build_inputs
from sco2.settings_splitflow import (ECON_BASE, LIMITS, HEATER_POSITION,
                                     NOT_MODELLED)

K = 273.15
PH, OPT, CHO, REF = "placeholder", "optimised", "design choice", "reference"

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
D = {k: v.default for k, v in inspect.signature(splitflow_adv).parameters.items()}
P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"])
ci, e = build_inputs(P, ECON_BASE)
R = run(ci, e)
st, W = P["res"]["states"], COOLING["wet_tower"]

A = []


def add(cat, text, value="", unit="", note=""):
    if isinstance(value, float):
        value = round(value, 4)
    A.append({"Category": cat, "Assumption": text, "Value": value,
              "Unit": unit, "Note": note})


add("General", "Steady-state, steady-flow operation; no dynamics", note="model scope")
add("General", "Kinetic and potential energy changes are neglected", note="model scope")
add("General", "Cycle quantities are per 1 kg of CO2 through the turbine; absolute values use the CO2 flow from the heat-source coupling", note="model scope")
add("General", "All cycle components are adiabatic; no leakage", note="model scope")

add("Heat source", HEATER_POSITION, note=CHO)
add("Heat source", "Gas inlet temperature", EAF["T_in"] - K, "C", PH)
add("Heat source", "Gas mass flow", EAF["m_gas"], "kg/s", PH)
add("Heat source", "Gas pressure (no pressure drop in the thermodynamic model)", EAF["p"] / 1e3, "kPa", PH)
add("Heat source", "Gas composition (mole fractions)", ", ".join(f"{k} {v}" for k, v in EAF["X"].items()), "", PH)
add("Heat source", "Ideal-gas mixture, NASA polynomials (Cantera gri30); dust and chemical exergy of CO neglected", note="model")
add("Heat source", "Minimum gas outlet (stack) temperature", EAF["T_stack_min"] - K, "C", PH)
add("Heat source", "Minimum gas-to-CO2 temperature difference in each heater", EAF["dT_pinch_heater"], "K", PH)
add("Heat source", "Gas passes the main heater first, then the branch heater", note=CHO)

add("Fluid", "CO2 with real-gas properties (CoolProp, Helmholtz EOS)", D["fluid"], note="model")
add("Cycle", "Layout: split-flow without recompressor; branch taken after the main compressor and rejoining at the mixer", note=CHO)
add("Cycle", "Compressor inlet temperature (set by the wet tower, at the floor of critical temperature + 1 K)", P["T1"] - K, "C", CHO)
add("Cycle", "Compressor inlet pressure (fixed)", D["p1"] / 1e6, "MPa", CHO)
add("Cycle", "Compressor pressure ratio", opt["pr"], "-", "optimised, at the pressure limit")
add("Cycle", "Turbine inlet temperature (fixed)", D["T7"] - K, "C", CHO)
add("Cycle", "Branch fraction y", opt["y"], "-", "optimised (flat optimum)")
add("Cycle", "Branch heater outlet temperature", opt["T_b"] - K, "C", "optimised (flat optimum)")
add("Cycle", "Main compressor isentropic efficiency", D["eta_mc"], "-", PH)
add("Cycle", "Turbine isentropic efficiency", D["eta_t"], "-", PH)
add("Cycle", "LTR and HTR effectiveness (enthalpy-based, at inlet pressures)", D["eps_ltr"], "-", PH)
add("Cycle", "Minimum temperature difference in HTR and LTR enforced in the optimisation", opt["dT_recuperator_min"], "K", CHO)
add("Cycle", "Branch heater pressure drop equals the LTR cold side drop (no throttling at the mixer)", note=CHO)
for label, key in [("Main heater", "dp_heater"), ("Cooler", "dp_cooler"),
                   ("LTR cold side", "dp_ltr_cold"), ("LTR hot side", "dp_ltr_hot"),
                   ("HTR cold side", "dp_htr_cold"), ("HTR hot side", "dp_htr_hot")]:
    add("Pressure drops", f"{label} (fraction of inlet pressure)", D[key] * 100, "%", PH)

add("Cooling", f"Cooling system: {W['label']}", note=CHO)
add("Cooling", "Design wet-bulb temperature", W["T_wb_C"], "C", PH)
add("Cooling", "Tower approach (cooling water minus wet bulb)", W["tower_approach"], "K", PH)
add("Cooling", "Heat exchanger approach (CO2 outlet minus cooling water inlet)", W["approach"], "K", PH)
add("Cooling", "Cooling water temperature rise", W["coolant_dT"], "K", PH)
add("Cooling", "Tower fan and pump power as a fraction of the cooler duty", W["fan_pump_fraction"] * 100, "%", PH)
add("Cooling", "Makeup water factor (makeup / evaporation)", W["makeup_factor"], "-", PH)
add("Cooling", "Lowest compressor inlet temperature allowed (critical temperature + 1 K)", COOLING["T1_floor_C"], "C", CHO)

add("Parasitics", "Compressor driven by an electric motor", DRIVE, "", CHO)
add("Parasitics", "Gearbox efficiency", PARASITIC["eta_gearbox"], "-", PH)
add("Parasitics", "Generator efficiency", PARASITIC["eta_generator"], "-", PH)
add("Parasitics", "Compressor motor efficiency", PARASITIC["eta_motor"], "-", PH)
add("Parasitics", "Gas-side pressure drop for the off-gas fan (heaters and ducts)", PARASITIC["gas_dp_total"], "Pa", PH)
add("Parasitics", "Off-gas fan efficiency", PARASITIC["eta_fan_gas"], "-", PH)

add("Operation", "Furnace operating days per year", OPERATION["days_per_year"], "days", PH)
add("Operation", "Furnace operating hours per day", OPERATION["hours_per_day"], "h", PH)
add("Operation", "Availability of the sCO2 unit", OPERATION["plant_availability"], "-", PH)
add("Operation", "Stable gas condition all year; the fluctuation phases are not used", note=CHO)

actual = {"p_max": max(s["p"] for s in st.values()) / 1e6,
          "T7": st[7]["T"] - K, "T8": st[8]["T"] - K}
for L in LIMITS:
    note = f"supplied list, not verified; {L['kind']}; {L['basis']}; {L['enforced']}"
    if L["key"]:
        a = actual[L["key"]]
        margin = L["value"] - a
        note += (f"; model value {a:.1f} {L['unit']}, margin {margin:+.1f} {L['unit']}"
                 + (" VIOLATED" if margin < 0 else ""))
    add("Limits", L["name"], L["value"], L["unit"], note)

desc = {"cepci_target": ("Cost year index (CEPCI of the cost year)", "-", REF),
        "f_inst": ("Installation and auxiliaries, fraction of purchased cost", "-", REF),
        "r": ("Discount rate", "1/yr", "base case, chosen"),
        "N": ("Economic life", "yr", "base case, chosen"),
        "hours": ("Operating hours (calendar days x 24 h x availability)", "h/yr", PH),
        "c_e": ("Value of the electricity (avoided purchase price)", "USD/kWh", "placeholder: replace with the site value"),
        "c_OM": ("O&M cost", "USD/kWe-yr", REF),
        "er": ("Escalation of O&M and water price", "1/yr", REF),
        "dr": ("Output degradation", "1/yr", REF),
        "salvage": ("Salvage value, fraction of total capital investment", "-", REF),
        "water_price": ("Makeup water price", "USD/m3", PH),
        "c_gas": ("Cost of the waste-heat exergy (treated as free)", "USD/GJ", CHO)}
for k, (text, unit, note) in desc.items():
    add("Economics", text, getattr(e, k), unit, note)
add("Economics", "Heat exchanger UA used for cost: larger of end-point and segmented UA", note=CHO)
add("Economics", "Currency: USD, nominal values of the cost references, no conversion", note=CHO)

extra = {"Main heater": "band widened upward (dust, fouling); not validated for EAF gas",
         "Branch heater": "band widened upward (dust, fouling); not validated for EAF gas",
         "Cooling tower": "formula reconstructed from the reference text; band assumed"}
for n, r in R["costs"]["rows"].items():
    add("Cost data", f"{n}: {r['source']}",
        f"{r['lo'] / r['PEC'] - 1:+.0%} / {r['hi'] / r['PEC'] - 1:+.0%}", "band",
        extra.get(n, REF))
for w in R["costs"]["warnings"]:
    add("Cost data", f"Validity warning: {w}", note="extrapolation")

for t in NOT_MODELLED:
    add("Not included", t, note="not modelled")

add("SPECO", "Physical exergy is not split into thermal and mechanical parts", note="simplification")
add("SPECO", "Cooler and cooling tower are dissipative: their cost, tower power and water are charged to the product", note="simplification")
add("SPECO", "O&M is shared in proportion to each component's purchased cost", note="simplification")
add("Numerics", "Convergence tolerance on h of state 5 (recuperator loop)", 1e-3, "J/kg", "model")
add("Numerics", "Recuperator pinch check with 30 segments; heaters and cooler with 20 to 30 segments; pressure linear along each side", note="model")

df = pd.DataFrame(A)
Path("tables").mkdir(exist_ok=True)
df.to_csv("tables/01_assumptions.csv", index=False, encoding="utf-8-sig")
print(f"{len(df)} assumptions written to tables/01_assumptions.csv\n")
print(df.groupby("Category", sort=False).size().to_string())
print("\nNotes:", df["Note"].str.split(";").str[0].value_counts().to_dict())