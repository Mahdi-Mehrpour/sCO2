import inspect
import json
from pathlib import Path
import pandas as pd

from sco2.cycle_adv import splitflow_adv
from sco2.heat_source_adv import EAF
from sco2.parasitics_splitflow import PARASITIC
from sco2.cooling_splitflow import COOLING
from sco2.operation_splitflow import OPERATION
from sco2.plant_splitflow import evaluate_plant, COOLING_OPTION, DRIVE

K = 273.15
with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)

P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"])
res, hs, par, cp = P["res"], P["hs"], P["par"], P["cooler"]
st = res["states"]
m = hs["m_co2"]

# ---------------- Table: plant performance ---------------------------------
net_shaft = par["net_shaft"]
parasitic_total = net_shaft - P["net_el"]
closure = (net_shaft - par["loss_gearbox"] - par["loss_generator"]
           - par["loss_motor"] - cp["P_aux"] - par["fan_gas"] - P["net_el"])
water_kt = cp["flow"] * 3600 * P["hours"] / 1e3 / 1e3          # thousand m3/yr
perf = [
    ("Gas inlet temperature", EAF["T_in"] - K, "C"),
    ("Gas mass flow", EAF["m_gas"], "kg/s"),
    ("Gas stack temperature", hs["T_out"] - K, "C"),
    ("Heat available down to the minimum stack temperature", hs["Q_max"] / 1e6, "MW"),
    ("Heat delivered to the cycle", hs["Q"] / 1e6, "MW"),
    ("CO2 mass flow (turbine)", m, "kg/s"),
    ("Compressor inlet temperature (set by the wet tower)", P["T1"] - K, "C"),
    ("Compressor pressure ratio", opt["pr"], "-"),
    ("Maximum cycle pressure", st[2]["p"] / 1e6, "MPa"),
    ("Turbine shaft power", par["W_turbine"] / 1e6, "MW"),
    ("Compressor shaft power", par["W_compressor"] / 1e6, "MW"),
    ("Net shaft power (cycle model)", net_shaft / 1e6, "MW"),
    ("Gearbox losses", par["loss_gearbox"] / 1e6, "MW"),
    ("Generator losses", par["loss_generator"] / 1e6, "MW"),
    ("Compressor motor losses", par["loss_motor"] / 1e6, "MW"),
    ("Cooling tower fan and pump", cp["P_aux"] / 1e6, "MW"),
    ("Off-gas fan", par["fan_gas"] / 1e6, "MW"),
    ("Total parasitic loads and losses", parasitic_total / 1e6, "MW"),
    ("Net electrical power", P["net_el"] / 1e6, "MW"),
    ("Cycle thermal efficiency (net shaft power / heat delivered)", res["eta"] * 100, "%"),
    ("Net electrical efficiency (net electrical power / heat delivered)", P["eta_el"] * 100, "%"),
    ("Heat recovery efficiency (net electrical power / available heat)",
     P["net_el"] / hs["Q_max"] * 100, "%"),
    ("Exergy of gas at inlet", hs["ex_in"] / 1e6, "MW"),
    ("System exergy efficiency (net electrical power / gas exergy)",
     P["net_el"] / hs["ex_in"] * 100, "%"),
    ("Cooler duty", cp["Q"] / 1e6, "MW"),
    ("Cooler UA", cp["UA"] / 1e3, "kW/K"),
    ("Cooler minimum temperature difference", cp["min_dT"], "K"),
    ("Makeup water", cp["flow"], "kg/s"),
    ("Main heater minimum temperature difference", hs["min_dT1"], "K"),
    ("Branch heater minimum temperature difference", hs["min_dT2"], "K"),
    ("HTR minimum temperature difference", res["pinch"]["HTR"]["min_dT"], "K"),
    ("LTR minimum temperature difference", res["pinch"]["LTR"]["min_dT"], "K"),
    ("Limiting constraint of the gas coupling", hs["binding"], ""),
    ("Operating hours (equivalent full load)", P["hours"], "h/yr"),
    ("Annual net electrical energy", P["energy_GWh"], "GWh/yr"),
    ("Annual makeup water", water_kt, "thousand m3/yr"),
    ("Closure check of the power balance", closure / 1e6, "MW"),
]
t_perf = pd.DataFrame([{"Parameter": a, "Value": (round(b, 4) if isinstance(b, float) else b),
                        "Unit": c} for a, b, c in perf])

# ---------------- Table: assumptions (read from the code) ------------------
D = {k: v.default for k, v in inspect.signature(splitflow_adv).parameters.items()}
W = COOLING["wet_tower"]
A = []


def add(cat, text, value="", unit=""):
    A.append({"Category": cat, "Assumption": text, "Value": value, "Unit": unit})


add("General", "Steady-state, steady-flow operation; no dynamics")
add("General", "Kinetic and potential energy changes are neglected")
add("General", "Cycle quantities per 1 kg of CO2 through the turbine; absolute values use the CO2 flow from the heat-source coupling")
add("General", "All cycle components are adiabatic; no leakage")
add("Heat source", "EAF off-gas at stable conditions (values are assumptions, not plant data)")
add("Heat source", "Gas inlet temperature", EAF["T_in"] - K, "C")
add("Heat source", "Gas mass flow", EAF["m_gas"], "kg/s")
add("Heat source", "Gas pressure (no pressure drop in the thermodynamic model)", EAF["p"] / 1e3, "kPa")
add("Heat source", "Gas composition (mole fractions)", ", ".join(f"{k} {v}" for k, v in EAF["X"].items()))
add("Heat source", "Ideal-gas mixture, NASA polynomials (Cantera gri30); dust and chemical exergy of CO neglected")
add("Heat source", "Minimum gas outlet (stack) temperature", EAF["T_stack_min"] - K, "C")
add("Heat source", "Minimum gas-to-CO2 temperature difference in each heater", EAF["dT_pinch_heater"], "K")
add("Heat source", "Gas passes the main heater first, then the branch heater")
add("Fluid", "CO2 with real-gas properties (CoolProp, Helmholtz EOS)", D["fluid"])
add("Cycle", "Layout: split-flow without recompressor; branch taken after the main compressor and rejoining at the mixer")
add("Cycle", "Compressor inlet temperature (from the wet tower, at the floor of critical temperature + 1 K)", P["T1"] - K, "C")
add("Cycle", "Compressor inlet pressure", D["p1"] / 1e6, "MPa")
add("Cycle", "Compressor pressure ratio (optimised at the pressure limit)", round(opt["pr"], 4), "-")
add("Cycle", "Maximum cycle pressure (limit applied at the compressor outlet)", opt["p_max_limit_Pa"] / 1e6, "MPa")
add("Cycle", "Turbine inlet temperature (fixed)", D["T7"] - K, "C")
add("Cycle", "Branch fraction y (optimised, flat optimum)", round(opt["y"], 4), "-")
add("Cycle", "Branch heater outlet temperature (optimised, flat optimum)", round(opt["T_b"] - K, 2), "C")
add("Cycle", "Main compressor isentropic efficiency", D["eta_mc"], "-")
add("Cycle", "Turbine isentropic efficiency", D["eta_t"], "-")
add("Cycle", "LTR and HTR effectiveness (enthalpy-based, at inlet pressures)", D["eps_ltr"], "-")
add("Cycle", "Minimum temperature difference in HTR and LTR enforced in the optimisation", opt["dT_recuperator_min"], "K")
add("Cycle", "Branch heater pressure drop equals the LTR cold side drop (no throttling at the mixer)")
for label, key in [("Main heater", "dp_heater"), ("Cooler", "dp_cooler"),
                   ("LTR cold side", "dp_ltr_cold"), ("LTR hot side", "dp_ltr_hot"),
                   ("HTR cold side", "dp_htr_cold"), ("HTR hot side", "dp_htr_hot")]:
    add("Pressure drops", f"{label} (fraction of inlet pressure)", D[key] * 100, "%")
add("Cooling", f"Cooling system: {W['label']}")
add("Cooling", "Design wet-bulb temperature", W["T_wb_C"], "C")
add("Cooling", "Tower approach (cooling water minus wet bulb)", W["tower_approach"], "K")
add("Cooling", "Heat exchanger approach (CO2 outlet minus cooling water inlet)", W["approach"], "K")
add("Cooling", "Cooling water temperature rise", W["coolant_dT"], "K")
add("Cooling", "Tower fan and pump power as a fraction of the cooler duty", W["fan_pump_fraction"] * 100, "%")
add("Cooling", "Makeup water factor (makeup / evaporation)", W["makeup_factor"], "-")
add("Cooling", "Lowest compressor inlet temperature allowed (critical temperature + 1 K)", COOLING["T1_floor_C"], "C")
add("Parasitics", "Compressor driven by an electric motor (drive option)", DRIVE)
add("Parasitics", "Gearbox efficiency", PARASITIC["eta_gearbox"], "-")
add("Parasitics", "Generator efficiency", PARASITIC["eta_generator"], "-")
add("Parasitics", "Compressor motor efficiency", PARASITIC["eta_motor"], "-")
add("Parasitics", "Gas-side pressure drop for the off-gas fan (heaters and ducts)", PARASITIC["gas_dp_total"], "Pa")
add("Parasitics", "Off-gas fan efficiency", PARASITIC["eta_fan_gas"], "-")
add("Operation", "Furnace operating days per year", OPERATION["days_per_year"], "days")
add("Operation", "Furnace operating hours per day", OPERATION["hours_per_day"], "h")
add("Operation", "Availability of the sCO2 unit", OPERATION["plant_availability"], "-")
add("Operation", "Stable gas condition all year; fluctuation phases are not used")
add("Exergy", "Dead state", "25 C, 101.325 kPa")
add("Exergy", "Only physical (thermo-mechanical) exergy is considered")
add("Numerics", "Convergence tolerance on h of state 5 (recuperator loop)", 1e-3, "J/kg")
add("Numerics", "Recuperator pinch check with 30 segments, pressure linear along each side")
add("Numerics", "Economic parameters: not included yet (cost model to be replaced)")
t_ass = pd.DataFrame(A)

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 110)
print(t_perf.to_string(index=False))
print(f"\n{len(t_ass)} assumptions written.")

out = Path("tables/splitflow")
out.mkdir(parents=True, exist_ok=True)
t_perf.to_csv(out / "plant_performance.csv", index=False, encoding="utf-8-sig")
t_ass.to_csv(out / "plant_assumptions.csv", index=False, encoding="utf-8-sig")
print(f"Saved: {out}/plant_performance.csv and plant_assumptions.csv")