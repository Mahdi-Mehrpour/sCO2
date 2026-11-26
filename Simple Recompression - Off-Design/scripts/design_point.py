"""Design point of the simple recompression cycle for the off-design model.

The design point is rebuilt from the saved optimum (tables/simple_optimum.json, copied from the design study): the
same cycle model, the same fixed conditions and the same gas-heater coupling as scripts/run_cycle_comparison.py.
Nothing is optimised here. The values below are the design basis; change them only together with a new optimum."""
import json
from pathlib import Path

from CoolProp.CoolProp import PropsSI

from sco2.cycle_adv import recompression_adv, _pinch_profile
from sco2.heat_source_adv import EAF, couple_heater
from sco2.plant_splitflow import get_gas
from heater import ua_design
from cooler import design_ua, CP_AIR, R_AIR, P_ATM

K = 273.15
FLUID = "CO2"
HERE = Path(__file__).resolve().parent
OFFDIR = HERE.parent                                  # "Simple Recompression - Off-Design"
TABLES, FIGURES = OFFDIR / "tables", OFFDIR / "figures"

# design basis (same as the design study)
T1, P1 = 32.0 + K, 7.7e6                              # main compressor inlet
P_MAX = 25e6
PR = P_MAX / P1
T7 = 550.0 + K
N_SEG = 30                                            # equal-duty segments of the recuperators (as in the cycle model)
DP = {"HTR": (0.01, 0.01), "LTR": (0.01, 0.01)}       # pressure drop (hot, cold) as a fraction of the inlet pressure
DP_COOLER = 0.01                                      # CO2-side pressure drop of the cooler (fraction of inlet pressure)
DP_HEATER = 0.02                                      # CO2-side pressure drop of the gas heater (fraction of inlet pressure)
GAS = get_gas()


def load_design(path=None):
    """Return (cycle result, heater coupling, x) of the design point."""
    path = Path(path) if path else TABLES / "simple_optimum.json"
    x = float(json.load(open(path))["z"][0])
    res = recompression_adv(x=x, pr=PR, T7=T7, T1=T1, p1=P1)
    hs = couple_heater(res, GAS, EAF["T_in"], EAF["m_gas"], dT_pinch=EAF["dT_pinch_heater"],
                       T_stack_min=EAF["T_stack_min"])
    return res, hs, x


def recuperator_design(res, hs, x):
    """Inputs and design UA of the two recuperators: dictionaries with the inlet states, the flows (kg/s), the
    pressure drops and the design duty (W) and UA (W/K). UA uses the same segment method as the cycle model."""
    st, m = res["states"], hs["m_co2"]
    out = {}
    for name, hot, cold, m_cold, q_kg in (("HTR", 8, 5, m, res["q_htr"]), ("LTR", 9, 2, (1.0 - x) * m, res["q_ltr"])):
        dp_h, dp_c = DP[name]
        prof = _pinch_profile(FLUID, st[hot], st[cold], q_kg, m_cold / m, dp_h, dp_c, N_SEG)
        out[name] = {"hot_in": st[hot], "cold_in": st[cold], "m_hot": m, "m_cold": m_cold, "dp_hot": dp_h,
                     "dp_cold": dp_c, "q": q_kg * m, "UA": prof["UA"] * m, "min_dT": prof["min_dT"]}
    return out


def heater_design(res, hs):
    """Inputs and design UA of the gas heater: gas inlet temperature and flow, CO2 inlet state (state 6) and flow."""
    return {"T_gas_in": EAF["T_in"], "m_gas": EAF["m_gas"], "co2_in": res["states"][6], "m_co2": hs["m_co2"],
            "dp": DP_HEATER, "UA": ua_design(hs), "Q": hs["Q"], "T_gas_out": hs["T_out"],
            "T7": res["states"][7]["T"], "min_dT": hs["min_dT"]}


# dry air cooler (placeholders, as in the earlier cooling module): design ambient 25 C, air warms by 10 K, fan 200 Pa, eta 0.65
AIR = {"T_in": 25.0 + K, "dT": 10.0, "fan_dp": 200.0, "eta_fan": 0.65}


def cooler_design(res, hs, x):
    """Inputs and design UA of the dry air cooler: CO2 inlet state (state 10) and flow (1 - x) m, air inlet temperature,
    design air flow and design fan power. UA by the same 30-segment method as the earlier cooling module."""
    st, m = res["states"], hs["m_co2"]
    m_co2 = (1.0 - x) * m
    UA, m_air, Q = design_ua(st[10], st[1], m_co2, DP_COOLER, AIR["T_in"], AIR["dT"])
    rho = P_ATM / (R_AIR * AIR["T_in"])
    P_fan = (m_air / rho) * AIR["fan_dp"] / AIR["eta_fan"]
    return {"co2_in": st[10], "m_co2": m_co2, "dp": DP_COOLER, "T_air_in": AIR["T_in"], "m_air": m_air, "UA": UA,
            "Q": Q, "T1": st[1]["T"], "P_fan": P_fan}
