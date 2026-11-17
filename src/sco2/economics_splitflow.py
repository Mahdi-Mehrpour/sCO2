"""Adapter between the plant evaluation (evaluate_plant) and the economic
module economics_eaf.py. Builds the module inputs directly from the model, so
cooling, parasitic and operating assumptions have a single source."""
from dataclasses import replace

from .economics_eaf import (State, CycleInputs, EconInputs, size,
                            equipment_costs, cash_flows, indicators)
import numpy as np
from .exergy_splitflow import exergy_splitflow
from .heat_source_adv import EAF
from .parasitics_splitflow import PARASITIC
from .cooling_splitflow import COOLING
from .plant_splitflow import get_gas


def ua_from_profile(Q, T_hot, T_cold):
    """UA (W/K) of a heat exchanger from its temperature profile.
    Q: total duty (W). T_hot, T_cold: arrays of n+1 temperatures (K) at the
    boundaries of n segments of equal duty. UA = sum of (dQ / dT_segment)."""
    dT = np.asarray(T_hot) - np.asarray(T_cold)
    if dT.min() <= 0.0:
        return float("inf")
    dT_mid = 0.5 * (dT[:-1] + dT[1:])
    return float(np.sum((Q / (len(dT) - 1)) / dT_mid))


K = 273.15


def build_inputs(P, econ_overrides=None):
    """P: dictionary from evaluate_plant(). Returns (CycleInputs, EconInputs)."""
    res, hs, par, cp = P["res"], P["hs"], P["par"], P["cooler"]
    st, y, m = res["states"], res["y"], hs["m_co2"]
    ex = exergy_splitflow(res, hs, get_gas(), EAF["m_gas"])
    fe = ex["flow_exergy"]
    frac = {3: 1.0 - y, 11: y}                       # all other states: full flow
    states = {k: State(m=m * frac.get(k, 1.0), T=x["T"] - K, P=x["p"] / 1e6,
                       h=x["h"] / 1e3, s=x["s"] / 1e3, ex=fe[k] / 1e3)
              for k, x in st.items()}

    # segmented UA (kW/K): captures internal pinches that end-point LMTD misses
    ua_seg = {
        "HTR": m * res["pinch"]["HTR"]["UA"] / 1e3,
        "LTR": m * res["pinch"]["LTR"]["UA"] / 1e3,
        "MH": ua_from_profile(hs["Q1"], hs["T_gas1"], hs["T_co2_1"]) / 1e3,
        "BH": ua_from_profile(hs["Q2"], hs["T_gas2"], hs["T_co2_2"]) / 1e3,
        "cool": cp["UA"] / 1e3,
    }
    w = COOLING["wet_tower"]
    ci = CycleInputs(
        st=states, y=y,
        T_g1=EAF["T_in"] - K, T_g2=hs["T_gas_mid"] - K, T_g3=hs["T_out"] - K,
        E_g1=hs["ex_in"] / 1e6, E_g2=ex["ex_mid_W"] / 1e6, E_g3=hs["ex_out"] / 1e6,
        W_fan_gas=par["fan_gas"] / 1e6,
        T_wb=w["T_wb_C"], tower_approach=w["tower_approach"],
        water_rise=w["coolant_dT"], tower_power_frac=w["fan_pump_fraction"],
        makeup_factor=w["makeup_factor"], h_fg=w["latent"] / 1e3,
        makeup_kg_s=cp["flow"],
        eta_gb=PARASITIC["eta_gearbox"], eta_gen=PARASITIC["eta_generator"],
        eta_mot=PARASITIC["eta_motor"],
        ua_basis="max", UA_seg=ua_seg)
    e = EconInputs(hours=P["hours"])
    if econ_overrides:
        e = replace(e, **econ_overrides)
    return ci, e


def fast_economics(ci, e, extra_capital=0.0):
    """Capital cost and economic indicators only (no cost rates, no SPECO).
    extra_capital: fraction added to the total capital investment."""
    d = size(ci)
    eq = equipment_costs(ci, d, e)
    PEC = sum(r["PEC"] for r in eq["rows"].values())
    TCI = PEC * (1.0 + e.f_inst) * (1.0 + extra_capital)
    water = d["makeup_kg_s"] * 3600.0 * e.hours / 1000.0
    out = indicators(TCI, cash_flows(TCI, d["W_net"], water, e), e)
    out.update(TCI=TCI, PEC=PEC, W_net=d["W_net"], size=d, rows=eq["rows"])
    return out