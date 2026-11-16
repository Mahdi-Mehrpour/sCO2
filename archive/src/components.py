"""Cycle components. All units SI: K, Pa, J/kg, J/kg.K"""
from CoolProp.CoolProp import PropsSI


def state_Tp(fluid, T, p):
    """Build a state dict from temperature and pressure."""
    return {"T": T, "p": p,
            "h": PropsSI("H", "T", T, "P", p, fluid),
            "s": PropsSI("S", "T", T, "P", p, fluid)}


def state_hp(fluid, h, p):
    """Build a state dict from enthalpy and pressure."""
    return {"T": PropsSI("T", "H", h, "P", p, fluid), "p": p, "h": h,
            "s": PropsSI("S", "H", h, "P", p, fluid)}


def compressor(fluid, inlet, p_out, eta_is):
    """Compressor with isentropic efficiency.
    Returns: (outlet state, specific work consumed in J/kg)"""
    h2s = PropsSI("H", "P", p_out, "S", inlet["s"], fluid)
    h2 = inlet["h"] + (h2s - inlet["h"]) / eta_is
    return state_hp(fluid, h2, p_out), h2 - inlet["h"]


def turbine(fluid, inlet, p_out, eta_is):
    """Turbine with isentropic efficiency.
    Returns: (outlet state, specific work produced in J/kg)"""
    hs = PropsSI("H", "P", p_out, "S", inlet["s"], fluid)
    h_out = inlet["h"] - eta_is * (inlet["h"] - hs)
    return state_hp(fluid, h_out, p_out), inlet["h"] - h_out



def recuperator(fluid, hot_in, cold_in, eff):
    """Recuperator using the effectiveness method, based on enthalpy
    (because cp of sCO2 is not constant near the critical point).
    Returns: (cold outlet state, hot outlet state, heat transferred in J/kg)"""
    # Maximum possible heat transfer from the hot side (hot stream cooled to cold inlet T)
    q_max_hot = hot_in["h"] - PropsSI("H", "T", cold_in["T"], "P", hot_in["p"], fluid)
    # Maximum possible heat transfer to the cold side (cold stream heated to hot inlet T)
    q_max_cold = PropsSI("H", "T", hot_in["T"], "P", cold_in["p"], fluid) - cold_in["h"]
    # Actual heat transfer = effectiveness x smaller of the two limits
    q = eff * min(q_max_hot, q_max_cold)
    cold_out = state_hp(fluid, cold_in["h"] + q, cold_in["p"])
    hot_out = state_hp(fluid, hot_in["h"] - q, hot_in["p"])
    return cold_out, hot_out, q

def recuperator_flows(fluid, hot_in, cold_in, eff, m_cold):
    """Recuperator with unequal flows (per 1 kg of hot-side flow).
    m_cold: cold-side mass flow as a fraction of the hot-side flow.
    Effectiveness is based on enthalpy, as in recuperator().
    Returns: (cold outlet state, hot outlet state, heat transferred in J/kg of hot flow)"""
    q_max_hot = hot_in["h"] - PropsSI("H", "T", cold_in["T"], "P", hot_in["p"], fluid)
    q_max_cold = m_cold * (PropsSI("H", "T", hot_in["T"], "P", cold_in["p"], fluid) - cold_in["h"])
    q = eff * min(q_max_hot, q_max_cold)
    cold_out = state_hp(fluid, cold_in["h"] + q / m_cold, cold_in["p"])
    hot_out = state_hp(fluid, hot_in["h"] - q, hot_in["p"])
    return cold_out, hot_out, q


def mixer(fluid, a, m_a, b, m_b):
    """Adiabatic mixing of two streams at the same pressure (energy balance on enthalpy)."""
    h = (m_a * a["h"] + m_b * b["h"]) / (m_a + m_b)
    return state_hp(fluid, h, a["p"])
