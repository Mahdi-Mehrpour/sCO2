"""Off-design model of a counterflow recuperator (HTR or LTR) with a fixed UA.

Given the inlet states, the two mass flows and the UA of the hardware, the duty q is the value for which the
segment-wise UA of the exchanger equals the fixed UA. The segment method is the one used in the cycle model
(sco2.cycle_adv._pinch_profile): the duty is split into n equal parts, CO2 properties are evaluated at every point
at the local enthalpy and pressure, and UA = sum of (segment duty / mean temperature difference). So the
effectiveness is not assumed: it follows from the flows and the CO2 properties, and the pinch is a result."""
import numpy as np
from CoolProp.CoolProp import PropsSI

from sco2.cycle_adv import _pinch_profile

FLUID = "CO2"


def _state(h, p):
    return {"h": h, "p": p, "T": PropsSI("T", "H", h, "P", p, FLUID)}


def q_max(hot_in, cold_in, m_hot, m_cold):
    """Thermodynamic limit of the duty (W): the smaller of the two sides cooled / heated to the other inlet temperature."""
    q_h = m_hot * (hot_in["h"] - PropsSI("H", "T", cold_in["T"], "P", hot_in["p"], FLUID))
    q_c = m_cold * (PropsSI("H", "T", hot_in["T"], "P", cold_in["p"], FLUID) - cold_in["h"])
    return min(q_h, q_c)


def ua_of_duty(q, hot_in, cold_in, m_hot, m_cold, dp_hot, dp_cold, n=30):
    """(UA in W/K, minimum temperature difference in K, profile) for a given duty q (W)."""
    prof = _pinch_profile(FLUID, hot_in, cold_in, q / m_hot, m_cold / m_hot, dp_hot, dp_cold, n)
    return prof["UA"] * m_hot, prof["min_dT"], prof


def solve(UA, hot_in, cold_in, m_hot, m_cold, dp_hot, dp_cold, n=30, rtol=1e-7):
    """Duty and outlet states of a recuperator with the fixed UA (W/K).

    UA(q) rises from 0 to infinity as q approaches the pinch limit, so the duty is found by bisection.
    Returns a dictionary with q (W), the effectiveness q / q_max, the minimum temperature difference, the outlet
    states and the profile."""
    qm = q_max(hot_in, cold_in, m_hot, m_cold)
    lo, hi = 0.0, qm
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        ua, dT, _ = ua_of_duty(mid, hot_in, cold_in, m_hot, m_cold, dp_hot, dp_cold, n)
        if dT <= 0.0 or ua > UA:
            hi = mid
        else:
            lo = mid
        if hi - lo < rtol * qm:
            break
    q = lo
    ua, dT, prof = ua_of_duty(q, hot_in, cold_in, m_hot, m_cold, dp_hot, dp_cold, n)
    hot_out = _state(hot_in["h"] - q / m_hot, hot_in["p"] * (1.0 - dp_hot))
    cold_out = _state(cold_in["h"] + q / m_cold, cold_in["p"] * (1.0 - dp_cold))
    return {"q": q, "eps": q / qm, "q_max": qm, "UA": ua, "min_dT": dT, "hot_out": hot_out, "cold_out": cold_out,
            "profile": prof}
