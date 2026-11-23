"""Off-design model of the gas heater (EAF off-gas -> CO2, counterflow) with a fixed UA.

Given the gas inlet temperature and flow, the CO2 inlet state and flow and the UA of the hardware, the duty Q is the
value for which the segment-wise UA equals the fixed UA. The segment method is the one of the cycle model
(sco2.heat_source_adv.couple_heater): the duty is split into n equal parts, the gas temperature follows from its
enthalpy table and the CO2 temperature from CO2 properties at the local enthalpy and pressure (pressure linear).
The turbine inlet temperature T7 (the CO2 outlet) is a result, not a specification."""
import numpy as np
from CoolProp.CoolProp import PropsSI

FLUID = "CO2"


def ua_segments(dT, Q, f):
    """UA = sum of (duty of each segment / mean temperature difference of the segment), W/K."""
    dT = np.asarray(dT, float)
    if dT.min() <= 0.0:
        return float("inf")
    return float(np.sum(Q * np.diff(np.asarray(f, float)) / (0.5 * (dT[:-1] + dT[1:]))))


def ua_design(hs):
    """Design UA (W/K) from the heater coupling of the design point."""
    return ua_segments(np.asarray(hs["T_gas"]) - np.asarray(hs["T_co2"]), hs["Q"], hs["f"])


def profile(Q, gas, T_gas_in, m_gas, co2_in, m_co2, dp, n=20):
    """(f, gas T, CO2 T, UA, min dT) for the duty Q (W). f = 0 is the CO2 inlet / gas outlet end."""
    f = np.linspace(0.0, 1.0, n + 1)
    p_out = co2_in["p"] * (1.0 - dp)
    h_c = co2_in["h"] + f * Q / m_co2
    p_c = co2_in["p"] + f * (p_out - co2_in["p"])
    T_c = np.array([PropsSI("T", "H", h, "P", p, FLUID) for h, p in zip(h_c, p_c)])
    h_in = float(gas.h(T_gas_in))
    T_g = np.asarray(gas.T_of_h((h_in - Q / m_gas) + f * (Q / m_gas)), float)
    dT = T_g - T_c
    return f, T_g, T_c, ua_segments(dT, Q, f), float(dT.min())


def q_max(gas, T_gas_in, m_gas, co2_in, m_co2, dp):
    """Thermodynamic limit of the duty (W): gas cooled to the CO2 inlet temperature, or CO2 heated to the gas inlet temperature."""
    q_g = m_gas * (float(gas.h(T_gas_in)) - float(gas.h(co2_in["T"])))
    q_c = m_co2 * (PropsSI("H", "T", T_gas_in, "P", co2_in["p"] * (1.0 - dp), FLUID) - co2_in["h"])
    return min(q_g, q_c)


def solve(UA, gas, T_gas_in, m_gas, co2_in, m_co2, dp, n=20, rtol=1e-7):
    """Duty and outlet states of the heater with the fixed UA (W/K), by bisection on the duty."""
    qm = q_max(gas, T_gas_in, m_gas, co2_in, m_co2, dp)
    lo, hi = 0.0, qm
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        _, _, _, ua, dT = profile(mid, gas, T_gas_in, m_gas, co2_in, m_co2, dp, n)
        if dT <= 0.0 or ua > UA:
            hi = mid
        else:
            lo = mid
        if hi - lo < rtol * qm:
            break
    Q = lo
    f, T_g, T_c, ua, dT = profile(Q, gas, T_gas_in, m_gas, co2_in, m_co2, dp, n)
    return {"Q": Q, "q_max": qm, "eps": Q / qm, "UA": ua, "min_dT": dT, "T_gas_out": float(T_g[0]),
            "T_co2_out": float(T_c[-1]), "f": f, "T_gas": T_g, "T_co2": T_c}
