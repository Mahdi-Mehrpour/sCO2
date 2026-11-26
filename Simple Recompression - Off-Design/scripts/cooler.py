"""Off-design model of the dry air cooler (CO2 from state 10 to state 1, counterflow idealisation) with a fixed UA.

Given the ambient (air inlet) temperature, the air flow, the CO2 inlet state and flow and the UA of the hardware, the
duty Q is the value for which the segment-wise UA equals the fixed UA. The air warms linearly with the heat absorbed;
CO2 properties are evaluated at every point at the local enthalpy and pressure (pressure linear). The compressor inlet
temperature T1 is a result: it is the CO2 outlet temperature of the cooler.
Air cooler assumptions (placeholders, as in the earlier cooling module): fan pressure rise 200 Pa, fan efficiency 0.65,
air at atmospheric pressure. The real air cooler is cross-flow with a correction factor below 1; counterflow is the
optimistic limit. Fan power follows the cube of the air mass flow and the air density."""
import numpy as np
from CoolProp.CoolProp import PropsSI

FLUID = "CO2"
CP_AIR, R_AIR, P_ATM = 1005.0, 287.05, 101325.0       # J/kg.K, J/kg.K, Pa


def ua_segments(dT, Q, f):
    dT = np.asarray(dT, float)
    if dT.min() <= 0.0:
        return float("inf")
    return float(np.sum(Q * np.diff(np.asarray(f, float)) / (0.5 * (dT[:-1] + dT[1:]))))


def profile(Q, T_air_in, m_air, co2_in, m_co2, dp, n=30):
    """(f, CO2 T, air T, UA, min dT); f = 0 is the CO2 outlet / air inlet end."""
    f = np.linspace(0.0, 1.0, n + 1)
    h_out = co2_in["h"] - Q / m_co2
    p_out = co2_in["p"] * (1.0 - dp)
    h = h_out + f * Q / m_co2
    p = p_out + f * (co2_in["p"] - p_out)
    T_c = np.array([PropsSI("T", "H", hi, "P", pi, FLUID) for hi, pi in zip(h, p)])
    T_a = T_air_in + f * Q / (m_air * CP_AIR)
    dT = T_c - T_a
    return f, T_c, T_a, ua_segments(dT, Q, f), float(dT.min())


def design_ua(co2_in, co2_out, m_co2, dp, T_air_in, dT_air, n=30):
    """Design UA (W/K) and design air flow (kg/s) of a cooler taking CO2 from co2_in to co2_out, with the air warming by
    dT_air. co2_out: state 1 (h, p). The duty follows from the CO2 side."""
    Q = m_co2 * (co2_in["h"] - co2_out["h"])
    m_air = Q / (CP_AIR * dT_air)
    f = np.linspace(0.0, 1.0, n + 1)
    h = co2_out["h"] + f * (co2_in["h"] - co2_out["h"])
    p = co2_out["p"] + f * (co2_in["p"] - co2_out["p"])
    T_c = np.array([PropsSI("T", "H", hi, "P", pi, FLUID) for hi, pi in zip(h, p)])
    T_a = T_air_in + dT_air * f
    return ua_segments(T_c - T_a, Q, f), m_air, Q


def q_max(T_air_in, m_air, co2_in, m_co2, dp):
    q_c = m_co2 * (co2_in["h"] - PropsSI("H", "T", T_air_in, "P", co2_in["p"] * (1.0 - dp), FLUID))
    q_a = m_air * CP_AIR * (co2_in["T"] - T_air_in)
    return min(q_c, q_a)


def solve(UA, T_air_in, m_air, co2_in, m_co2, dp, n=30, rtol=1e-6):
    """Duty, T1 and the profile of the cooler with the fixed UA (W/K), by bisection on the duty."""
    qm = q_max(T_air_in, m_air, co2_in, m_co2, dp)
    lo, hi = 0.0, qm
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        _, _, _, ua, dT = profile(mid, T_air_in, m_air, co2_in, m_co2, dp, n)
        if dT <= 0.0 or ua > UA:
            hi = mid
        else:
            lo = mid
        if hi - lo < rtol * qm:
            break
    Q = lo
    f, T_c, T_a, ua, dT = profile(Q, T_air_in, m_air, co2_in, m_co2, dp, n)
    return {"Q": Q, "q_max": qm, "eps": Q / qm, "UA": ua, "min_dT": dT, "T1": float(T_c[0]),
            "T_air_out": float(T_a[-1]), "f": f, "T_co2": T_c, "T_air": T_a}


def fan_power(m_air, m_air0, T_air_in, P0, T_air0):
    """Fan power (W): P0 * (m / m0)**3 * (rho0 / rho)**2 for the same fan, air density from the ambient temperature."""
    rho, rho0 = P_ATM / (R_AIR * T_air_in), P_ATM / (R_AIR * T_air0)
    return P0 * (m_air / m_air0) ** 3 * (rho0 / rho) ** 2
