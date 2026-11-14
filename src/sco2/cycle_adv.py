"""Advanced recompression sCO2 Brayton cycle (the file that keeps evolving).

Fully self-contained: it only depends on CoolProp, so it can be changed
freely without touching components.py, cycle.py or cycle_rc.py.

Current features:
  - recompression with LTR and HTR recuperators
  - pressure drops in the heater, cooler and both sides of both recuperators

States: 1 MC inlet | 2 MC outlet | 3 LTR cold outlet | 4 RC outlet
        5 mixer outlet (HTR cold inlet) | 6 HTR cold outlet | 7 turbine inlet
        8 turbine outlet | 9 HTR hot outlet | 10 LTR hot outlet (split point)

All flows are per 1 kg of CO2 through the turbine. Units are SI.
Pressure drops are fractions of the inlet pressure of each component."""
from CoolProp.CoolProp import PropsSI


# ----------------------------------------------------------------------
# Local components (own copies, independent from components.py)
# ----------------------------------------------------------------------
def _state_Tp(fluid, T, p):
    return {"T": T, "p": p,
            "h": PropsSI("H", "T", T, "P", p, fluid),
            "s": PropsSI("S", "T", T, "P", p, fluid)}


def _state_hp(fluid, h, p):
    return {"T": PropsSI("T", "H", h, "P", p, fluid), "p": p, "h": h,
            "s": PropsSI("S", "H", h, "P", p, fluid)}


def _compressor(fluid, inlet, p_out, eta_is):
    """Returns (outlet state, specific work consumed in J/kg)."""
    h2s = PropsSI("H", "P", p_out, "S", inlet["s"], fluid)
    h2 = inlet["h"] + (h2s - inlet["h"]) / eta_is
    return _state_hp(fluid, h2, p_out), h2 - inlet["h"]


def _turbine(fluid, inlet, p_out, eta_is):
    """Returns (outlet state, specific work produced in J/kg)."""
    hs = PropsSI("H", "P", p_out, "S", inlet["s"], fluid)
    h_out = inlet["h"] - eta_is * (inlet["h"] - hs)
    return _state_hp(fluid, h_out, p_out), inlet["h"] - h_out


def _recuperator(fluid, hot_in, cold_in, eff, m_cold, dp_hot, dp_cold):
    """Enthalpy-based effectiveness recuperator with unequal flows and
    pressure drops. m_cold is the cold-side flow as a fraction of the hot flow.
    Returns (cold outlet, hot outlet, heat transferred in J/kg of hot flow)."""
    q_max_hot = hot_in["h"] - PropsSI("H", "T", cold_in["T"], "P", hot_in["p"], fluid)
    q_max_cold = m_cold * (PropsSI("H", "T", hot_in["T"], "P", cold_in["p"], fluid)
                           - cold_in["h"])
    q = eff * min(q_max_hot, q_max_cold)
    cold_out = _state_hp(fluid, cold_in["h"] + q / m_cold, cold_in["p"] * (1 - dp_cold))
    hot_out = _state_hp(fluid, hot_in["h"] - q, hot_in["p"] * (1 - dp_hot))
    return cold_out, hot_out, q


def _mixer(fluid, a, m_a, b, m_b):
    """Adiabatic mixing of two streams at the same pressure."""
    h = (m_a * a["h"] + m_b * b["h"]) / (m_a + m_b)
    return _state_hp(fluid, h, a["p"])


# ----------------------------------------------------------------------
# Cycle
# ----------------------------------------------------------------------
def recompression_adv(fluid="CO2", T1=305.15, p1=7.7e6, pr=2.6,
                      T7=823.15, eta_mc=0.85, eta_rc=0.85, eta_t=0.90,
                      eps_ltr=0.90, eps_htr=0.90, x=0.3,
                      dp_heater=0.02, dp_cooler=0.01,
                      dp_ltr_cold=0.01, dp_ltr_hot=0.01,
                      dp_htr_cold=0.01, dp_htr_hot=0.01):
    # --- Pressure levels (known before any thermodynamic calculation) ---
    # High-pressure side, forward from the main compressor outlet
    p2 = p1 * pr
    p3 = p2 * (1 - dp_ltr_cold)
    p4 = p3                       # recompressor must match state 3 for mixing
    p5 = p3
    p6 = p5 * (1 - dp_htr_cold)
    p7 = p6 * (1 - dp_heater)
    # Low-pressure side, backward from the main compressor inlet p1
    p10 = p1 / (1 - dp_cooler)
    p9 = p10 / (1 - dp_ltr_hot)
    p8 = p9 / (1 - dp_htr_hot)

    # --- Main compressor and turbine ---
    s1 = _state_Tp(fluid, T1, p1)
    s2, w_mc = _compressor(fluid, s1, p2, eta_mc)
    s7 = _state_Tp(fluid, T7, p7)
    s8, w_t = _turbine(fluid, s7, p8, eta_t)

    # --- Recuperators and recompressor (circular, solved by iteration) ---
    s5 = _state_hp(fluid, s2["h"], p5)       # initial guess
    for _ in range(200):
        s6, s9, q_htr = _recuperator(fluid, s8, s5, eps_htr, 1.0,
                                     dp_htr_hot, dp_htr_cold)
        s3, s10, q_ltr = _recuperator(fluid, s9, s2, eps_ltr, 1 - x,
                                      dp_ltr_hot, dp_ltr_cold)
        s4, w_rc = _compressor(fluid, s10, p4, eta_rc)
        s5_new = _mixer(fluid, s3, 1 - x, s4, x)
        done = abs(s5_new["h"] - s5["h"]) < 1e-3   # J/kg
        s5 = s5_new
        if done:
            break
    else:
        raise RuntimeError("Recuperator iteration did not converge")

    q_in = s7["h"] - s6["h"]                  # heater
    q_out = (1 - x) * (s10["h"] - s1["h"])    # cooler (only the 1-x stream)
    w_net = w_t - (1 - x) * w_mc - x * w_rc
    return {"states": {1: s1, 2: s2, 3: s3, 4: s4, 5: s5,
                       6: s6, 7: s7, 8: s8, 9: s9, 10: s10},
            "w_mc": w_mc, "w_rc": w_rc, "w_t": w_t, "w_net": w_net,
            "q_in": q_in, "q_out": q_out, "q_htr": q_htr, "q_ltr": q_ltr,
            "x": x, "pr_turbine": p7 / p8, "eta": w_net / q_in}
