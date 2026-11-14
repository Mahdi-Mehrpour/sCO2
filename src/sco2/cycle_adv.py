"""Advanced recompression sCO2 Brayton cycle (the file that keeps evolving).

Fully self-contained: it only depends on CoolProp and NumPy, so it can be
changed freely without touching components.py, cycle.py or cycle_rc.py.

Current features:
  - recompression with LTR and HTR recuperators
  - pressure drops in the heater, cooler and both sides of both recuperators
  - recuperator pinch check (discretised counterflow profile) and an optional
    minimum temperature difference constraint (dT_pinch)

States: 1 MC inlet | 2 MC outlet | 3 LTR cold outlet | 4 RC outlet
        5 mixer outlet (HTR cold inlet) | 6 HTR cold outlet | 7 turbine inlet
        8 turbine outlet | 9 HTR hot outlet | 10 LTR hot outlet (split point)

All flows are per 1 kg of CO2 through the turbine. Units are SI.
Pressure drops are fractions of the inlet pressure of each component."""
import numpy as np
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


def _pinch_profile(fluid, hot_in, cold_in, q, m_cold, dp_hot, dp_cold, n):
    """Counterflow temperature profile of a recuperator transferring q (J/kg of
    hot flow), split into n segments of equal heat duty.

    f = 0 is the cold end (cold inlet / hot outlet), f = 1 is the hot end
    (cold outlet / hot inlet). Pressure varies linearly with f on each side.
    Returns a dictionary with the profiles, the minimum temperature difference
    and UA per kg of hot flow (J/kg.K)."""
    f = np.linspace(0.0, 1.0, n + 1)
    p_c = cold_in["p"] * (1.0 - dp_cold * f)
    p_h_out = hot_in["p"] * (1.0 - dp_hot)
    p_h = p_h_out + f * (hot_in["p"] - p_h_out)
    h_c = cold_in["h"] + f * q / m_cold
    h_h = (hot_in["h"] - q) + f * q
    T_c = np.array([PropsSI("T", "H", h, "P", p, fluid) for h, p in zip(h_c, p_c)])
    T_h = np.array([PropsSI("T", "H", h, "P", p, fluid) for h, p in zip(h_h, p_h)])
    dT = T_h - T_c
    i = int(np.argmin(dT))

    # UA = integral of dQ / dT, trapezoid on the segments
    if q <= 0.0:
        UA = 0.0
    elif dT.min() <= 0.0:
        UA = float("inf")
    else:
        dT_mid = 0.5 * (dT[:-1] + dT[1:])
        UA = float(np.sum((q / n) / dT_mid))

    return {"f": f, "Q": f * q, "T_hot": T_h, "T_cold": T_c, "dT": dT,
            "min_dT": float(dT[i]), "f_min": float(f[i]),
            "dT_cold_end": float(dT[0]), "dT_hot_end": float(dT[-1]),
            "UA": UA, "q": q}


def _recuperator(fluid, hot_in, cold_in, eff, m_cold, dp_hot, dp_cold,
                 dT_pinch=None, n_seg=30):
    """Enthalpy-based effectiveness recuperator with unequal flows and
    pressure drops. m_cold is the cold-side flow as a fraction of the hot flow.
    If dT_pinch (K) is given, the heat duty is reduced until the minimum
    temperature difference inside the exchanger is at least dT_pinch.
    Returns (cold outlet, hot outlet, heat transferred in J/kg of hot flow)."""
    q_max_hot = hot_in["h"] - PropsSI("H", "T", cold_in["T"], "P", hot_in["p"], fluid)
    q_max_cold = m_cold * (PropsSI("H", "T", hot_in["T"], "P", cold_in["p"], fluid)
                           - cold_in["h"])
    q = eff * min(q_max_hot, q_max_cold)

    if dT_pinch is not None:
        def min_dT(qq):
            return _pinch_profile(fluid, hot_in, cold_in, qq, m_cold,
                                  dp_hot, dp_cold, n_seg)["min_dT"]
        if min_dT(q) < dT_pinch:
            # Bisection: largest q in [0, q] that still respects the pinch.
            # Assumes the minimum temperature difference decreases with q.
            lo, hi = 0.0, q
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if min_dT(mid) >= dT_pinch:
                    lo = mid
                else:
                    hi = mid
            q = lo

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
                      dp_htr_cold=0.01, dp_htr_hot=0.01,
                      dT_pinch=None, n_seg=30):
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
                                     dp_htr_hot, dp_htr_cold, dT_pinch, n_seg)
        s3, s10, q_ltr = _recuperator(fluid, s9, s2, eps_ltr, 1 - x,
                                      dp_ltr_hot, dp_ltr_cold, dT_pinch, n_seg)
        s4, w_rc = _compressor(fluid, s10, p4, eta_rc)
        s5_new = _mixer(fluid, s3, 1 - x, s4, x)
        done = abs(s5_new["h"] - s5["h"]) < 1e-3   # J/kg
        s5 = s5_new
        if done:
            break
    else:
        raise RuntimeError("Recuperator iteration did not converge")

    # --- Pinch check of the converged recuperators ---
    pinch = {"HTR": _pinch_profile(fluid, s8, s5, q_htr, 1.0,
                                   dp_htr_hot, dp_htr_cold, n_seg),
             "LTR": _pinch_profile(fluid, s9, s2, q_ltr, 1 - x,
                                   dp_ltr_hot, dp_ltr_cold, n_seg)}

    q_in = s7["h"] - s6["h"]                  # heater
    q_out = (1 - x) * (s10["h"] - s1["h"])    # cooler (only the 1-x stream)
    w_net = w_t - (1 - x) * w_mc - x * w_rc
    return {"states": {1: s1, 2: s2, 3: s3, 4: s4, 5: s5,
                       6: s6, 7: s7, 8: s8, 9: s9, 10: s10},
            "w_mc": w_mc, "w_rc": w_rc, "w_t": w_t, "w_net": w_net,
            "q_in": q_in, "q_out": q_out, "q_htr": q_htr, "q_ltr": q_ltr,
            "pinch": pinch,
            "x": x, "pr_turbine": p7 / p8, "eta": w_net / q_in}