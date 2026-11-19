"""Off-design model of the split-flow plant: fixed hardware, variable gas.

The hardware is frozen at the design point (the design gas of heat_source_adv.EAF):
  - UA of the five heat exchangers (HTR, LTR, main heater, branch heater),
    calibrated from the design states with the same segmented method used here
  - turbine swallowing capacity (ellipse law, nozzle area fixed)
  - compressor and turbine isentropic efficiencies (constant, no maps yet)
  - compressor inlet fixed at T1, p1 (cooling system assumed to hold it)

What varies: gas temperature, gas flow, CO2 flow, branch fraction y, and the
gas bypass around the heaters.

Operation is SLIDING PRESSURE: the high pressure follows from the turbine law
for the chosen CO2 flow (inventory control). The pressure limit is NOT imposed
on the solution; it is reported as a flag.

ASSUMPTIONS (placeholders until real component data exist):
  - UA scales with flow^N_UA (N_UA = 0.8). Heaters: gas-side flow. HTR: CO2 flow.
    LTR: mean of the hot-side and cold-side flow ratios.
  - Turbine: m = Kt * sqrt(p_in^2 - p_out^2) / sqrt(T_in) (Stodola ellipse law).
  - Constant isentropic efficiencies (no part-load degradation).
  - Gas-side fan power scales with (gas flow)^3.
Units: SI (W, K, Pa, J/kg, kg/s). Cycle states are in J/kg."""
import inspect
import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import root, brentq

from .co2_tables import get_table
from .cycle_adv import splitflow_adv, _state_Tp, _state_hp, _compressor, _turbine
from .heat_source_adv import EAF
from .parasitics_splitflow import PARASITIC, parasitic_loads
from .cooling_splitflow import COOLING

FLUID = "CO2"
N_UA = 0.8                  # exponent of the UA-versus-flow law
M_MIN_FRAC = 0.30           # PLACEHOLDER: minimum CO2 flow / design flow
DT_REC_MIN = 10.0           # K, minimum temperature difference HTR, LTR


# ---------------------------------------------------------------------------
# Property helpers (scalars or arrays of equal length)
# ---------------------------------------------------------------------------
def T_hp(h, p):
    return PropsSI("T", "H", h, "P", p, FLUID)


def h_Tp(T, p):
    return PropsSI("H", "T", T, "P", p, FLUID)


def _T_arr(h, p):
    """Temperature of CO2 along a profile (arrays): interpolated table."""
    return get_table().T(h, p)


# ---------------------------------------------------------------------------
# Heat exchangers with fixed UA: find the duty Q that needs exactly that UA
# ---------------------------------------------------------------------------
def _profile_dT(hot_T, cold_T, n):
    f = np.linspace(0.0, 1.0, n + 1)           # f = 0 at the cold end
    return hot_T(f) - cold_T(f)


def _ua_from_dT(Q, dT):
    """UA = sum of (Q/n) / mean dT of each segment (same formula as the
    design-point UA)."""
    if dT.min() <= 0.0:
        return np.inf
    dTm = 0.5 * (dT[:-1] + dT[1:])
    return float(np.sum((Q / (len(dT) - 1)) / dTm))


def _solve_Q(ua_fix, Q_hi, ua_of_Q):
    """Largest Q in [0, Q_hi] with UA(Q) <= ua_fix (UA(Q) increases with Q).
    Brent's method on g(Q) = 1 - ua_fix / UA(Q), which is bounded."""
    if Q_hi <= 0.0:
        return 0.0
    if ua_of_Q(Q_hi) <= ua_fix:
        return Q_hi
    Q_lo = 1e-6 * Q_hi

    def g(Q):
        ua = ua_of_Q(Q)
        return 1.0 if not np.isfinite(ua) else 1.0 - ua_fix / max(ua, 1e-30)
    if g(Q_lo) >= 0.0:
        return Q_lo
    return brentq(g, Q_lo, Q_hi, xtol=1e-12 * Q_hi, rtol=1e-12, maxiter=100)


def _recup_profiles(Q, hot_in, cold_in, m_h, m_c, p_ho, p_co):
    h_hi, p_hi = hot_in
    h_ci, p_ci = cold_in
    h_ho = h_hi - Q / m_h
    h_co = h_ci + Q / m_c

    def hot_T(f):
        return _T_arr(h_ho + f * (h_hi - h_ho), p_ho + f * (p_hi - p_ho))

    def cold_T(f):
        return _T_arr(h_ci + f * (h_co - h_ci), p_ci + f * (p_co - p_ci))
    return hot_T, cold_T


def _recup_Qmax(hot_in, cold_in, m_h, m_c, p_ho, p_co):
    h_hi, p_hi = hot_in
    h_ci, p_ci = cold_in
    T_hi, T_ci = float(T_hp(h_hi, p_hi)), float(T_hp(h_ci, p_ci))
    if T_hi <= T_ci:
        return 0.0
    Q_h = m_h * (h_hi - float(h_Tp(T_ci, p_ho)))
    Q_c = m_c * (float(h_Tp(T_hi, p_co)) - h_ci)
    return min(Q_h, Q_c)


def recuperator_Q(ua, hot_in, cold_in, m_h, m_c, p_ho, p_co, n):
    """Duty (W) and minimum temperature difference (K) of a counterflow
    CO2-CO2 recuperator with fixed UA (segmented, linear pressure)."""
    Q_hi = _recup_Qmax(hot_in, cold_in, m_h, m_c, p_ho, p_co)

    def ua_of_Q(Q):
        hT, cT = _recup_profiles(Q, hot_in, cold_in, m_h, m_c, p_ho, p_co)
        return _ua_from_dT(Q, _profile_dT(hT, cT, n))
    Q = _solve_Q(ua, Q_hi, ua_of_Q)
    hT, cT = _recup_profiles(Q, hot_in, cold_in, m_h, m_c, p_ho, p_co)
    return Q, float(_profile_dT(hT, cT, n).min())


def _heater_profiles(Q, co2_in, m_side, p_out, gas, T_g_in, m_g):
    h_ci, p_ci = co2_in
    h_g_in = float(gas.h(T_g_in))
    h_g_out = h_g_in - Q / m_g
    h_co = h_ci + Q / m_side

    def hot_T(f):
        return gas.T_of_h(h_g_out + f * (h_g_in - h_g_out))

    def cold_T(f):
        return _T_arr(h_ci + f * (h_co - h_ci), p_ci + f * (p_out - p_ci))
    return hot_T, cold_T


def heater_Q(ua, co2_in, m_side, p_out, gas, T_g_in, m_g, T_stack_min, n):
    """Duty (W) and minimum temperature difference (K) of a gas-to-CO2
    heater with fixed UA. The gas outlet is not allowed below T_stack_min."""
    h_ci, p_ci = co2_in
    if m_g <= 1e-9 or m_side <= 1e-9:
        return 0.0, float("nan")
    T_ci = float(T_hp(h_ci, p_ci))
    if T_g_in <= T_ci + 0.01:
        return 0.0, float("nan")
    h_g_in = float(gas.h(T_g_in))
    Q_gas = m_g * (h_g_in - float(gas.h(max(T_stack_min, T_ci))))
    Q_co2 = m_side * (float(h_Tp(T_g_in, p_out)) - h_ci)
    Q_hi = min(Q_gas, Q_co2)

    def ua_of_Q(Q):
        hT, cT = _heater_profiles(Q, co2_in, m_side, p_out, gas, T_g_in, m_g)
        return _ua_from_dT(Q, _profile_dT(hT, cT, n))
    Q = _solve_Q(ua, Q_hi, ua_of_Q)
    hT, cT = _heater_profiles(Q, co2_in, m_side, p_out, gas, T_g_in, m_g)
    return Q, float(_profile_dT(hT, cT, n).min())


# ---------------------------------------------------------------------------
# Hardware: frozen at the design point
# ---------------------------------------------------------------------------
def design_hardware(P, gas, n_seg=20):
    """P: dictionary from evaluate_plant() at the design point.
    UA values are recomputed from the design states with the segmented method
    of this file, so the design point is reproduced exactly."""
    res, hs = P["res"], P["hs"]
    st, m, y = res["states"], hs["m_co2"], res["y"]
    D = {k: v.default for k, v in
         inspect.signature(splitflow_adv).parameters.items()}
    T_g, m_g = P["T_in"], P["m_gas"]

    def ua_rec(hot, cold, m_h, m_c, Q, p_ho, p_co):
        hT, cT = _recup_profiles(Q, (st[hot]["h"], st[hot]["p"]),
                                 (st[cold]["h"], st[cold]["p"]),
                                 m_h, m_c, p_ho, p_co)
        return _ua_from_dT(Q, _profile_dT(hT, cT, n_seg))

    def ua_heat(i_in, m_side, p_out, Q, T_gin):
        hT, cT = _heater_profiles(Q, (st[i_in]["h"], st[i_in]["p"]), m_side,
                                  p_out, gas, T_gin, m_g)
        return _ua_from_dT(Q, _profile_dT(hT, cT, n_seg))

    Q_htr = m * (st[8]["h"] - st[9]["h"])
    Q_ltr = m * (st[9]["h"] - st[10]["h"])
    Q_mh = m * (st[7]["h"] - st[6]["h"])
    Q_bh = y * m * (st[11]["h"] - st[2]["h"])
    p7, p8 = st[7]["p"], st[8]["p"]
    return {
        "n_seg": n_seg, "m_des": m, "y_des": y,
        "T_gas_des": T_g, "m_gas_des": m_g,
        "UA_HTR": ua_rec(8, 5, m, m, Q_htr, st[9]["p"], st[6]["p"]),
        "UA_LTR": ua_rec(9, 2, m, (1 - y) * m, Q_ltr, st[10]["p"], st[3]["p"]),
        "UA_MH": ua_heat(6, m, st[7]["p"], Q_mh, T_g),
        "UA_BH": ua_heat(2, y * m, st[11]["p"], Q_bh, hs["T_gas_mid"]),
        "Kt": m * np.sqrt(st[7]["T"]) / np.sqrt(p7 ** 2 - p8 ** 2),
        "T1": st[1]["T"], "p1": st[1]["p"],
        "p8": p8, "p9": st[9]["p"], "p10": st[10]["p"],
        "dp_heater": D["dp_heater"], "dp_htr_cold": D["dp_htr_cold"],
        "dp_ltr_cold": D["dp_ltr_cold"],
        "eta_mc": D["eta_mc"], "eta_t": D["eta_t"],
        "T7_max": D["T7"], "p_max": st[2]["p"],
        "T_stack_min": EAF["T_stack_min"],
        "h5_des": st[5]["h"], "T7_des": st[7]["T"],
        "net_el_des": P["net_el"],
        # total parasitic load at the design point (generator output minus net)
        "P_par_des": (P["par"]["W_turbine"] * PARASITIC["eta_gearbox"]
                      * PARASITIC["eta_generator"] - P["net_el"]),
    }


# ---------------------------------------------------------------------------
# One steady off-design operating point
# ---------------------------------------------------------------------------
def offdesign_point(hw, gas, T_gas, m_gas, m_co2, y, m_bypass=0.0, x0=None):
    """Solve the plant for the given gas (T_gas in K, m_gas in kg/s), CO2
    flow m_co2 (kg/s), branch fraction y and gas bypass m_bypass (kg/s, gas
    that does not pass the heaters). x0: starting guess [h5 in kJ/kg, T7 in K],
    e.g. the 'x' of the previous point (continuation)."""
    n = hw["n_seg"]
    m_g = m_gas - m_bypass
    if not (0.02 <= y <= 0.98):
        raise ValueError("y must be between 0.02 and 0.98")
    r_co2 = m_co2 / hw["m_des"]
    r_c = (1 - y) * m_co2 / ((1 - hw["y_des"]) * hw["m_des"])
    r_g = max(m_g, 1e-9) / hw["m_gas_des"]
    UA = {"HTR": hw["UA_HTR"] * r_co2 ** N_UA,
          "LTR": hw["UA_LTR"] * (0.5 * (r_co2 + r_c)) ** N_UA,
          "MH": hw["UA_MH"] * r_g ** N_UA,
          "BH": hw["UA_BH"] * r_g ** N_UA}
    T1, p1, p8, p9, p10 = hw["T1"], hw["p1"], hw["p8"], hw["p9"], hw["p10"]
    h_gin = float(gas.h(T_gas))

    def cycle(u):
        h5, T7 = u[0] * 1e3, u[1]
        p7 = np.sqrt(p8 ** 2 + (m_co2 * np.sqrt(max(T7, 250.0)) / hw["Kt"]) ** 2)
        p6 = p7 / (1 - hw["dp_heater"])
        p5 = p6 / (1 - hw["dp_htr_cold"])
        p3 = p5
        p2 = p3 / (1 - hw["dp_ltr_cold"])
        s1 = _state_Tp(FLUID, T1, p1)
        s2, w_mc = _compressor(FLUID, s1, p2, hw["eta_mc"])
        s7 = _state_Tp(FLUID, T7, p7)
        s8, w_t = _turbine(FLUID, s7, p8, hw["eta_t"])
        # HTR
        Q_htr, dT_htr = recuperator_Q(UA["HTR"], (s8["h"], p8), (h5, p5),
                                      m_co2, m_co2, p9, p6, n)
        s6 = _state_hp(FLUID, h5 + Q_htr / m_co2, p6)
        s9 = _state_hp(FLUID, s8["h"] - Q_htr / m_co2, p9)
        # main heater
        Q_mh, dT_mh = heater_Q(UA["MH"], (s6["h"], p6), m_co2, p7, gas,
                               T_gas, m_g, hw["T_stack_min"], n)
        T7_calc = float(T_hp(s6["h"] + Q_mh / m_co2, p7))
        h_g2 = h_gin - Q_mh / max(m_g, 1e-9)
        T_g2 = float(gas.T_of_h(h_g2))
        # branch heater
        Q_bh, dT_bh = heater_Q(UA["BH"], (s2["h"], p2), y * m_co2, p3, gas,
                               T_g2, m_g, hw["T_stack_min"], n)
        s11 = _state_hp(FLUID, s2["h"] + Q_bh / (y * m_co2), p3)
        # LTR
        Q_ltr, dT_ltr = recuperator_Q(UA["LTR"], (s9["h"], p9), (s2["h"], p2),
                                      m_co2, (1 - y) * m_co2, p10, p3, n)
        s3 = _state_hp(FLUID, s2["h"] + Q_ltr / ((1 - y) * m_co2), p3)
        s10 = _state_hp(FLUID, s9["h"] - Q_ltr / m_co2, p10)
        h5_calc = (1 - y) * s3["h"] + y * s11["h"]
        return {"resid": np.array([(h5_calc - h5) / 1e3, T7_calc - T7]),
                "states": {1: s1, 2: s2, 3: s3, 5: _state_hp(FLUID, h5, p5),
                           6: s6, 7: s7, 8: s8, 9: s9, 10: s10, 11: s11},
                "w_mc": w_mc, "w_t": w_t, "Q_htr": Q_htr, "Q_ltr": Q_ltr,
                "Q_mh": Q_mh, "Q_bh": Q_bh, "h_g_out": h_g2 - Q_bh / max(m_g, 1e-9),
                "dT": {"HTR": dT_htr, "LTR": dT_ltr, "MH": dT_mh, "BH": dT_bh}}

    def resid(u):
        try:
            return cycle(u)["resid"]
        except Exception:
            return np.array([1e3, 1e3])

    if x0 is None:
        x0 = [hw["h5_des"] / 1e3, hw["T7_des"]]
    sol = root(resid, x0, method="hybr", options={"xtol": 1e-10, "eps": 1e-6})
    c = cycle(sol.x)
    err = float(np.max(np.abs(c["resid"])))
    if err > 1e-3:
        raise RuntimeError(f"Off-design solution did not converge "
                           f"(residual {err:.2e}): {sol.message}")

    st = c["states"]
    # gas leaving the heaters, and the stack after mixing with the bypass
    h_stack = (m_g * c["h_g_out"] + m_bypass * h_gin) / m_gas
    T_stack = float(gas.T_of_h(h_stack))
    T_g3 = float(gas.T_of_h(c["h_g_out"]))
    Q_use = c["Q_mh"] + c["Q_bh"]
    Q_avail = m_gas * (h_gin - float(gas.h(hw["T_stack_min"])))

    # electrical output with the same loss formulas as the design point
    res_like = {"w_t": c["w_t"], "w_mc": c["w_mc"],
                "q_out": st[10]["h"] - st[1]["h"],
                "q_in": st[7]["h"] - st[6]["h"],
                "q_b": y * (st[11]["h"] - st[2]["h"])}
    hs_like = {"m_co2": m_co2, "T_out": T_stack}
    par = parasitic_loads(res_like, hs_like, gas, m_gas, EAF["p"],
                          dict(PARASITIC, drive="motor"))
    Q_cool = m_co2 * res_like["q_out"]
    P_tower = COOLING["wet_tower"]["fan_pump_fraction"] * Q_cool
    net_el = par["net_el"] + par["fan_air"] - P_tower

    flags = []
    if st[7]["T"] > hw["T7_max"] + 0.5:
        flags.append("T7 above the limit")
    if st[2]["p"] > hw["p_max"] * (1 + 1e-4):
        flags.append("pressure above the design limit")
    if T_g3 < hw["T_stack_min"] - 0.5:
        flags.append("gas below the minimum stack temperature")
    if min(c["dT"]["HTR"], c["dT"]["LTR"]) < DT_REC_MIN - 1e-6:
        flags.append("recuperator pinch below the minimum")
    if m_co2 < M_MIN_FRAC * hw["m_des"]:
        flags.append("CO2 flow below the minimum")
    return {
        "x": [sol.x[0], sol.x[1]], "residual": err, "states": st,
        "m_co2": m_co2, "y": y, "m_gas": m_gas, "m_bypass": m_bypass,
        "T7": st[7]["T"], "T_b": st[11]["T"],
        "p7": st[7]["p"], "p2": st[2]["p"],
        "W_t": m_co2 * c["w_t"], "W_mc": m_co2 * c["w_mc"],
        "net_shaft": m_co2 * (c["w_t"] - c["w_mc"]),
        "net_el": net_el, "eta_el": net_el / Q_use if Q_use > 0 else float("nan"),
        "Q_mh": c["Q_mh"], "Q_bh": c["Q_bh"], "Q_use": Q_use,
        "Q_avail": Q_avail, "heat_used_pct": Q_use / Q_avail * 100.0,
        "T_gas_out_heaters": T_g3, "T_stack": T_stack,
        "min_dT": c["dT"], "UA": UA, "par": par, "flags": flags}