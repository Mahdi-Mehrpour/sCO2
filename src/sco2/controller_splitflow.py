"""Steady-state controller of the split-flow plant ("follow the gas").

For a given gas condition (T_gas, m_gas) it chooses the CO2 flow, the branch
fraction y and the gas bypass, using the frozen hardware of offdesign_splitflow.

Logic for a given y:
  1. HOT gas (the maximum CO2 flow, which sets the 25 MPa limit, cannot absorb
     all the heat at T7 = 550 C): CO2 flow at the maximum; the gas bypass is
     opened just enough to hold T7 at the limit.
  2. Otherwise the CO2 flow is the free variable. T7 falls as the flow rises,
     so the flow that gives T7 = 550 C is the lowest allowed (m*, or the
     minimum flow if the gas is too cool to reach 550 C even there). Between m*
     and the maximum flow the flow with the highest net power is chosen.
y is either fixed at the design value or optimised (bounded 1-D search).
The plant idles if the best net electrical power is below net_el_min_run or T7
is below T7_min_run.

ALL CONTROL PARAMETERS BELOW ARE PLACEHOLDERS (no compressor map, no vendor
data): minimum flow, run threshold, idle consumption."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .offdesign_splitflow import offdesign_point

CONTROL = {
    "m_min_frac": 0.30,             # minimum CO2 flow / design flow (surge limit)
    "net_el_min_run_W": 0.1e6,      # W, below this the plant idles
    "T7_min_run_K": 623.15,         # K (350 C), below this turbine inlet temperature the plant idles
    "idle_parasitic_frac": 0.25,    # idle consumption / design parasitic load
    "y_bounds": (0.08, 0.55),
    "y_xatol": 0.015,
    "m_xatol_frac": 0.01,           # tolerance of the flow search / design flow
    "b_xatol": 0.002,               # kg/s, tolerance of the bypass search
    "b_max_frac": 0.90,             # largest bypass / gas flow
}


def operate(hw, gas, T_gas, m_gas, y_mode="optimise", ctrl=CONTROL, x0=None):
    """Steady operating point chosen by the controller.
    y_mode: 'fixed' (design y) or 'optimise'. x0: starting guess for the
    solver (the 'x' of a nearby point)."""
    m_des, T7_max = hw["m_des"], hw["T7_max"]
    m_min = ctrl["m_min_frac"] * m_des
    st = {"x": x0, "n": 0}

    def pt(m, y, b=0.0):
        """One off-design solution, or None if the solver fails."""
        st["n"] += 1
        for guess in (st["x"], None):
            try:
                r = offdesign_point(hw, gas, T_gas, m_gas, m, y, b, x0=guess)
                st["x"] = r["x"]
                return r
            except (RuntimeError, ValueError):
                continue
        return None

    def T7_of(m, y, b=0.0):
        r = pt(m, y, b)
        return np.nan if r is None else r["T7"]

    def inner(y):
        """Best point for this y (None if nothing converges)."""
        r_cap = pt(m_des, y)
        if r_cap is None:
            return None
        # 1) hot gas: maximum flow, bypass to hold T7
        if r_cap["T7"] > T7_max + 0.05:
            b_hi = ctrl["b_max_frac"] * m_gas
            if not T7_of(m_des, y, b_hi) < T7_max:
                return r_cap
            b = brentq(lambda b: T7_of(m_des, y, b) - T7_max, 0.0, b_hi,
                       xtol=ctrl["b_xatol"])
            return pt(m_des, y, b)
        # 2) flow is the free variable, between m_star and the maximum
        r_min = pt(m_min, y)
        f_hi = r_cap["T7"] - T7_max          # <= 0 here (about 0 at the design gas)
        if f_hi > -1e-3:
            m_star = m_des                   # the maximum flow already holds the limit
        elif r_min is not None and r_min["T7"] - T7_max > 0.0:
            f_lo = r_min["T7"] - T7_max      # > 0

            def g(m):
                # end values come from the points already solved, so the bracket
                # is consistent even if a fresh solve differs in the last digits
                if m >= m_des:
                    return f_hi
                if m <= m_min:
                    return f_lo
                v = T7_of(m, y)
                return f_lo if np.isnan(v) else v - T7_max
            try:
                m_star = brentq(g, m_min, m_des, xtol=ctrl["m_xatol_frac"] * m_des)
            except (ValueError, RuntimeError):
                m_star = m_min               # wide search range, still safe
        else:
            m_star = m_min
        cands = [r_cap]
        r_star = pt(m_star, y)
        if r_star is not None:
            cands.append(r_star)
        if m_des - m_star > 2 * ctrl["m_xatol_frac"] * m_des:
            def f(m):
                r = pt(m, y)
                return 1e3 if r is None else -r["net_el"] / 1e6
            o = minimize_scalar(f, bounds=(m_star, m_des), method="bounded",
                                options={"xatol": ctrl["m_xatol_frac"] * m_des})
            r_o = pt(o.x, y)
            if r_o is not None:
                cands.append(r_o)
        return max(cands, key=lambda r: r["net_el"])

    best = {"r": None}

    def y_obj(y):
        r = inner(y)
        if r is None:
            return 1e3
        if best["r"] is None or r["net_el"] > best["r"]["net_el"]:
            best["r"] = r
        return -r["net_el"] / 1e6

    if y_mode == "fixed":
        y_obj(hw["y_des"])
    elif y_mode == "optimise":
        y_obj(hw["y_des"])
        lo, hi = ctrl["y_bounds"]
        minimize_scalar(y_obj, bounds=(lo, hi), method="bounded",
                        options={"xatol": ctrl["y_xatol"]})
    else:
        raise ValueError("y_mode must be 'fixed' or 'optimise'")

    r = best["r"]
    P_idle = -ctrl["idle_parasitic_frac"] * hw["P_par_des"]
    if r is None:
        return {"mode": "idle", "net_el": P_idle, "m_co2": 0.0, "y": float("nan"),
                "m_bypass": m_gas, "n_eval": st["n"], "point": None,
                "x": st["x"], "flags": ["no converged operating point"]}
    mode = ("run" if (r["net_el"] >= ctrl["net_el_min_run_W"]
                      and r["T7"] >= ctrl["T7_min_run_K"]) else "idle")
    return {"mode": mode,
            "net_el": r["net_el"] if mode == "run" else P_idle,
            "net_el_if_run": r["net_el"],
            "m_co2": r["m_co2"], "y": r["y"], "m_bypass": r["m_bypass"],
            "T7": r["T7"], "p2": r["p2"], "T_b": r["T_b"],
            "T_stack": r["T_stack"], "heat_used_pct": r["heat_used_pct"],
            "flags": r["flags"], "n_eval": st["n"], "point": r,
            "x": r["x"]}