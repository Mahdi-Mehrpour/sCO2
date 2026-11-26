"""Off-design model of the water-cooled main cooler: straight-channel PCHE (CO2 -> cooling water) fed by a wet cooling tower.

Same approach as the gas heater and the recuperators. The hardware is fixed from the design: n_seg equal-duty segments with
their design conductance UA_i,des (log-mean definition) and resistance shares (CO2 / wall / water) read from
tables/cooler_resistance_shares.csv, written by cooler_resistance_shares.py from the cycle design point. In every segment

    1/UA_i = (1/UA_i,des) [ f_co2 / R_co2 + f_wall + f_water / R_water ],   Q_i = UA_i * LMTD_i  (exact counterflow)

R_co2 and R_water are the ratios of the local heat-transfer coefficients to the design ones:
  CO2:   h ~ Re^0.8 Pr^0.4 k   (local CO2 properties at the cold end of the segment, so the pseudo-critical rise of cp
         is followed; the wall-temperature effect of Dang & Hihara is not repeated here)
  water: h = Nu(Re, Pr) k / Dh; Nu = 4.089 (laminar) below Re 2300, Gnielinski above 4000, linear (VDI) interpolation between.
         Most segments are laminar at design, so the water-side coefficient hardly responds to the water flow.
CO2 pressure drop per segment: dp = dp_des (rho_des / rho) (m / m_des)^2. Water pressure drop: friction factor with Re^-1
(laminar) turning into Re^-0.2 (turbulent) with the same interpolation weight.

Rating mode (solve): given the water inlet temperature and flow, the CO2 inlet enthalpy and flow and the CO2 outlet pressure
p1, find the CO2 outlet temperature T1, the duty, the water outlet temperature and the CO2 inlet pressure. The segments are
marched from the cold end (CO2 outlet / water inlet) for a guessed CO2 outlet enthalpy; brentq matches the CO2 inlet enthalpy.
Coupled mode (solve_coupled): the water inlet temperature comes from the cooling tower (cooling_tower.py) for the wet-bulb
temperature, water flow and tower air flow; the water loop is closed by fixed-point iteration on the water outlet temperature.
Control: water flow (and tower air flow at constant L/G) up to the pump and fan limits; beyond them T1 rises (flow_for_T1).
Auxiliaries: pump (cooler + loop pressure drop) and tower fan (cube of the air flow). Evaporation estimate Q / h_fg."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq

FLUID = "CO2"
CPW = 4180.0                                           # J/kg.K, water (the same value as in the design script)
K = 273.15


def _lmtd(a, b):
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return a if abs(a - b) < 1e-9 * max(a, b) else (b - a) / math.log(b / a)


def _gnielinski(Re, Pr):
    f = (0.79 * math.log(Re) - 1.64) ** -2
    return (f / 8) * (Re - 1000) * Pr / (1.0 + 12.7 * math.sqrt(f / 8) * (Pr ** (2 / 3) - 1))


RE_LAM, RE_TURB = 2300.0, 4000.0                       # laminar below RE_LAM, turbulent above RE_TURB (set from the reference data)


def turb_weight(Re):
    """0 below RE_LAM (laminar), 1 above RE_TURB (turbulent), linear in between."""
    return min(1.0, max(0.0, (Re - RE_LAM) / (RE_TURB - RE_LAM)))


def nu_water(Re, Pr, nu_lam=4.089):
    """Water-side Nusselt number: laminar constant, Gnielinski above RE_TURB, VDI-type interpolation between."""
    w = turb_weight(Re)
    if w <= 0.0:
        return nu_lam
    return (1.0 - w) * nu_lam + w * _gnielinski(max(Re, 4000.0), Pr)


class CO2Table:
    """Cubic-spline tables of T, rho and mu of CO2 on an (h, p) grid, built once from CoolProp and cached in an .npz file.
    The temperature is the property the segment marching asks for thousands of times per solution; the table makes it about ten
    times cheaper. k and cp vary too sharply near the pseudo-critical line for a spline (a first version showed errors of
    several percent in k and several hundred percent in cp at some points), so they are always taken from CoolProp. Outside
    the grid CoolProp is used. accuracy() compares the table with CoolProp."""
    H = np.concatenate([np.arange(190e3, 400e3, 0.5e3), np.arange(400e3, 560.1e3, 2e3)])
    P = np.arange(7.2e6, 10.21e6, 0.1e6)
    NAMES = ("T", "D", "V")

    def __init__(self, cache=None):
        Z = None
        if cache is not None and Path(cache).exists():
            z = np.load(cache)
            if z["H"].shape == self.H.shape and z["P"].shape == self.P.shape:
                Z = {k: z[k] for k in self.NAMES}
        if Z is None:
            Z = {k: np.zeros((len(self.H), len(self.P))) for k in self.NAMES}
            for j, p in enumerate(self.P):
                for i, h in enumerate(self.H):
                    for k in self.NAMES:
                        Z[k][i, j] = PropsSI(k, "H", h, "P", p, FLUID)
            if cache is not None:
                np.savez(cache, H=self.H, P=self.P, **Z)
        self.sp = {k: RectBivariateSpline(self.H, self.P, Z[k], kx=3, ky=3) for k in self.NAMES}

    def inside(self, h, p):
        return self.H[0] <= h <= self.H[-1] and self.P[0] <= p <= self.P[-1]

    def T(self, h, p):
        if not self.inside(h, p):
            return PropsSI("T", "H", h, "P", p, FLUID)
        return float(self.sp["T"].ev(h, p))

    def props(self, h, p):
        if not self.inside(h, p):
            return co2_props(h, p)
        e = lambda k: float(self.sp[k].ev(h, p))
        return {"rho": e("D"), "mu": e("V"), "k": PropsSI("L", "H", h, "P", p, FLUID), "cp": PropsSI("C", "H", h, "P", p, FLUID)}

    def accuracy(self, n=400, seed=1):
        """Largest relative deviation from CoolProp at random points of the cooler range (T in K, others relative)."""
        rng = np.random.default_rng(seed)
        err = {k: 0.0 for k in self.NAMES}
        for h, p in zip(rng.uniform(260e3, 520e3, n), rng.uniform(7.6e6, 9.5e6, n)):
            t = self.props(h, p)
            ref = co2_props(h, p)
            for k, name in (("rho", "D"), ("mu", "V")):
                err[name] = max(err[name], abs(t[k] / ref[k] - 1.0))
            err["T"] = max(err["T"], abs(self.T(h, p) - PropsSI("T", "H", h, "P", p, FLUID)))
        return err


def water_props(T, p):
    g = lambda q: PropsSI(q, "P", p, "T", T, "Water")
    return {"rho": g("D"), "mu": g("V"), "k": g("L")}


def co2_props(h, p):
    g = lambda q: PropsSI(q, "P", p, "H", h, FLUID)
    return {"rho": g("D"), "mu": g("V"), "k": g("L"), "cp": g("C")}


def load_shares(path):
    return pd.read_csv(path)


class WaterCooler:
    def __init__(self, shares, design, ref, tower=None, table_path=None):
        """shares: DataFrame of cooler_resistance_shares.csv; design: design_point.water_cooler_design(); ref: Series of
        cooler_reference_data.csv (float values by name); tower: CoolingTower (needed by solve_coupled only)."""
        global RE_LAM, RE_TURB
        self.tower, self.ref = tower, ref
        RE_LAM, RE_TURB = float(ref.get("re_lam_max", 2300.0)), float(ref.get("re_turb_full", 4000.0))
        self.tab = CO2Table(table_path)
        self.P_W, self.dT_loop = float(ref["p_water"]), float(ref["dT_loop"])
        d_ch = float(ref["d_ch"])
        self.Dh = 4.0 * (math.pi * d_ch ** 2 / 8) / (math.pi * d_ch / 2 + d_ch)
        s = shares
        self.N = len(s)
        self.Q_i = s["Q_i (W)"].to_numpy(float)
        self.Q_des = float(self.Q_i.sum())
        self.UA_des = s["UA_i (W/K)"].to_numpy(float)
        self.fc, self.fw, self.fwat = (s["f_co2"].to_numpy(float), s["f_wall"].to_numpy(float), s["f_water"].to_numpy(float))
        self.m_c_des = design["m_co2"]
        # design water flow and inlet temperature, rebuilt from the segment-mean water temperatures of the shares table
        Tm = s["T_water_mean (C)"].to_numpy(float) + K
        self.m_w_des = float(self.Q_i[0] / (CPW * (Tm[1] - Tm[0])))
        self.T_w_in_des = float(Tm[0] - 0.5 * self.Q_i[0] / (self.m_w_des * CPW))
        Tw_b = self.T_w_in_des + np.concatenate([[0.0], np.cumsum(self.Q_i)]) / (self.m_w_des * CPW)
        self.T_w_out_des = float(Tw_b[-1])
        # design CO2 state at the cold end of every segment (f = 0 is the CO2 outlet)
        co_i, co_o = design["co2_in"], design["co2_out"]
        f = np.concatenate([s["f_from"].to_numpy(float), [1.0]])
        h_b = co_o["h"] + f * (co_i["h"] - co_o["h"])
        p_b = co_o["p"] + f * (co_i["p"] - co_o["p"])
        self.co2_des = [self.tab.props(h_b[i], p_b[i]) for i in range(self.N)]
        self.wat_des = [water_props(Tw_b[i], self.P_W) for i in range(self.N)]
        self.Re_w_des = s["Re_water"].to_numpy(float)
        for i in range(self.N):
            self.wat_des[i]["cp"] = CPW
        self.Pr_w_des = np.array([CPW * w["mu"] / w["k"] for w in self.wat_des])
        self.dp_c_seg = (co_i["p"] - co_o["p"]) / self.N      # design CO2 pressure drop per segment (cycle value, equal split)
        self.dp_w_des = self._water_dp_des(s)
        self.Nu_w_des = np.array([nu_water(self.Re_w_des[i], self.Pr_w_des[i]) for i in range(self.N)])
        self.UA_total_des = float(self.UA_des.sum())
        self.pump_eta, self.loop_dp = float(ref["pump_eta"]), float(ref["loop_dp_des"])
        self.m_w_max = float(ref["pump_max_frac"]) * self.m_w_des
        self.fan_max_frac = float(ref["fan_max_frac"])
        self.h_fg = float(ref["h_fg"])
        self.rho_air = float(ref["air_density_tower"])
        self.fan_dp, self.fan_eta = float(ref["tower_fan_dp"]), float(ref["tower_fan_eta"])
        self.LG_des = float(ref["LG_des"])
        self.m_air_des = self.m_w_des / self.LG_des
        self.P_fan_des = (self.m_air_des / self.rho_air) * self.fan_dp / self.fan_eta
        rho_w = self.wat_des[0]["rho"]
        self.P_pump_des = (self.m_w_des / rho_w) * (float(self.dp_w_des.sum()) + self.loop_dp) / self.pump_eta

    def _water_dp_des(self, s):
        """Design water-side friction pressure drop of every segment (Pa) from the shares table (same rules as the design script)."""
        out = np.zeros(self.N)
        for i in range(self.N):
            Re, L = float(s["Re_water"][i]), float(s["L_seg (m)"][i])
            rho, mu = float(s["rho_water"][i]), float(s["mu_water"][i])
            G = Re * mu / self.Dh
            fw = (0.79 * math.log(Re) - 1.64) ** -2 / 4 if Re > 2300 else 15.78 / Re
            out[i] = 4.0 * fw * (L / self.Dh) * G ** 2 / (2.0 * rho)
        return out

    # ------------------------------------------------------------------ one pass through the segments
    def _march(self, h_out, p_out, T_w_in, m_w, m_c):
        N = self.N
        r_c, r_w = m_c / self.m_c_des, m_w / self.m_w_des
        h, p = h_out, p_out
        Tc = self.tab.T(h, p)
        Tw = T_w_in
        o = {k: np.zeros(N + 1) for k in ("Tc", "Tw", "p", "h")}
        o["Tc"][0], o["Tw"][0], o["p"][0], o["h"][0] = Tc, Tw, p, h
        Qs, UAs, Re_w, dpw, cp_pk, cross = np.zeros(N), np.zeros(N), np.zeros(N), 0.0, np.zeros(N), False
        for i in range(N):
            c = self.tab.props(h, p)
            w = water_props(Tw, self.P_W)
            d, dw = self.co2_des[i], self.wat_des[i]
            Re_ratio_c = r_c * d["mu"] / c["mu"]
            Pr_c, Pr_c_des = c["cp"] * c["mu"] / c["k"], d["cp"] * d["mu"] / d["k"]
            R_c = Re_ratio_c ** 0.8 * (Pr_c / Pr_c_des) ** 0.4 * (c["k"] / d["k"])
            Re_w_i = self.Re_w_des[i] * r_w * dw["mu"] / w["mu"]
            Nu = nu_water(Re_w_i, CPW * w["mu"] / w["k"])
            R_w = (Nu / self.Nu_w_des[i]) * (w["k"] / dw["k"])
            UA = self.UA_des[i] / (self.fc[i] / R_c + self.fw[i] + self.fwat[i] / R_w)
            p2 = p + self.dp_c_seg * (d["rho"] / c["rho"]) * r_c ** 2
            dT0 = Tc - Tw

            def hot(Q):
                return self.tab.T(h + Q / m_c, p2), Tw + Q / (m_w * CPW)

            def g(Q):
                try:
                    Tc2, Tw2 = hot(Q)
                except ValueError:
                    return 1.0
                dTh = Tc2 - Tw2
                return Q if dTh <= 0.0 else Q - UA * _lmtd(dT0, dTh)
            Q = 0.0
            if dT0 > 0.0:
                hi = UA * dT0
                for _ in range(80):
                    if g(hi) > 0.0:
                        break
                    hi *= 2.0
                Q = brentq(g, 0.0, hi, xtol=1.0, rtol=1e-9)
            Tc2, Tw2 = hot(Q)
            if Tc2 - Tw2 <= 0.0:
                cross = True
            wt = turb_weight(Re_w_i)
            m_exp = 1.0 - 0.8 * wt
            dpw += self.dp_w_des[i] * r_w ** (2.0 - m_exp) * (w["mu"] / dw["mu"]) ** m_exp * (dw["rho"] / w["rho"])
            Qs[i], UAs[i], Re_w[i], cp_pk[i] = Q, UA, Re_w_i, c["cp"]
            h, p, Tc, Tw = h + Q / m_c, p2, Tc2, Tw2
            o["Tc"][i + 1], o["Tw"][i + 1], o["p"][i + 1], o["h"][i + 1] = Tc, Tw, p, h
        return h, {"Q_i": Qs, "UA_i": UAs, "Re_w": Re_w, "dp_w": dpw, "cp": cp_pk, "cross": cross, **o}

    def solve(self, T_w_in, m_w, h_in, p_out, m_c):
        """Rating mode. T_w_in (K), m_w, m_c (kg/s), h_in (J/kg, CO2 inlet), p_out = p1 (Pa)."""
        lo = PropsSI("H", "T", T_w_in + 1e-6, "P", p_out, FLUID)

        def F(h_out):
            try:
                return self._march(h_out, p_out, T_w_in, m_w, m_c)[0] - h_in
            except ValueError:
                return 1e9
        if F(lo) >= 0.0:
            h_out = lo
        else:
            h_out = brentq(F, lo, h_in, xtol=0.02, rtol=1e-10)
        h_top, m = self._march(h_out, p_out, T_w_in, m_w, m_c)
        Q = float(m["Q_i"].sum())
        dT = m["Tc"] - m["Tw"]
        return {"Q": Q, "T1": float(m["Tc"][0]), "h1": h_out, "T_w_in": T_w_in, "T_w_out": float(m["Tw"][-1]),
                "p_in": float(m["p"][-1]), "dp_co2": float(m["p"][-1] - p_out), "dp_water": float(m["dp_w"]),
                "min_dT": float(dT.min()), "UA_total": float(m["UA_i"].sum()), "cross": m["cross"],
                "Re_w_max": float(m["Re_w"].max()), "n_turb": int((m["Re_w"] > 2300.0).sum()),
                "f_pc": float(m["cp"].argmax() / self.N),
                "Q_i": m["Q_i"], "UA_i": m["UA_i"], "Re_w": m["Re_w"], "T_co2": m["Tc"], "T_water": m["Tw"], "p": m["p"]}

    def aux(self, m_w, m_air, dp_water):
        """Pump power (W) for the cooler plus loop pressure drop (loop part scales with the flow squared), tower fan power (W)."""
        rho = self.wat_des[0]["rho"]
        P_pump = (m_w / rho) * (dp_water + self.loop_dp * (m_w / self.m_w_des) ** 2) / self.pump_eta
        return P_pump, self.P_fan_des * (m_air / self.m_air_des) ** 3

    def solve_coupled(self, T_wb, m_w, m_air, h_in, p_out, m_c, tol=1e-2, max_iter=12, T_w_out0=None):
        """Cooler plus cooling tower. T_wb in C. The water loop is closed by iteration on the water outlet temperature
        (fixed point, then secant). Returns the rating result with the auxiliaries, the tower approach and the evaporation
        rate added."""
        T_out = self.T_w_out_des if T_w_out0 is None else T_w_out0
        prev = None
        for _ in range(max_iter):
            T_in = self.tower.cold_water(T_wb, m_w, m_air, T_out - K - self.dT_loop) + K + self.dT_loop
            r = self.solve(T_in, m_w, h_in, p_out, m_c)
            gap = r["T_w_out"] - T_out
            if abs(gap) < tol:
                break
            if prev is not None and abs(gap - prev[1]) > 1e-9:
                T_next = T_out - gap * (T_out - prev[0]) / (gap - prev[1])      # secant on gap(T_out) = 0
                T_next = min(max(T_next, T_in + 0.1), T_in + 80.0)
            else:
                T_next = r["T_w_out"]
            prev = (T_out, gap)
            T_out = T_next
        P_pump, P_fan = self.aux(m_w, m_air, r["dp_water"])
        r.update({"T_wb": T_wb, "m_w": m_w, "m_air": m_air, "P_pump": P_pump, "P_fan_tower": P_fan, "P_aux": P_pump + P_fan,
                  "approach": T_in - K - self.dT_loop - T_wb, "m_evap": r["Q"] / self.h_fg})
        return r

    def flow_for_T1(self, T1_target, T_wb, h_in, p_out, m_c, r0=1.0, r_min=0.3, tol=5e-3):
        """Flow (water and tower air at constant L/G, relative to design) that holds T1 = T1_target, limited by the pump and the
        fan. Returns (result, limited). If even the largest flow leaves T1 above the target, the largest flow is returned.
        r0: first guess (use the previous point of a sweep). Illinois regula falsi, about six coupled solutions per point."""
        r_max = min(self.m_w_max / self.m_w_des, self.fan_max_frac)
        last = {}

        def run(r):
            res = self.solve_coupled(T_wb, r * self.m_w_des, r * self.m_air_des, h_in, p_out, m_c, T_w_out0=last.get("T"))
            last["T"] = res["T_w_out"]
            return res
        top = run(r_max)
        if top["T1"] > T1_target:
            return top, True
        hi, f_hi, res_hi = r_max, top["T1"] - T1_target, top            # T1 below the target: enough flow
        r = min(max(r0, r_min), r_max * 0.999)
        while True:                                                     # find a flow with T1 above the target
            res = run(r)
            f = res["T1"] - T1_target
            if abs(f) < tol:
                return res, False
            if f > 0.0:
                lo, f_lo = r, f
                break
            hi, f_hi, res_hi = r, f, res
            if r <= r_min:
                return res, False                                       # even the smallest flow holds T1
            r = max(r * 0.8, r_min)
        side = 0
        for _ in range(14):
            c = hi - f_hi * (hi - lo) / (f_hi - f_lo)
            res = run(c)
            fc = res["T1"] - T1_target
            if abs(fc) < tol or abs(hi - lo) < 1e-3:
                return res, False
            if fc > 0.0:
                lo, f_lo = c, fc
                if side == 1:
                    f_hi *= 0.5
                side = 1
            else:
                hi, f_hi, res_hi = c, fc, res
                if side == -1:
                    f_lo *= 0.5
                side = -1
        return res_hi, False
