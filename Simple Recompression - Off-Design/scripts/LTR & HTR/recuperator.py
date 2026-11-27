"""Off-design model of a counterflow recuperator (HTR or LTR) with a fixed UA.

Given the inlet states, the two mass flows and the UA of the hardware, the duty q is the value for which the
segment-wise UA of the exchanger equals the fixed UA. The segment method is the one used in the cycle model
(sco2.cycle_adv._pinch_profile): the duty is split into n equal parts, CO2 properties are evaluated at every point
at the local enthalpy and pressure, and UA = sum of (segment duty / mean temperature difference). So the
effectiveness is not assumed: it follows from the flows and the CO2 properties, and the pinch is a result.

SegmentedRecuperator (below) is the model used for off-design work: the fixed hardware is the set of design segment conductances
and resistance shares (tables/HTR_resistance_shares.csv, LTR_resistance_shares.csv), and every segment conductance follows the
local flows and properties. The lumped fixed-UA functions are kept for comparison."""
from pathlib import Path

import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq

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


# ---------------------------------------------------------------------------------------------------------------------
# Segmented off-design model
# ---------------------------------------------------------------------------------------------------------------------
class CO2Table:
    """Cubic-spline tables of T, rho and mu of CO2 on an (h, p) grid (built once from CoolProp, cached in an .npz file).
    k and cp are always taken from CoolProp (splines were not accurate enough for them near the pseudo-critical line)."""
    H = np.arange(250e3, 1000.1e3, 2e3)
    P = np.arange(7.0e6, 27.01e6, 0.5e6)
    NAMES = ("T", "D", "V")

    def __init__(self, cache=None):
        Z = None
        if cache is not None and Path(cache).exists():
            z = np.load(cache)
            if z["H"].shape == self.H.shape and z["P"].shape == self.P.shape:
                Z = {k: z[k] for k in self.NAMES}
        if Z is None:
            Z = {k: np.full((len(self.H), len(self.P)), np.nan) for k in self.NAMES}
            for j, p in enumerate(self.P):
                for i, h in enumerate(self.H):
                    try:
                        for k in self.NAMES:
                            Z[k][i, j] = PropsSI(k, "H", h, "P", p, FLUID)
                    except ValueError:
                        pass
            for k in self.NAMES:                        # points outside the property range: copy the neighbour at lower h
                for i in range(1, len(self.H)):
                    bad = np.isnan(Z[k][i])
                    Z[k][i][bad] = Z[k][i - 1][bad]
                Z[k][0][np.isnan(Z[k][0])] = 0.0
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
        g = lambda q: PropsSI(q, "H", h, "P", p, FLUID)
        if not self.inside(h, p):
            return {"rho": g("D"), "mu": g("V"), "k": g("L"), "cp": g("C")}
        return {"rho": float(self.sp["D"].ev(h, p)), "mu": float(self.sp["V"].ev(h, p)), "k": g("L"), "cp": g("C")}

    def accuracy(self, n=300, seed=1, p_lo=7.6e6):
        rng = np.random.default_rng(seed)
        err = {k: 0.0 for k in self.NAMES}
        for h, p in zip(rng.uniform(300e3, 900e3, n), rng.uniform(p_lo, 26e6, n)):
            try:
                ref = {k: PropsSI(k, "H", h, "P", p, FLUID) for k in self.NAMES}
            except ValueError:
                continue
            err["T"] = max(err["T"], abs(self.T(h, p) - ref["T"]))
            t = self.props(h, p)
            err["D"] = max(err["D"], abs(t["rho"] / ref["D"] - 1.0))
            err["V"] = max(err["V"], abs(t["mu"] / ref["V"] - 1.0))
        return err


def _lmtd(a, b):
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return a if abs(a - b) < 1e-9 * max(a, b) else (b - a) / np.log(b / a)


class SegmentedRecuperator:
    def __init__(self, shares, design, hot_out_des, cold_out_des, ref, table):
        """shares: DataFrame of <name>_resistance_shares.csv; design: design_point.recuperator_design()[name]; hot_out_des,
        cold_out_des: design outlet states (h, p); ref: reference-data Series (a = Nu_a, b = Nu_b, f_d, limits); table: CO2Table.
        Per segment (i = 0 is the cold end: hot outlet, cold inlet):  1/UA_i = (1/UA_i,des) [ f_hot R_hot + f_cold R_cold + f_wall ]
        with R = h_des / h and h / h_des = (Re / Re_des)^a (Pr / Pr_des)^b (k / k_des), Re / Re_des = (m / m_des)(mu_des / mu);
        local properties are taken at the cold-end boundary of the segment on each side, design values at the same place.
        Every segment obeys Q = UA_i LMTD_i (exact counterflow, equal-duty segments of the design are not assumed off-design:
        the duty of each segment is solved). Pressure drop per segment: dp = dp_des (rho_des/rho) (m/m_des)^(2+f_d) (mu/mu_des)^(-f_d)."""
        self.tab, self.N = table, len(shares)
        self.a, self.b, self.fd = float(ref["Nu_a"]), float(ref["Nu_b"]), float(ref["f_d"])
        self.Re_lim, self.Pr_lim = (float(ref["Re_min"]), float(ref["Re_max"])), (float(ref["Pr_min"]), float(ref["Pr_max"]))
        s = shares
        self.UA_des = s["UA_i (W/K)"].to_numpy(float)
        self.fh, self.fc, self.fw = (s["f_hot"].to_numpy(float), s["f_cold"].to_numpy(float), s["f_wall"].to_numpy(float))
        self.Q_des = float(s["Q_i (W)"].sum())
        self.m_h_des, self.m_c_des = design["m_hot"], design["m_cold"]
        hi, ci = design["hot_in"], design["cold_in"]
        f = np.concatenate([s["f_from"].to_numpy(float), [1.0]])
        h_h = hot_out_des["h"] + f * (hi["h"] - hot_out_des["h"])
        p_h = hot_out_des["p"] + f * (hi["p"] - hot_out_des["p"])
        h_c = ci["h"] + f * (cold_out_des["h"] - ci["h"])
        p_c = ci["p"] + f * (cold_out_des["p"] - ci["p"])
        self.hot_des = [table.props(h_h[i], p_h[i]) for i in range(self.N)]
        self.cold_des = [table.props(h_c[i], p_c[i]) for i in range(self.N)]
        self.dp_h_des = (hi["p"] - hot_out_des["p"]) / self.N
        self.dp_c_des = (ci["p"] - cold_out_des["p"]) / self.N
        self.p_h_in_des, self.p_c_in_des = hi["p"], ci["p"]
        self.UA_total_des = float(self.UA_des.sum())
        self.name = None

    def _ratio(self, m, m_des, pr, des):
        Re_r = (m / m_des) * des["mu"] / pr["mu"]
        Pr, Pr_d = pr["cp"] * pr["mu"] / pr["k"], des["cp"] * des["mu"] / des["k"]
        return Re_r ** self.a * (Pr / Pr_d) ** self.b * (pr["k"] / des["k"])

    def _march(self, h_ho, m_h, m_c, h_hot_in, cold_in, p_h, p_c):
        N = self.N
        h_h, h_c = h_ho, cold_in["h"]
        Th, Tc = self.tab.T(h_h, p_h[0]), self.tab.T(h_c, p_c[0])
        o = {k: np.zeros(N + 1) for k in ("Th", "Tc", "hh", "hc")}
        o["Th"][0], o["Tc"][0], o["hh"][0], o["hc"][0] = Th, Tc, h_h, h_c
        Q_i, UA_i, Re_h, Re_c, Pr_h, Pr_c = (np.zeros(N) for _ in range(6))
        rho_h, rho_c, mu_h, mu_c = (np.zeros(N) for _ in range(4))
        cross = False
        for i in range(N):
            ph, pc = self.tab.props(h_h, p_h[i]), self.tab.props(h_c, p_c[i])
            Rh = self._ratio(m_h, self.m_h_des, ph, self.hot_des[i])
            Rc = self._ratio(m_c, self.m_c_des, pc, self.cold_des[i])
            UA = self.UA_des[i] / (self.fh[i] / Rh + self.fc[i] / Rc + self.fw[i])
            dT0 = Th - Tc
            ph2, pc2 = p_h[i + 1], p_c[i + 1]

            def ends(Q):
                return self.tab.T(h_h + Q / m_h, ph2), self.tab.T(h_c + Q / m_c, pc2)

            def g(Q):
                try:
                    Th2, Tc2 = ends(Q)
                except ValueError:
                    return 1.0
                dTh = Th2 - Tc2
                return Q if dTh <= 0.0 else Q - UA * _lmtd(dT0, dTh)
            Q = 0.0
            if dT0 > 0.0:
                hi = UA * dT0
                for _ in range(80):
                    if g(hi) > 0.0:
                        break
                    hi *= 2.0
                Q = brentq(g, 0.0, hi, xtol=1.0, rtol=1e-9)
            Th2, Tc2 = ends(Q)
            cross = cross or (Th2 - Tc2 <= 0.0)
            Q_i[i], UA_i[i] = Q, UA
            Re_h[i] = self.hot_des[i]["mu"] / ph["mu"] * m_h / self.m_h_des * self._Re_des_h[i]
            Re_c[i] = self.cold_des[i]["mu"] / pc["mu"] * m_c / self.m_c_des * self._Re_des_c[i]
            Pr_h[i], Pr_c[i] = ph["cp"] * ph["mu"] / ph["k"], pc["cp"] * pc["mu"] / pc["k"]
            rho_h[i], rho_c[i], mu_h[i], mu_c[i] = ph["rho"], pc["rho"], ph["mu"], pc["mu"]
            h_h, h_c, Th, Tc = h_h + Q / m_h, h_c + Q / m_c, Th2, Tc2
            o["Th"][i + 1], o["Tc"][i + 1], o["hh"][i + 1], o["hc"][i + 1] = Th, Tc, h_h, h_c
        return h_h, {"Q_i": Q_i, "UA_i": UA_i, "Re_h": Re_h, "Re_c": Re_c, "Pr_h": Pr_h, "Pr_c": Pr_c, "rho_h": rho_h,
                     "rho_c": rho_c, "mu_h": mu_h, "mu_c": mu_c, "cross": cross, **o}

    def attach_reynolds(self, shares):
        """Design Reynolds numbers per segment (from the shares table), used only to report the Reynolds number off-design."""
        self._Re_des_h = shares["Re_hot"].to_numpy(float)
        self._Re_des_c = shares["Re_cold"].to_numpy(float)

    def _dp_profiles(self, m_h, m_c, hot_in_p, cold_in_p, m=None):
        N = self.N
        if m is None:                                          # first guess: design split, scaled by the flow only
            dh = np.full(N, self.dp_h_des * (m_h / self.m_h_des) ** (2.0 + self.fd))
            dc = np.full(N, self.dp_c_des * (m_c / self.m_c_des) ** (2.0 + self.fd))
        else:
            dh = np.array([self.dp_h_des * (self.hot_des[i]["rho"] / m["rho_h"][i]) * (m_h / self.m_h_des) ** (2.0 + self.fd)
                           * (m["mu_h"][i] / self.hot_des[i]["mu"]) ** (-self.fd) for i in range(N)])
            dc = np.array([self.dp_c_des * (self.cold_des[i]["rho"] / m["rho_c"][i]) * (m_c / self.m_c_des) ** (2.0 + self.fd)
                           * (m["mu_c"][i] / self.cold_des[i]["mu"]) ** (-self.fd) for i in range(N)])
        p_h = np.concatenate([[0.0], np.zeros(N)])
        p_h[N] = hot_in_p
        for i in range(N - 1, -1, -1):
            p_h[i] = p_h[i + 1] - dh[i]
        p_c = np.zeros(N + 1)
        p_c[0] = cold_in_p
        for i in range(N):
            p_c[i + 1] = p_c[i] - dc[i]
        return p_h, p_c

    def solve(self, hot_in, cold_in, m_hot, m_cold, passes=3):
        """Rating mode. hot_in, cold_in: dicts with h, p (T optional); flows in kg/s. Returns duty q, effectiveness, outlet states,
        minimum dT, total UA, Reynolds and Prandtl ranges and the number of segments outside the correlation range."""
        cold_in = dict(cold_in)
        cold_in["T"] = self.tab.T(cold_in["h"], cold_in["p"])
        p_h, p_c = self._dp_profiles(m_hot, m_cold, hot_in["p"], cold_in["p"])
        for _ in range(passes):
            lo = PropsSI("H", "T", cold_in["T"] + 1e-6, "P", p_h[0], FLUID)

            def F(h_ho):
                try:
                    return self._march(h_ho, m_hot, m_cold, hot_in["h"], cold_in, p_h, p_c)[0] - hot_in["h"]
                except ValueError:
                    return 1e9
            h_ho = lo if F(lo) >= 0.0 else brentq(F, lo, hot_in["h"], xtol=0.05, rtol=1e-10)
            _, m = self._march(h_ho, m_hot, m_cold, hot_in["h"], cold_in, p_h, p_c)
            p_h_new, p_c_new = self._dp_profiles(m_hot, m_cold, hot_in["p"], cold_in["p"], m)
            change = max(abs(p_h_new[0] - p_h[0]), abs(p_c_new[-1] - p_c[-1]))
            p_h, p_c = p_h_new, p_c_new
            if change < 200.0:
                break
        h_ho = lo if F(lo) >= 0.0 else brentq(F, lo, hot_in["h"], xtol=0.05, rtol=1e-10)
        _, m = self._march(h_ho, m_hot, m_cold, hot_in["h"], cold_in, p_h, p_c)
        q = float(m["Q_i"].sum())
        qm = q_max({"h": hot_in["h"], "p": p_h[-1], "T": self.tab.T(hot_in["h"], p_h[-1])}, cold_in, m_hot, m_cold)
        dT = m["Th"] - m["Tc"]
        out_re = int(((m["Re_h"] < self.Re_lim[0]) | (m["Re_h"] > self.Re_lim[1])).sum() + ((m["Re_c"] < self.Re_lim[0]) | (m["Re_c"] > self.Re_lim[1])).sum())
        return {"q": q, "eps": q / qm, "q_max": qm, "UA": float(m["UA_i"].sum()), "min_dT": float(dT.min()), "cross": m["cross"],
                "hot_out": {"h": float(m["hh"][0]), "p": float(p_h[0]), "T": float(m["Th"][0])},
                "cold_out": {"h": float(m["hc"][-1]), "p": float(p_c[-1]), "T": float(m["Tc"][-1])},
                "dp_hot": float(hot_in["p"] - p_h[0]), "dp_cold": float(cold_in["p"] - p_c[-1]),
                "Re_hot_min": float(m["Re_h"].min()), "Re_cold_min": float(m["Re_c"].min()),
                "Re_hot_max": float(m["Re_h"].max()), "Re_cold_max": float(m["Re_c"].max()),
                "Pr_hot_max": float(m["Pr_h"].max()), "Pr_cold_max": float(m["Pr_c"].max()), "n_outside_Re": out_re,
                "Q_i": m["Q_i"], "UA_i": m["UA_i"], "T_hot": m["Th"], "T_cold": m["Tc"], "Re_hot": m["Re_h"], "Re_cold": m["Re_c"]}
