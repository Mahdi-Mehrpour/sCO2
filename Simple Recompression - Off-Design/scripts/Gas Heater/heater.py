"""Off-design model of the gas heater (EAF off-gas -> CO2, counterflow) with a fixed UA.

Given the gas inlet temperature and flow, the CO2 inlet state and flow and the UA of the hardware, the duty Q is the
value for which the segment-wise UA equals the fixed UA. The segment method is the one of the cycle model
(sco2.heat_source_adv.couple_heater): the duty is split into n equal parts, the gas temperature follows from its
enthalpy table and the CO2 temperature from CO2 properties at the local enthalpy and pressure (pressure linear).
The turbine inlet temperature T7 (the CO2 outlet) is a result, not a specification.

Two models live here:
  solve(...)           lumped: one UA for the whole heater (kept for comparison and for design_point).
  SegmentedHeater      the off-design model: the 20 design segments are fixed blocks of fixed area; the UA of every segment
                       follows the gas and CO2 flows through the thermal-resistance shares of the design
                       (heater_resistance_shares.py); the CO2 pressure drop follows flow and density; a gas bypass damper
                       is a control variable; the tube-wall temperature is a monitored result."""
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

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


# ----------------------------------------------------------------------------------------------------------------------
# Segmented off-design model
# ----------------------------------------------------------------------------------------------------------------------
GAS_NAME = {"N2": "Nitrogen", "CO2": "CarbonDioxide", "H2O": "Water", "O2": "Oxygen", "CO": "Nitrogen"}   # CO lumped into N2
P_ATM = 101325.0


def _lmtd(a, b):
    """Log-mean temperature difference of the end differences a and b."""
    return a if abs(a - b) <= 1e-9 * max(abs(a), abs(b)) else (b - a) / np.log(b / a)


class GasTransport:
    """Transport properties of the off-gas (ideal-gas mixture of pure CoolProp fluids at 1 atm, Wilke mixing for viscosity and
    conductivity; CO is lumped into N2 because CoolProp has no viscosity model for it). Also the property group that sets
    the gas-side heat transfer coefficient at fixed mass flux, h ~ k Pr^(1/3) mu^(-0.681) (Briggs & Young), tabulated on a
    temperature grid so that the heater solver does not call CoolProp for it."""

    def __init__(self, X, T_grid=None):
        self.x = {}
        for k, v in X.items():
            self.x[GAS_NAME[k]] = self.x.get(GAS_NAME[k], 0.0) + v
        self.MW = {k: PropsSI("M", k) for k in self.x}
        self.Mmix = sum(self.x[k] * self.MW[k] for k in self.x)
        self._T = np.linspace(300.0, 1500.0, 41) if T_grid is None else np.asarray(T_grid, float)
        self._G = np.array([self.group(T) for T in self._T])

    def props(self, T):
        xs, MW = self.x, self.MW
        cp = sum(xs[k] * MW[k] * PropsSI("Cp0mass", "T", T, "P", P_ATM, k) for k in xs) / self.Mmix
        mu = {k: PropsSI("V", "T", T, "P", P_ATM, k) for k in xs}
        kk = {k: PropsSI("L", "T", T, "P", P_ATM, k) for k in xs}

        def wilke(prop):
            tot = 0.0
            for i in xs:
                den = 0.0
                for j in xs:
                    phi = (1 + (mu[i] / mu[j]) ** 0.5 * (MW[j] / MW[i]) ** 0.25) ** 2 / (8 * (1 + MW[i] / MW[j])) ** 0.5
                    den += xs[j] * phi
                tot += xs[i] * prop[i] / den
            return tot
        return {"cp": cp, "mu": wilke(mu), "k": wilke(kk), "rho": P_ATM * self.Mmix / (8.314462 * T)}

    def group(self, T):
        p = self.props(T)
        return p["k"] * (p["cp"] * p["mu"] / p["k"]) ** (1.0 / 3.0) * p["mu"] ** (-0.681)

    def G(self, T):
        """Property group by interpolation on the grid."""
        return float(np.interp(T, self._T, self._G))


def load_shares(path):
    """Resistance shares per segment from heater_resistance_shares.py: dict of arrays f_g, f_c, f_w (wall), f_foul."""
    d = pd.read_csv(path)
    return {k: d[k].to_numpy(float) for k in ("f_g", "f_c", "f_w", "f_foul")}


class SegmentedHeater:
    def __init__(self, hs, co2_in, p_out, m_co2, m_gas, shares, gas, transport=None, n_g=0.65, n_c=0.8):
        """Fix the hardware from the design point.
        hs: heater coupling of the design (T_gas, T_co2, f, Q); co2_in: CO2 inlet state (h, p); p_out: design CO2 outlet
        pressure (Pa); m_co2, m_gas: design flows (kg/s); shares: dict from load_shares(); gas: gas table (h, T_of_h);
        transport: GasTransport for the property correction of the gas-side coefficient (None = no correction);
        n_g, n_c: flow exponents of the gas-side and CO2-side coefficients.
        Every segment obeys Q = UA_i * LMTD (exact counterflow relation, no limit on the NTU of a segment). The design UA_i is
        defined with the same log-mean difference, so the design point is reproduced exactly; its sum differs by about 0.2 %
        from the cycle's arithmetic-mean UA (attribute UA_cycle_total)."""
        self.gas, self.tr, self.n_g, self.n_c = gas, transport, n_g, n_c
        self.m_co2_des, self.m_gas_des = m_co2, m_gas
        f = np.asarray(hs["f"], float)
        self.N = len(f) - 1
        T_g, T_c = np.asarray(hs["T_gas"], float), np.asarray(hs["T_co2"], float)
        self.Q_des = hs["Q"]
        self.Q_i = hs["Q"] * np.diff(f)
        dT = T_g - T_c
        self.UA_cycle_total = ua_segments(dT, hs["Q"], f)                # cycle definition (arithmetic mean dT), for reference
        self.UA_des = self.Q_i / np.array([_lmtd(a, b) for a, b in zip(dT[:-1], dT[1:])])   # segment conductance, log-mean dT
        self.Tg_cold_des = T_g[:-1]
        h_b = co2_in["h"] + f * hs["Q"] / m_co2
        p_b = co2_in["p"] + f * (p_out - co2_in["p"])
        self.rho_des = np.array([PropsSI("D", "H", h, "P", p, FLUID) for h, p in zip(h_b[:-1], p_b[:-1])])
        self.dp_des = np.full(self.N, (co2_in["p"] - p_out) / self.N)
        self.fg, self.fc = shares["f_g"], shares["f_c"]
        self.fw, self.ff = shares["f_w"], shares["f_foul"]
        self.G_des = None if transport is None else np.array([transport.G(T) for T in self.Tg_cold_des])
        self.UA_total_des = float(self.UA_des.sum())

    def _march(self, Tg_out, m_g, co2_in, m_c):
        """March the segments from the cold end for a guessed gas outlet temperature. Returns the gas temperature found at the
        hot end and the segment results."""
        gas, N = self.gas, self.N
        r_g, r_c = m_g / self.m_gas_des, m_c / self.m_co2_des
        h_c, p, Tc, Tg = co2_in["h"], co2_in["p"], co2_in["T"], Tg_out
        hg = float(gas.h(Tg))
        out = {k: np.zeros(N + 1) for k in ("Tg", "Tc", "p")}
        out["Tg"][0], out["Tc"][0], out["p"][0] = Tg, Tc, p
        Qs, UAs, Tw, cross = np.zeros(N), np.zeros(N), np.zeros(N), False
        for i in range(N):
            rho = PropsSI("D", "H", h_c, "P", p, FLUID)
            Fg = 1.0 if self.tr is None else self.G_des[i] / self.tr.G(Tg)
            rc_term = self.fc[i] * r_c ** (-self.n_c)
            s = self.fg[i] * r_g ** (-self.n_g) * Fg + rc_term + self.fw[i] + self.ff[i]
            UA = self.UA_des[i] / s
            p2 = p - self.dp_des[i] * (self.rho_des[i] / rho) * r_c ** 2
            dT0 = Tg - Tc

            def hot(Q):
                return float(gas.T_of_h(hg + Q / m_g)), PropsSI("T", "H", h_c + Q / m_c, "P", p2, FLUID)

            def g(Q):
                """Segment equation Q = UA * LMTD(Q) (exact for counterflow with constant cp inside the segment)."""
                try:
                    Tg2, Tc2 = hot(Q)
                except ValueError:                              # CO2 state out of the property range: Q far too large
                    return 1.0
                dTh = Tg2 - Tc2
                return Q if dTh <= 0.0 else Q - UA * _lmtd(dT0, dTh)
            Q = 0.0
            if dT0 > 0.0:
                hi = UA * dT0
                for _ in range(80):
                    if g(hi) > 0.0:
                        break
                    hi *= 2.0
                Q = brentq(g, 0.0, hi, xtol=1e-3, rtol=1e-12)
            Tg2, Tc2 = hot(Q)
            if Tg2 - Tc2 <= 0.0:
                cross = True
            Qs[i], UAs[i] = Q, UA
            Tw[i] = Tc2 + (Tg2 - Tc2) * (rc_term + self.fw[i]) / s       # outer wall of the tube at the hot end of the segment
            h_c, p, Tc, Tg, hg = h_c + Q / m_c, p2, Tc2, Tg2, hg + Q / m_g
            out["Tg"][i + 1], out["Tc"][i + 1], out["p"][i + 1] = Tg, Tc, p
        return Tg, {"Q_i": Qs, "UA_i": UAs, "T_wall": Tw, "cross": cross, **out}

    def solve(self, T_gas_in, m_gas, co2_in, m_co2, bypass=0.0):
        """Rating mode: duty, CO2 outlet temperature (T7), gas outlet and stack temperatures, pressure drop and wall
        temperature for the gas inlet temperature T_gas_in (K), gas flow m_gas (kg/s), CO2 inlet state (T, h, p) and flow.
        bypass: fraction of the gas flow that bypasses the heater and rejoins downstream (control variable)."""
        gas = self.gas
        m_g = (1.0 - bypass) * m_gas
        lo, hi = co2_in["T"] + 1e-7, T_gas_in

        def f(Tgo):
            try:
                return self._march(Tgo, m_g, co2_in, m_co2)[0] - T_gas_in
            except ValueError:                                  # CO2 state out of the property range: the guess is far too high
                return 1e6
        if f(lo) >= 0.0:                                        # gas leaves at the CO2 inlet temperature: thermodynamic limit
            Tg_out = lo
        else:
            Tg_out = brentq(f, lo, hi, xtol=1e-9, rtol=1e-13)
        _, m = self._march(Tg_out, m_g, co2_in, m_co2)
        Q = float(m["Q_i"].sum())
        h_stack = (1.0 - bypass) * float(gas.h(Tg_out)) + bypass * float(gas.h(T_gas_in))
        dT = m["Tg"] - m["Tc"]
        return {"Q": Q, "T7": float(m["Tc"][-1]), "T_gas_out": float(Tg_out), "T_stack": float(gas.T_of_h(h_stack)),
                "min_dT": float(dT.min()), "p_out": float(m["p"][-1]), "dp": float(co2_in["p"] - m["p"][-1]),
                "T_wall_max": float(m["T_wall"].max()), "UA_total": float(m["UA_i"].sum()), "cross": m["cross"],
                "bypass": bypass, "Q_i": m["Q_i"], "UA_i": m["UA_i"], "T_gas": m["Tg"], "T_co2": m["Tc"], "T_wall": m["T_wall"],
                "p": m["p"]}

    def bypass_for_T7(self, T7_target, T_gas_in, m_gas, co2_in, m_co2, b_max=0.9):
        """Bypass fraction that brings T7 down to T7_target (0 if T7 is already at or below it)."""
        r0 = self.solve(T_gas_in, m_gas, co2_in, m_co2, 0.0)
        if r0["T7"] <= T7_target:
            return 0.0, r0
        if self.solve(T_gas_in, m_gas, co2_in, m_co2, b_max)["T7"] > T7_target:
            raise ValueError("T7 cannot be held even with the largest bypass")
        b = brentq(lambda b: self.solve(T_gas_in, m_gas, co2_in, m_co2, b)["T7"] - T7_target, 0.0, b_max, xtol=1e-7)
        return b, self.solve(T_gas_in, m_gas, co2_in, m_co2, b)
