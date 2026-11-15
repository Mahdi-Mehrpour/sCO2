"""Heat source model: electric arc furnace (EAF) off-gas, constant conditions.

The gas is an ideal-gas mixture (Cantera, NASA polynomials), which is valid
for exhaust gas near atmospheric pressure over a wide temperature range.
Only the physical (thermo-mechanical) exergy is considered; the chemical
exergy of the CO and the dust are neglected. Units are SI."""
import numpy as np
import cantera as ct
from CoolProp.CoolProp import PropsSI

# Default EAF case (all values are assumptions, to be replaced by real data)
EAF = {
    "T_in": 1073.15,          # K, gas temperature at the heat exchanger inlet
    "p": 101325.0,            # Pa
    "m_gas": 25.0,            # kg/s
    "X": {"N2": 0.68, "CO2": 0.12, "H2O": 0.10, "O2": 0.09, "CO": 0.01},
    "T_stack_min": 423.15,    # K, lowest allowed gas outlet temperature
    "dT_pinch_heater": 20.0,  # K, minimum gas-to-CO2 temperature difference
}


class GasStream:
    """Gas mixture of fixed composition. h and s are tabulated once on a
    temperature grid and then interpolated (fast)."""

    def __init__(self, X, p=101325.0, T0=298.15, T_min=280.0, T_max=1600.0, n=600):
        g = ct.Solution("gri30.yaml")
        self.T_grid = np.linspace(T_min, T_max, n)
        h, s = [], []
        for T in self.T_grid:
            g.TPX = T, p, X
            h.append(g.enthalpy_mass)
            s.append(g.entropy_mass)
        self.h_grid = np.array(h)
        self.s_grid = np.array(s)
        g.TPX = T0, p, X
        self.h0, self.s0, self.T0 = g.enthalpy_mass, g.entropy_mass, T0
        self.M = g.mean_molecular_weight       # kg/kmol
        self.T_max = T_max

    def h(self, T):
        return np.interp(T, self.T_grid, self.h_grid)

    def s(self, T):
        return np.interp(T, self.T_grid, self.s_grid)

    def T_of_h(self, h):
        return np.interp(h, self.h_grid, self.T_grid)

    def ex(self, T):
        """Physical flow exergy (J/kg) relative to the dead state (T0, p)."""
        return (self.h(T) - self.h0) - self.T0 * (self.s(T) - self.s0)


def couple_heater(res, gas, T_in, m_gas, dT_pinch=20.0, T_stack_min=423.15,
                  n_seg=20, fluid="CO2"):
    """Link the cycle heater (states 6 -> 7) to the gas stream.

    The gas outlet temperature is the lowest one (>= T_stack_min) for which the
    minimum gas-to-CO2 temperature difference in the heater is at least dT_pinch.
    The CO2 flow follows from the energy balance of the heater.
    res: dictionary from recompression_adv()."""
    if T_in > gas.T_max:
        raise ValueError("Gas inlet temperature is above the gas table range")
    s6, s7 = res["states"][6], res["states"][7]
    q_in = res["q_in"]                                   # J per kg of CO2

    # CO2 side profile (per kg CO2), f = 0 cold end, f = 1 hot end
    f = np.linspace(0.0, 1.0, n_seg + 1)
    p_c = s6["p"] + f * (s7["p"] - s6["p"])
    h_c = s6["h"] + f * q_in
    T_c = np.array([PropsSI("T", "H", h, "P", p, fluid) for h, p in zip(h_c, p_c)])
    h_in = float(gas.h(T_in))

    def profile(T_out):
        h_out = float(gas.h(T_out))
        T_g = gas.T_of_h(h_out + f * (h_in - h_out))
        return T_g, T_g - T_c

    def min_dT(T_out):
        return profile(T_out)[1].min()

    lo, hi = T_stack_min, T_in - 1.0
    if min_dT(hi) < dT_pinch:
        raise ValueError("Gas is too cold: the heater cannot respect the pinch")
    if min_dT(lo) >= dT_pinch:
        T_out = lo
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if min_dT(mid) >= dT_pinch:
                hi = mid
            else:
                lo = mid
        T_out = hi

    h_out = float(gas.h(T_out))
    T_g, dT = profile(T_out)
    Q = m_gas * (h_in - h_out)                           # W, heat to the cycle
    return {
        "T_out": T_out, "Q": Q, "m_co2": Q / q_in,
        "Q_max": m_gas * (h_in - float(gas.h(T_stack_min))),
        "ex_in": m_gas * float(gas.ex(T_in)),
        "ex_out": m_gas * float(gas.ex(T_out)),
        "f": f, "T_gas": T_g, "T_co2": T_c, "dT": dT,
        "min_dT": float(dT.min()), "f_min": float(f[int(np.argmin(dT))]),
    }



def couple_cascade(res, gas, T_in, m_gas, dT_pinch1=20.0, dT_pinch2=20.0,
                   T_stack_min=423.15, n_seg=20, fluid="CO2"):
    """Link the cascade cycle to the gas stream (gas passes the main heater
    first, then the branch heater).

    Finds the largest CO2 flow (turbine flow, kg/s) for which the minimum
    gas-to-CO2 temperature difference is at least dT_pinch1 in the main heater
    and dT_pinch2 in the branch heater, and the gas outlet is at least
    T_stack_min. res: dictionary from recompression_cascade()."""
    st = res["states"]
    s2, s6, s7, sb = st[2], st[6], st[7], st[11]
    q1, q2 = res["q_in"], res["q_b"]           # J per kg of turbine flow
    f = np.linspace(0.0, 1.0, n_seg + 1)

    # CO2 temperature profiles (independent of the CO2 flow)
    p_c1 = s6["p"] + f * (s7["p"] - s6["p"])
    h_c1 = s6["h"] + f * q1
    T_c1 = np.array([PropsSI("T", "H", h, "P", p, fluid) for h, p in zip(h_c1, p_c1)])
    p_c2 = s2["p"] + f * (sb["p"] - s2["p"])
    h_c2 = s2["h"] + f * (sb["h"] - s2["h"])
    T_c2 = np.array([PropsSI("T", "H", h, "P", p, fluid) for h, p in zip(h_c2, p_c2)])

    h_in = float(gas.h(T_in))
    h_stack = float(gas.h(T_stack_min))
    m_max = m_gas * (h_in - h_stack) / (q1 + q2)     # limit from stack temperature

    def check(m):
        d1, d2 = m * q1 / m_gas, m * q2 / m_gas      # gas enthalpy drops, J/kg gas
        hg1 = h_in - d1
        hg2 = hg1 - d2
        T_g1 = gas.T_of_h(hg1 + f * d1)
        T_g2 = gas.T_of_h(hg2 + f * d2)
        return float((T_g1 - T_c1).min()), float((T_g2 - T_c2).min()), hg1, hg2, T_g1, T_g2

    def ok(m):
        a, b = check(m)[:2]
        return a >= dT_pinch1 and b >= dT_pinch2

    if not ok(1e-6):
        m = 0.0
    elif ok(m_max):
        m = m_max
    else:
        lo, hi = 1e-6, m_max
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if ok(mid):
                lo = mid
            else:
                hi = mid
        m = lo

    d1c, d2c, hg1, hg2, T_g1, T_g2 = check(m)
    if m >= m_max * (1 - 1e-9):
        binding = "stack temperature"
    else:
        binding = ("main heater pinch" if d1c - dT_pinch1 < d2c - dT_pinch2
                   else "branch heater pinch")
    T_out = float(gas.T_of_h(hg2))
    return {
        "m_co2": m, "Q1": m * q1, "Q2": m * q2, "Q": m * (q1 + q2),
        "Q_max": m_gas * (h_in - h_stack),
        "T_gas_mid": float(gas.T_of_h(hg1)), "T_out": T_out,
        "ex_in": m_gas * float(gas.ex(T_in)),
        "ex_out": m_gas * float(gas.ex(T_out)),
        "min_dT1": d1c, "min_dT2": d2c, "binding": binding,
        "f": f, "T_gas1": T_g1, "T_gas2": T_g2, "T_co2_1": T_c1, "T_co2_2": T_c2,
    }
