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