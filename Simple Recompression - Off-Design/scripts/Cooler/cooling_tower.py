"""Wet cooling tower, Merkel model (4-point Chebyshev integration), for the water-cooled main cooler.

The tower is sized at the design point: the Merkel constant c follows from the design water range, wet bulb and L/G,
Me = c (L/G)^-n. Off-design, tower_cold_water() returns the cold-water temperature (cooler inlet water) for a given
wet-bulb temperature, water flow, air flow and hot-water temperature. Temperatures in C unless stated; air at 1 atm."""
import CoolProp.CoolProp as CP

P_ATM = 101325.0
CPW = 4180.0                                           # J/kg.K, liquid water


def h_sat_air(T_C):
    """Enthalpy of saturated moist air (J per kg dry air) at T_C (C)."""
    return CP.HAPropsSI("H", "T", T_C + 273.15, "P", P_ATM, "R", 1.0)


def merkel_number(T_wb, T_cold, T_hot, LG):
    """Merkel number Me = cpw * range / 4 * sum(1/(h_sat - h_air)) at the Chebyshev points 0.1, 0.4, 0.6, 0.9."""
    h_a_in = h_sat_air(T_wb)
    s = 0.0
    for fr in (0.1, 0.4, 0.6, 0.9):
        T = T_cold + fr * (T_hot - T_cold)
        dh = h_sat_air(T) - (h_a_in + LG * CPW * (T - T_cold))
        if dh <= 0.0:
            return float("inf")
        s += 1.0 / dh
    return CPW * (T_hot - T_cold) / 4.0 * s


class CoolingTower:
    def __init__(self, T_wb_des, T_cold_des, T_hot_des, m_w_des, LG_des, n_merkel):
        self.n, self.m_w_des, self.m_air_des = n_merkel, m_w_des, m_w_des / LG_des
        self.T_wb_des, self.T_cold_des, self.T_hot_des, self.LG_des = T_wb_des, T_cold_des, T_hot_des, LG_des
        self.Me_des = merkel_number(T_wb_des, T_cold_des, T_hot_des, LG_des)
        self.c = self.Me_des * LG_des ** n_merkel

    def cold_water(self, T_wb, m_w, m_air, T_hot):
        """Cold-water temperature (C) leaving the tower, by bisection (Me falls as the cold-water temperature rises)."""
        LG = m_w / m_air
        Me_avail = self.c * LG ** (-self.n)
        lo, hi = T_wb + 0.05, T_hot - 1e-3
        if hi <= lo:
            return hi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if merkel_number(T_wb, mid, T_hot, LG) > Me_avail:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def fan_power_ratio(self, m_air):
        """Tower fan power relative to design: proportional to the cube of the air flow (constant density)."""
        return (m_air / self.m_air_des) ** 3
