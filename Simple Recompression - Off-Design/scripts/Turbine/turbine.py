"""Off-design model of the radial-inflow turbine at constant speed (Dyreby 2014, Section 3.3).

Fixed hardware, fixed shaft speed (gearbox and synchronous generator), so the velocity ratio changes only through the
isentropic enthalpy drop:
    spouting velocity   C_s = sqrt(2 dh_s)                       dh_s: isentropic enthalpy drop to the outlet pressure
    velocity ratio      nu  = nu_des * C_s_des / C_s             nu_des = 0.7071 (efficiency maximum)
    efficiency          eta = eta_des * 2 nu sqrt(1 - nu**2) / (2 nu_des sqrt(1 - nu_des**2))      (= eta_des at design)
    mass flow (nozzle)  m   = A_eff * rho_out * C_s              un-choked nozzle, rho_out = density at the turbine outlet
    mass flow (Stodola) m   = K_t * sqrt(p_in * rho_in) * sqrt(1 - 1/beta**2)        alternative law, beta = p_in / p_out
A_eff (or K_t) is fixed so that the model passes exactly through the design point.
Shaft power only: gearbox, generator and bearing losses are not included."""
import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

FLUID = "CO2"
NU_DES = 1.0 / np.sqrt(2.0)                    # velocity ratio of maximum efficiency


def state_Tp(T, p):
    return {"T": T, "p": p, "h": PropsSI("H", "T", T, "P", p, FLUID), "s": PropsSI("S", "T", T, "P", p, FLUID),
            "rho": PropsSI("D", "T", T, "P", p, FLUID)}


def eta_ratio(nu):
    """eta / eta_des for the velocity ratio nu: 2 nu sqrt(1 - nu^2), normalised to 1 at nu_des. Zero for nu >= 1."""
    nu = np.asarray(nu, float)
    f = np.where(nu < 1.0, 2.0 * nu * np.sqrt(np.clip(1.0 - nu ** 2, 0.0, None)), 0.0)
    return f / (2.0 * NU_DES * np.sqrt(1.0 - NU_DES ** 2))


class RadialTurbine:
    def __init__(self, T_in, p_in, p_out, m, eta, law="nozzle"):
        """Fix the hardware from the design point: inlet T (K) and p (Pa), outlet p (Pa), mass flow m (kg/s), isentropic
        efficiency eta. law: 'nozzle' (Dyreby) or 'stodola'."""
        if law not in ("nozzle", "stodola"):
            raise ValueError("law must be 'nozzle' or 'stodola'")
        self.law, self.eta_des, self.m_des = law, eta, m
        s_in = state_Tp(T_in, p_in)
        h_out_s = PropsSI("H", "P", p_out, "S", s_in["s"], FLUID)
        self.dh_s_des = s_in["h"] - h_out_s
        self.Cs_des = float(np.sqrt(2.0 * self.dh_s_des))
        h_out = s_in["h"] - eta * self.dh_s_des
        self.rho_out_des = PropsSI("D", "H", h_out, "P", p_out, FLUID)
        self.beta_des = p_in / p_out
        self.A_eff = m / (self.rho_out_des * self.Cs_des)
        self.K_t = m / (np.sqrt(p_in * s_in["rho"]) * np.sqrt(1.0 - 1.0 / self.beta_des ** 2))
        self.design = {"T_in": T_in, "p_in": p_in, "p_out": p_out, "h_in": s_in["h"], "dh_s": self.dh_s_des,
                       "Cs": self.Cs_des, "rho_out": self.rho_out_des, "A_eff": self.A_eff, "K_t": self.K_t,
                       "w": eta * self.dh_s_des, "power": m * eta * self.dh_s_des}

    def expand(self, T_in, p_in, p_out):
        """Operating point for the inlet state (T_in, p_in) and the outlet pressure: mass flow, efficiency, work, outlet state."""
        s_in = state_Tp(T_in, p_in)
        h_out_s = PropsSI("H", "P", p_out, "S", s_in["s"], FLUID)
        dh_s = s_in["h"] - h_out_s
        Cs = float(np.sqrt(2.0 * dh_s))
        nu = NU_DES * self.Cs_des / Cs
        eta = float(self.eta_des * eta_ratio(nu))
        w = eta * dh_s
        h_out = s_in["h"] - w
        out = {"T_out": PropsSI("T", "H", h_out, "P", p_out, FLUID), "h_out": h_out,
               "rho_out": PropsSI("D", "H", h_out, "P", p_out, FLUID)}
        beta = p_in / p_out
        if self.law == "nozzle":
            m = self.A_eff * out["rho_out"] * Cs
        else:
            m = self.K_t * np.sqrt(p_in * s_in["rho"]) * np.sqrt(max(1.0 - 1.0 / beta ** 2, 0.0))
        return {"m": float(m), "eta": eta, "nu": float(nu), "Cs": Cs, "dh_s": dh_s, "w": w, "power": float(m) * w,
                "beta": beta, **out}

    def inlet_pressure(self, m, T_in, p_out, p_hi=80e6):
        """Inlet pressure at which the turbine passes the mass flow m (kg/s) for the inlet temperature and outlet pressure:
        the turbine's swallowing capacity. Mass flow rises with the inlet pressure, so the root is found by bisection."""
        f = lambda p: self.expand(T_in, p, p_out)["m"] - m
        lo = p_out * 1.02
        if f(lo) > 0.0:
            raise ValueError("flow below the turbine's flow at the lowest pressure ratio considered")
        return brentq(f, lo, p_hi, xtol=1.0, rtol=1e-10)
