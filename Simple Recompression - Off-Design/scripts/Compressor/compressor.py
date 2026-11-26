"""Off-design model of the centrifugal compressors (Dyreby 2014, Sandia compressor correlations).

One stage is described by two dimensionless numbers and a speed correction:
    flow coefficient   phi = m / (rho_in * U * D**2)            U = (D / 2) * omega  (tip speed)
    head coefficient   psi = dh_s / U**2                        dh_s: isentropic enthalpy rise to the outlet pressure
    speed correction   phi* = phi * (N / N_des)**0.2
    psi*(phi*), eta*(phi*)  fourth-order polynomials fitted to the Sandia data
    psi     = psi* / (N_des / N)**((20 phi*)**3)
    eta     = eta_des * 1.47528 * eta* / (N_des / N)**((20 phi*)**5)      (= eta_des at phi_des and N = N_des)
The design point fixes the hardware: phi_des = 0.02971 gives psi_des = 0.4588, so the design head fixes U and the design
flow fixes D; the design speed follows from U and D. The speed N is an input of the operating point, so a constant-speed
shaft and a variable-speed drive are the same model with a different N.
The recompressor is two such stages in series on one shaft (same speed, different diameters). Shaft power only: motor,
variable-speed-drive and bearing losses are not included."""
import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

FLUID = "CO2"
PHI_DES = 0.02971                       # design flow coefficient (efficiency maximum of the Sandia map)
PHI_SURGE = 0.02                        # surge limit on phi*
N_RANGE = (0.6, 1.2)                    # speed ratios N / N_des covered by the Sandia data


def state_Tp(T, p):
    return {"T": T, "p": p, "h": PropsSI("H", "T", T, "P", p, FLUID), "s": PropsSI("S", "T", T, "P", p, FLUID),
            "rho": PropsSI("D", "T", T, "P", p, FLUID)}


def state_hp(h, p):
    return {"T": PropsSI("T", "H", h, "P", p, FLUID), "p": p, "h": h, "s": PropsSI("S", "H", h, "P", p, FLUID),
            "rho": PropsSI("D", "H", h, "P", p, FLUID)}


def psi_star(f):
    return (((-498626.0 * f + 53224.0) * f - 2505.0) * f + 54.6) * f + 0.04049


def eta_star(f):
    return (((-1.638e6 * f + 182725.0) * f - 8089.0) * f + 168.6) * f - 0.7069


PSI_DES = float(psi_star(PHI_DES))


def map_point(phi, r):
    """Head coefficient psi and efficiency ratio eta / eta_des for the flow coefficient phi and the speed ratio r = N / N_des.
    Also returns phi* and the flags: surge (phi* < 0.02), overload (psi* <= 0, flow beyond the end of the map) and
    outside (speed ratio outside the range of the data)."""
    fs = phi * r ** 0.2
    ps, es = psi_star(fs), eta_star(fs)
    psi = ps / (1.0 / r) ** ((20.0 * fs) ** 3)
    er = 1.47528 * es / (1.0 / r) ** ((20.0 * fs) ** 5)
    return {"psi": float(psi), "eta_ratio": float(er), "phi_star": float(fs), "surge": bool(fs < PHI_SURGE),
            "overload": bool(ps <= 0.0), "outside": bool(r < N_RANGE[0] or r > N_RANGE[1])}


class CompressorStage:
    def __init__(self, D, N_des, eta_des):
        """Hardware: rotor diameter D (m), design speed N_des (rpm), design isentropic efficiency."""
        self.D, self.N_des, self.eta_des = D, N_des, eta_des

    @classmethod
    def from_design(cls, T_in, p_in, p_out, m, eta):
        """Size one stage for the design point: inlet T (K) and p (Pa), outlet p (Pa), mass flow (kg/s), efficiency."""
        s = state_Tp(T_in, p_in)
        dh_s = PropsSI("H", "P", p_out, "S", s["s"], FLUID) - s["h"]
        U = np.sqrt(dh_s / PSI_DES)
        D = np.sqrt(m / (PHI_DES * s["rho"] * U))
        return cls(D, 2.0 * U / D * 60.0 / (2.0 * np.pi), eta)

    @property
    def U_des(self):
        return 0.5 * self.D * self.N_des * 2.0 * np.pi / 60.0

    def operate(self, st_in, m, N):
        """Operating point for the inlet state st_in (dict with h, p, s, rho), mass flow m (kg/s) and speed N (rpm):
        flow and head coefficients, efficiency, outlet pressure, work, outlet state and the map flags."""
        r = N / self.N_des
        U = 0.5 * self.D * N * 2.0 * np.pi / 60.0
        phi = m / (st_in["rho"] * U * self.D ** 2)
        mp = map_point(phi, r)
        eta = self.eta_des * mp["eta_ratio"]
        dh_s = max(mp["psi"], 0.0) * U ** 2
        p_out = PropsSI("P", "H", st_in["h"] + dh_s, "S", st_in["s"], FLUID)
        w = dh_s / eta
        out = state_hp(st_in["h"] + w, p_out)
        return {"phi": phi, "psi": mp["psi"], "phi_star": mp["phi_star"], "eta": eta, "U": U, "dh_s": dh_s, "w": w,
                "power": m * w, "p_out": p_out, "T_out": out["T"], "h_out": out["h"], "state_out": out,
                "surge": mp["surge"], "overload": mp["overload"], "outside": mp["outside"], "N": N, "N_ratio": r}

    def speed_for(self, st_in, m, p_out, r_lo=0.4, r_hi=1.5):
        """Speed (rpm) at which the stage delivers the outlet pressure p_out for the inlet state and mass flow m. The outlet
        pressure rises with speed, so the root is found by bisection; the map flags of the result say whether it is usable."""
        f = lambda N: self.operate(st_in, m, N)["p_out"] - p_out
        a, b = r_lo * self.N_des, r_hi * self.N_des
        if f(a) > 0.0 or f(b) < 0.0:
            raise ValueError("outlet pressure not reachable within the speed range searched")
        return brentq(f, a, b, xtol=1e-3, rtol=1e-10)


class Recompressor:
    """Two stages in series on one shaft: same speed, different diameters. The interstage pressure follows from the
    design point; at part load it is found by the stages themselves (stage 2 takes the outlet state of stage 1)."""

    def __init__(self, st_in, p_out, m, h_out):
        """Size for the design point: inlet state, outlet pressure, mass flow and the design outlet enthalpy (J/kg).
        The stage efficiency is chosen so that the two stages together reach h_out (the single-stage design efficiency
        of the cycle model cannot be used per stage: the second stage compresses a state warmed by the first)."""
        self.m_des, self.p_out_des = m, p_out

        def design(eta_s):
            def stage2_p(p_mid):
                dh1 = PropsSI("H", "P", p_mid, "S", st_in["s"], FLUID) - st_in["h"]
                U1 = np.sqrt(dh1 / PSI_DES)
                D1 = np.sqrt(m / (PHI_DES * st_in["rho"] * U1))
                w_ang = 2.0 * U1 / D1
                mid = state_hp(st_in["h"] + dh1 / eta_s, p_mid)
                D2 = (2.0 * m / (PHI_DES * mid["rho"] * w_ang)) ** (1.0 / 3.0)
                U2 = 0.5 * w_ang * D2
                dh2 = PSI_DES * U2 ** 2
                p2 = PropsSI("P", "H", mid["h"] + dh2, "S", mid["s"], FLUID)
                return p2, D1, D2, w_ang, mid, dh2
            p_mid = brentq(lambda p: stage2_p(p)[0] - p_out, st_in["p"] * 1.001, p_out * 0.999, rtol=1e-12)
            p2, D1, D2, w_ang, mid, dh2 = stage2_p(p_mid)
            return p_mid, D1, D2, w_ang, mid["h"] + dh2 / eta_s

        eta_s = brentq(lambda e: design(e)[4] - h_out, 0.6, 0.98, xtol=1e-12)
        p_mid, D1, D2, w_ang, _ = design(eta_s)
        N_des = w_ang * 60.0 / (2.0 * np.pi)
        self.stages = [CompressorStage(D1, N_des, eta_s), CompressorStage(D2, N_des, eta_s)]
        self.N_des, self.eta_stage, self.p_mid_des = N_des, eta_s, p_mid

    def operate(self, st_in, m, N):
        a = self.stages[0].operate(st_in, m, N)
        b = self.stages[1].operate(a["state_out"], m, N)
        return {"p_mid": a["p_out"], "p_out": b["p_out"], "T_out": b["T_out"], "h_out": b["h_out"], "state_out": b["state_out"],
                "w": a["w"] + b["w"], "power": a["power"] + b["power"], "eta": (a["dh_s"] + b["dh_s"]) / max(a["w"] + b["w"], 1e-9),
                "phi": (a["phi"], b["phi"]), "surge": a["surge"] or b["surge"], "overload": a["overload"] or b["overload"],
                "outside": a["outside"] or b["outside"], "N": N, "N_ratio": N / self.N_des, "stage": (a, b)}

    def speed_for(self, st_in, m, p_out, r_lo=0.4, r_hi=1.5):
        f = lambda N: self.operate(st_in, m, N)["p_out"] - p_out
        a, b = r_lo * self.N_des, r_hi * self.N_des
        if f(a) > 0.0 or f(b) < 0.0:
            raise ValueError("outlet pressure not reachable within the speed range searched")
        return brentq(f, a, b, xtol=1e-3, rtol=1e-10)
