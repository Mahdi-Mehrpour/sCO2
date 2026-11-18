"""Tabulated CO2 properties for fast heat exchanger profiles.

Two tables are built once from CoolProp (a few seconds) and interpolated
linearly afterwards:
  T(h, p)  for the temperature profiles of the CO2 side
  h(T, p)  (kept for completeness and for the accuracy check)

Range: p = 7.6 to 27.6 MPa, T = 285 to 1250 K. Outside the range the tables
extrapolate linearly (not accurate): check the 'verify' output.
Scalar state calculations in the cycle model still use CoolProp directly, so
only the segment profiles inside the heat exchangers use the tables."""
import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.interpolate import RegularGridInterpolator

P_GRID = np.concatenate([np.arange(7.6e6, 8.4e6, 0.1e6),
                         np.arange(8.4e6, 27.61e6, 0.4e6)])
T_GRID = np.concatenate([np.arange(285.0, 330.0, 0.25),
                         np.arange(330.0, 1250.01, 2.0)])
N_H = 900                      # points of the enthalpy grid


class CO2Table:
    def __init__(self, fluid="CO2"):
        self.fluid = fluid
        P, T = np.meshgrid(P_GRID, T_GRID, indexing="ij")
        H = PropsSI("H", "T", T.ravel(), "P", P.ravel(), fluid).reshape(P.shape)
        self._h = RegularGridInterpolator((P_GRID, T_GRID), H,
                                          bounds_error=False, fill_value=None)
        # common enthalpy range that every pressure column covers
        self.h_grid = np.linspace(float(H[:, 0].max()), float(H[:, -1].min()), N_H)
        PP, HH = np.meshgrid(P_GRID, self.h_grid, indexing="ij")
        TT = PropsSI("T", "H", HH.ravel(), "P", PP.ravel(), fluid).reshape(PP.shape)
        self._T = RegularGridInterpolator((P_GRID, self.h_grid), TT,
                                          bounds_error=False, fill_value=None)

    def T(self, h, p):
        """Temperature (K) from enthalpy (J/kg) and pressure (Pa); arrays."""
        h = np.asarray(h, float)
        p = np.broadcast_to(np.asarray(p, float), h.shape)
        pts = np.column_stack([p.ravel(), h.ravel()])
        return self._T(pts).reshape(h.shape)

    def h(self, T, p):
        """Enthalpy (J/kg) from temperature (K) and pressure (Pa); arrays."""
        T = np.asarray(T, float)
        p = np.broadcast_to(np.asarray(p, float), T.shape)
        pts = np.column_stack([p.ravel(), T.ravel()])
        return self._h(pts).reshape(T.shape)

    def verify(self, n=4000, seed=1):
        """Interpolation error against CoolProp on random points: the whole
        working range, and the near-critical corner (worst case)."""
        rng = np.random.default_rng(seed)
        out = {}
        for name, (T0, T1, p0, p1) in {
                "working range": (305.0, 1100.0, 7.7e6, 25.5e6),
                "near-critical corner": (304.5, 325.0, 7.7e6, 9.0e6)}.items():
            T = rng.uniform(T0, T1, n)
            p = rng.uniform(p0, p1, n)
            h = PropsSI("H", "T", T, "P", p, self.fluid)
            dT = np.abs(self.T(h, p) - T)
            dh = np.abs(self.h(T, p) - h)
            out[name] = {"max dT (K)": float(dT.max()),
                         "99 % dT (K)": float(np.percentile(dT, 99)),
                         "max dh (kJ/kg)": float(dh.max() / 1e3)}
        return out


_TABLE = []


def get_table():
    """The table, built on first use."""
    if not _TABLE:
        _TABLE.append(CO2Table())
    return _TABLE[0]