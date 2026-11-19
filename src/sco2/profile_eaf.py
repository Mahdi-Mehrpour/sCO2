"""Gas temperature and flow over one tap-to-tap cycle.

Measured reference cycle (Dal Magro 2017, Fig. 3.40), digitised by eye, plus a
constant-gas profile for reference cases. Time unit: min."""
import numpy as np


def make_profile_steady(T_C, m, cycle_min=70.0, dt_min=0.5):
    """Constant gas over one cycle (reference case)."""
    t = np.arange(0.0, cycle_min, dt_min)
    return {"t_min": t, "T_C": np.full_like(t, T_C), "m": np.full_like(t, m),
            "dt_min": dt_min, "cycle_min": cycle_min}


# ---------------------------------------------------------------------------
# Reference cycle WITH measured flow (digitised by eye)
# ---------------------------------------------------------------------------
# Source: Dal Magro (2017), "Energy recovery systems based on high-temperature
# phase change materials", Univ. of Udine, Fig. 3.40 "The reference cycle"
# (most energy-intensive 40-min tap-to-tap cycle, off-gas after the scrap
# preheater of a continuous-charge EAF). Digitised by eye: T about +-30 C,
# flow about +-0.2e5 Nm3/h, smoothed to about 1 min (the measured flow is very
# noisy; the figure's dashed averages are about 690 C and 1.34e5 Nm3/h).
# Columns: (time min, gas temperature C, flow in 1e5 Nm3/h).
EAF_REF_T = [
    (0, 350), (1, 320), (2, 285), (3, 255), (3.5, 250), (4.3, 305), (5.4, 417),
    (6.2, 487), (7, 640), (7.7, 770), (8.5, 720), (9.3, 775), (10, 740),
    (11, 720), (11.5, 760), (12, 800), (12.5, 900), (13, 950), (13.8, 860),
    (14.6, 955), (15.8, 820), (16.5, 880), (17.5, 860), (18.5, 760), (19.3, 710),
    (20, 800), (21, 870), (22, 900), (23, 930), (23.5, 900), (24, 920),
    (24.8, 880), (25.5, 800), (26.5, 720), (27, 705), (27.6, 780), (28.5, 880),
    (29, 890), (29.7, 880), (30.5, 810), (31.5, 740), (32.5, 700), (33.5, 650),
    (34.5, 600), (35.5, 570), (36.5, 520), (37.5, 480), (39, 430), (40, 400)]
EAF_REF_F = [
    (0, 0.9), (0.5, 1.5), (1, 2.5), (1.5, 2.8), (2, 2.6), (3, 2.3), (3.5, 2.0),
    (4, 1.7), (4.5, 1.5), (5, 1.35), (5.5, 1.3), (6, 1.25), (7, 0.9), (7.5, 0.6),
    (8, 1.2), (9, 1.4), (10, 1.5), (11, 1.3), (12, 1.4), (13, 1.6), (14, 1.7),
    (15, 1.4), (16, 1.5), (17, 1.8), (18, 1.4), (19, 1.2), (20, 1.5), (21, 1.5),
    (22, 1.4), (23, 1.5), (24, 1.5), (25, 1.4), (26, 1.5), (27, 1.4), (28, 1.6),
    (29, 1.6), (30, 1.4), (31, 1.4), (32, 1.3), (33, 1.0), (33.5, 0.6), (34, 1.0),
    (35, 1.3), (36, 1.15), (37, 1.1), (38, 1.0), (39, 1.05), (40, 1.1)]


def make_profile_ref(m_mean=25.0, mean_C=None, flow="data", dt_min=0.5):
    """Reference-cycle profile (40 min). The flow is the measured SHAPE scaled
    to a time mean of m_mean (the measured flow is digitised in 1e5 Nm3/h, so only
    the relative shape is used). flow = 'data' (measured shape) or 'const'.
    mean_C: shift the temperature curve to this time mean (None = as measured).
    The cycle is not exactly periodic (ends at 400 C, starts at 350 C)."""
    cyc = float(EAF_REF_T[-1][0])
    t = np.arange(0.0, cyc, dt_min)
    T = np.interp(t, *zip(*EAF_REF_T))
    if mean_C is not None:
        T = T + (mean_C - T.mean())
    if flow == "data":
        f = np.interp(t, *zip(*EAF_REF_F))
        m = m_mean * f / f.mean()
    elif flow == "const":
        m = np.full_like(t, m_mean)
    else:
        raise ValueError("flow must be 'data' or 'const'")
    return {"t_min": t, "T_C": T, "m": m, "dt_min": dt_min, "cycle_min": cyc}
