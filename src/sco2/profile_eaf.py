"""Gas temperature and flow over one tap-to-tap cycle (PLACEHOLDER SHAPE).

This is a first, simple generator, to be replaced by a profile fitted to real
EAF data. One shared process driver s(t), between -1 and +1 with zero time
mean, moves both signals:
    T(t) = T_mean * (1 + a_T * amp * s(t))
    m(t) = m_mean * (1 + a_m * amp * s(t - lag))      (flow lags temperature)
amp = 1 is the "weak" case (up to +-10 % in temperature, +-15 % in flow);
amp = 0 gives steady gas. An optional power-off gap (charging, tapping)
replaces the gas by cold gas for a few minutes, centred on the start of the
cycle. The time means of T and m are the design values (without a gap).

ALL NUMBERS ARE PLACEHOLDERS, not furnace data."""
import numpy as np

PROFILE = {
    "cycle_min": 60.0,                  # min, tap-to-tap time
    "dt_min": 0.5,                      # min, time step
    # (fraction of the cycle, driver): ramp up, plateau, ramp down, low phase
    "knots": [(0.00, -1.0), (0.20, 1.0), (0.50, 1.0), (0.70, -1.0), (1.00, -1.0)],
    "a_T": 0.10,                        # relative temperature amplitude at amp = 1
    "a_m": 0.15,                        # relative flow amplitude at amp = 1
    "lag_min": 3.0,                     # min, flow lags temperature
    "gap_min": 8.0,                     # min, duration of the power-off gap
    "gap_ramp_min": 2.0,                # min, ramp into and out of the gap
    "gap_T_C": 200.0,                   # C, gas temperature in the gap
    "gap_m_frac": 0.5,                  # gas flow in the gap / mean flow
}


def _raw(x, prm):
    kx = [k for k, _ in prm["knots"]]
    kv = [v for _, v in prm["knots"]]
    return np.interp(np.mod(x, 1.0), kx, kv)


def driver(x, prm=PROFILE):
    """Periodic driver of the cycle (x = fraction of the cycle), centred to
    zero time mean and scaled so that its largest deviation is 1."""
    xs = np.linspace(0.0, 1.0, 4001)[:-1]
    raw = _raw(xs, prm)
    mean = raw.mean()
    scale = np.max(np.abs(raw - mean))
    return (_raw(x, prm) - mean) / scale


def make_profile(prm=PROFILE, amp=1.0, T_mean_C=800.0, m_mean=25.0, gap=False):
    """Time series over one cycle: dict with t_min, T_C (C), m (kg/s)."""
    cyc, dt = prm["cycle_min"], prm["dt_min"]
    t = np.arange(0.0, cyc, dt)
    x = t / cyc
    s_T = driver(x, prm)
    s_m = driver(x - prm["lag_min"] / cyc, prm)
    T = T_mean_C * (1.0 + prm["a_T"] * amp * s_T)
    m = m_mean * (1.0 + prm["a_m"] * amp * s_m)
    if gap:
        d = np.minimum(t, cyc - t)                      # distance to the cycle start
        w = np.clip((prm["gap_min"] / 2.0 - d) / prm["gap_ramp_min"], 0.0, 1.0)
        T = (1.0 - w) * T + w * prm["gap_T_C"]
        m = (1.0 - w) * m + w * prm["gap_m_frac"] * m_mean
    return {"t_min": t, "T_C": T, "m": m, "dt_min": dt, "cycle_min": cyc}


# ---------------------------------------------------------------------------
# Real temperature profile (digitised by eye from a published figure)
# ---------------------------------------------------------------------------
# EAF exhaust-gas temperature over one complete melting cycle (about 70 min),
# read from the published figure (ref. [31] of the source paper, whose text
# quotes an average of about 600 C). Accuracy about +-30 C; time mean 590.8 C.
# There is NO flow information in the figure.
EAF_T_POINTS = [
    (0, 225), (1, 225), (2, 230), (3, 245), (4, 330), (5, 400), (6, 440),
    (7, 480), (8, 520), (9, 570), (10, 575), (11, 562), (12, 566), (13, 590),
    (14, 640), (15, 700), (16, 780), (17, 850), (18, 920), (19, 955),
    (19.5, 960), (20, 930), (20.5, 880), (21, 790), (22, 640), (22.5, 580),
    (23, 530), (24, 495), (25, 485), (26, 500), (27, 530), (28, 570),
    (29, 610), (30, 630), (31, 645), (32, 660), (33, 665), (34, 650),
    (35, 700), (36, 760), (37, 800), (38, 850), (39, 880), (40, 900),
    (41, 895), (42, 890), (43, 860), (44, 800), (45, 750), (46, 710),
    (47, 700), (48, 700), (49, 715), (50, 735), (51, 735), (52, 710),
    (53, 690), (54, 670), (55, 645), (56, 610), (57, 570), (58, 520),
    (59, 470), (60, 420), (61, 380), (62, 330), (63, 310), (64, 285),
    (65, 265), (66, 250), (67, 240), (68, 232), (69, 228), (70, 225)]


def make_profile_real(points=EAF_T_POINTS, m_mean=25.0, a_flow=0.0,
                      lag_min=PROFILE["lag_min"], mean_C=None, dt_min=0.5):
    """Real temperature profile with an ASSUMED flow model.
    T(t): linear interpolation of the points (periodic, last point = first).
    mean_C: if given, the whole curve is shifted so that its time mean is mean_C.
    Flow: m = m_mean * (1 + a_flow * z(t - lag)), z = normalised temperature
    deviation (largest |z| = 1, zero mean). a_flow = 0 gives constant flow."""
    tp = np.array([p[0] for p in points], float)
    Tp = np.array([p[1] for p in points], float)
    cyc = float(tp[-1])
    t = np.arange(0.0, cyc, dt_min)
    T = np.interp(t, tp, Tp)
    if mean_C is not None:
        T = T + (mean_C - T.mean())
    dev = T - T.mean()
    z = dev / np.max(np.abs(dev))
    zl = np.interp(t - lag_min, t, z, period=cyc)
    m = m_mean * (1.0 + a_flow * zl)
    return {"t_min": t, "T_C": T, "m": m, "dt_min": dt_min, "cycle_min": cyc}


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
    to a time mean of m_mean (the real flow is about 2x our 25 kg/s, so only
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