"""Quasi-steady response of the plant to a time-varying EAF gas.

1. build_map(): the controller (controller_splitflow.operate) is run once on a
   grid of gas temperature and flow. The result is the "power map".
2. PowerMap: interpolates the map along a gas profile (cheap).
3. simulate(): applies the run / idle / restart logic along the profile.
4. summarise(): energy per cycle, per year, heat used, losses.

Quasi-steady means no thermal inertia: the plant follows the gas instantly.
The only dynamic element is a placeholder restart time after idle.
ALL CONTROL PARAMETERS ARE PLACEHOLDERS (see controller_splitflow.CONTROL)."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .controller_splitflow import operate, CONTROL
from .operation_splitflow import OPERATION, calendar_hours

K = 273.15
T_GRID_C = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]   # C
M_GRID = [5, 8, 12, 16, 20, 25, 30, 36, 42]                       # kg/s
RESTART = {"t_restart_min": 3.0}      # PLACEHOLDER: idle to full operation
FIELDS = ["net_if_run", "T7", "m_co2", "bypass", "Q_use", "Q_avail",
          "T_stack", "p2"]


# ---------------------------------------------------------------------------
# Power map
# ---------------------------------------------------------------------------
def build_map(hw, gas, y_mode="fixed", log=print, T_grid=None, m_grid=None):
    """Run the controller on the grid. Returns a dict of arrays [T, m].
    T_grid (C) and m_grid (kg/s) default to the module grids."""
    T_GRID_C_, M_GRID_ = (T_grid or T_GRID_C), (m_grid or M_GRID)
    nT, nm = len(T_GRID_C_), len(M_GRID_)
    arr = {k: np.full((nT, nm), np.nan) for k in FIELDS}
    order_T = sorted(range(nT), key=lambda i: abs(T_GRID_C_[i] - (hw["T_gas_des"] - K)))
    order_m = sorted(range(nm), key=lambda j: abs(M_GRID_[j] - hw["m_gas_des"]))
    x, t0, done = None, time.time(), 0
    for i in order_T:
        for j in order_m:
            Tc, mg = T_GRID_C_[i], M_GRID_[j]
            r = operate(hw, gas, Tc + K, mg, y_mode=y_mode, x0=x)
            Q_av = max(mg * (float(gas.h(Tc + K)) - float(gas.h(hw["T_stack_min"]))), 0.0)
            p = r["point"]
            if p is None:        # nothing converged: treat as not runnable
                v = dict(net_if_run=-1e6, T7=300.0, m_co2=0.0, bypass=mg,
                         Q_use=0.0, T_stack=Tc + K, p2=0.0)
            else:
                v = dict(net_if_run=p["net_el"], T7=p["T7"], m_co2=p["m_co2"],
                         bypass=p["m_bypass"], Q_use=p["Q_use"],
                         T_stack=p["T_stack"], p2=p["p2"])
                x = r["x"]
            v["Q_avail"] = Q_av
            for k, val in v.items():
                arr[k][i, j] = val
            done += 1
            log(f"  map point {done}/{nT * nm}: {Tc} C, {mg} kg/s -> {r['mode']}, "
                f"{r['net_el'] / 1e6:.3f} MW ({time.time() - t0:.0f} s)")
    return arr


def save_map(path, arr, hw, T_grid=None, m_grid=None):
    meta = {"m_des": hw["m_des"], "net_el_des": hw["net_el_des"],
            "control": {k: v for k, v in CONTROL.items() if not isinstance(v, tuple)}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, T=np.array(T_grid or T_GRID_C), m=np.array(m_grid or M_GRID),
             meta=json.dumps(meta), **arr)


def load_map(path, hw, T_grid=None, m_grid=None):
    """Returns the arrays, or None if the file is missing or does not match
    the current design point."""
    if not Path(path).exists():
        return None
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    if (abs(meta["m_des"] - hw["m_des"]) > 1e-6 * hw["m_des"]
            or abs(meta["net_el_des"] - hw["net_el_des"]) > 1e-6 * hw["net_el_des"]
            or list(d["T"]) != list(T_grid or T_GRID_C)
            or list(d["m"]) != list(m_grid or M_GRID)):
        return None
    return {k: d[k] for k in FIELDS}


class PowerMap:
    """Interpolation of the map. Gas conditions outside the grid are clipped
    to its edge (counted in 'n_clipped')."""

    def __init__(self, arr, hw, ctrl=CONTROL, T_grid=None, m_grid=None):
        self.T_c = np.asarray(T_grid or T_GRID_C, float)
        self.m = np.asarray(m_grid or M_GRID, float)
        self.ctrl = ctrl
        self.P_idle = -ctrl["idle_parasitic_frac"] * hw["P_par_des"]
        self.f = {k: RegularGridInterpolator((self.T_c, self.m), arr[k],
                                             bounds_error=False, fill_value=None)
                  for k in FIELDS}

    def at(self, T_C, m):
        T_C, m = np.atleast_1d(np.asarray(T_C, float)), np.atleast_1d(np.asarray(m, float))
        T = np.clip(T_C, self.T_c[0], self.T_c[-1])
        mm = np.clip(m, self.m[0], self.m[-1])
        pts = np.column_stack([T, mm])
        out = {k: self.f[k](pts) for k in FIELDS}
        out["run"] = ((out["net_if_run"] >= self.ctrl["net_el_min_run_W"])
                      & (out["T7"] >= self.ctrl["T7_min_run_K"]))
        out["n_clipped"] = int(np.sum((T != T_C) | (mm != m)))
        return out


# ---------------------------------------------------------------------------
# Simulation along a profile
# ---------------------------------------------------------------------------
def simulate(pm, prof, t_restart_min=RESTART["t_restart_min"], n_cycles=2):
    """Quasi-steady run with the idle / restart logic. The cycle is repeated
    n_cycles times and the last one is returned (periodic steady state)."""
    n1 = len(prof["T_C"])
    o = pm.at(np.tile(prof["T_C"], n_cycles), np.tile(prof["m"], n_cycles))
    dt, n = prof["dt_min"], n1 * n_cycles
    run_raw = o["run"]
    state = np.empty(n, dtype=object)
    running = bool(run_raw[0])
    ready = t_restart_min if running else 0.0
    for k in range(n):
        if not run_raw[k]:
            running, ready, state[k] = False, 0.0, "idle"
        elif running:
            state[k] = "run"
        else:
            ready += dt
            if ready >= t_restart_min:
                running, state[k] = True, "run"
            else:
                state[k] = "restart"
    s = slice(n - n1, n)
    run = state[s] == "run"
    P = np.where(run, o["net_if_run"][s], pm.P_idle)
    Q_use = np.where(run, o["Q_use"][s], 0.0)
    return {"t_min": prof["t_min"], "T_C": prof["T_C"], "m_gas": prof["m"],
            "state": state[s], "P": P, "Q_use": Q_use,
            "Q_avail": o["Q_avail"][s],
            "m_co2": np.where(run, o["m_co2"][s], 0.0),
            "bypass": np.where(run, o["bypass"][s], prof["m"]),
            "T7": np.where(run, o["T7"][s], np.nan),
            "n_clipped": o["n_clipped"]}


def hours_per_year():
    return calendar_hours(OPERATION) * OPERATION["plant_availability"]


def summarise(pm, prof, sim, hw, label=""):
    """Energy and heat accounting of one simulated cycle."""
    hy = hours_per_year()
    P_mean = float(np.mean(sim["P"]))
    cyc_h = prof["cycle_min"] / 60.0
    Tm, mm = float(np.mean(prof["T_C"])), float(np.mean(prof["m"]))
    ref = pm.at(Tm, mm)
    P_ref = (float(ref["net_if_run"][0]) if bool(ref["run"][0]) else pm.P_idle)
    st = sim["state"]
    Qa, Qu = float(np.mean(sim["Q_avail"])), float(np.mean(sim["Q_use"]))
    run = st == "run"
    return {
        "Scenario": label,
        "Mean gas temperature (C)": Tm, "Mean gas flow (kg/s)": mm,
        "Gas temperature min / max (C)": f"{prof['T_C'].min():.0f} / {prof['T_C'].max():.0f}",
        "Gas flow min / max (kg/s)": f"{prof['m'].min():.1f} / {prof['m'].max():.1f}",
        "Mean net electrical power (MW)": P_mean / 1e6,
        "Peak net electrical power (MW)": float(np.max(sim["P"])) / 1e6,
        "Energy per cycle (MWh)": P_mean / 1e6 * cyc_h,
        "Annual energy (GWh/yr)": P_mean / 1e6 * hy / 1e3,
        "Steady equivalent at the mean gas (MW)": P_ref / 1e6,
        "Fluctuation penalty vs steady equivalent (%)":
            (1.0 - P_mean / P_ref) * 100.0 if P_ref > 0 else float("nan"),
        "Mean power / design steady power (%)": P_mean / hw["net_el_des"] * 100.0,
        "Heat available, mean (MW)": Qa / 1e6,
        "Heat used, mean (MW)": Qu / 1e6,
        "Heat used (% of available)": Qu / Qa * 100.0 if Qa > 0 else float("nan"),
        "Time running (%)": float(np.mean(run)) * 100.0,
        "Time idle (%)": float(np.mean(st == "idle")) * 100.0,
        "Time in restart (%)": float(np.mean(st == "restart")) * 100.0,
        "Mean gas bypass while running (kg/s)":
            float(np.mean(sim["bypass"][run])) if run.any() else float("nan"),
        "CO2 flow range while running (% of design)":
            (f"{sim['m_co2'][run].min() / hw['m_des'] * 100:.0f} - "
             f"{sim['m_co2'][run].max() / hw['m_des'] * 100:.0f}") if run.any() else "-",
        "T7 range while running (C)":
            (f"{np.nanmin(sim['T7'][run]) - K:.0f} - {np.nanmax(sim['T7'][run]) - K:.0f}")
            if run.any() else "-",
        "Points clipped to the map edge": sim["n_clipped"],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(results, hw, path_series, path_summary, show_labels=None):
    """results: list of (label, prof, sim, summary). show_labels: scenarios
    drawn in the time-series figure (default: the three amplitude cases)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if show_labels is None:
        show_labels = ("Weak (amp 1)", "Amp 3", "Amp 3 + gap")
    show = [r for r in results if r[0] in show_labels]
    if not show:
        show = results[:3]
    fig, axes = plt.subplots(3, len(show), figsize=(5.2 * len(show), 9),
                             sharex=True, squeeze=False)
    for c, (label, prof, sim, _) in enumerate(show):
        t = sim["t_min"]
        a = axes[0, c]
        a.plot(t, sim["T_C"], color="#c0392b", label="Gas temperature (C)")
        a.set_ylabel("Gas temperature (C)", color="#c0392b")
        a2 = a.twinx()
        a2.plot(t, sim["m_gas"], color="#1f4e79", label="Gas flow (kg/s)")
        a2.set_ylabel("Gas flow (kg/s)", color="#1f4e79")
        a.set_title(label)
        b = axes[1, c]
        b.plot(t, sim["P"] / 1e6, color="#2e7d32")
        b.axhline(hw["net_el_des"] / 1e6, color="gray", ls=":", label="Design steady")
        idle = sim["state"] != "run"
        b.fill_between(t, b.get_ylim()[0], b.get_ylim()[1], where=idle,
                       color="#bbbbbb", alpha=0.5, label="Idle or restart")
        b.set_ylabel("Net electrical power (MW)")
        b.legend(loc="lower right", fontsize=8)
        d = axes[2, c]
        d.plot(t, sim["m_co2"] / hw["m_des"] * 100, color="#6a1b9a")
        d.set_ylabel("CO2 flow (% of design)", color="#6a1b9a")
        d2 = d.twinx()
        d2.plot(t, sim["T7"] - K, color="#e67e22")
        d2.set_ylabel("Turbine inlet T7 (C)", color="#e67e22")
        d.set_xlabel("Time in the cycle (min)")
        for ax in (a, b, d):
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_series, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r[0] for r in results]
    E = [r[3]["Annual energy (GWh/yr)"] for r in results]
    ref = [r[3]["Steady equivalent at the mean gas (MW)"] * hours_per_year() / 1e3
           for r in results]
    xs = np.arange(len(results))
    ax.bar(xs, E, color="#2e7d32", label="Simulated profile")
    ax.plot(xs, ref, "k_", markersize=26, mew=2, label="Steady at the mean gas")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Annual net electrical energy (GWh/yr)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_summary, dpi=150)
    plt.close(fig)