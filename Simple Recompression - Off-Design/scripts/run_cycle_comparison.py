"""Compare three cycle layouts under the SAME fixed conditions as the current design:
  simple      : recompression cycle, one heater          (variables x, pr)
  cascade     : recompression cycle with a branch heater (variables x, y, T_b, pr)
  split       : split-flow without recompressor (current layout)  (variables y, T_b, pr)

Fixed conditions: gas of the measured reference cycle (EAF["T_in"], EAF["m_gas"]), T7 = 550 C,
T1 = 32 C, p1 = 7.7 MPa, maximum pressure 25 MPa (pr <= 3.2468), heater pinch EAF["dT_pinch_heater"],
HTR/LTR minimum temperature difference >= 10 K, gas stack temperature >= EAF["T_stack_min"].

Each layout is optimised for maximum net power (shaft power of the cycle: turbine - compressors;
gearbox, generator and cooling-tower losses are NOT included, so the numbers are a little higher
than the electrical figures of the plant model). Net power is proportional to the system exergy
efficiency here, because the gas is the same.

Steps (run one at a time or all):  python scripts/run_cycle_comparison.py [--steps simple,cascade,split,compare]
Extra step: --steps t7 / --steps prx  -> net power, efficiencies and stack temperature against T7 / against the pressure ratio
(x re-optimised at each point; Simple-Recompression/simple_t7_sweep.png, simple_pr_sweep.png)
Extra step: --steps assumptions  -> table of the design assumptions of the simple recompression cycle and a
summary of the waste heat condition (Simple-Recompression/simple_assumptions.png)
Results:
  Simple-Recompression/              the simple recompression cycle
  Recompression with branch heater/  the recompression cycle with a branch heater
  Cycle-Comparison/                  split-flow reference and the comparison of the three"""
import inspect
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from scipy.optimize import minimize

plt.rcParams.update({"font.family": "serif",           # Times New Roman for all figure text
                     "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": False})
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

from sco2.cycle_adv import recompression_adv, recompression_cascade, splitflow_adv
from sco2.heat_source_adv import EAF, couple_cascade, couple_heater
from sco2.plant_splitflow import get_gas

K = 273.15
T7 = 550.0 + K
T1K = 32.0 + K                               # main compressor inlet temperature (K); changed only by the t1 step
P1 = 7.7e6
P_MAX = 25e6
PR_MAX = P_MAX / P1                        # 3.2468
DT_REC = 10.0                              # K, HTR and LTR
PEN = 1.0                                  # MW of penalty per K of violation
GAS = get_gas()
DTP, TMIN, MG, TIN = EAF["dT_pinch_heater"], EAF["T_stack_min"], EAF["m_gas"], EAF["T_in"]
DIRS = {"simple": Path("Simple-Recompression"), "cascade": Path("Recompression with branch heater"),
        "split": Path("Cycle-Comparison"), "compare": Path("Cycle-Comparison")}
TITLE = {"simple": "Simple recompression (one heater)", "cascade": "Recompression with a branch heater",
         "split": "Split-flow, no recompressor"}
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#888888"
EXERGY_C = "#7a4fd1"                        # exergy efficiency curves
# Efficiencies, all with the net shaft power W as the output:
#   energy efficiency, captured heat      = W / Q taken by the cycle from the gas
#   energy efficiency, available heat     = W / Q the gas could give down to the stack limit (150 C)
#   exergy efficiency, captured exergy    = W / exergy given up by the gas in the heater(s)
#   exergy efficiency, available exergy   = W / exergy the gas could give down to the stack limit
EFF_COLS = ["Energy efficiency, captured heat (%)", "Energy efficiency, available heat (%)",
            "Exergy efficiency, captured exergy (%)", "Exergy efficiency, available exergy (%)"]

# ---- the three layouts: variables, bounds, and how to run them -----------------------------------------
# The pressure ratio is NOT optimised: it is fixed by the maximum pressure, pr = P_MAX / P1 (no cost model yet).
# z vectors: simple [x]; cascade [x, y, T_b (C)]; split [y, T_b (C)]
LAYOUT = {
    "simple": dict(names=["x"], lb=[0.0], ub=[0.6]),
    "cascade": dict(names=["x", "y", "T_b"], lb=[0.0, 0.02, 200.0], ub=[0.6, 0.6, 550.0]),
    "split": dict(names=["y", "T_b"], lb=[0.02, 200.0], ub=[0.6, 550.0])}


def run(kind, z, pr=None):
    """(cycle result, gas-coupling result) for one design vector (pr defaults to the fixed PR_MAX)."""
    pr = PR_MAX if pr is None else pr
    if kind == "simple":
        res = recompression_adv(x=z[0], pr=pr, T7=T7, T1=T1K, p1=P1)
        return res, couple_heater(res, GAS, TIN, MG, dT_pinch=DTP, T_stack_min=TMIN)
    if kind == "cascade":
        res = recompression_cascade(x=z[0], y=z[1], T_b=z[2] + K, pr=pr, T7=T7, check_pinch=True)
    else:
        res = splitflow_adv(y=z[0], T_b=z[1] + K, pr=pr, T7=T7, check_pinch=True)
    return res, couple_cascade(res, GAS, TIN, MG, DTP, DTP, TMIN)


def violation(res):
    p = res["pinch"]
    return max(0.0, DT_REC - p["HTR"]["min_dT"]) + max(0.0, DT_REC - p["LTR"]["min_dT"])


def objective(kind, z, pr=None):
    """Minimise: - net power (MW) + penalty for HTR/LTR pinch below the minimum."""
    try:
        res, hs = run(kind, z, pr)
        return -hs["m_co2"] * res["w_net"] / 1e6 + PEN * violation(res)
    except Exception:
        return 10.0


def search(fun, lb, ub, starts=(), n_lhs=200, n_nm=3, maxfev=200, seed=1):
    """Global-then-local search: Latin hypercube sample, then Nelder-Mead from the best points."""
    lb, ub = np.array(lb, float), np.array(ub, float)
    rng = np.random.default_rng(seed)
    d = len(lb)
    u = np.column_stack([(rng.permutation(n_lhs) + rng.random(n_lhs)) / n_lhs for _ in range(d)])
    pts = [lb + (ub - lb) * r for r in u] + [np.clip(np.array(s, float), lb, ub) for s in starts]
    vals = np.array([fun(p) for p in pts])
    best = []
    for k in np.argsort(vals):
        if all(np.linalg.norm((pts[k] - b) / (ub - lb)) > 0.05 for b in best):
            best.append(pts[k])
        if len(best) == n_nm:
            break
    out = (np.inf, None)
    for z0 in best:
        verts = [z0]
        for j in range(d):
            v = z0.copy()
            v[j] += 0.05 * (ub[j] - lb[j]) if z0[j] + 0.05 * (ub[j] - lb[j]) <= ub[j] else -0.05 * (ub[j] - lb[j])
            verts.append(v)
        o = minimize(fun, z0, method="Nelder-Mead", bounds=[(a, b) for a, b in zip(lb, ub)],
                     options={"initial_simplex": np.array(verts), "maxfev": maxfev, "xatol": 1e-4, "fatol": 1e-6})
        if o.fun < out[0]:
            out = (o.fun, o.x)
    return out[1], out[0]


def metrics(kind, z, pr=None):
    pr = PR_MAX if pr is None else pr
    res, hs = run(kind, z, pr)
    W = hs["m_co2"] * res["w_net"]
    names = LAYOUT[kind]["names"]
    zz = dict(zip(names, [float(v) for v in z]))
    ex_gas_in = MG * float(GAS.ex(TIN))                                  # physical exergy of the gas at the inlet
    ex_taken = ex_gas_in - MG * float(GAS.ex(hs["T_out"]))               # exergy given up by the gas in the cycle heater(s)
    ex_avail = ex_gas_in - MG * float(GAS.ex(TMIN))                      # exergy given up if the gas cooled to the stack limit
    return {"kind": kind, "x": zz.get("x", 0.0), "y": zz.get("y", 0.0),
            "T_b (C)": zz.get("T_b", float("nan")), "pr": pr, "p_max (MPa)": pr * P1 / 1e6,
            "Net power, shaft (MW)": W / 1e6, "System exergy efficiency (%)": W / hs["ex_in"] * 100,
            "Cycle thermal efficiency (%)": res["eta"] * 100,
            EFF_COLS[0]: W / hs["Q"] * 100, EFF_COLS[1]: W / hs["Q_max"] * 100,
            EFF_COLS[2]: W / ex_taken * 100, EFF_COLS[3]: W / ex_avail * 100,
            "Heat taken from the gas (MW)": hs["Q"] / 1e6,
            "Share of the available heat (%)": hs["Q"] / hs["Q_max"] * 100,
            "Gas stack temperature (C)": hs["T_out"] - K, "CO2 mass flow (kg/s)": hs["m_co2"],
            "HTR min dT (K)": res["pinch"]["HTR"]["min_dT"], "LTR min dT (K)": res["pinch"]["LTR"]["min_dT"],
            "Main heater min dT (K)": float(hs.get("min_dT", hs.get("min_dT1", float("nan")))),
            "Branch heater min dT (K)": float(hs.get("min_dT2", float("nan"))),
            "Limiting constraint": hs.get("binding", "heater pinch or stack")}, res, hs


def optimum(kind):
    lay = LAYOUT[kind]
    print(f"\n{TITLE[kind]}: searching the optimum ({len(lay['names'])} variable(s); "
          f"pr fixed at {PR_MAX:.4f} = {P_MAX / 1e6:.0f} MPa / {P1 / 1e6:.1f} MPa)...", flush=True)
    t0 = time.time()
    extra = {"simple": [[0.0], [0.27]], "cascade": [[0.0, 0.40, 354.0]], "split": [[0.39, 354.0]]}[kind]           # start points on the x = 0 edge / near the known optimum
    z, f = search(lambda z: objective(kind, z), lay["lb"], lay["ub"], starts=extra)
    m, res, hs = metrics(kind, z)
    print(f"  net power {m['Net power, shaft (MW)']:.4f} MW at "
          + ", ".join(f"{n} = {v:.4g}" for n, v in zip(lay["names"], z)) + f"  ({(time.time() - t0) / 60:.1f} min)")
    for n, v, lo, hi in zip(lay["names"], z, lay["lb"], lay["ub"]):
        if abs(v - lo) < 1e-3 * (hi - lo + 1) or abs(v - hi) < 1e-3 * (hi - lo + 1):
            print(f"  note: {n} is at its bound ({lo if abs(v - lo) < abs(v - hi) else hi:g})")
    return z, m, res, hs


def x_scan(kind, z_opt):
    """Net power against the recompression fraction x, other variables re-optimised."""
    lay = LAYOUT[kind]
    rows, zs = [], []
    others = np.array([i for i, n in enumerate(lay["names"]) if n != "x"], dtype=int)
    lb, ub = np.array(lay["lb"], float)[others], np.array(lay["ub"], float)[others]
    prev = np.array(z_opt, float)[others]
    for x in np.linspace(0.0, 0.6, 13):
        def f(w, x=x):
            z = np.array(z_opt, float)
            z[others] = w
            z[0] = x
            return objective(kind, z)
        if len(others):
            w, fv = search(f, lb, ub, starts=[prev], n_lhs=40, n_nm=1, maxfev=120)
        else:                                              # x is the only variable: nothing else to re-optimise
            w = np.array([])
        z = np.array(z_opt, float)
        z[others] = w
        z[0] = x
        prev = w
        zs.append(z.copy())
        try:
            m, _, _ = metrics(kind, z)
            rows.append({"x": x, "Net power (MW)": m["Net power, shaft (MW)"],
                         "System exergy efficiency (%)": m["System exergy efficiency (%)"],
                         "Cycle thermal efficiency (%)": m["Cycle thermal efficiency (%)"],
                         **{c: m[c] for c in EFF_COLS},
                         "Gas stack temperature (C)": m["Gas stack temperature (C)"],
                         "CO2 mass flow (kg/s)": m["CO2 mass flow (kg/s)"],
                         "pr": m["pr"], "HTR min dT (K)": m["HTR min dT (K)"], "LTR min dT (K)": m["LTR min dT (K)"],
                         "Main heater min dT (K)": m["Main heater min dT (K)"],
                         "Branch heater min dT (K)": m["Branch heater min dT (K)"],
                         "Recuperator limit met": violation(run(kind, z)[0]) < 1e-6})
        except Exception:
            rows.append({"x": x, "Net power (MW)": np.nan, "Recuperator limit met": False})
        print(f"    x = {x:.2f}: {rows[-1]['Net power (MW)']:.3f} MW", flush=True)
    return pd.DataFrame(rows), zs


def pr_scan(kind, z, m, folder):
    """Effect of the pressure ratio (maximum pressure = pr x p1) at the optimum design (x, y, T_b fixed)."""
    prs = np.unique(np.round(np.concatenate([np.linspace(2.0, 30e6 / P1, 11), [PR_MAX]]), 5))
    rows = []
    for pr in prs:
        try:
            mm, res, _ = metrics(kind, z, pr)
            rows.append({"pr": pr, "p_max (MPa)": pr * P1 / 1e6, "Net power (MW)": mm["Net power, shaft (MW)"],
                         "Energy efficiency (%)": mm["Cycle thermal efficiency (%)"],
                         "System exergy efficiency (%)": mm["System exergy efficiency (%)"],
                         **{c: mm[c] for c in EFF_COLS},
                         "Gas stack temperature (C)": mm["Gas stack temperature (C)"],
                         "HTR min dT (K)": mm["HTR min dT (K)"], "LTR min dT (K)": mm["LTR min dT (K)"],
                         "Recuperator limit met": violation(res) < 1e-6})
        except Exception:
            rows.append({"pr": pr, "p_max (MPa)": pr * P1 / 1e6, "Net power (MW)": np.nan,
                         "Recuperator limit met": False})
    sc = pd.DataFrame(rows)
    sc.round(5).to_csv(folder / f"{kind}_pr_scan.csv", index=False, encoding="utf-8-sig")
    plot_pr(kind, sc, z, folder)


def plot_pr(kind, sc, z, folder):
    """Energy and exergy efficiency (left axis) and net power (right axis) against the pressure ratio."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax2 = ax.twinx()
    ax.plot(sc["pr"], sc["Energy efficiency (%)"], color=AQUA, lw=2, marker="^", ms=5,
            label="energy efficiency, captured (left axis)")
    ax.plot(sc["pr"], sc[EFF_COLS[2]], color=EXERGY_C, lw=1.6, ls="--", marker="s", ms=7,
            mfc="none", label="exergy efficiency, captured (left axis)")
    ax2.plot(sc["pr"], sc["Net power (MW)"], color=BLUE, lw=2, marker="o", ms=5, label="net power (right axis)")
    bad = ~sc["Recuperator limit met"].astype(bool).to_numpy()
    handles = ax.get_lines() + ax2.get_lines()
    if bad.any():
        ring, = ax2.plot(sc["pr"][bad], sc["Net power (MW)"][bad], ls="none", marker="o", ms=10, mfc="none",
                         mec="#e34948", mew=1.6, label="HTR or LTR pinch below the limit")
        handles.append(ring)
    ax.axvline(PR_MAX, color=GREY, lw=1, ls="--")
    ax.annotate(f"design: pr = {PR_MAX:.3f} ({P_MAX / 1e6:.0f} MPa)", (PR_MAX, 0.5), xycoords=("data", "axes fraction"),
                xytext=(-5, 0), textcoords="offset points", fontsize=8, rotation=90, va="center", ha="right")
    ax.set_xlabel("Pressure ratio (maximum pressure = pr x p1)")
    ax.set_ylabel("Efficiency (%)")
    ax2.set_ylabel("Net power, shaft (MW)")
    top = ax.secondary_xaxis("top", functions=(lambda r: r * P1 / 1e6, lambda p: p * 1e6 / P1))
    top.set_xlabel("Maximum pressure (MPa)")
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ttl = ", ".join(f"{n} = {v:.3g}" for n, v in zip(LAYOUT[kind]["names"], z))
    fig.suptitle(f"{TITLE[kind]}: effect of the pressure ratio at the optimum design ({ttl})", fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / f"{kind}_pr_scan.png", dpi=170)
    plt.close(fig)


def save_table(kind, m, z, folder):
    names = LAYOUT[kind]["names"]
    unit = {"x": "-", "y": "-", "T_b": "C", "pr": "-"}
    rows = [{"Quantity": f"Decision variable {n}", "Unit": unit[n], "Value": float(v)} for n, v in zip(names, z)]
    for k, v in m.items():
        if k != "kind":
            name, _, u = k.partition(" (")
            rows.append({"Quantity": name, "Unit": u.rstrip(")") if u else "-", "Value": v})
    pd.DataFrame(rows).round(5).to_csv(folder / f"{kind}_optimum.csv", index=False, encoding="utf-8-sig")
    with open(folder / f"{kind}_optimum.json", "w") as f:
        json.dump({"z": [float(v) for v in z], "names": names, "metrics": m}, f, indent=2)


def exchanger_profile(res, hot, cold, n=20, props=False):
    """Temperature profiles of a recuperator in n equal steps of its duty, from the hot end.
    hot = (inlet state, outlet state) of the hot side, cold = (inlet, outlet) of the cold side;
    the enthalpy of each side changes linearly with the duty, the pressure linearly between its states."""
    st = res["states"]
    f = np.linspace(0.0, 1.0, n + 1)
    (hi, ho), (ci, co) = hot, cold
    h_h = st[hi]["h"] - f * (st[hi]["h"] - st[ho]["h"])
    p_h = st[hi]["p"] + f * (st[ho]["p"] - st[hi]["p"])
    h_c = st[co]["h"] - f * (st[co]["h"] - st[ci]["h"])            # the cold outlet sits at the hot end
    p_c = st[co]["p"] + f * (st[ci]["p"] - st[co]["p"])
    T_h = np.array([PropsSI("T", "H", h, "P", p, "CO2") for h, p in zip(h_h, p_h)])
    T_c = np.array([PropsSI("T", "H", h, "P", p, "CO2") for h, p in zip(h_c, p_c)])
    if props:
        cp_h = np.array([PropsSI("C", "H", h, "P", q, "CO2") for h, q in zip(h_h, p_h)])
        cp_c = np.array([PropsSI("C", "H", h, "P", q, "CO2") for h, q in zip(h_c, p_c)])
        d_h = np.array([PropsSI("D", "H", h, "P", q, "CO2") for h, q in zip(h_h, p_h)])
        d_c = np.array([PropsSI("D", "H", h, "P", q, "CO2") for h, q in zip(h_c, p_c)])
        return f, T_h, T_c, cp_h, cp_c, d_h, d_c
    return f, T_h, T_c


def flow_line(ax, f, y, color, leftward, label):
    """Plot a curve and put one small filled arrowhead on every segment, aligned with the local slope of the
    line, to show the flow direction (leftward = towards x = 0). Returns the legend handle."""
    f, y = np.asarray(f, float), np.asarray(y, float)
    ax.plot(f, y, color=color, lw=1.8)
    for i in range(len(f) - 1):
        a, b = (f[i], y[i]), (f[i + 1], y[i + 1])
        m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        tail = (m[0] - 0.22 * (b[0] - a[0]), m[1] - 0.22 * (b[1] - a[1]))
        head = (m[0] + 0.22 * (b[0] - a[0]), m[1] + 0.22 * (b[1] - a[1]))
        tail, head = (head, tail) if leftward else (tail, head)
        ax.annotate("", xy=head, xytext=tail, arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.25", color=color, lw=0.4,
                    mutation_scale=11, shrinkA=0, shrinkB=0))
    return Line2D([0], [0], color=color, lw=1.8, marker="<" if leftward else ">", ms=5, label=label)


def plot_ltr_properties(kind, res, folder, m):
    """Specific heat capacity and density of both LTR streams in the same 20 equal steps of duty."""
    f, th, tc, cph, cpc, dh, dc = exchanger_profile(res, hot=(9, 10), cold=(2, 3), props=True)
    st = res["states"]
    ph, pc = st[9]["p"] / 1e6, st[2]["p"] / 1e6
    pd.DataFrame({"Fraction of the LTR duty": f, "Hot side T (C)": th - K, "Cold side T (C)": tc - K,
                  "Hot side cp (kJ/kg K)": cph / 1e3, "Cold side cp (kJ/kg K)": cpc / 1e3,
                  "Hot side density (kg/m3)": dh, "Cold side density (kg/m3)": dc}).round(4).to_csv(
        folder / f"{kind}_ltr_properties.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    red = "#e34948"
    for ax, (yh, yc, yl, ttl) in zip(axes, ((cph / 1e3, cpc / 1e3, "Specific heat capacity cp (kJ/kg K)", "Heat capacity"),
                                            (dh, dc, "Density (kg/m3)", "Density"))):
        hh = flow_line(ax, f, yh, red, False, f"hot side ({ph:.1f} MPa)")
        hc = flow_line(ax, f, yc, BLUE, True, f"cold side ({pc:.1f} MPa)")
        ax.set_xlabel("Fraction of the LTR duty (from the hot end)")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10)
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.legend(handles=[hh, hc], frameon=False, fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    k = int(np.argmax(cpc))
    axes[0].annotate(f"cold-side cp peak {cpc[k] / 1e3:.2f}\nat {tc[k] - K:.0f} C", (f[k], cpc[k] / 1e3),
                     xytext=(-8 if f[k] > 0.5 else 8, -30), textcoords="offset points", fontsize=8,
                     va="top", ha="right" if f[k] > 0.5 else "left",
                     arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    fig.suptitle(f"{TITLE[kind]}: LTR stream properties at the optimum ({m['Net power, shaft (MW)']:.2f} MW)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / f"{kind}_ltr_properties.png", dpi=170)
    plt.close(fig)


def plot_ltr_capacity(kind, res, hs, folder, m):
    """Heat-capacity rate (mass flow x cp) of both LTR streams along the same 20 duty steps.
    The hot side carries the whole CO2 flow; the cold-side flow follows from the energy balance of the
    (adiabatic) LTR, which also gives the cold-side share of the flow."""
    st = res["states"]
    f, th, tc, cph, cpc, dh, dc = exchanger_profile(res, hot=(9, 10), cold=(2, 3), props=True)
    m_h = hs["m_co2"]
    m_c = m_h * (st[9]["h"] - st[10]["h"]) / (st[3]["h"] - st[2]["h"])
    c_h, c_c = m_h * cph / 1e3, m_c * cpc / 1e3                      # kW/K
    ratio = c_c / c_h
    pd.DataFrame({"Fraction of the LTR duty": f, "Hot side m*cp (kW/K)": c_h, "Cold side m*cp (kW/K)": c_c,
                  "Cold/hot ratio": ratio, "Hot side flow (kg/s)": m_h, "Cold side flow (kg/s)": m_c}).round(4).to_csv(
        folder / f"{kind}_ltr_capacity.csv", index=False, encoding="utf-8-sig")
    print(f"  LTR flows: hot side {m_h:.2f} kg/s, cold side {m_c:.2f} kg/s ({m_c / m_h * 100:.1f} % of the hot side); "
          f"m*cp ratio cold/hot from {ratio.min():.2f} to {ratio.max():.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    red = "#e34948"
    hh = flow_line(axes[0], f, c_h, red, False, f"hot side ({m_h:.1f} kg/s)")
    hc = flow_line(axes[0], f, c_c, BLUE, True, f"cold side ({m_c:.1f} kg/s)")
    axes[0].set_ylabel("Heat-capacity rate  m cp (kW/K)")
    axes[0].set_title("Heat-capacity rate", fontsize=10)
    axes[0].legend(handles=[hh, hc], frameon=False, fontsize=9, loc="best")
    axes[1].plot(f, ratio, color=AQUA, lw=1.8, marker="o", ms=4)
    axes[1].axhline(1.0, color=GREY, lw=1, ls="--")
    axes[1].annotate("balanced (ratio = 1)", (0.02, 1.0), xytext=(0, 4), textcoords="offset points",
                     fontsize=8, color=GREY)
    k = int(np.argmax(ratio))
    axes[1].annotate(f"largest ratio {ratio[k]:.2f}", (f[k], ratio[k]), xytext=(0, 9),
                     textcoords="offset points", fontsize=8, ha="center", va="bottom")   # above the peak, off the line
    lo, hi = min(1.0, ratio.min()), ratio.max()
    axes[1].set_ylim(lo - 0.04 * (hi - lo), hi + 0.16 * (hi - lo))                       # headroom for the label
    axes[1].set_ylabel("Cold-side / hot-side  m cp")
    axes[1].set_title("Heat-capacity-rate ratio", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Fraction of the LTR duty (from the hot end)")
        ax.grid(True, color="#dddddd", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"{TITLE[kind]}: LTR heat-capacity rates at the optimum ({m['Net power, shaft (MW)']:.2f} MW)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / f"{kind}_ltr_capacity.png", dpi=170)
    plt.close(fig)


def plot_heater(kind, hs, folder, m, res=None):
    """Temperature profiles at the optimum: heater(s) (gas and CO2) and the LTR (hot and cold CO2),
    each in 20 equal steps of duty."""
    two = "T_gas1" in hs
    sets = ([("Main heater", hs["f"], hs["T_gas1"], hs["T_co2_1"], "gas", "CO2", "Fraction of the heater duty"),
             ("Branch heater", hs["f"], hs["T_gas2"], hs["T_co2_2"], "gas", "CO2", "Fraction of the heater duty")]
            if two else [("Heater", hs["f"], hs["T_gas"], hs["T_co2"], "gas", "CO2", "Fraction of the heater duty")])
    ltr = None
    if res is not None and "h" in res["states"][2]:
        f, th, tc = exchanger_profile(res, hot=(9, 10), cold=(2, 3))
        ltr = (f, th, tc)
        sets.append(("LTR", f, th, tc, "hot side", "cold side", "Fraction of the LTR duty (from the hot end)"))
        pd.DataFrame({"Fraction of the LTR duty": f, "Hot side (C)": th - K, "Cold side (C)": tc - K,
                      "Temperature difference (K)": th - tc}).round(4).to_csv(
            folder / f"{kind}_ltr_profile.csv", index=False, encoding="utf-8-sig")
        plot_ltr_properties(kind, res, folder, m)
        plot_ltr_capacity(kind, res, hs, folder, m)
        k = int(np.argmin(th - tc))
        print(f"  LTR profile (20 steps): minimum difference {np.min(th - tc):.2f} K at {f[k]:.2f} of the duty; "
              f"cycle model {res['pinch']['LTR']['min_dT']:.2f} K")
    n = len(sets)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), squeeze=False)
    for ax, (name, f, ta, tb, la, lb, xl) in zip(axes.ravel(), sets):
        # filled arrows give the flow direction: gas and the LTR cold side flow towards the left (x = 0),
        # CO2 in the heater and the LTR hot side flow towards the right (x = 1)
        h_a = flow_line(ax, f, np.asarray(ta) - K, ORANGE if la == "gas" else "#e34948", la == "gas", la)
        h_b = flow_line(ax, f, np.asarray(tb) - K, BLUE, lb == "cold side", lb)
        ax.set_xlabel(xl)
        ax.set_ylabel("Temperature (C)")
        ax.set_title(name, fontsize=10)
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.legend(handles=[h_a, h_b], frameon=False, fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        # minimum temperature difference: segment at the pinch and a two-line label below the lower curve
        ta_a, tb_a = np.asarray(ta) - K, np.asarray(tb) - K
        d = ta_a - tb_a
        k = int(np.argmin(d))
        right = f[k] > 0.5                                   # keep the label inside the panel
        ax.plot([f[k], f[k]], [tb_a[k], ta_a[k]], color=GREY, lw=1.4)
        ax.annotate(f"minimum difference\n{d[k]:.1f} K", (f[k], tb_a[k]), xytext=(-3 if right else 3, -8),
                    textcoords="offset points", fontsize=8, va="top", ha="right" if right else "left",
                    linespacing=1.15)
        lo, hi = min(ta_a.min(), tb_a.min()), max(ta_a.max(), tb_a.max())
        ax.set_ylim(lo - 0.22 * (hi - lo), hi + 0.04 * (hi - lo))   # room under the lower curve for the label
    fig.suptitle(f"{TITLE[kind]}: heater and LTR temperature profiles at the optimum "
                 f"({m['Net power, shaft (MW)']:.2f} MW)", fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / f"{kind}_heater_profiles.png", dpi=170)
    plt.close(fig)


def plot_scan(kind, sc, m, folder):
    """Left panel: energy (cycle thermal) efficiency on the left axis and net power on the right
    axis against x; right panel: gas stack temperature. Dashed line = optimum (all limits met)."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    x = sc["x"]
    ax, ax2 = axes[0], axes[0].twinx()
    ax.plot(x, sc["Cycle thermal efficiency (%)"], color=AQUA, lw=2, marker="^", ms=5,
            label="energy efficiency, captured (left axis)")
    ax.plot(x, sc[EFF_COLS[2]], color=EXERGY_C, lw=1.6, ls="--", marker="s", ms=7, mfc="none",
            label="exergy efficiency, captured (left axis)")
    ax2.plot(x, sc["Net power (MW)"], color=BLUE, lw=2, marker="o", ms=5, label="net power (right axis)")
    ax.set_ylabel("Efficiency (%)")
    ax2.set_ylabel("Net power, shaft (MW)")
    ax.legend(handles=ax.get_lines() + ax2.get_lines(), frameon=False, fontsize=8, loc="lower left")
    axes[1].plot(x, sc["Gas stack temperature (C)"], color=ORANGE, lw=2, marker="s", ms=5,
                 label="stack temperature (left)")
    axes[1].axhline(TMIN - K, color=GREY, lw=1, ls=":")
    axes[1].set_ylabel("Gas stack temperature (C)")
    axes[1].set_ylim(0, None)
    ax3 = axes[1].twinx()                                  # efficiencies on the total heat and exergy available in the gas
    ax3.plot(x, sc[EFF_COLS[1]], color=AQUA, lw=2, marker="^", ms=5,
             label=f"energy eff., available to {TMIN - K:.0f} C (right)")
    ax3.plot(x, sc[EFF_COLS[3]], color=EXERGY_C, lw=1.6, ls="--", marker="s", ms=7, mfc="none",
             label=f"exergy eff., available to {TMIN - K:.0f} C (right)")
    ax3.set_ylabel("Efficiency on available heat / exergy (%)")
    ax3.set_ylim(0, None)
    ax3.spines["top"].set_visible(False)
    axes[1].legend(handles=axes[1].get_lines()[:1] + ax3.get_lines(), frameon=False, fontsize=8, loc="lower left")
    axes[2].plot(x, sc["CO2 mass flow (kg/s)"], color="#6250d6", lw=2, marker="D", ms=5)
    axes[2].set_ylabel("CO2 mass flow (kg/s)")
    for a in (ax, axes[1], axes[2]):
        a.axvline(m["x"], color=GREY, lw=1, ls="--")
        a.set_xlabel("Recompression fraction x")
        a.grid(True, color="#dddddd", lw=0.6)
    for txt, dx, ha in ((f"optimum x = {m['x']:.3f}", -4, "right"),
                        (f"LTR pinch = {m['LTR min dT (K)']:.1f} K", 4, "left")):
        ax.annotate(txt, (m["x"], 0.62), xycoords=("data", "axes fraction"), xytext=(dx, 0),
                    textcoords="offset points", fontsize=8, rotation=90, va="center", ha=ha)
    for a in (ax, ax2, axes[1], axes[2]):
        a.spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
    fig.suptitle(f"{TITLE[kind]}: effect of the recompression fraction (other variables re-optimised)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / f"{kind}_x_scan.png", dpi=170)
    plt.close(fig)


def plot_pinch(kind, sc, m, folder):
    """Recuperator pinch points against x (limit 10 K, points below it ringed in red)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = sc["x"]
    ax.plot(x, sc["HTR min dT (K)"], color=BLUE, lw=2, marker="o", ms=5, label="HTR")
    ax.plot(x, sc["LTR min dT (K)"], color=AQUA, lw=2, marker="^", ms=5, label="LTR")
    ax.axhline(DT_REC, color="#e34948", lw=1.2, ls="--", label=f"limit {DT_REC:g} K")
    for col in ("HTR min dT (K)", "LTR min dT (K)"):
        v = sc[col].to_numpy()
        low = v < DT_REC - 1e-6
        if low.any():
            ax.plot(x[low], v[low], ls="none", marker="o", ms=9, mfc="none", mec="#e34948", mew=1.6)
    ax.axvline(m["x"], color=GREY, lw=1, ls="--")
    ax.set_xlabel("Recompression fraction x")
    ax.set_ylabel("Recuperator minimum temperature difference (K)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color="#dddddd", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{TITLE[kind]}: recuperator pinch points\n(red rings: below the limit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / f"{kind}_x_scan_pinch.png", dpi=170)
    plt.close(fig)


def do_cycle(kind):
    folder = DIRS[kind]
    folder.mkdir(parents=True, exist_ok=True)
    z, m, res, hs = optimum(kind)
    sc = None
    if "x" in LAYOUT[kind]["names"]:
        print("  scan of the recompression fraction x ...", flush=True)
        sc, zs = x_scan(kind, z)
        feas = sc["Recuperator limit met"].astype(bool) & sc["Net power (MW)"].notna()
        k = int(sc["Net power (MW)"].where(feas).idxmax()) if feas.any() else None
        best_ok = violation(res) < 1e-6                        # the first search result respects the limits
        if k is not None and (not best_ok or sc["Net power (MW)"].iloc[k] > m["Net power, shaft (MW)"] + 1e-6):
            lay = LAYOUT[kind]
            print(f"  the scan found a better design that respects the pinch limits "
                  f"({sc['Net power (MW)'].iloc[k]:.4f} MW at x = {sc['x'].iloc[k]:.2f}); polishing from that point ...")
            z2, _ = search(lambda zz: objective(kind, zz), lay["lb"], lay["ub"], starts=[zs[k]],
                           n_lhs=40, n_nm=1, maxfev=300)
            m2, res2, hs2 = metrics(kind, z2)
            if violation(res2) < 1e-6 and m2["Net power, shaft (MW)"] >= sc["Net power (MW)"].iloc[k] - 1e-6:
                z, m, res, hs = z2, m2, res2, hs2
            else:
                z = zs[k]
                m, res, hs = metrics(kind, z)
                res = run(kind, z)[0]
            print(f"  corrected optimum: {m['Net power, shaft (MW)']:.4f} MW at "
                  + ", ".join(f"{n} = {v:.4g}" for n, v in zip(lay["names"], z)))
        print(f"  optimum respects the limits: HTR {m['HTR min dT (K)']:.2f} K, LTR {m['LTR min dT (K)']:.2f} K "
              f"(limit {DT_REC:g} K)")
        sc.round(5).to_csv(folder / f"{kind}_x_scan.csv", index=False, encoding="utf-8-sig")
        plot_scan(kind, sc, m, folder)
        plot_pinch(kind, sc, m, folder)
    save_table(kind, m, z, folder)
    plot_heater(kind, hs, folder, m, res)
    pr_scan(kind, z, m, folder)
    print(f"  saved in '{folder}'")
    return m


def do_split():
    """Split-flow reference: re-optimised, and the saved current design, same method."""
    folder = DIRS["split"]
    folder.mkdir(parents=True, exist_ok=True)
    z, m, res, hs = optimum("split")
    save_table("split", m, z, folder)
    plot_heater("split", hs, folder, m, res)
    pr_scan("split", z, m, folder)
    with open("data/results/splitflow_optimum_ref.json") as f:
        o = json.load(f)
    zc = [o["y"], (o["T_b"] - K) if o["T_b"] > 500 else o["T_b"]]
    mc, _, _ = metrics("split", zc, o["pr"])
    with open(folder / "split_current_design.json", "w") as f:
        json.dump({"z": zc, "names": LAYOUT["split"]["names"], "metrics": mc}, f, indent=2)
    print(f"  current design (saved): {mc['Net power, shaft (MW)']:.4f} MW")
    return m


def do_compare():
    folder = DIRS["compare"]
    folder.mkdir(parents=True, exist_ok=True)
    files = [("simple", DIRS["simple"] / "simple_optimum.json", "Simple recompression"),
             ("cascade", DIRS["cascade"] / "cascade_optimum.json", "Recompression + branch heater"),
             ("split", DIRS["split"] / "split_optimum.json", "Split-flow (re-optimised)"),
             ("split", DIRS["split"] / "split_current_design.json", "Split-flow (current design)")]
    cols = {}
    for kind, path, label in files:
        if path.exists():
            cols[label] = json.load(open(path))["metrics"]
        else:
            print(f"  missing: {path} (run that step first)")
    if len(cols) < 2:
        raise SystemExit("Not enough results to compare; run the other steps first.")
    keys = [k for k in next(iter(cols.values())) if k != "kind"]
    tab = pd.DataFrame({"Quantity": keys, **{lab: [m.get(k, float('nan')) for k in keys] for lab, m in cols.items()}})
    base = next(iter(cols))
    tab.loc[len(tab)] = ["Net power gain over " + base + " (%)"] + [
        (m["Net power, shaft (MW)"] / cols[base]["Net power, shaft (MW)"] - 1) * 100 for m in cols.values()]
    tab.round(4).to_csv(folder / "comparison_table.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print("\nCOMPARISON (same fixed conditions, shaft net power)\n")
    print(tab.round(3).to_string(index=False))

    labels = list(cols)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, key, ttl in ((axes[0], "Net power, shaft (MW)", "Net power, shaft (MW)"),
                         (axes[1], "Gas stack temperature (C)", "Gas stack temperature (C)"),
                         (axes[2], "x", "Recompression fraction x")):
        vals = [cols[lab][key] for lab in labels]
        bars = ax.bar(range(len(labels)), vals, color=[BLUE, ORANGE, AQUA, GREY][:len(labels)], width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}" if key != "x" else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(" (", "\n(").replace(" + ", "\n+ ") for l in labels], fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.grid(True, axis="y", color="#dddddd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Cycle layouts under the same fixed conditions (maximum net power)", fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=170)
    print(f"\nSaved in '{folder}': comparison_table.csv, comparison.png")


def do_assumptions():
    """One figure: table of the design assumptions of the simple recompression cycle, below it a summary of
    the waste heat condition. Values are read from the model (defaults, EAF data, saved optimum)."""
    folder = DIRS["simple"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "simple_optimum.json"
    if not path.exists():
        raise SystemExit("Run the 'simple' step first (it writes simple_optimum.json).")
    m = json.load(open(path))["metrics"]
    D = {k: v.default for k, v in inspect.signature(recompression_adv).parameters.items()}
    rows = [
        ("Working fluid", "CO2 (real gas)", "properties from CoolProp"),
        ("Layout", "Simple recompression", "one heater"),
        ("Turbomachinery", "Main compressor, recompressor, turbine", ""),
        ("Heat exchangers", "LTR, HTR, gas heater, cooler", ""),
        ("Compressor inlet temperature, T1", f"{D['T1'] - K:.1f}", "C (wet cooling tower)"),
        ("Compressor inlet pressure, p1", f"{D['p1'] / 1e6:.1f}", "MPa"),
        ("Maximum pressure (compressor outlet)", f"{m['p_max (MPa)']:.1f}", "MPa (fixed)"),
        ("Pressure ratio, pr", f"{m['pr']:.3f}", "- (fixed by the maximum pressure)"),
        ("Turbine inlet temperature, T7", f"{T7 - K:.0f}", "C"),
        ("Recompression fraction, x", f"{m['x']:.3f}", "- (optimised)"),
        ("Isentropic efficiency, main compressor / recompressor", f"{D['eta_mc']:.2f} / {D['eta_rc']:.2f}", "-"),
        ("Isentropic efficiency, turbine", f"{D['eta_t']:.2f}", "-"),
        ("Effectiveness, HTR / LTR", f"{D['eps_htr']:.2f} / {D['eps_ltr']:.2f}", "- (enthalpy based)"),
        ("Minimum temperature difference, HTR and LTR", f">= {DT_REC:g}", "K (enforced)"),
        ("Minimum temperature difference, heater (gas - CO2)", f"{DTP:g}", "K"),
        ("Relative pressure drop: heater / cooler", f"{D['dp_heater'] * 100:.0f} % / {D['dp_cooler'] * 100:.0f} %", "of inlet pressure"),
        ("Relative pressure drop: LTR, HTR (each side)", f"{D['dp_ltr_cold'] * 100:.0f} %, {D['dp_htr_cold'] * 100:.0f} %", "of inlet pressure"),
        ("Minimum gas stack temperature", f"{TMIN - K:.0f}", "C"),
        ("Objective", "Maximum net shaft power", "shaft power, before generator losses"),
    ]
    t0 = 298.15
    q_max = MG * (float(GAS.h(TIN)) - float(GAS.h(TMIN)))
    ex_in = MG * float(GAS.ex(TIN))
    comp = ", ".join(f"{k} {100 * v:.1f} %" for k, v in sorted(EAF["X"].items(), key=lambda kv: -kv[1]) if v > 5e-4)
    lines = [
        ("Source", "Electric-arc-furnace off-gas, mean of the measured reference cycle (Dal Magro 2017, Fig. 3.40)"),
        ("Inlet temperature / mass flow", f"{TIN - K:.2f} C  /  {MG:.3f} kg/s  (1.34e5 Nm3/h x 1.3 kg/Nm3)"),
        ("Gas pressure, composition (mole)", f"{EAF['p'] / 1e3:.1f} kPa;  {comp}"),
        ("Heat available down to 150 C", f"{q_max / 1e6:.2f} MW  (physical exergy {ex_in / 1e6:.2f} MW; dead state 25 C, 101.325 kPa)"),
        ("Taken by this cycle", f"{m['Heat taken from the gas (MW)']:.2f} MW ({m['Share of the available heat (%)']:.0f} % of the available), "
                                f"stack gas {m['Gas stack temperature (C)']:.0f} C"),
        ("Result", f"net shaft power {m['Net power, shaft (MW)']:.2f} MW, energy efficiency {m['Cycle thermal efficiency (%)']:.1f} %, "
                   f"system exergy efficiency {m['System exergy efficiency (%)']:.1f} %"),
    ]
    n = len(rows)
    row_h, line_h, W = 0.27, 0.27, 9.2                       # inches
    t_h = (n + 1) * row_h                                    # table height
    H = 0.55 + t_h + 0.35 + (len(lines) + 1) * line_h + 0.25
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.03, 1 - (0.55 + t_h) / H, 0.94, t_h / H])
    ax.axis("off")
    tab = ax.table(cellText=[list(r) for r in rows], colLabels=["Assumption", "Value", "Unit / note"],
                   colWidths=[0.39, 0.33, 0.28], cellLoc="left", bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#e9e9e9")
            cell.set_text_props(fontweight="bold")
    fig.text(0.5, 1 - 0.28 / H, "Simple recompression cycle: design assumptions", fontsize=12,
             fontweight="bold", ha="center", va="center")
    y = 1 - (0.55 + t_h + 0.35) / H
    fig.text(0.03, y, "Waste heat condition", fontsize=11, fontweight="bold", va="center")
    for i, (k, v) in enumerate(lines, 1):
        fig.text(0.03, y - i * line_h / H, k + ":", fontsize=9.5, fontweight="bold", va="center")
        fig.text(0.29, y - i * line_h / H, v, fontsize=9.5, va="center")
    fig.savefig(folder / "simple_assumptions.png", dpi=200)
    pd.DataFrame(rows, columns=["Assumption", "Value", "Unit / note"]).to_csv(
        folder / "simple_assumptions.csv", index=False, encoding="utf-8-sig")
    plt.close(fig)
    print(f"Saved in '{folder}': simple_assumptions.png, simple_assumptions.csv")


STATE_TEXT = {1: "Main compressor inlet (cooler outlet)", 2: "Main compressor outlet, LTR cold inlet",
              3: "LTR cold outlet", 4: "Recompressor outlet", 5: "Mixer outlet, HTR cold inlet",
              6: "HTR cold outlet, heater inlet", 7: "Heater outlet, turbine inlet",
              8: "Turbine outlet, HTR hot inlet", 9: "HTR hot outlet, LTR hot inlet",
              10: "LTR hot outlet, cooler and recompressor inlet"}


def state_table(res, hs, x):
    """One row per state of the simple recompression cycle. T, s, density and exergy are evaluated with
    CoolProp from the (h, p) pair of the model; the mass flow of each stream follows from the split x."""
    st, m = res["states"], hs["m_co2"]
    h0, s0 = PropsSI("H", "T", 298.15, "P", 101325.0, "CO2"), PropsSI("S", "T", 298.15, "P", 101325.0, "CO2")
    rows = []
    for k in range(1, 11):
        if k not in st:
            continue
        h, p = float(st[k]["h"]), float(st[k]["p"])
        T = PropsSI("T", "H", h, "P", p, "CO2")
        s = PropsSI("S", "H", h, "P", p, "CO2")
        rho = PropsSI("D", "H", h, "P", p, "CO2")
        flow = (1 - x) * m if k in (1, 2, 3) else (x * m if k == 4 else m)
        rows.append({"State": k, "Location": STATE_TEXT[k], "T (C)": T - K, "p (MPa)": p / 1e6,
                     "h (kJ/kg)": h / 1e3, "s (kJ/kg K)": s / 1e3, "Density (kg/m3)": rho,
                     "Mass flow (kg/s)": flow, "Exergy (kJ/kg)": ((h - h0) - 298.15 * (s - s0)) / 1e3})
    return pd.DataFrame(rows)


def plot_state_table(tab, m, folder):
    cols = ["State", "Location", "T (C)", "p (MPa)", "h (kJ/kg)", "s (kJ/kg K)", "Density (kg/m3)",
            "Mass flow (kg/s)", "Exergy (kJ/kg)"]
    head = ["State", "Location", "T (°C)", "p (MPa)", "h (kJ/kg)", "s (kJ/kg·K)", "Density (kg/m³)",
            "Mass flow (kg/s)", "Exergy (kJ/kg)"]
    fmt = {"T (C)": "{:.1f}", "p (MPa)": "{:.3f}", "h (kJ/kg)": "{:.1f}", "s (kJ/kg K)": "{:.4f}",
           "Density (kg/m3)": "{:.1f}", "Mass flow (kg/s)": "{:.2f}", "Exergy (kJ/kg)": "{:.1f}"}
    cells = [[str(int(r["State"])), r["Location"]] + [fmt[c].format(r[c]) for c in cols[2:]]
             for _, r in tab.iterrows()]
    n = len(cells)
    W, row_h = 12.6, 0.36
    H = 0.75 + (n + 1) * row_h + 0.65
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.02, 0.65 / H, 0.96, (n + 1) * row_h / H])
    ax.axis("off")
    widths = [0.06, 0.29, 0.07, 0.08, 0.09, 0.09, 0.10, 0.11, 0.11]
    t = ax.table(cellText=cells, colLabels=head, colWidths=widths, cellLoc="center", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        cell.set_linewidth(0.6)
        if c == 1 and r > 0:
            cell._loc = "left"
            cell.PAD = 0.03
        if r == 0:
            cell.set_facecolor("#e9e9e9")
            cell.set_text_props(fontweight="bold")
    fig.text(0.5, 1 - 0.3 / H, f"Simple recompression cycle: thermodynamic states "
             f"(x = {m['x']:.3f}, net shaft power {m['Net power, shaft (MW)']:.2f} MW)",
             fontsize=12.5, fontweight="bold", ha="center", va="center")
    fig.text(0.02, 0.25 / H, "CO2 properties from CoolProp. Exergy: dead state 25 °C and 101.325 kPa. Mass flow: "
             "(1 − x) through states 1–3, x through state 4, the full flow elsewhere.", fontsize=9, va="center")
    fig.savefig(folder / "simple_state_table.png", dpi=220)
    plt.close(fig)


def _pipe(ax, pts, color, arrows=(), lw=2.4):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=1)
    for seg, frac in arrows:
        (x0, y0), (x1, y1) = pts[seg], pts[seg + 1]
        L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        xm, ym = x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
        ax.annotate("", xy=(xm + 0.12 * ux, ym + 0.12 * uy), xytext=(xm - 0.12 * ux, ym - 0.12 * uy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=15), zorder=2)


def plot_schematic(res, hs, m, folder):
    """Process schematic of the simple recompression cycle with the EAF gas side. State numbers follow
    recompression_adv(); line colours follow the temperature of the stream (blue < 120 C, orange < 400 C,
    red above)."""
    from matplotlib.patches import Rectangle, Polygon, Circle
    st = res["states"]
    x = m["x"]

    def col(k):
        T = PropsSI("T", "H", float(st[k]["h"]), "P", float(st[k]["p"]), "CO2") - K if k in st else 200.0
        return "#2a78d6" if T < 120 else ("#eb6834" if T < 400 else "#e34948")

    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    ax.set_xlim(0.2, 20.6)
    ax.set_ylim(0.5, 11.3)
    ax.set_aspect("equal")
    ax.axis("off")
    GAS_C = "#6b6b6b"

    def hx(x0, x1, yc, up, low, name, lab_xy):
        ax.add_patch(Rectangle((x0, yc - 0.7), x1 - x0, 1.4, fc="#f2f2f2", ec="black", lw=1.5, zorder=3))
        ax.plot([x0, x1], [yc + 0.3] * 2, color=up, lw=2.4, zorder=4)
        ax.plot([x0, x1], [yc - 0.3] * 2, color=low, lw=2.4, zorder=4)
        ax.text(*lab_xy, name, ha="center", va="bottom", fontsize=13, fontweight="bold")

    def machine(pts, name, lab_xy):
        ax.add_patch(Polygon(pts, closed=True, fc="#f2f2f2", ec="black", lw=1.5, zorder=3))
        ax.text(*lab_xy, name, ha="center", va="center", fontsize=12, fontweight="bold")

    def num(k, xy):
        ax.text(*xy, str(k), ha="center", va="center", fontsize=11, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec="black", lw=1.2))

    def dot(xy):
        ax.plot(*xy, "o", color="black", ms=7, zorder=6)

    # recuperators (cold stream on the upper line, hot stream on the lower line)
    hx(3.9, 7.1, 7.4, col(3), col(9), "LTR", (5.5, 8.2))
    hx(9.7, 12.9, 7.4, col(6), col(8), "HTR", (11.3, 8.2))
    # gas heater: CO2 on the lower line, gas on the upper line
    ax.add_patch(Rectangle((13.9, 8.65), 2.6, 1.8, fc="#f2f2f2", ec="black", lw=1.5, zorder=3))
    ax.plot([13.9, 16.5], [9.1, 9.1], color=col(7), lw=2.4, zorder=4)
    ax.text(15.2, 10.6, "Gas heater", ha="center", va="bottom", fontsize=13, fontweight="bold")
    # cooler
    ax.add_patch(Rectangle((3.0, 4.2), 0.8, 1.0, fc="#f2f2f2", ec="black", lw=1.5, zorder=3))
    ax.text(2.85, 4.7, "Cooler", ha="right", va="center", fontsize=13, fontweight="bold")
    # turbomachines on one shaft (compressors: inlet on the tall side; turbine: inlet on the short side)
    machine([(3.0, 4.1), (3.0, 2.3), (1.3, 2.8), (1.3, 3.6)], "", (2.15, 3.2))
    machine([(6.1, 4.1), (6.1, 2.3), (7.7, 2.8), (7.7, 3.6)], "", (6.9, 3.2))
    machine([(14.7, 4.1), (14.7, 2.3), (16.3, 2.8), (16.3, 3.6)], "", (15.5, 3.2))
    ax.plot([2.15, 17.6], [1.7, 1.7], color="black", lw=4, solid_capstyle="butt", zorder=2)
    for xs in (2.15, 6.9, 15.5):
        ax.plot([xs, xs], [1.7, 2.55], color="black", lw=4, solid_capstyle="butt", zorder=2)
    ax.add_patch(Circle((18.0, 1.7), 0.4, fc="#e6e6e6", ec="black", lw=1.5, zorder=3))
    ax.text(18.0, 1.7, "G", ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)
    for xs, name in ((2.15, "Main\ncompressor"), (6.9, "Recompressor"), (15.5, "Turbine"), (18.0, "Generator")):
        ax.text(xs, 1.05, name, ha="center", va="center", fontsize=12, fontweight="bold")

    # CO2 pipes
    _pipe(ax, [(3.9, 7.1), (3.4, 7.1), (3.4, 6.0)], col(10), [(1, 0.5)])
    _pipe(ax, [(3.4, 6.0), (3.4, 5.2)], col(10), [(0, 0.5)])
    _pipe(ax, [(3.4, 4.2), (3.4, 3.2), (3.0, 3.2)], col(1), [(0, 0.5)])
    _pipe(ax, [(3.4, 6.0), (5.6, 6.0), (5.6, 3.2), (6.1, 3.2)], col(10), [(0, 0.55), (1, 0.5)])
    _pipe(ax, [(1.3, 3.2), (0.9, 3.2), (0.9, 7.7), (3.9, 7.7)], col(2), [(1, 0.5), (2, 0.6)])
    _pipe(ax, [(7.1, 7.7), (8.4, 7.7)], col(3), [(0, 0.55)])
    _pipe(ax, [(8.4, 7.7), (9.7, 7.7)], col(5), [(0, 0.7)])
    _pipe(ax, [(7.7, 3.2), (8.4, 3.2), (8.4, 7.7)], col(4), [(1, 0.5)])
    _pipe(ax, [(9.7, 7.1), (8.55, 7.1)], col(9), [(0, 0.5)])
    _pipe(ax, [(8.25, 7.1), (7.1, 7.1)], col(9), [(0, 0.5)])
    ax.add_patch(plt.matplotlib.patches.Arc((8.4, 7.1), 0.3, 0.3, theta1=0, theta2=180, color=col(9), lw=2.4, zorder=1))
    _pipe(ax, [(12.9, 7.7), (13.5, 7.7), (13.5, 9.1), (13.9, 9.1)], col(6), [(1, 0.5)])
    _pipe(ax, [(16.5, 9.1), (17.1, 9.1), (17.1, 3.2), (16.3, 3.2)], col(7), [(1, 0.5)])
    _pipe(ax, [(14.7, 3.2), (14.1, 3.2), (14.1, 7.1), (12.9, 7.1)], col(8), [(1, 0.5)])
    _pipe(ax, [(18.0, 10.0), (12.4, 10.0)], GAS_C, [(0, 0.12), (0, 0.9)])
    ax.text(18.1, 10.0, f"EAF off-gas\n{TIN - K:.1f} °C, {MG:.1f} kg/s", ha="left", va="center", fontsize=11, color=GAS_C)
    ax.text(12.3, 10.0, f"To stack\n{m['Gas stack temperature (C)']:.0f} °C", ha="right", va="center", fontsize=11,
            color=GAS_C)
    for xy in ((3.4, 6.0), (8.4, 7.7)):
        dot(xy)
    ax.text(4.5, 6.15, "x", fontsize=12, ha="center", va="bottom", style="italic")
    ax.text(3.55, 5.6, "1 − x", fontsize=12, ha="left", va="center", style="italic")
    # state numbers
    for k, xy in {1: (3.4, 3.75), 2: (0.9, 5.0), 3: (7.75, 7.7), 4: (8.4, 5.2), 5: (9.2, 7.7), 6: (13.5, 8.4),
                  7: (17.1, 6.0), 8: (14.1, 5.2), 9: (7.75, 7.1), 10: (3.4, 6.6)}.items():
        num(k, xy)
    ax.text(0.3, 11.1, f"Simple recompression cycle (x = {x:.3f}, one gas heater)", fontsize=15, fontweight="bold",
            ha="left", va="top")
    ax.legend(handles=[Line2D([0], [0], color="#2a78d6", lw=3, label="CO2 below 120 °C"),
                       Line2D([0], [0], color="#eb6834", lw=3, label="CO2 120–400 °C"),
                       Line2D([0], [0], color="#e34948", lw=3, label="CO2 above 400 °C"),
                       Line2D([0], [0], color=GAS_C, lw=3, label="off-gas")],
              loc="center", bbox_to_anchor=(0.535, 0.40), frameon=False, fontsize=11, ncol=1)
    fig.savefig(folder / "simple_schematic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def do_states():
    """Schematic and state table of the simple recompression cycle at its saved optimum."""
    folder = DIRS["simple"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "simple_optimum.json"
    if not path.exists():
        raise SystemExit("Run the 'simple' step first (it writes simple_optimum.json).")
    z = json.load(open(path))["z"]
    m, res, hs = metrics("simple", z)
    tab = state_table(res, hs, m["x"])
    st = res["states"]
    print("State check: T7 =", f"{tab.loc[tab['State'] == 7, 'T (C)'].iloc[0]:.1f}" if 7 in set(tab["State"]) else "n/a",
          f"C (expected {T7 - K:.0f}); p2 = {tab.loc[tab['State'] == 2, 'p (MPa)'].iloc[0]:.2f} MPa "
          f"(expected {m['p_max (MPa)']:.1f}); states found: {sorted(st.keys())}")
    tab.round(4).to_csv(folder / "simple_state_table.csv", index=False, encoding="utf-8-sig")
    plot_state_table(tab, m, folder)
    plot_schematic(res, hs, m, folder)
    print(f"Saved in '{folder}': simple_schematic.png, simple_state_table.png, simple_state_table.csv")


# ---- T-s diagram of the simple recompression cycle -------------------------------------------------------
TS_OFFSET = {1: (-11, -7), 2: (-11, 5), 3: (-11, -8), 4: (10, 7), 5: (-12, 8), 6: (-10, 6), 7: (7, 7),
             8: (9, 4), 9: (9, -8), 10: (9, -8)}           # label offsets in points, chosen for the design point


def _ts_path(st, a, b, mode, n=30):
    """(T in C, s in kJ/kg K) of the process a -> b in n steps: enthalpy linear, pressure linear (heat
    exchangers) or geometric (compressors and turbine)."""
    f = np.linspace(0.0, 1.0, n + 1)
    h = st[a]["h"] + f * (st[b]["h"] - st[a]["h"])
    p = st[a]["p"] * (st[b]["p"] / st[a]["p"]) ** f if mode == "geo" else st[a]["p"] + f * (st[b]["p"] - st[a]["p"])
    T = np.array([PropsSI("T", "H", hh, "P", pp, "CO2") for hh, pp in zip(h, p)]) - K
    s = np.array([PropsSI("S", "H", hh, "P", pp, "CO2") for hh, pp in zip(h, p)]) / 1e3
    return T, s


def plot_ts(res, m, folder):
    """T-s diagram: saturation dome, the 7.7 and 25 MPa isobars, the cycle processes and the state numbers."""
    st = res["states"]
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    Tc = PropsSI("Tcrit", "CO2")
    Tsat = np.linspace(273.15, Tc - 0.02, 80)
    for q in (0, 1):
        ax.plot([PropsSI("S", "T", t, "Q", q, "CO2") / 1e3 for t in Tsat], Tsat - K, color="#999999", lw=1.0)
    ax.plot(PropsSI("S", "T", Tc, "Q", 0, "CO2") / 1e3, Tc - K, marker="*", ms=12, color="#444444", ls="none",
            zorder=4)
    Tg = np.linspace(273.15, 873.15, 60)
    for pp in (P1, P_MAX):
        sg = [PropsSI("S", "T", t, "P", pp, "CO2") / 1e3 for t in Tg]
        ax.plot(sg, Tg - K, color="#bbbbbb", lw=0.9, ls=":")
        ax.annotate(f"{pp / 1e6:g} MPa", (sg[-1], Tg[-1] - K), xytext=(0, 4), textcoords="offset points",
                    fontsize=8, color="#777777", ha="center")
    heater, rec_c, cool = "#e34948", ORANGE, BLUE
    proc = [((6, 7), "lin", heater), ((2, 3), "lin", rec_c), ((5, 6), "lin", rec_c),
            ((8, 9), "lin", cool), ((9, 10), "lin", cool), ((10, 1), "lin", cool),
            ((1, 2), "geo", "black"), ((10, 4), "geo", "black"), ((7, 8), "geo", "black")]
    for (a, b), mode, col in proc:
        T, s = _ts_path(st, a, b, mode)
        ax.plot(s, T, color=col, lw=2.0)
    for a in (3, 4):                                        # mixing of the two streams at state 5
        T, s = _ts_path(st, a, 5, "lin", n=6)
        ax.plot(s, T, color=GREY, lw=1.4, ls="--")
    pts = {}
    for k in range(1, 11):
        h, pk = float(st[k]["h"]), float(st[k]["p"])
        pts[k] = (PropsSI("S", "H", h, "P", pk, "CO2") / 1e3, PropsSI("T", "H", h, "P", pk, "CO2") - K)
        ax.plot(*pts[k], marker="o", ms=4.5, color="black", zorder=5)
        dx, dy = TS_OFFSET[k]
        ax.annotate(str(k), pts[k], xytext=(dx, dy), textcoords="offset points", ha="center", va="center",
                    fontsize=9, zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="black", lw=0.9))
    ax.set_xlabel("Specific entropy s (kJ/kg K)")
    ax.set_ylabel("Temperature (C)")
    ax.set_ylim(0, 620)
    ax.grid(True, color="#dddddd", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], color=heater, lw=2, label="gas heater (CO2 side)"),
                       Line2D([0], [0], color=rec_c, lw=2, label="recuperators, cold side"),
                       Line2D([0], [0], color=cool, lw=2, label="recuperators, hot side and cooler"),
                       Line2D([0], [0], color="black", lw=2, label="compression and expansion"),
                       Line2D([0], [0], color=GREY, lw=1.4, ls="--", label="mixing at state 5"),
                       Line2D([0], [0], marker="*", ms=12, color="#444444", ls="none", label="critical point")],
              frameon=False, fontsize=9, loc="upper left")
    ax.set_title(f"Simple recompression cycle: T-s diagram (x = {m['x']:.3f}, 25 MPa maximum pressure)", fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "simple_ts_diagram.png", dpi=200)
    plt.close(fig)


def do_ts():
    folder = DIRS["simple"]
    path = folder / "simple_optimum.json"
    if not path.exists():
        raise SystemExit("Run the 'simple' step first (it writes simple_optimum.json).")
    m, res, hs = metrics("simple", json.load(open(path))["z"])
    plot_ts(res, m, folder)
    print(f"Saved in '{folder}': simple_ts_diagram.png")


def do_replot():
    """Redraw the x-scan, pinch and pressure-ratio figures from the saved CSV files (no new optimisation);
    the four efficiencies are recomputed for each saved point when the CSV files do not have them yet."""
    folder = DIRS["simple"]
    d = json.load(open(folder / "simple_optimum.json"))
    m = metrics("simple", d["z"])[0]                      # design point, recomputed (includes the four efficiencies)
    sc = pd.read_csv(folder / "simple_x_scan.csv")
    prs = pd.read_csv(folder / "simple_pr_scan.csv")
    if not all(c in sc.columns for c in EFF_COLS):
        vals = []
        for x in sc["x"]:
            try:
                mm = metrics("simple", [float(x)])[0]
                vals.append([mm[c] for c in EFF_COLS])
            except Exception:
                vals.append([np.nan] * 4)
        sc[EFF_COLS] = vals
        sc.round(5).to_csv(folder / "simple_x_scan.csv", index=False, encoding="utf-8-sig")
    if not all(c in prs.columns for c in EFF_COLS):
        vals = []
        for pr in prs["pr"]:
            try:
                mm = metrics("simple", d["z"], float(pr))[0]
                vals.append([mm[c] for c in EFF_COLS])
            except Exception:
                vals.append([np.nan] * 4)
        prs[EFF_COLS] = vals
        prs.round(5).to_csv(folder / "simple_pr_scan.csv", index=False, encoding="utf-8-sig")
    plot_scan("simple", sc, m, folder)
    plot_pinch("simple", sc, m, folder)
    plot_pr("simple", prs, d["z"], folder)
    cols = ["Heat taken from the gas (MW)", "Net power, shaft (MW)"] + EFF_COLS
    print("At the design point:")
    for c in cols:
        print(f"  {c}: {m[c]:.2f}" if c in m else f"  {c}: (run the 'simple' step again to store it)")
    print(f"Saved in '{folder}': simple_x_scan.png, simple_x_scan_pinch.png, simple_pr_scan.png")


# ---- sensitivity to the gas inlet temperature (flow fixed), x and T7 re-optimised --------------------------
T7_CAP = 700.0            # C, highest turbine inlet temperature allowed: the STEP 10 MWe sCO2 turbine targets >= 700 C
GAS_T_GRID = [600.0, 650.0, 684.54, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0]     # C (684.54 = design gas)


def do_gas():
    """For each gas inlet temperature (mass flow and composition fixed): find x and the turbine inlet temperature T7
    (limited by T7_CAP and by the heater pinch) that give the maximum net power, then report the four efficiencies.
    The module values TIN and T7 are changed temporarily and restored at the end."""
    folder = DIRS["simple"]
    folder.mkdir(parents=True, exist_ok=True)
    tin0, t70 = globals()["TIN"], globals()["T7"]
    rows, prev = [], None
    try:
        for Tc in GAS_T_GRID:
            globals()["TIN"] = Tc + K
            hi7 = min(T7_CAP, Tc - DTP - 1.0)
            if hi7 < 450.0:
                print(f"  gas {Tc:.0f} C: too cold for a turbine inlet of 450 C with a {DTP:g} K pinch; skipped")
                continue

            def f(z):
                globals()["T7"] = z[1] + K
                return objective("simple", [z[0]])
            starts = [[0.27, hi7]] + ([prev] if prev is not None else [])
            z, fv = search(f, [0.0, 450.0], [0.6, hi7], starts=starts, n_lhs=80, n_nm=2, maxfev=150)
            globals()["T7"] = z[1] + K
            m, res, hs = metrics("simple", [z[0]])
            ok = violation(res) < 1e-6
            prev = [float(z[0]), float(z[1])]
            row = {"Gas inlet T (C)": Tc, "x": float(z[0]), "T7 (C)": float(z[1]), "T7 at the limit": bool(z[1] > hi7 - 0.5),
                   "Net power (MW)": m["Net power, shaft (MW)"], "Heat taken from the gas (MW)": m["Heat taken from the gas (MW)"],
                   "Share of the available heat (%)": m["Share of the available heat (%)"],
                   "Gas stack temperature (C)": m["Gas stack temperature (C)"],
                   **{c: m[c] for c in EFF_COLS}, "Recuperator limit met": ok}
            rows.append(row)
            print(f"  gas {Tc:6.1f} C: x = {z[0]:.3f}, T7 = {z[1]:.0f} C, net power {row['Net power (MW)']:.2f} MW, "
                  f"energy eff. {m[EFF_COLS[0]]:.1f} / {m[EFF_COLS[1]]:.1f} %, exergy eff. {m[EFF_COLS[2]]:.1f} / "
                  f"{m[EFF_COLS[3]]:.1f} %  (captured / available)", flush=True)
            pd.DataFrame(rows).round(4).to_csv(folder / "simple_gas_temperature.csv", index=False, encoding="utf-8-sig")
    finally:
        globals()["TIN"], globals()["T7"] = tin0, t70
    sc = pd.DataFrame(rows)
    plot_gas(sc, folder)
    print(f"Saved in '{folder}': simple_gas_temperature.csv, simple_gas_temperature.png")


def plot_gas(sc, folder):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    x = sc["Gas inlet T (C)"]
    ax, ax2 = axes[0], axes[0].twinx()
    ax.plot(x, sc["Net power (MW)"], color=BLUE, lw=2, marker="o", ms=5, label="net power (left axis)")
    ax2.plot(x, sc["T7 (C)"], color=ORANGE, lw=1.8, ls="--", marker="s", ms=6, mfc="none",
             label="turbine inlet T7 (right axis)")
    ax2.axhline(T7_CAP, color=GREY, lw=1, ls=":")
    ax.set_ylabel("Net power, shaft (MW)")
    ax2.set_ylabel("Turbine inlet temperature T7 (C)")
    ax.legend(handles=ax.get_lines() + ax2.get_lines()[:1], frameon=False, fontsize=8, loc="lower right")
    axes[1].plot(x, sc[EFF_COLS[0]], color=AQUA, lw=2, marker="^", ms=5, label="energy efficiency")
    axes[1].plot(x, sc[EFF_COLS[2]], color=EXERGY_C, lw=1.6, ls="--", marker="s", ms=7, mfc="none", label="exergy efficiency")
    axes[1].set_title("Input: heat and exergy captured by the cycle", fontsize=10)
    axes[2].plot(x, sc[EFF_COLS[1]], color=AQUA, lw=2, marker="^", ms=5, label="energy efficiency")
    axes[2].plot(x, sc[EFF_COLS[3]], color=EXERGY_C, lw=1.6, ls="--", marker="s", ms=7, mfc="none", label="exergy efficiency")
    axes[2].set_title(f"Input: heat and exergy available down to {TMIN - K:.0f} C", fontsize=10)
    for a in axes[1:]:
        a.set_ylabel("Efficiency (%)")
        a.legend(frameon=False, fontsize=9, loc="best")
    for a in (ax, axes[1], axes[2]):
        a.axvline(TIN - K, color=GREY, lw=1, ls="--")                       # the design gas
        a.set_xlabel("Gas inlet temperature (C)")
        a.grid(True, color="#dddddd", lw=0.6)
    for a in (ax, ax2, axes[1], axes[2]):
        a.spines["top"].set_visible(False)
    fig.suptitle(f"Simple recompression: effect of the gas inlet temperature (flow {MG:.1f} kg/s; x and T7 re-optimised, "
                 f"T7 <= {T7_CAP:.0f} C; dashed line = design gas)", fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / "simple_gas_temperature.png", dpi=170)
    plt.close(fig)


# ---- one-parameter sweeps at the design gas with x re-optimised: T7 and the pressure ratio -----------------
STACK_C = "#c0392b"


def plot_three(sc, xcol, xlabel, folder, fname, title, design, design_label, top_axis=None,
               leg3=("center left", (0.04, 0.63))):
    """Three panels against one parameter: net power (+ heat taken), efficiencies on the captured basis,
    efficiencies on the available basis (+ gas stack temperature on the right axis)."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    x = sc[xcol]
    ax, ax2 = axes[0], axes[0].twinx()
    ax.plot(x, sc["Net power (MW)"], color=BLUE, lw=2, marker="o", ms=5, label="net power (left axis)")
    ax2.plot(x, sc["Heat taken from the gas (MW)"], color=ORANGE, lw=1.8, ls="--", marker="s", ms=6, mfc="none",
             label="heat taken from the gas (right axis)")
    ax.set_ylabel("Net power, shaft (MW)")
    ax2.set_ylabel("Heat taken from the gas (MW)")
    ax.legend(handles=ax.get_lines() + ax2.get_lines()[:1], frameon=False, fontsize=8, loc="best")
    axes[1].plot(x, sc[EFF_COLS[0]], color=AQUA, lw=2, marker="^", ms=5, label="energy efficiency")
    axes[1].plot(x, sc[EFF_COLS[2]], color=EXERGY_C, lw=1.6, ls="--", marker="s", ms=7, mfc="none", label="exergy efficiency")
    axes[1].set_title("Input: heat and exergy captured by the cycle", fontsize=10)
    axes[1].set_ylabel("Efficiency (%)")
    axes[1].legend(frameon=False, fontsize=9, loc="best")
    a3, a3b = axes[2], axes[2].twinx()
    a3.plot(x, sc[EFF_COLS[1]], color=AQUA, lw=2, marker="^", ms=5, label="energy eff. (left)")
    a3.plot(x, sc[EFF_COLS[3]], color=EXERGY_C, lw=1.6, ls="--", marker="s", ms=7, mfc="none",
            label="exergy eff. (left)")
    a3b.plot(x, sc["Gas stack temperature (C)"], color=STACK_C, lw=1.8, ls=":", marker="d", ms=6,
             label="stack T (right)")
    a3.set_title(f"Input: heat and exergy available down to {TMIN - K:.0f} C", fontsize=10)
    a3.set_ylabel("Efficiency (%)")
    a3b.set_ylabel("Gas stack temperature (C)")
    a3.legend(handles=a3.get_lines() + a3b.get_lines(), frameon=True, framealpha=0.9, edgecolor="none", fontsize=7,
              handlelength=1.6, labelspacing=0.3, borderpad=0.3, loc=leg3[0], bbox_to_anchor=leg3[1])   # in the free area
    for a in (ax, axes[1], a3):
        a.axvline(design, color=GREY, lw=1, ls="--")
        a.set_xlabel(xlabel)
        a.grid(True, color="#dddddd", lw=0.6)
        a.spines["top"].set_visible(False)
        if top_axis is not None:
            t = a.secondary_xaxis("top", functions=top_axis[0])
            t.set_xlabel(top_axis[1])
    for a in (ax2, a3b):
        a.spines["top"].set_visible(False)
    fig.suptitle(f"{title}; dashed line = {design_label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / fname, dpi=170)
    plt.close(fig)


def _ua_segments(dT, Q, f=None):
    """UA = sum of (duty of each equal-duty segment) / (mean temperature difference of the segment), in W/K."""
    dT = np.asarray(dT, float)
    if dT.min() <= 0.0:
        return float("inf")
    w = np.full(len(dT) - 1, 1.0 / (len(dT) - 1)) if f is None else np.diff(np.asarray(f, float))
    return float(np.sum(Q * w / (0.5 * (dT[:-1] + dT[1:]))))


def size_indicators(res, hs, x):
    """Indicators of the hardware size of the simple recompression cycle at one design point: UA of the HTR, LTR
    and gas heater (20 equal-duty segments, the same method as the profile figures), cooler duty and the powers of
    the turbine and the two compressors. All from the state table (enthalpies, flows (1-x) m, x m and m)."""
    st, m = res["states"], hs["m_co2"]
    h = lambda i: st[i]["h"]
    f, th, tc = exchanger_profile(res, hot=(8, 9), cold=(5, 6))
    q_htr = m * (h(8) - h(9))
    ua_htr = _ua_segments(th - tc, q_htr)
    f, th, tc = exchanger_profile(res, hot=(9, 10), cold=(2, 3))
    q_ltr = m * (h(9) - h(10))
    ua_ltr = _ua_segments(th - tc, q_ltr)
    ua_heat = _ua_segments(np.asarray(hs["T_gas"]) - np.asarray(hs["T_co2"]), hs["Q"], hs["f"])
    w_t, w_mc, w_rc = m * (h(7) - h(8)), (1 - x) * m * (h(2) - h(1)), x * m * (h(4) - h(10))
    q_cool = (1 - x) * m * (h(10) - h(1))
    w_net = w_t - w_mc - w_rc
    return {"CO2 mass flow (kg/s)": m, "UA HTR (MW/K)": ua_htr / 1e6, "UA LTR (MW/K)": ua_ltr / 1e6,
            "UA heater (MW/K)": ua_heat / 1e6, "UA total (MW/K)": (ua_htr + ua_ltr + ua_heat) / 1e6,
            "HTR duty (MW)": q_htr / 1e6, "LTR duty (MW)": q_ltr / 1e6,
            "Cooler duty (MW)": q_cool / 1e6, "Rejected heat per MW net (-)": q_cool / w_net,
            "Turbine power (MW)": w_t / 1e6, "Main compressor power (MW)": w_mc / 1e6,
            "Recompressor power (MW)": w_rc / 1e6, "Compressor share of turbine power (%)": (w_mc + w_rc) / w_t * 100,
            "Energy balance check (%)": (hs["Q"] - q_cool - w_net) / w_net * 100}


def _sweep_row(par, val, m, res, hs):
    return {par: val, "x": None, "Net power (MW)": m["Net power, shaft (MW)"],
            "Heat taken from the gas (MW)": m["Heat taken from the gas (MW)"],
            "Gas stack temperature (C)": m["Gas stack temperature (C)"],
            **{c: m[c] for c in EFF_COLS}, "HTR min dT (K)": m["HTR min dT (K)"], "LTR min dT (K)": m["LTR min dT (K)"],
            **size_indicators(res, hs, m["x"]),
            "Recuperator limit met": bool(violation(res) < 1e-6)}


def plot_size(sc, xcol, xlabel, folder, fname, title, design, design_label, top_axis=None):
    """Hardware size indicators against one parameter: CO2 flow and x, UA of the heat exchangers, cooler duty."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    x = sc[xcol]
    a, a2 = axes[0], axes[0].twinx()
    a.plot(x, sc["CO2 mass flow (kg/s)"], color=BLUE, lw=2, marker="o", ms=5, label="CO2 mass flow (left)")
    a2.plot(x, sc["x"], color=ORANGE, lw=1.8, ls="--", marker="s", ms=6, mfc="none", label="recompression x (right)")
    a.set_ylabel("CO2 mass flow (kg/s)")
    a2.set_ylabel("Recompression fraction x")
    a.legend(handles=a.get_lines() + a2.get_lines(), frameon=False, fontsize=8, loc="best")
    b = axes[1]
    b.plot(x, sc["UA heater (MW/K)"], color=ORANGE, lw=1.8, marker="s", ms=5, mfc="none", label="gas heater")
    b.plot(x, sc["UA HTR (MW/K)"], color=BLUE, lw=1.8, marker="o", ms=5, label="HTR")
    b.plot(x, sc["UA LTR (MW/K)"], color=AQUA, lw=1.8, marker="^", ms=5, label="LTR")
    b.plot(x, sc["UA total (MW/K)"], color="black", lw=1.6, ls="--", label="total")
    b.set_ylabel("UA (MW/K)")
    b.set_title("Heat exchanger size (UA)", fontsize=10)
    b.legend(frameon=False, fontsize=8, loc="best")
    c, c2 = axes[2], axes[2].twinx()
    c.plot(x, sc["Cooler duty (MW)"], color=BLUE, lw=2, marker="o", ms=5, label="cooler duty (left)")
    c2.plot(x, sc["Rejected heat per MW net (-)"], color=EXERGY_C, lw=1.8, ls="--", marker="d", ms=6, mfc="none",
            label="rejected heat per MW of net power (right)")
    c.set_ylabel("Cooler duty (MW)")
    c2.set_ylabel("Rejected heat per MW net power (MW/MW)")
    c.set_title("Heat rejection", fontsize=10)
    c.legend(handles=c.get_lines() + c2.get_lines(), frameon=False, fontsize=8, loc="best")
    for ax in (a, b, c):
        ax.axvline(design, color=GREY, lw=1, ls="--")
        ax.set_xlabel(xlabel)
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.spines["top"].set_visible(False)
        if top_axis is not None:
            t = ax.secondary_xaxis("top", functions=top_axis[0])
            t.set_xlabel(top_axis[1])
    for ax in (a2, c2):
        ax.spines["top"].set_visible(False)
    fig.suptitle(f"{title}; dashed line = {design_label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / fname, dpi=170)
    plt.close(fig)


def _print_row(label, r):
    print(f"  {label}: x = {r['x']:.3f}, net power {r['Net power (MW)']:.2f} MW, heat taken "
          f"{r['Heat taken from the gas (MW)']:.2f} MW, stack {r['Gas stack temperature (C)']:.0f} C, "
          f"energy eff. {r[EFF_COLS[0]]:.1f} / {r[EFF_COLS[1]]:.1f} %, exergy eff. {r[EFF_COLS[2]]:.1f} / "
          f"{r[EFF_COLS[3]]:.1f} %  (captured / available)"
          + ("" if r["Recuperator limit met"] else "  [recuperator below 10 K]"), flush=True)
    if "UA total (MW/K)" in r:
        print(f"      size: CO2 {r['CO2 mass flow (kg/s)']:.1f} kg/s, UA heater {r['UA heater (MW/K)']:.3f} + HTR "
              f"{r['UA HTR (MW/K)']:.3f} + LTR {r['UA LTR (MW/K)']:.3f} = {r['UA total (MW/K)']:.3f} MW/K, cooler "
              f"{r['Cooler duty (MW)']:.2f} MW ({r['Rejected heat per MW net (-)']:.2f} per MW net), turbine "
              f"{r['Turbine power (MW)']:.2f} MW, compressors {r['Main compressor power (MW)'] + r['Recompressor power (MW)']:.2f} MW; "
              f"energy balance {r['Energy balance check (%)']:.3f} %", flush=True)


def do_t7():
    """Sweep T7 from 400 C up to the heater-pinch limit (gas inlet - pinch - 1 K, at most T7_CAP); x is re-optimised
    for maximum net power at every T7. The module value T7 is restored at the end."""
    folder = DIRS["simple"]
    folder.mkdir(parents=True, exist_ok=True)
    x_des = float(json.load(open(folder / "simple_optimum.json"))["z"][0])
    t70 = globals()["T7"]
    top = min(T7_CAP, TIN - K - DTP - 1.0)
    grid = sorted(set([float(t) for t in np.arange(400.0, top, 25.0)] + [550.0, round(top, 1)]))
    rows, prev = [], x_des
    try:
        for t7 in grid:
            globals()["T7"] = t7 + K
            z, fv = search(lambda z: objective("simple", [z[0]]), [0.0], [0.6], starts=[[prev], [x_des]],
                           n_lhs=40, n_nm=2, maxfev=100)
            m, res, hs = metrics("simple", [z[0]])
            prev = float(z[0])
            r = _sweep_row("T7 (C)", t7, m, res, hs)
            r["x"] = prev
            rows.append(r)
            _print_row(f"T7 {t7:5.0f} C", r)
            pd.DataFrame(rows).round(4).to_csv(folder / "simple_t7_sweep.csv", index=False, encoding="utf-8-sig")
    finally:
        globals()["T7"] = t70
    sc = pd.DataFrame(rows)
    plot_three(sc, "T7 (C)", "Turbine inlet temperature T7 (C)", folder, "simple_t7_sweep.png",
               f"Simple recompression, effect of T7 (gas {TIN - K:.1f} C, {MG:.1f} kg/s, p_max {P_MAX / 1e6:.0f} MPa; "
               f"x re-optimised at each T7)", T7 - K, f"design T7 = {T7 - K:.0f} C")
    k = int(sc["Net power (MW)"].idxmax())
    print(f"Highest net power: {sc.loc[k, 'Net power (MW)']:.2f} MW at T7 = {sc.loc[k, 'T7 (C)']:.0f} C "
          f"(x = {sc.loc[k, 'x']:.3f}); at the top of the range: {sc['Net power (MW)'].iloc[-1]:.2f} MW")
    plot_size(sc, "T7 (C)", "Turbine inlet temperature T7 (C)", folder, "simple_t7_size.png",
              f"Simple recompression, hardware size against T7 (gas {TIN - K:.1f} C, p_max {P_MAX / 1e6:.0f} MPa; x re-optimised)",
              T7 - K, f"design T7 = {T7 - K:.0f} C")
    print(f"Saved in '{folder}': simple_t7_sweep.csv, simple_t7_sweep.png, simple_t7_size.png")


def do_prx():
    """Sweep the pressure ratio (maximum pressure = pr x p1) at the design T7; x is re-optimised for maximum net
    power at every pressure ratio. Same three panels as the T7 sweep."""
    folder = DIRS["simple"]
    folder.mkdir(parents=True, exist_ok=True)
    x_des = float(json.load(open(folder / "simple_optimum.json"))["z"][0])
    prs = np.unique(np.round(np.concatenate([np.linspace(2.0, 30e6 / P1, 12), [PR_MAX]]), 4))
    rows, prev = [], x_des
    for pr in prs:
        z, fv = search(lambda z: objective("simple", [z[0]], pr), [0.0], [0.6], starts=[[prev], [x_des]],
                       n_lhs=40, n_nm=2, maxfev=100)
        m, res, hs = metrics("simple", [z[0]], pr)
        prev = float(z[0])
        r = _sweep_row("pr", float(pr), m, res, hs)
        r["x"] = prev
        r["p_max (MPa)"] = pr * P1 / 1e6
        rows.append(r)
        _print_row(f"pr {pr:.3f} ({pr * P1 / 1e6:4.1f} MPa)", r)
        pd.DataFrame(rows).round(4).to_csv(folder / "simple_pr_sweep.csv", index=False, encoding="utf-8-sig")
    sc = pd.DataFrame(rows)
    plot_three(sc, "pr", "Pressure ratio (maximum pressure = pr x p1)", folder, "simple_pr_sweep.png",
               f"Simple recompression, effect of the pressure ratio (gas {TIN - K:.1f} C, {MG:.1f} kg/s, T7 {T7 - K:.0f} C; "
               f"x re-optimised at each pressure ratio)", PR_MAX, f"design pr = {PR_MAX:.3f} ({P_MAX / 1e6:.0f} MPa)",
               top_axis=((lambda r: r * P1 / 1e6, lambda p: p * 1e6 / P1), "Maximum pressure (MPa)"),
               leg3=("center right", (0.99, 0.60)))
    plot_size(sc, "pr", "Pressure ratio (maximum pressure = pr x p1)", folder, "simple_pr_size.png",
              f"Simple recompression, hardware size against the pressure ratio (gas {TIN - K:.1f} C, T7 {T7 - K:.0f} C; x re-optimised)",
              PR_MAX, f"design pr = {PR_MAX:.3f} ({P_MAX / 1e6:.0f} MPa)",
              top_axis=((lambda r: r * P1 / 1e6, lambda p: p * 1e6 / P1), "Maximum pressure (MPa)"))
    print(f"Saved in '{folder}': simple_pr_sweep.csv, simple_pr_sweep.png, simple_pr_size.png")


def do_replot3():
    """Redraw the T7 and pressure-ratio sweep figures from their saved CSV files (no new optimisation)."""
    folder = DIRS["simple"]
    sc = pd.read_csv(folder / "simple_t7_sweep.csv")
    plot_three(sc, "T7 (C)", "Turbine inlet temperature T7 (C)", folder, "simple_t7_sweep.png",
               f"Simple recompression, effect of T7 (gas {TIN - K:.1f} C, {MG:.1f} kg/s, p_max {P_MAX / 1e6:.0f} MPa; "
               f"x re-optimised at each T7)", T7 - K, f"design T7 = {T7 - K:.0f} C")
    sc = pd.read_csv(folder / "simple_pr_sweep.csv")
    plot_three(sc, "pr", "Pressure ratio (maximum pressure = pr x p1)", folder, "simple_pr_sweep.png",
               f"Simple recompression, effect of the pressure ratio (gas {TIN - K:.1f} C, {MG:.1f} kg/s, T7 {T7 - K:.0f} C; "
               f"x re-optimised at each pressure ratio)", PR_MAX, f"design pr = {PR_MAX:.3f} ({P_MAX / 1e6:.0f} MPa)",
               top_axis=((lambda r: r * P1 / 1e6, lambda p: p * 1e6 / P1), "Maximum pressure (MPa)"),
               leg3=("center right", (0.99, 0.60)))
    print(f"Saved in '{folder}': simple_t7_sweep.png, simple_pr_sweep.png")
    for fn, xc, xl, out, ttl, des, dl, ta in (
            ("simple_t7_sweep.csv", "T7 (C)", "Turbine inlet temperature T7 (C)", "simple_t7_size.png",
             f"Simple recompression, hardware size against T7 (gas {TIN - K:.1f} C, p_max {P_MAX / 1e6:.0f} MPa; x re-optimised)",
             T7 - K, f"design T7 = {T7 - K:.0f} C", None),
            ("simple_pr_sweep.csv", "pr", "Pressure ratio (maximum pressure = pr x p1)", "simple_pr_size.png",
             f"Simple recompression, hardware size against the pressure ratio (gas {TIN - K:.1f} C, T7 {T7 - K:.0f} C; x re-optimised)",
             PR_MAX, f"design pr = {PR_MAX:.3f} ({P_MAX / 1e6:.0f} MPa)",
             ((lambda r: r * P1 / 1e6, lambda p: p * 1e6 / P1), "Maximum pressure (MPa)"))):
        d = pd.read_csv(folder / fn)
        if "UA total (MW/K)" in d.columns:
            plot_size(d, xc, xl, folder, out, ttl, des, dl, top_axis=ta)
            print(f"Saved in '{folder}': {out}")
        else:
            print(f"  {fn} has no size columns yet: run the t7 / prx step again to add them")


# ---- compressor inlet temperature T1 and pressure p1 ---------------------------------------------------------
T1_GRID = [28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 45.0]          # C
P1_GRID = [7.5, 7.7, 7.9, 8.1, 8.3, 8.6, 9.0]                              # MPa, candidates for p1 (pr = P_MAX / p1)
P1_SHOW = (7.7, 8.3)                                                        # the two p1 values drawn as separate curves
BLACK = "black"


def do_t1():
    """Sweep the main compressor inlet temperature T1 at several compressor inlet pressures p1. The maximum pressure
    stays at P_MAX, so pr = P_MAX / p1 changes with p1; x is re-optimised for maximum net power at every (T1, p1).
    Besides the net power the file holds the density and cp at state 1, the margin to saturation (p1 - psat(T1) for
    T1 below the critical temperature) and the main compressor work. The best p1 of each T1 is marked in the plot.
    The module values T1K and P1 are restored at the end."""
    folder = DIRS["simple"]
    folder.mkdir(parents=True, exist_ok=True)
    x_des = float(json.load(open(folder / "simple_optimum.json"))["z"][0])
    t10, p10 = globals()["T1K"], globals()["P1"]
    rows, prev = [], {p: x_des for p in P1_GRID}
    try:
        for t1 in T1_GRID:
            globals()["T1K"] = t1 + K
            for p1m in P1_GRID:
                globals()["P1"] = p1m * 1e6
                pr = P_MAX / globals()["P1"]
                try:
                    z, fv = search(lambda z: objective("simple", [z[0]], pr), [0.0], [0.6],
                                   starts=[[prev[p1m]], [x_des]], n_lhs=40, n_nm=2, maxfev=100)
                    if fv >= 5.0:
                        raise ValueError("no feasible point")
                    m, res, hs = metrics("simple", [z[0]], pr)
                except Exception as e:
                    print(f"  T1 {t1:4.1f} C, p1 {p1m:.1f} MPa: failed ({e})", flush=True)
                    continue
                prev[p1m] = float(z[0])
                r = _sweep_row("T1 (C)", t1, m, res, hs)
                st = res["states"]
                r["x"], r["p1 (MPa)"], r["pr"] = prev[p1m], p1m, pr
                r["Density at state 1 (kg/m3)"] = PropsSI("D", "T", t1 + K, "P", globals()["P1"], "CO2")
                r["cp at state 1 (kJ/kg.K)"] = PropsSI("C", "T", t1 + K, "P", globals()["P1"], "CO2") / 1e3
                r["Margin to saturation (MPa)"] = (globals()["P1"] - PropsSI("P", "T", t1 + K, "Q", 0, "CO2")) / 1e6 \
                    if t1 + K < PropsSI("Tcrit", "CO2") else float("nan")
                r["Main compressor work (kJ/kg)"] = (st[2]["h"] - st[1]["h"]) / 1e3
                rows.append(r)
                print(f"  T1 {t1:4.1f} C, p1 {p1m:.1f} MPa (pr {pr:.3f}): net power {r['Net power (MW)']:.3f} MW, x = {r['x']:.3f}, "
                      f"density {r['Density at state 1 (kg/m3)']:.0f} kg/m3, main compressor {r['Main compressor power (MW)']:.2f} MW, "
                      f"cooler {r['Cooler duty (MW)']:.2f} MW"
                      + ("" if r["Recuperator limit met"] else "  [recuperator below 10 K]"), flush=True)
                pd.DataFrame(rows).round(4).to_csv(folder / "simple_t1_sweep.csv", index=False, encoding="utf-8-sig")
    finally:
        globals()["T1K"], globals()["P1"] = t10, p10
    sc = pd.DataFrame(rows)
    plot_t1(sc, folder)
    print_t1_summary(sc)


def print_t1_summary(sc):
    """Net power at each T1 for p1 = 7.7 and 8.3 MPa and for the best p1, relative to the design (32 C, 7.7 MPa)."""
    ref = sc[(sc["T1 (C)"] == 32.0) & (sc["p1 (MPa)"] == 7.7)]
    w0 = float(ref["Net power (MW)"].iloc[0]) if len(ref) else float("nan")
    print(f"\nNet power (MW) against T1; design (32 C, 7.7 MPa) = {w0:.3f} MW")
    print("  T1 (C)   " + "   ".join(f"p1 {p:.1f}" for p in P1_SHOW) + "   best p1 (MPa)")
    for t1, g in sc.groupby("T1 (C)"):
        cells = []
        for p in P1_SHOW:
            q = g[g["p1 (MPa)"] == p]
            cells.append(f"{float(q['Net power (MW)'].iloc[0]):7.3f}" if len(q) else "    n/a")
        b = g.loc[g["Net power (MW)"].idxmax()]
        print(f"  {t1:5.1f}   " + "   ".join(cells) + f"   {b['Net power (MW)']:.3f} at {b['p1 (MPa)']:.1f}")


def plot_t1(sc, folder):
    """Four panels against T1: net power, density at state 1, main compressor power, best p1."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    tc = PropsSI("Tcrit", "CO2") - K
    cols = {P1_SHOW[0]: BLUE, P1_SHOW[1]: ORANGE}
    best = sc.loc[sc.groupby("T1 (C)")["Net power (MW)"].idxmax()]

    def lines(a, col, ylabel, with_best=False):
        for p, c in cols.items():
            d = sc[sc["p1 (MPa)"] == p]
            a.plot(d["T1 (C)"], d[col], color=c, lw=2, marker="o", ms=5, label=f"p1 = {p:.1f} MPa")
        if with_best:
            a.plot(best["T1 (C)"], best[col], color=BLACK, lw=1.4, ls="--", label="best p1 at each T1")
        a.axvline(tc, color=BLACK, lw=1.2, ls="--")
        a.axvline(32.0, color=GREY, lw=1, ls=":")
        a.set_ylabel(ylabel)
        a.set_xlabel("Compressor inlet temperature T1 (C)")
        a.grid(True, color="#dddddd", lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    lines(axes[0, 0], "Net power (MW)", "Net power, shaft (MW)", with_best=True)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[0, 0].set_title("Net power (x re-optimised)", fontsize=10)
    lines(axes[0, 1], "Density at state 1 (kg/m3)", "Density at state 1 (kg/m3)")
    axes[0, 1].set_title("Compressor inlet density", fontsize=10)
    lines(axes[1, 0], "Main compressor power (MW)", "Main compressor power (MW)")
    axes[1, 0].set_title("Main compressor power", fontsize=10)
    a = axes[1, 1]
    a.plot(best["T1 (C)"], best["p1 (MPa)"], color=BLACK, lw=1.4, ls="--", marker="o", ms=6, label="best p1 (net power)")
    for p, c in cols.items():
        a.axhline(p, color=c, lw=1, ls=":")
    a.axvline(tc, color=BLACK, lw=1.2, ls="--")
    a.axvline(32.0, color=GREY, lw=1, ls=":")
    a.set_xlabel("Compressor inlet temperature T1 (C)")
    a.set_ylabel("p1 giving the highest net power (MPa)")
    a.set_title(f"Best p1 (candidates {P1_GRID[0]:.1f} to {P1_GRID[-1]:.1f} MPa)", fontsize=10)
    a.grid(True, color="#dddddd", lw=0.6)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle(f"Simple recompression, effect of the compressor inlet temperature (gas {TIN - K:.1f} C, T7 {T7 - K:.0f} C, "
                 f"p_max {P_MAX / 1e6:.0f} MPa; black dashed = critical temperature {tc:.2f} C, grey dotted = T1 design 32 C)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "simple_t1_sweep.png", dpi=170)
    plt.close(fig)
    print(f"Saved in '{folder}': simple_t1_sweep.csv, simple_t1_sweep.png")


def do_replot_t1():
    folder = DIRS["simple"]
    sc = pd.read_csv(folder / "simple_t1_sweep.csv")
    plot_t1(sc, folder)
    print_t1_summary(sc)


STEPS = (sys.argv[sys.argv.index("--steps") + 1].split(",") if "--steps" in sys.argv
         else ["simple", "cascade", "split", "compare"])
print(f"Fixed conditions: gas {TIN - K:.2f} C, {MG:.3f} kg/s; T7 {T7 - K:.0f} C; p1 {P1 / 1e6:.1f} MPa; "
      f"p_max {P_MAX / 1e6:.0f} MPa; heater pinch {DTP:g} K; stack >= {TMIN - K:.0f} C")
for s in STEPS:
    {"simple": lambda: do_cycle("simple"), "cascade": lambda: do_cycle("cascade"),
     "split": do_split, "compare": do_compare, "assumptions": do_assumptions, "states": do_states, "ts": do_ts,
     "replot": do_replot, "gas": do_gas, "t7": do_t7, "prx": do_prx, "replot3": do_replot3,
     "t1": do_t1, "replot_t1": do_replot_t1}[s]()
