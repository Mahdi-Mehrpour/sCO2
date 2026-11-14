"""Plotting helpers for the cycle."""
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from .components import state_hp


def _process_path(fluid, a, b, n=40):
    """Intermediate states between state a and state b.
    Enthalpy and pressure are both varied linearly, so this works for
    constant-pressure processes (heater, cooler, recuperator) and for
    compression/expansion (an approximation of the real path)."""
    h = np.linspace(a["h"], b["h"], n)
    p = np.linspace(a["p"], b["p"], n)
    pts = [state_hp(fluid, hi, pi) for hi, pi in zip(h, p)]
    s = np.array([x["s"] for x in pts]) / 1e3          # kJ/kg.K
    T = np.array([x["T"] for x in pts]) - 273.15       # deg C
    return s, T


def plot_ts(res, fluid="CO2", save_path=None):
    """T-s diagram of the recuperated Brayton cycle.
    res: dictionary returned by recuperated_brayton()."""
    st = res["states"]
    fig, ax = plt.subplots(figsize=(8, 6))

    # Saturation dome (liquid and vapour lines up to the critical point)
    T_triple = PropsSI("Ttriple", fluid)
    T_crit = PropsSI("Tcrit", fluid)
    T_sat = np.linspace(T_triple + 1, T_crit - 0.01, 200)
    s_liq = [PropsSI("S", "T", T, "Q", 0, fluid) / 1e3 for T in T_sat]
    s_vap = [PropsSI("S", "T", T, "Q", 1, fluid) / 1e3 for T in T_sat]
    ax.plot(s_liq, T_sat - 273.15, color="gray", lw=1)
    ax.plot(s_vap, T_sat - 273.15, color="gray", lw=1)

    # Cycle processes: (from state, to state, label, color)
    processes = [
        (1, 2, "Compressor", "tab:blue"),
        (2, 3, "Recuperator (cold side)", "tab:orange"),
        (3, 4, "Heater", "tab:red"),
        (4, 5, "Turbine", "tab:green"),
        (5, 6, "Recuperator (hot side)", "tab:purple"),
        (6, 1, "Cooler", "tab:cyan"),
    ]
    for i, j, label, color in processes:
        s, T = _process_path(fluid, st[i], st[j])
        ax.plot(s, T, color=color, lw=2, label=label)

    # State points with their numbers
    for k, x in st.items():
        ax.plot(x["s"] / 1e3, x["T"] - 273.15, "ko", ms=5)
        ax.annotate(str(k), (x["s"] / 1e3, x["T"] - 273.15),
                    textcoords="offset points", xytext=(6, 6), fontsize=11)

    ax.set_xlabel("Specific entropy (kJ/kg.K)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"T-s diagram, recuperated {fluid} Brayton cycle")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax
