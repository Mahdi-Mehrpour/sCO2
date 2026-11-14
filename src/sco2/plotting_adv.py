"""T-s diagram for the advanced cycle (self-contained, edit freely)."""
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI


def _path(fluid, a, b, n=40):
    """Intermediate states between a and b (h and p varied linearly)."""
    h = np.linspace(a["h"], b["h"], n)
    p = np.linspace(a["p"], b["p"], n)
    s = [PropsSI("S", "H", hi, "P", pi, fluid) / 1e3 for hi, pi in zip(h, p)]
    T = [PropsSI("T", "H", hi, "P", pi, fluid) - 273.15 for hi, pi in zip(h, p)]
    return s, T


def _dome(ax, fluid):
    T_sat = np.linspace(PropsSI("Ttriple", fluid) + 1,
                        PropsSI("Tcrit", fluid) - 0.01, 200)
    ax.plot([PropsSI("S", "T", T, "Q", 0, fluid) / 1e3 for T in T_sat],
            T_sat - 273.15, color="gray", lw=1)
    ax.plot([PropsSI("S", "T", T, "Q", 1, fluid) / 1e3 for T in T_sat],
            T_sat - 273.15, color="gray", lw=1)


def plot_ts_adv(res, fluid="CO2", save_path=None):
    """res: dictionary returned by recompression_adv()."""
    st = res["states"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    _dome(ax, fluid)

    # (from state, to state, label, color, line style)
    processes = [
        (1, 2, "Main compressor", "tab:blue", "-"),
        (2, 3, "LTR (cold side)", "tab:orange", "-"),
        (10, 4, "Recompressor", "tab:brown", "-"),
        (3, 5, "Mixing", "black", "--"),
        (4, 5, "Mixing", "black", "--"),
        (5, 6, "HTR (cold side)", "gold", "-"),
        (6, 7, "Heater", "tab:red", "-"),
        (7, 8, "Turbine", "tab:green", "-"),
        (8, 9, "HTR (hot side)", "tab:purple", "-"),
        (9, 10, "LTR (hot side)", "tab:pink", "-"),
        (10, 1, "Cooler", "tab:cyan", "-"),
    ]
    used = set()
    for i, j, label, color, ls in processes:
        s, T = _path(fluid, st[i], st[j])
        ax.plot(s, T, color=color, lw=2, ls=ls,
                label=label if label not in used else None)
        used.add(label)

    for k, x in st.items():
        ax.plot(x["s"] / 1e3, x["T"] - 273.15, "ko", ms=5)
        ax.annotate(str(k), (x["s"] / 1e3, x["T"] - 273.15),
                    textcoords="offset points", xytext=(6, 6), fontsize=11)

    ax.set_xlabel("Specific entropy (kJ/kg.K)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"T-s diagram, advanced {fluid} cycle (x = {res['x']})")
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax
