"""T-s diagram of the split-flow cycle (self-contained, edit freely)."""
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# State label positions: state -> (right, up) offset in points from its marker.
# Edit a pair to move one label.
LABEL_OFFSETS = {
    1: (-2, -17),
    2: (13, 11),
    3: (-17, 12),
    5: (16, -11),
    6: (-17, 12),
    7: (14, 9),
    8: (16, 0),
    9: (16, -11),
    10: (15, -12),
    11: (16, -12),
}


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
    for q in (0, 1):
        ax.plot([PropsSI("S", "T", T, "Q", q, fluid) / 1e3 for T in T_sat],
                T_sat - 273.15, color="gray", lw=1)


def plot_ts_splitflow(res, fluid="CO2", save_path=None, label_offsets=None):
    """res: dictionary returned by splitflow_adv()."""
    offsets = dict(LABEL_OFFSETS)
    if label_offsets:
        offsets.update(label_offsets)
    st = res["states"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    _dome(ax, fluid)

    # (from state, to state, label, color, line style)
    processes = [
        (1, 2, "Main compressor", "tab:blue", "-"),
        (2, 3, "LTR (cold side)", "tab:orange", "-"),
        (2, 11, "Branch heater", "tab:brown", "-"),
        (3, 5, "Mixing", "black", "--"),
        (11, 5, "Mixing", "black", "--"),
        (5, 6, "HTR (cold side)", "gold", "-"),
        (6, 7, "Main heater", "tab:red", "-"),
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

    # State markers with numbered circles and thin leader lines
    for k, x in st.items():
        sx, Tx = x["s"] / 1e3, x["T"] - 273.15
        ax.plot(sx, Tx, "ko", ms=5, zorder=5)
        dx, dy = offsets.get(k, (14, 10))
        ax.annotate(str(k), (sx, Tx), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=0.9),
                    arrowprops=dict(arrowstyle="-", color="0.35", lw=0.7,
                                    shrinkA=0, shrinkB=2))

    ax.set_xlabel("Specific entropy (kJ/kg.K)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"T-s diagram, split-flow {fluid} cycle "
                 f"(y = {res['y']:.3f}, pr = {st[2]['p'] / st[1]['p']:.2f})")
    ax.grid(True)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax