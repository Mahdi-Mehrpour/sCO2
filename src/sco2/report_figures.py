"""Report figures of one simulated cycle (three PNG + PDF files).

1. Gas temperature (left axis) and gas flow (right axis) over the loop.
2. Turbine CO2 flow (left) and cycle thermal efficiency (right).
3. Net electrical power, with the idle and restart periods shaded.

Cycle thermal efficiency = net shaft power / heat added to the CO2 (idle and
restart periods are not drawn). Colours: two validated categorical slots
(blue, orange) and neutral grey for the non-running states."""
import numpy as np

K = 273.15
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, GREY = "#0b0b0b", "#52514e", "#d9d8d4", "#b9b8b3"


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _twin(sim, y_left, y_right, lab_left, lab_right, title, path, fmt_right=None,
          y_right_lim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = sim["t_min"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    _style(ax)
    l1, = ax.plot(t, y_left, color=BLUE, lw=1.8, label=lab_left)
    ax.set_ylabel(lab_left, color=INK)
    ax.set_xlabel("Time in the cycle (min)", color=INK)
    ax.set_xlim(t[0], t[-1] + (t[1] - t[0]))
    ax2 = ax.twinx()
    for sp in ("top",):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(colors=INK2, labelsize=10)
    l2, = ax2.plot(t, y_right, color=ORANGE, lw=1.8, label=lab_right)
    ax2.set_ylabel(lab_right, color=INK)
    if y_right_lim:
        ax2.set_ylim(*y_right_lim)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.legend([l1, l2], [lab_left + "  (left axis)", lab_right + "  (right axis)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
              fontsize=9.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=200)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def make_report_figures(sim, hw, prefix, title_suffix=""):
    """sim: result of fluctuation_splitflow.simulate (measured profile);
    prefix: path prefix, e.g. 'figures/ref_size1.0_'. Returns the file stems."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    suffix = f" - {title_suffix}" if title_suffix else ""
    p1 = prefix + "1_gas_temperature_flow"
    _twin(sim, sim["T_C"], sim["m_gas"], "Flue-gas temperature (C)", "Flue-gas mass flow (kg/s)",
          "Flue gas over the loop" + suffix, p1)

    run = sim["state"] == "run"
    m_turb = np.where(run, sim["m_co2"], np.nan)
    eta = sim["eta_cyc"] * 100.0
    p2 = prefix + "2_co2_flow_efficiency"
    _twin(sim, m_turb, eta, "Turbine CO2 mass flow (kg/s)", "Cycle thermal efficiency (%)",
          "CO2 flow and cycle efficiency (gaps = idle or restart)" + suffix, p2)

    t, dt = sim["t_min"], sim["t_min"][1] - sim["t_min"][0]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    _style(ax)
    for state, colour, alpha, hatch in (("idle", GREY, 0.55, None), ("restart", GREY, 0.30, "///")):
        m = sim["state"] == state
        i = 0
        while i < len(t):
            if m[i]:
                j = i
                while j + 1 < len(t) and m[j + 1]:
                    j += 1
                ax.axvspan(t[i], t[j] + dt, color=colour, alpha=alpha, lw=0, hatch=hatch)
                i = j + 1
            else:
                i += 1
    ax.plot(t, sim["P"] / 1e6, color=BLUE, lw=1.8)
    ax.axhline(hw["net_el_des"] / 1e6, color=INK2, lw=1.0, ls=":")
    ax.axhline(0.0, color=GRID, lw=0.8)
    ax.set_xlim(t[0], t[-1] + dt)
    ax.set_xlabel("Time in the cycle (min)", color=INK)
    ax.set_ylabel("Net electrical power (MW)", color=INK)
    ax.set_title("Net electrical power over the loop" + suffix, color=INK, fontsize=12, loc="left")
    handles = [plt.Line2D([0], [0], color=BLUE, lw=1.8),
               plt.Line2D([0], [0], color=INK2, lw=1.0, ls=":"),
               Patch(facecolor=GREY, alpha=0.55, label="Idle"),
               Patch(facecolor=GREY, alpha=0.30, hatch="///", label="Restart")]
    ax.legend(handles, ["Net electrical power", f"Design steady ({hw['net_el_des'] / 1e6:.2f} MW)",
                        "Idle", "Restart"], loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=4, frameon=False, fontsize=9.5, labelcolor=INK2)
    fig.tight_layout()
    p3 = prefix + "3_net_power_states"
    fig.savefig(p3 + ".png", dpi=200)
    fig.savefig(p3 + ".pdf")
    plt.close(fig)
    return [p1, p2, p3]
