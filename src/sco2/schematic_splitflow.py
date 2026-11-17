"""Process schematic of the split-flow sCO2 cycle with the EAF gas side.

Drawn with matplotlib on a fixed grid. State numbers follow splitflow_adv().
Temperatures, pressures, powers and heat duties are read from the model
results, so the figure updates when the cycle changes. The electrical machines
(generator, compressor motor), the off-gas fan and the electrical power
balance are drawn when the dictionary `elec` is given (motor-driven compressor
only). If the layout changes (new components), edit the coordinates here."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon, Circle

K = 273.15
C_COLD, C_WARM, C_HOT, C_GAS = "tab:blue", "tab:orange", "tab:red", "dimgray"


def _pipe(ax, pts, color, arrows=(), lw=2.2):
    """Polyline with optional flow arrows: arrows = [(segment index, fraction)]."""
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=1)
    for seg, frac in arrows:
        (x0, y0), (x1, y1) = pts[seg], pts[seg + 1]
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        xm, ym = x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
        ax.annotate("", xy=(xm + 0.15 * ux, ym + 0.15 * uy),
                    xytext=(xm - 0.15 * ux, ym - 0.15 * uy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=15), zorder=2)


def _hx(ax, x0, x1, yc, h, upper, lower, lw_up=2.2, lw_low=2.2, dy=0.3):
    """Heat exchanger box with two parallel lines (upper and lower stream)."""
    ax.add_patch(Rectangle((x0, yc - h / 2), x1 - x0, h, fc="#f2f2f2",
                           ec="black", lw=1.5, zorder=3))
    ax.plot([x0, x1], [yc + dy] * 2, color=upper, lw=lw_up, zorder=4)
    ax.plot([x0, x1], [yc - dy] * 2, color=lower, lw=lw_low, zorder=4)


def _machine(ax, x, y, letter, fc):
    """Circle with a letter: generator (G), motor (M) or fan (F)."""
    ax.add_patch(Circle((x, y), 0.33, fc=fc, ec="black", lw=1.5, zorder=3))
    ax.text(x, y, letter, ha="center", va="center", fontsize=10,
            fontweight="bold", zorder=4)


def _shaft(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color="black", lw=3.5, solid_capstyle="butt",
            zorder=2)


def _state(ax, k, st, x, y, vx, vy, ha, values):
    """Circled state number on the pipe, with T and p next to it."""
    ax.text(x, y, str(k), ha="center", va="center", fontsize=9,
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="black", lw=1.2))
    if values:
        s = st[k]
        ax.text(vx, vy, f"{s['T'] - K:.0f} °C\n{s['p'] / 1e6:.2f} MPa",
                ha=ha, va="center", fontsize=7.5, zorder=6)


def draw_schematic_splitflow(res, hs=None, eaf=None, elec=None,
                             show_values=True, save_path=None):
    """res: splitflow_adv() result; hs: couple_cascade() result; eaf: EAF dict.
    elec: dictionary with the electrical side (see run_schematic_splitflow.py):
    P_gen, P_motor, fan_tower, fan_gas, loss_gb, loss_gen, loss_mot, net_shaft,
    net_el, eta_el, makeup (all in W, kg/s or -). Without it the figure shows
    the shaft powers only."""
    st, y = res["states"], res["y"]
    m = hs["m_co2"] if hs is not None else None
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(-1.5, 20.2)
    ax.set_ylim(0.2, 10.4)
    ax.set_aspect("equal")
    ax.axis("off")

    def name(x, yy, s, **kw):
        ax.text(x, yy, s, ha=kw.pop("ha", "center"), va="center",
                fontsize=9, zorder=6, **kw)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------
    ax.add_patch(Polygon([(1.1, 2.5), (2.5, 2.5), (2.2, 4.1), (1.4, 4.1)],
                         closed=True, fc="#cfe2f3", ec="black", lw=1.5, zorder=3))
    ax.text(1.8, 3.3, "MC", ha="center", va="center", fontsize=10,
            fontweight="bold", zorder=4)
    ax.add_patch(Polygon([(16.2, 7.8), (17.0, 7.8), (17.4, 6.2), (15.8, 6.2)],
                         closed=True, fc="#f4cccc", ec="black", lw=1.5, zorder=3))
    ax.text(16.6, 7.0, "T", ha="center", va="center", fontsize=10,
            fontweight="bold", zorder=4)

    _hx(ax, 4.2, 7.2, 4.8, 1.2, upper=C_WARM, lower=C_COLD)             # LTR
    _hx(ax, 8.8, 11.8, 4.8, 1.2, upper=C_HOT, lower=C_WARM)             # HTR
    _hx(ax, 13.0, 15.4, 9.0, 1.4, upper=C_GAS, lower=C_HOT, lw_up=3)    # main heater
    _hx(ax, 4.8, 7.2, 9.0, 1.4, upper=C_GAS, lower=C_WARM, lw_up=3)     # branch heater
    ax.add_patch(Rectangle((3.2, 1.4), 1.0, 1.2, fc="#f2f2f2", ec="black",
                           lw=1.5, zorder=3))                           # cooler
    ax.plot([3.7, 3.7], [1.4, 2.6], color=C_COLD, lw=2.2, zorder=4)
    ax.add_patch(Circle((8.0, 5.1), 0.15, fc="white", ec="black", lw=1.5, zorder=5))
    ax.add_patch(Circle((3.3, 5.1), 0.07, fc="black", ec="black", zorder=5))

    if elec is not None:
        _machine(ax, 0.45, 3.3, "M", "#fff2cc")                 # compressor motor
        _shaft(ax, 0.78, 3.3, 1.26, 3.3)
        _machine(ax, 18.65, 7.0, "G", "#d9ead3")                # generator
        _shaft(ax, 17.2, 7.0, 18.32, 7.0)
        _machine(ax, 2.0, 9.3, "F", "#eeeeee")                  # off-gas fan

    # ------------------------------------------------------------------
    # CO2 pipes
    # ------------------------------------------------------------------
    _pipe(ax, [(3.7, 1.4), (3.7, 0.9), (1.8, 0.9), (1.8, 2.5)], C_COLD, [(1, 0.5)])
    _pipe(ax, [(1.8, 4.1), (1.8, 5.1), (4.2, 5.1)], C_COLD, [(1, 0.8)])
    _pipe(ax, [(3.3, 5.1), (3.3, 8.7), (4.8, 8.7)], C_COLD, [(0, 0.5)])
    _pipe(ax, [(7.2, 5.1), (7.85, 5.1)], C_WARM)
    _pipe(ax, [(8.15, 5.1), (8.8, 5.1)], C_WARM)
    _pipe(ax, [(7.2, 8.7), (8.0, 8.7), (8.0, 5.25)], C_WARM, [(1, 0.3)])
    _pipe(ax, [(11.8, 5.1), (12.5, 5.1), (12.5, 8.7), (13.0, 8.7)], C_HOT, [(1, 0.85)])
    _pipe(ax, [(15.4, 8.7), (16.6, 8.7), (16.6, 7.8)], C_HOT, [(1, 0.5)])
    _pipe(ax, [(16.6, 6.2), (16.6, 4.5), (11.8, 4.5)], C_HOT, [(0, 0.5), (1, 0.3)])
    _pipe(ax, [(8.8, 4.5), (7.2, 4.5)], C_WARM, [(0, 0.8)])
    _pipe(ax, [(4.2, 4.5), (3.7, 4.5), (3.7, 2.6)], C_COLD, [(1, 0.8)])

    # EAF gas pipes (right to left along the top)
    _pipe(ax, [(17.4, 9.3), (15.4, 9.3)], C_GAS, [(0, 0.5)], lw=3)
    _pipe(ax, [(13.0, 9.3), (7.2, 9.3)], C_GAS, [(0, 0.8)], lw=3)
    if elec is not None:
        _pipe(ax, [(4.8, 9.3), (2.33, 9.3)], C_GAS, [(0, 0.6)], lw=3)
        _pipe(ax, [(1.67, 9.3), (0.7, 9.3)], C_GAS, [(0, 0.5)], lw=3)
    else:
        _pipe(ax, [(4.8, 9.3), (2.4, 9.3)], C_GAS, [(0, 0.6)], lw=3)

    # ------------------------------------------------------------------
    # State labels: k: (x, y, value text x, value text y, alignment)
    # ------------------------------------------------------------------
    spec = {1: (1.8, 1.7, 2.1, 1.7, "left"),
            2: (2.55, 5.1, 2.55, 5.65, "center"),
            3: (7.55, 5.1, 7.45, 5.7, "center"),
            5: (8.5, 5.1, 8.6, 5.75, "center"),
            6: (12.5, 6.6, 12.75, 6.6, "left"),
            7: (16.0, 8.7, 16.0, 8.15, "center"),
            8: (14.0, 4.5, 14.0, 3.95, "center"),
            9: (8.0, 4.5, 8.0, 3.95, "center"),
            10: (3.7, 3.6, 3.95, 3.6, "left"),
            11: (8.0, 7.0, 8.25, 7.0, "left")}
    for k, (x, yy, vx, vy, ha) in spec.items():
        _state(ax, k, st, x, yy, vx, vy, ha, show_values)

    # ------------------------------------------------------------------
    # Names and numbers read from the model
    # ------------------------------------------------------------------
    name(5.7, 3.85, "LTR")
    name(10.3, 3.85, "HTR")
    name(8.12, 6.3, "Mixer", ha="left")
    name(3.45, 7.3, f"y = {y:.3f}", ha="left")

    mc, tb = "Main\ncompressor", "Turbine"
    mh, bh, cl = "Main heater", "Branch heater", "Cooler"
    if hs is not None:
        mc += f"\nW = {m * res['w_mc'] / 1e6:.2f} MW"
        tb += f"\nW = {m * res['w_t'] / 1e6:.2f} MW"
        mh += f"\nQ = {hs['Q1'] / 1e6:.2f} MW"
        bh += f"\nQ = {hs['Q2'] / 1e6:.2f} MW"
        cl += f"\nQ = {m * res['q_out'] / 1e6:.2f} MW"
    name(14.2, 7.95, mh)
    name(6.0, 7.95, bh)

    if elec is not None:
        name(0.3, 2.1, mc)
        name(0.45, 3.95, f"Motor\n{elec['P_motor'] / 1e6:.2f} MWe")
        name(17.55, 5.35, tb, ha="left")
        name(18.65, 6.3, f"Generator\n{elec['P_gen'] / 1e6:.2f} MWe")
        name(2.0, 8.6, f"Off-gas fan\n{elec['fan_gas'] / 1e6:.2f} MWe")
        name(4.4, 2.0, cl.replace("Cooler", "Cooler (water loop\nto wet tower)")
             + f"\nfan + pump {elec['fan_tower'] / 1e6:.2f} MWe"
             + f"\nmakeup water {elec['makeup']:.1f} kg/s", ha="left")
    else:
        name(0.95, 3.3, mc, ha="right")
        name(17.5, 7.0, tb, ha="left")
        name(4.4, 2.0, cl, ha="left")

    if hs is not None and eaf is not None:
        name(17.55, 9.3, f"EAF off-gas\n{eaf['T_in'] - K:.0f} °C, "
             f"{eaf['m_gas']:.0f} kg/s", ha="left")
        ax.text(10.1, 9.6, f"{hs['T_gas_mid'] - K:.0f} °C", ha="center",
                va="center", fontsize=8, color=C_GAS, zorder=6)
        stack_txt = f"Stack\n{hs['T_out'] - K:.0f} °C"
        if elec is not None:
            ax.text(0.55, 9.3, stack_txt, ha="right", va="center", fontsize=8,
                    color=C_GAS, zorder=6)
        else:
            ax.text(3.6, 9.6, f"Stack {hs['T_out'] - K:.0f} °C", ha="center",
                    va="center", fontsize=8, color=C_GAS, zorder=6)
        pr = st[2]["p"] / st[1]["p"]
        if elec is not None:
            fans = (elec["fan_tower"] + elec["fan_gas"]) / 1e6
            info = (f"Net electrical power: {elec['net_el'] / 1e6:.2f} MW\n"
                    f"   = generator {elec['P_gen'] / 1e6:.2f} - motor "
                    f"{elec['P_motor'] / 1e6:.2f} - fans {fans:.2f} MW\n"
                    f"Net shaft power: {elec['net_shaft'] / 1e6:.2f} MW\n"
                    f"Losses: gearboxes {elec['loss_gb'] / 1e6:.2f}, generator "
                    f"{elec['loss_gen'] / 1e6:.2f}, motor {elec['loss_mot'] / 1e6:.2f} MW\n"
                    f"Net electrical efficiency: {elec['eta_el'] * 100:.1f} % "
                    f"(cycle {res['eta'] * 100:.1f} %)\n"
                    f"CO2 flow: {m:.1f} kg/s\n"
                    f"Heat used: {hs['Q'] / 1e6:.2f} of {hs['Q_max'] / 1e6:.2f} MW\n"
                    f"Pressure ratio {pr:.2f}, maximum pressure "
                    f"{st[2]['p'] / 1e6:.1f} MPa")
        else:
            info = (f"Net power: {m * res['w_net'] / 1e6:.2f} MW\n"
                    f"Cycle efficiency: {res['eta'] * 100:.1f} %\n"
                    f"CO2 flow: {m:.1f} kg/s\n"
                    f"Heat used: {hs['Q'] / 1e6:.2f} of {hs['Q_max'] / 1e6:.2f} MW\n"
                    f"Compressor pressure ratio: {pr:.2f}")
        ax.text(9.0, 2.2, info, ha="left", va="center", fontsize=9, zorder=6,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray"))

    handles = [Line2D([0], [0], color=C_COLD, lw=2.5, label="CO2, cold"),
               Line2D([0], [0], color=C_WARM, lw=2.5, label="CO2, warm"),
               Line2D([0], [0], color=C_HOT, lw=2.5, label="CO2, hot"),
               Line2D([0], [0], color=C_GAS, lw=3, label="EAF gas")]
    if elec is not None:
        handles.append(Line2D([0], [0], color="black", lw=3.5, label="Shaft"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_title("Split-flow sCO2 cycle with EAF off-gas heat source", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax