"""Checks of the off-design gas-heater model (20 fixed segments, UA per segment from the resistance shares).

A  Design: the segmented model must return the design duty, stack temperature, T7, minimum dT and CO2 pressure drop.
B  Gas flow x factor (gas temperature, CO2 inlet and CO2 flow kept): segmented model clean and fouled, against the lumped model
   with a fixed UA and with a UA that follows the gas flow (90 % gas-side resistance, the earlier placeholder).
C  Gas inlet temperature, flows kept: T7 without and with the bypass damper that holds T7 at its design value; bypass fraction
   and the tube-wall temperature at the hot end.
D  CO2 flow x factor at the design gas: T7, CO2 pressure drop, segmented against the lumped fixed-UA model.
E  Property correction of the gas-side coefficient (k Pr^(1/3) mu^-0.681): its effect on T7 at low gas flow and at hot and cold gas.
F  Design figure (from the resistance-share tables): temperature profile, segment UA, resistance budget, gas Reynolds number.
Output: tables/heater_checks.csv, figures/heater_checks.png, figures/heater_design.png.
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/Gas Heater/check_heater.py" """
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                  # scripts/: design_point.py, ua_scaling.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import design_point as dp
import heater as ht
from ua_scaling import ua_ratio

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": False})
BLUE, ORANGE, AQUA, GREY, DARK = "#2a78d6", "#eb6834", "#1baf7a", "#888888", "#0f3f7d"
K = dp.K
RSHARE_GAS_OLD = 0.9                                                  # earlier placeholder of the lumped model
T7_DES = dp.T7

res, hs, x = dp.load_design()
H = dp.heater_design(res, hs)
st6, st7 = res["states"][6], res["states"][7]
m_c, m_g, T_in = H["m_co2"], H["m_gas"], H["T_gas_in"]
tr = ht.GasTransport(dp.EAF["X"])
sh_clean = ht.load_shares(dp.TABLES / "heater_resistance_shares.csv")
sh_foul = ht.load_shares(dp.TABLES / "heater_resistance_shares_fouled.csv")
mk = lambda sh, t=tr: ht.SegmentedHeater(hs, st6, st7["p"], m_c, m_g, sh, dp.GAS, transport=t)
seg, segF, segN = mk(sh_clean), mk(sh_foul), mk(sh_clean, None)
print(f"Design point: CO2 {m_c:.2f} kg/s, gas {m_g:.2f} kg/s at {T_in - K:.2f} C, UA {seg.UA_cycle_total / 1e6:.4f} MW/K (cycle definition), {seg.UA_total_des / 1e6:.4f} MW/K with the log-mean segments of the model, {seg.N} segments")
rows = []


def old(UAscale, rg=1.0, rc=1.0, Tg=T_in):
    UA = H["UA"] * (ua_ratio(rg, rc, rshare_hot=1.0 - RSHARE_GAS_OLD) if UAscale else 1.0)
    return ht.solve(UA, dp.GAS, Tg, m_g * rg, st6, m_c * rc, H["dp"])


print("\nA. Design check")
for tag, m in (("clean", seg), ("fouled", segF), ("clean, no property correction", segN)):
    r = m.solve(T_in, m_g, st6, m_c)
    print(f"  {tag}: duty {r['Q'] / 1e6:.4f} MW (design {H['Q'] / 1e6:.4f}), stack {r['T_stack'] - K:.3f} C ({H['T_gas_out'] - K:.3f}), "
          f"T7 {r['T7'] - K:.3f} C ({H['T7'] - K:.3f}), min dT {r['min_dT']:.3f} K ({H['min_dT']:.3f}), "
          f"dp {r['dp'] / 1e3:.1f} kPa ({(st6['p'] - st7['p']) / 1e3:.1f}), wall {r['T_wall_max'] - K:.1f} C")
r0 = seg.solve(T_in, m_g, st6, m_c)
rows.append({"case": "A design", "Q (MW)": r0["Q"] / 1e6, "T7 (C)": r0["T7"] - K, "T stack (C)": r0["T_stack"] - K, "min dT (K)": r0["min_dT"]})

print("\nB. Gas flow x factor (gas temperature, CO2 flow fixed)")
bf = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
B = []
for f in bf:
    a, b = seg.solve(T_in, m_g * f, st6, m_c), segF.solve(T_in, m_g * f, st6, m_c)
    c, d = old(False, f), old(True, f)
    B.append({"case": "B gas flow", "factor": f, "clean T7 (C)": a["T7"] - K, "clean min dT (K)": a["min_dT"], "clean Q (MW)": a["Q"] / 1e6,
              "fouled T7 (C)": b["T7"] - K, "fouled min dT (K)": b["min_dT"], "fouled Q (MW)": b["Q"] / 1e6,
              "lumped fixed UA T7 (C)": c["T_co2_out"] - K, "lumped fixed UA min dT (K)": c["min_dT"],
              "lumped scaled UA T7 (C)": d["T_co2_out"] - K, "lumped scaled UA min dT (K)": d["min_dT"],
              "clean stack (C)": a["T_stack"] - K, "clean UA (MW/K)": a["UA_total"] / 1e6})
    print(f"  gas {f:.1f}: clean T7 {a['T7'] - K:5.1f} C, dT {a['min_dT']:5.1f} K, Q {a['Q'] / 1e6:5.2f} MW | fouled T7 {b['T7'] - K:5.1f} C, "
          f"dT {b['min_dT']:5.1f} K | lumped fixed {c['T_co2_out'] - K:5.1f} C ({c['min_dT']:4.1f} K), scaled {d['T_co2_out'] - K:5.1f} C ({d['min_dT']:4.1f} K) "
          f"| UA {a['UA_total'] / seg.UA_total_des:.3f} of design")
rows += B
Bd = pd.DataFrame(B)

print("\nC. Gas inlet temperature (flows kept): T7 free and with the bypass damper holding T7 at design")
C = []
for Tc_ in (535, 585, 635, 685, 735, 785, 835, 885, 935):
    Tg = Tc_ + K
    a = seg.solve(Tg, m_g, st6, m_c)
    try:
        bp, rb = seg.bypass_for_T7(T7_DES, Tg, m_g, st6, m_c)
    except ValueError as err:
        print(f"    bypass at {Tc_} C failed: {err}")
        bp, rb = np.nan, a
    C.append({"case": "C gas T", "T gas (C)": Tc_, "T7 (C)": a["T7"] - K, "Q (MW)": a["Q"] / 1e6, "T stack (C)": a["T_stack"] - K,
              "min dT (K)": a["min_dT"], "wall max (C)": a["T_wall_max"] - K, "bypass for T7 design": bp, "T7 with bypass (C)": rb["T7"] - K,
              "Q with bypass (MW)": rb["Q"] / 1e6, "stack with bypass (C)": rb["T_stack"] - K, "wall with bypass (C)": rb["T_wall_max"] - K})
    print(f"  gas {Tc_} C: T7 {a['T7'] - K:5.1f} C, Q {a['Q'] / 1e6:5.2f} MW, stack {a['T_stack'] - K:5.1f} C, wall {a['T_wall_max'] - K:5.1f} C | "
          f"bypass {bp * 100:5.1f} % -> Q {rb['Q'] / 1e6:5.2f} MW, stack {rb['T_stack'] - K:5.1f} C, wall {rb['T_wall_max'] - K:5.1f} C")
rows += C
Cd = pd.DataFrame(C)

print("\nD. CO2 flow x factor (design gas)")
D = []
for f in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
    a, c = seg.solve(T_in, m_g, st6, m_c * f), old(False, 1.0, f)
    D.append({"case": "D CO2 flow", "factor": f, "T7 (C)": a["T7"] - K, "Q (MW)": a["Q"] / 1e6, "dp (kPa)": a["dp"] / 1e3, "min dT (K)": a["min_dT"],
              "lumped T7 (C)": c["T_co2_out"] - K, "UA (MW/K)": a["UA_total"] / 1e6})
    print(f"  CO2 {f:.1f}: T7 {a['T7'] - K:5.1f} C (lumped {c['T_co2_out'] - K:5.1f}), Q {a['Q'] / 1e6:5.2f} MW, dp {a['dp'] / 1e3:6.1f} kPa "
          f"({a['dp'] / st6['p'] * 100:.2f} % of inlet), dT {a['min_dT']:5.1f} K")
rows += D
Dd = pd.DataFrame(D)

print("\nE. Effect of the property correction of the gas-side coefficient on T7 (K)")
for Tc_, f in ((585, 1.0), (685, 0.6), (885, 1.0), (885, 0.6)):
    a, b = seg.solve(Tc_ + K, m_g * f, st6, m_c), segN.solve(Tc_ + K, m_g * f, st6, m_c)
    print(f"  gas {Tc_} C, flow {f:.1f}: with {a['T7'] - K:.2f} C, without {b['T7'] - K:.2f} C, difference {a['T7'] - b['T7']:+.2f} K, "
          f"UA ratio {a['UA_total'] / b['UA_total']:.3f}")
    rows.append({"case": "E property correction", "T gas (C)": Tc_, "factor": f, "T7 with (C)": a["T7"] - K, "T7 without (C)": b["T7"] - K})

pd.DataFrame(rows).to_csv(dp.TABLES / "heater_checks.csv", index=False)

fig, ax = plt.subplots(2, 4, figsize=(22, 9.5))
a_ = ax[0, 0]
a_.plot(Bd["factor"], Bd["clean T7 (C)"], "o-", color=BLUE, label="segmented, clean")
a_.plot(Bd["factor"], Bd["fouled T7 (C)"], "o--", color=DARK, label="segmented, fouled")
a_.plot(Bd["factor"], Bd["lumped fixed UA T7 (C)"], "s:", color=GREY, label="lumped, fixed UA")
a_.plot(Bd["factor"], Bd["lumped scaled UA T7 (C)"], "^:", color=ORANGE, label="lumped, UA follows gas (90 %)")
a_.set_xlabel("gas flow / design"); a_.set_ylabel("T7 (°C)"); a_.set_title("(a) T7 against gas flow"); a_.legend(frameon=False, fontsize=9); a_.grid(alpha=0.3)
a_ = ax[0, 1]
a_.plot(Bd["factor"], Bd["clean min dT (K)"], "o-", color=BLUE, label="segmented, clean")
a_.plot(Bd["factor"], Bd["fouled min dT (K)"], "o--", color=DARK, label="segmented, fouled")
a_.plot(Bd["factor"], Bd["lumped fixed UA min dT (K)"], "s:", color=GREY, label="lumped, fixed UA")
a_.plot(Bd["factor"], Bd["lumped scaled UA min dT (K)"], "^:", color=ORANGE, label="lumped, UA follows gas (90 %)")
a_.set_xlabel("gas flow / design"); a_.set_ylabel("minimum temperature difference (K)"); a_.set_title("(b) The 20 K pinch becomes a result")
a_.legend(frameon=False, fontsize=9); a_.grid(alpha=0.3)
a_ = ax[0, 2]
a_.plot(Cd["T gas (C)"], Cd["T7 (C)"], "o-", color=BLUE, label="no bypass")
a_.plot(Cd["T gas (C)"], Cd["T7 with bypass (C)"], "o-", color=ORANGE, label="bypass holds T7")
a_.axhline(T7_DES - K, color=GREY, lw=0.8, ls="--")
a_.set_xlabel("gas inlet temperature (°C)"); a_.set_ylabel("T7 (°C)"); a_.set_title("(c) T7 against gas temperature")
a_.legend(frameon=False, fontsize=9); a_.grid(alpha=0.3)
a_ = ax[1, 0]
a_.plot(Cd["T gas (C)"], Cd["bypass for T7 design"] * 100, "o-", color=ORANGE)
a_.set_xlabel("gas inlet temperature (°C)"); a_.set_ylabel("gas bypass needed to hold T7 (%)"); a_.set_title("(d) Bypass damper opening"); a_.grid(alpha=0.3)
a_ = ax[1, 1]
a_.plot(Cd["T gas (C)"], Cd["wall max (C)"], "o-", color=BLUE, label="no bypass")
a_.plot(Cd["T gas (C)"], Cd["wall with bypass (C)"], "o-", color=ORANGE, label="bypass holds T7")
a_.set_xlabel("gas inlet temperature (°C)"); a_.set_ylabel("hottest tube wall (°C)"); a_.set_title("(e) Tube-wall temperature at the hot end")
a_.legend(frameon=False, fontsize=9); a_.grid(alpha=0.3)
a_ = ax[0, 3]
a_.plot(Dd["factor"], Dd["T7 (C)"], "o-", color=BLUE, label="segmented")
a_.plot(Dd["factor"], Dd["lumped T7 (C)"], "s:", color=GREY, label="lumped, fixed UA")
a_.set_xlabel("CO$_2$ flow / design"); a_.set_ylabel("T7 (°C)"); a_.set_title("(f) T7 against CO$_2$ flow")
a_.legend(frameon=False, fontsize=9); a_.grid(alpha=0.3)
a_ = ax[1, 2]
a_.plot(Dd["factor"], Dd["dp (kPa)"], "o-", color=ORANGE)
a_.set_xlabel("CO$_2$ flow / design"); a_.set_ylabel("CO$_2$ pressure drop (kPa)"); a_.set_title("(g) CO$_2$-side pressure drop"); a_.grid(alpha=0.3)
a_ = ax[1, 3]
a_.plot(Bd["factor"], Bd["clean UA (MW/K)"], "o-", color=BLUE)
a_.axhline(seg.UA_total_des / 1e6, color=GREY, lw=0.8, ls="--")
a_.set_xlabel("gas flow / design"); a_.set_ylabel("heater UA (MW/K)"); a_.set_title("(h) UA follows the gas flow (dashed: design)"); a_.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(dp.FIGURES / "heater_checks.png", dpi=200)
print(f"\nSaved: {dp.TABLES / 'heater_checks.csv'} and {dp.FIGURES / 'heater_checks.png'}")


def design_figure():
    """Design-point figure: temperature profile, segment conductances, resistance budget, gas Reynolds number."""
    clean = pd.read_csv(dp.TABLES / "heater_resistance_shares.csv")
    foul = pd.read_csv(dp.TABLES / "heater_resistance_shares_fouled.csv")
    n = len(clean)
    mid = 0.5 * (clean["f_from"] + clean["f_to"])
    UA = clean["UA_i (W/K)"] / 1e3
    frac_first5 = UA.iloc[:5].sum() / UA.sum()

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    a.plot(mid, clean["T_gas_mean (C)"], "o-", color=ORANGE, ms=4, label="off-gas")
    a.plot(mid, clean["T_co2_mean (C)"], "o-", color=BLUE, ms=4, label="CO$_2$")
    dT = clean["T_gas_mean (C)"] - clean["T_co2_mean (C)"]
    a.annotate(f"segment-mean $\\Delta T$ = {dT.iloc[0]:.0f} K at the cold end,\n{dT.iloc[-1]:.0f} K at the hot end", xy=(0.02, 0.52),
               xycoords="axes fraction", fontsize=10)
    a.set_xlabel("fraction of the heater duty (0 = CO$_2$ inlet / gas outlet)"); a.set_ylabel("segment-mean temperature (°C)")
    a.set_title("(a) Temperature profile of the 20 segments"); a.legend(frameon=False, loc="lower right"); a.grid(alpha=0.3)

    b = ax[0, 1]
    b.bar(clean["seg"], UA, color=BLUE, edgecolor="white", linewidth=1.0, width=0.85)
    b.set_xlabel("segment (1 = cold end)"); b.set_ylabel("segment conductance UA$_i$ (kW/K)")
    b.set_title("(b) Equal duty per segment, very unequal area")
    b.annotate(f"first 5 segments hold {frac_first5 * 100:.0f} % of the UA\ntotal UA = {UA.sum() / 1e3:.3f} MW/K", xy=(0.45, 0.8),
               xycoords="axes fraction", fontsize=10)
    b.set_xticks(range(1, n + 1, 2)); b.grid(alpha=0.3, axis="y")

    c = ax[1, 0]
    labels = ["clean", "fouled\n(R$_f''$ = 0.0015)"]
    parts = [("gas side", "f_g", ORANGE), ("CO$_2$ side", "f_c", BLUE), ("wall", "f_w", GREY), ("fouling", "f_foul", DARK)]
    bottoms = np.zeros(2)
    for name, col, colr in parts:
        v = np.array([clean[col].mean(), foul[col].mean()])
        bars = c.bar(labels, v, bottom=bottoms, color=colr, edgecolor="white", linewidth=1.5, width=0.5, label=name)
        for i, (val, bt) in enumerate(zip(v, bottoms)):
            if val > 0.03:
                c.text(i, bt + val / 2, f"{val:.2f}", ha="center", va="center", color="white", fontsize=11)
        bottoms += v
    c.set_ylabel("share of the design thermal resistance"); c.set_ylim(0, 1.0)
    c.set_title("(c) Resistance budget (mean of the segments)")
    c.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)

    d = ax[1, 1]
    d.plot(clean["seg"], clean["Re_g"], "o-", color=ORANGE, ms=4)
    d.axhspan(1100, 18000, color=GREY, alpha=0.15)
    d.text(10.5, 11000, "validity range of the gas-side correlation\n(Briggs & Young, Re about 1100 to 18000)", ha="center", fontsize=10)
    d.set_ylim(0, 20000); d.set_xlabel("segment (1 = cold end)"); d.set_ylabel("gas Reynolds number (-)")
    d.set_title("(d) Gas-side Reynolds number"); d.set_xticks(range(1, n + 1, 2)); d.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(dp.FIGURES / "heater_design.png", dpi=200)
    print(f"Wrote {dp.FIGURES / 'heater_design.png'}")


design_figure()
