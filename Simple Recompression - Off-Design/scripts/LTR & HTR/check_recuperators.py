"""Checks of the segmented off-design recuperator model (HTR and LTR) against the design point and the lumped fixed-UA model.

The segmented model keeps the design segment conductances and resistance shares (tables/HTR_resistance_shares.csv, LTR_...) and
lets every segment follow the local flows and properties (Ngo zigzag exponents, Re^0.629 Pr^0.317). Both models use the same design.
A  Design: duty, outlet temperatures, minimum dT and pressure drops of the segmented model at the design inlets and flows.
B  Part load: both flows scaled together (inlet states kept): effectiveness, minimum dT, UA relative to design, and the number of
   segments outside the correlation range (Re 3500-22000, Pr 0.75-2.2).
C  Recompression fraction: only the LTR cold flow (1 - x) m changes (component response, inlet states kept).
D  Design figure (from the resistance-share tables).
Output: tables/recuperator_checks.csv, figures/recuperator_checks.png, figures/recuperator_design.png,
tables/co2_recuperator_table.npz (CO2 property table, built on the first run).
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/LTR & HTR/check_recuperators.py"
Run recuperator_resistance_shares.py first. """
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                  # scripts/: design_point.py, ua_scaling.py
sys.path.insert(0, str(HERE))                                         # this folder first, so its own recuperator.py is the one imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import design_point as dp
import recuperator as rc

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": False})
BLUE, ORANGE, GREY, AQUA = "#2a78d6", "#eb6834", "#888888", "#1baf7a"
K = dp.K
t0 = time.time()

res, hs, x = dp.load_design()
m = hs["m_co2"]
D = dp.recuperator_design(res, hs, x)
st = res["states"]
OUT = {"HTR": (9, 6), "LTR": (10, 3)}                                 # (hot outlet state, cold outlet state)
ref = pd.read_csv(HERE / "recuperator_reference_data.csv").set_index("parameter")["value"].astype(float)
for name in ("HTR", "LTR"):
    if not (dp.TABLES / f"{name}_resistance_shares.csv").exists():
        sys.exit(f"{name}_resistance_shares.csv is missing: run recuperator_resistance_shares.py first.")
dp.TABLES.mkdir(parents=True, exist_ok=True)
dp.FIGURES.mkdir(parents=True, exist_ok=True)
tab = rc.CO2Table(dp.TABLES / "co2_recuperator_table.npz")
acc = tab.accuracy()
print(f"CO2 property table ready in {time.time() - t0:.0f} s; largest deviation from CoolProp: T {acc['T']:.4f} K, rho {acc['D']:.1e}, "
      f"mu {acc['V']:.1e} (relative); k and cp come from CoolProp")
SEG = {}
for name in ("HTR", "LTR"):
    sh = pd.read_csv(dp.TABLES / f"{name}_resistance_shares.csv")
    ho, co = st[OUT[name][0]], st[OUT[name][1]]
    SEG[name] = rc.SegmentedRecuperator(sh, D[name], ho, co, ref, tab)
    SEG[name].attach_reynolds(sh)
print(f"Design point: x = {x:.4f}, CO2 {m:.2f} kg/s, net power {m * res['w_net'] / 1e6:.3f} MW")


def seg_run(name, f_hot=1.0, f_cold=1.0):
    d = D[name]
    return SEG[name].solve(d["hot_in"], d["cold_in"], d["m_hot"] * f_hot, d["m_cold"] * f_cold)


def lump_run(name, f_hot=1.0, f_cold=1.0):
    d = D[name]
    return rc.solve(d["UA"], d["hot_in"], d["cold_in"], d["m_hot"] * f_hot, d["m_cold"] * f_cold, d["dp_hot"], d["dp_cold"])


def row(case, name, fh, fc, a, b):
    return {"case": case, "exchanger": name, "flow factor hot": fh, "flow factor cold": fc,
            "seg duty (MW)": a["q"] / 1e6, "seg effectiveness": a["eps"], "seg min dT (K)": a["min_dT"], "seg UA / UA design": a["UA"] / SEG[name].UA_total_des,
            "seg hot outlet (C)": a["hot_out"]["T"] - K, "seg cold outlet (C)": a["cold_out"]["T"] - K, "seg dp hot (kPa)": a["dp_hot"] / 1e3,
            "seg dp cold (kPa)": a["dp_cold"] / 1e3, "Re hot min": a["Re_hot_min"], "Re cold min": a["Re_cold_min"], "segments outside Re range": a["n_outside_Re"],
            "lumped duty (MW)": b["q"] / 1e6, "lumped effectiveness": b["eps"], "lumped min dT (K)": b["min_dT"],
            "lumped hot outlet (C)": b["hot_out"]["T"] - K, "lumped cold outlet (C)": b["cold_out"]["T"] - K}


rows = []
print("\nA. Design check (segmented model, design inlets and flows)")
for name in ("HTR", "LTR"):
    a, b = seg_run(name), lump_run(name)
    d, ho, co = D[name], st[OUT[name][0]], st[OUT[name][1]]
    print(f"  {name}: duty {a['q'] / 1e6:.4f} MW (design {d['q'] / 1e6:.4f}), hot outlet {a['hot_out']['T'] - K:.3f} C ({ho['T'] - K:.3f}), "
          f"cold outlet {a['cold_out']['T'] - K:.3f} C ({co['T'] - K:.3f}), min dT {a['min_dT']:.3f} K ({d['min_dT']:.3f}), "
          f"UA {a['UA'] / 1e6:.4f} MW/K ({SEG[name].UA_total_des / 1e6:.4f})")
    print(f"       dp hot {a['dp_hot'] / 1e3:.1f} kPa (design {(d['hot_in']['p'] - ho['p']) / 1e3:.1f}), cold {a['dp_cold'] / 1e3:.1f} kPa "
          f"({(d['cold_in']['p'] - co['p']) / 1e3:.1f}); Re hot {a['Re_hot_min']:.0f}-{a['Re_hot_max']:.0f}, cold {a['Re_cold_min']:.0f}-{a['Re_cold_max']:.0f}; "
          f"{a['n_outside_Re']} segment-sides outside the Re range")
    rows.append(row("A design", name, 1.0, 1.0, a, b))

print("\nB. Part load: both flows x factor, inlet states fixed (segmented | lumped fixed UA)")
fl = [1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
for name in ("HTR", "LTR"):
    for f in fl:
        a, b = seg_run(name, f, f), lump_run(name, f, f)
        rows.append(row("B part load", name, f, f, a, b))
        print(f"  {name} flow {f:.1f}: eff {a['eps']:.3f} | {b['eps']:.3f}, min dT {a['min_dT']:5.2f} | {b['min_dT']:5.2f} K, UA {a['UA'] / SEG[name].UA_total_des:.3f} of design, "
              f"Re cold min {a['Re_cold_min']:6.0f}, outside range {a['n_outside_Re']:2d}")

print("\nC. LTR cold flow (1 - x) m, hot flow and inlet states fixed (segmented | lumped)")
for xv in [0.15, 0.20, 0.25, x, 0.30, 0.35]:
    fc = (1.0 - xv) / (1.0 - x)
    a, b = seg_run("LTR", 1.0, fc), lump_run("LTR", 1.0, fc)
    r = row("C recompression fraction", "LTR", 1.0, fc, a, b)
    r["x"] = xv
    rows.append(r)
    print(f"  x = {xv:.3f}: eff {a['eps']:.3f} | {b['eps']:.3f}, min dT {a['min_dT']:5.2f} | {b['min_dT']:5.2f} K, hot outlet {a['hot_out']['T'] - K:.1f} | {b['hot_out']['T'] - K:.1f} C, "
          f"Re cold min {a['Re_cold_min']:.0f}")

df = pd.DataFrame(rows)
df.round(5).to_csv(dp.TABLES / "recuperator_checks.csv", index=False, encoding="utf-8-sig")
print(f"\nRun time {time.time() - t0:.0f} s")

fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.6))
for name, c in (("HTR", BLUE), ("LTR", ORANGE)):
    d = df[(df["case"] == "B part load") & (df["exchanger"] == name)].sort_values("flow factor hot")
    f_ = d["flow factor hot"]
    ax[0, 0].plot(f_, d["seg effectiveness"], color=c, lw=2, marker="o", ms=5, label=f"{name} segmented")
    ax[0, 0].plot(f_, d["lumped effectiveness"], color=c, lw=1.4, ls=":", marker="s", ms=4, mfc="none", label=f"{name} lumped, fixed UA")
    ax[0, 1].plot(f_, d["seg min dT (K)"], color=c, lw=2, marker="o", ms=5, label=f"{name} segmented")
    ax[0, 1].plot(f_, d["lumped min dT (K)"], color=c, lw=1.4, ls=":", marker="s", ms=4, mfc="none", label=f"{name} lumped, fixed UA")
    ax[0, 2].plot(f_, d["seg UA / UA design"], color=c, lw=2, marker="o", ms=5, label=name)
    ax[1, 1].plot(f_, d["Re cold min"], color=c, lw=2, marker="o", ms=5, label=f"{name} cold side")
    ax[1, 2].plot(f_, d["segments outside Re range"], color=c, lw=2, marker="o", ms=5, label=name)
ax[0, 0].set_ylabel("Effectiveness (-)"); ax[0, 1].set_ylabel("Minimum temperature difference (K)")
ax[0, 2].set_ylabel("UA relative to design"); ax[1, 1].set_ylabel("Lowest Reynolds number (cold side)")
ax[1, 1].axhline(rc_re := float(ref["Re_min"]), color=GREY, ls="--", lw=1); ax[1, 1].text(0.41, rc_re * 1.03, "correlation limit 3500", fontsize=8, color=GREY)
ax[1, 2].set_ylabel("Segment-sides outside the Ngo range (of 120)")
titles = ["(a) Part load, both flows scaled", "(b) Part load, both flows scaled", "(c) Segment UA follows the flow", "", "(e) Reynolds number of the cold side", "(f) Validity of the correlation"]
for a, t in zip(ax.ravel(), titles):
    if t:
        a.set_title(t, fontsize=10)
for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 1], ax[1, 2]):
    a.set_xlabel("Flow relative to design (both sides)"); a.axvline(1.0, color=GREY, lw=1, ls="--"); a.legend(frameon=False, fontsize=8)
dC = df[df["case"] == "C recompression fraction"]
ax[1, 0].plot(dC["x"], dC["seg min dT (K)"], color=ORANGE, lw=2, marker="o", ms=5, label="segmented: min dT (left)")
ax[1, 0].plot(dC["x"], dC["lumped min dT (K)"], color=ORANGE, lw=1.4, ls=":", marker="s", ms=4, mfc="none", label="lumped: min dT (left)")
ax[1, 0].axvline(x, color=GREY, lw=1, ls="--"); ax[1, 0].set_xlabel("Recompression fraction x"); ax[1, 0].set_ylabel("LTR minimum temperature difference (K)")
bx = ax[1, 0].twinx()
bx.plot(dC["x"], dC["seg effectiveness"], color=BLUE, lw=1.8, ls="--", marker="s", ms=5, label="segmented: effectiveness (right)")
bx.set_ylabel("LTR effectiveness (-)")
ax[1, 0].legend(handles=ax[1, 0].get_lines() + bx.get_lines(), frameon=False, fontsize=8, loc="best")
ax[1, 0].set_title("(d) LTR response to x (other states fixed)", fontsize=10)
for a in list(ax.ravel()) + [bx]:
    a.spines["top"].set_visible(False)
for a in ax.ravel():
    a.grid(True, color="#dddddd", lw=0.6)
fig.suptitle("Off-design recuperators: segmented model against the lumped fixed-UA model (dashed line = design point)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(dp.FIGURES / "recuperator_checks.png", dpi=170)
print(f"Saved: {dp.TABLES / 'recuperator_checks.csv'} and {dp.FIGURES / 'recuperator_checks.png'}")


def design_figure():
    """Design-point figure: segment conductances, resistance shares, Reynolds and Prandtl numbers against the correlation range."""
    GREY, DARK = "#888888", "#0f3f7d"
    RE_RANGE, PR_RANGE = (3500, 22000), (0.75, 2.2)                       # Ngo et al. (2007) validity range
    if not ((dp.TABLES / "HTR_resistance_shares.csv").exists() and (dp.TABLES / "LTR_resistance_shares.csv").exists()):
        print("\nDesign figure skipped: run recuperator_resistance_shares.py first")
        return
    htr = pd.read_csv(dp.TABLES / "HTR_resistance_shares.csv")
    ltr = pd.read_csv(dp.TABLES / "LTR_resistance_shares.csv")
    n = len(htr)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    a.plot(htr["seg"], htr["UA_i (W/K)"] / 1e3, "o-", color=BLUE, ms=4, label=f"HTR (UA {htr['UA_i (W/K)'].sum() / 1e6:.3f} MW/K)")
    a.plot(ltr["seg"], ltr["UA_i (W/K)"] / 1e3, "o-", color=ORANGE, ms=4, label=f"LTR (UA {ltr['UA_i (W/K)'].sum() / 1e6:.3f} MW/K)")
    a.set_xlabel("segment (1 = cold end)"); a.set_ylabel("segment conductance UA$_i$ (kW/K)")
    a.set_title("(a) Equal duty per segment, unequal area"); a.legend(frameon=False); a.grid(alpha=0.3)

    b = ax[0, 1]
    for df, name, c in ((htr, "HTR", BLUE), (ltr, "LTR", ORANGE)):
        b.plot(df["seg"], df["f_hot"], "-", color=c, label=f"{name} hot side")
        b.plot(df["seg"], df["f_cold"], "--", color=c, label=f"{name} cold side")
        b.plot(df["seg"], df["f_wall"], ":", color=c, label=f"{name} wall")
    b.set_xlabel("segment (1 = cold end)"); b.set_ylabel("share of the segment resistance"); b.set_ylim(0, 0.65)
    b.set_title("(b) Resistance shares"); b.legend(frameon=False, fontsize=9, ncol=2); b.grid(alpha=0.3)

    c = ax[1, 0]
    for df, name, col in ((htr, "HTR", BLUE), (ltr, "LTR", ORANGE)):
        c.plot(df["seg"], df["Re_hot"], "-", color=col, label=f"{name} hot side")
        c.plot(df["seg"], df["Re_cold"], "--", color=col, label=f"{name} cold side")
    c.axhspan(*RE_RANGE, color=GREY, alpha=0.15)
    c.set_ylim(0, 24000); c.set_xlabel("segment (1 = cold end)"); c.set_ylabel("Reynolds number (-)")
    c.text(15.5, 22500, "validity range of the correlation", ha="center", fontsize=10, color=DARK)
    c.set_title("(c) Reynolds number"); c.legend(frameon=False, fontsize=9, loc="lower right", ncol=2); c.grid(alpha=0.3)

    d = ax[1, 1]
    for df, name, col in ((htr, "HTR", BLUE), (ltr, "LTR", ORANGE)):
        d.plot(df["seg"], df["Pr_hot"], "-", color=col, label=f"{name} hot side")
        d.plot(df["seg"], df["Pr_cold"], "--", color=col, label=f"{name} cold side")
    d.axhspan(*PR_RANGE, color=GREY, alpha=0.15)
    d.set_ylim(0.5, 2.4); d.set_xlabel("segment (1 = cold end)"); d.set_ylabel("Prandtl number (-)")
    d.text(15.5, 2.28, "validity range of the correlation", ha="center", fontsize=10, color=DARK)
    d.set_title("(d) Prandtl number"); d.legend(frameon=False, fontsize=9, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 0.92)); d.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(dp.FIGURES / "recuperator_design.png", dpi=200)
    print(f"Wrote {dp.FIGURES / 'recuperator_design.png'}")


design_figure()
