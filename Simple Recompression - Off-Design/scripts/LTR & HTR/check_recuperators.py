"""Checks of the off-design recuperator model against the design point.

A  Design: with the design UA and the design inlets and flows the model must return the design duty and outlet
   temperatures (it uses the same method, so the match should be to the solver tolerance).
B  Part load: both flows scaled together (inlet states kept), UA fixed: the effectiveness rises as the flow falls.
C  Recompression fraction: only the LTR cold flow (1 - x) m changes, hot flow and inlet states kept. This is a
   component response, not a cycle: in the cycle the inlet states move as well.
D  Design figure (from the resistance-share tables, if present): segment UA, resistance shares, Re and Pr against the correlation range.
Output: tables/recuperator_checks.csv, figures/recuperator_checks.png, figures/recuperator_design.png.
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/LTR & HTR/check_recuperators.py" """
import sys
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
BLUE, ORANGE = "#2a78d6", "#eb6834"
K = dp.K

res, hs, x = dp.load_design()
m = hs["m_co2"]
D = dp.recuperator_design(res, hs, x)
print(f"Design point: x = {x:.4f}, CO2 {m:.2f} kg/s, net power {m * res['w_net'] / 1e6:.3f} MW")


def run(name, f_hot=1.0, f_cold=1.0, label=""):
    d = D[name]
    r = rc.solve(d["UA"], d["hot_in"], d["cold_in"], d["m_hot"] * f_hot, d["m_cold"] * f_cold, d["dp_hot"], d["dp_cold"])
    return {"case": label, "exchanger": name, "flow factor hot": f_hot, "flow factor cold": f_cold,
            "UA (MW/K)": r["UA"] / 1e6, "duty (MW)": r["q"] / 1e6, "effectiveness": r["eps"],
            "min dT (K)": r["min_dT"], "hot outlet (C)": r["hot_out"]["T"] - K, "cold outlet (C)": r["cold_out"]["T"] - K}


rows = []
print("\nA. Design check (design UA, design inlets and flows)")
for name in ("HTR", "LTR"):
    r = run(name, label="A design")
    d = D[name]
    st = res["states"]
    ho, co = ((9, 6) if name == "HTR" else (10, 3))
    print(f"  {name}: duty {r['duty (MW)']:.4f} MW (design {d['q'] / 1e6:.4f}), hot outlet {r['hot outlet (C)']:.3f} C "
          f"(design {st[ho]['T'] - K:.3f}), cold outlet {r['cold outlet (C)']:.3f} C (design {st[co]['T'] - K:.3f}), "
          f"min dT {r['min dT (K)']:.3f} K (design {d['min_dT']:.3f}), UA {r['UA (MW/K)']:.4f} MW/K")
    rows.append(r)

print("\nB. Part load: both flows x factor, inlet states and UA fixed")
fl = [1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
for name in ("HTR", "LTR"):
    for f in fl:
        r = run(name, f, f, "B part load")
        rows.append(r)
        print(f"  {name} flow {f:.1f}: duty {r['duty (MW)']:.3f} MW, effectiveness {r['effectiveness']:.3f}, "
              f"min dT {r['min dT (K)']:.2f} K")

print("\nC. LTR cold flow (1 - x) m, hot flow and inlet states fixed")
xs = [0.15, 0.20, 0.25, x, 0.30, 0.35]
for xv in xs:
    r = run("LTR", 1.0, (1.0 - xv) / (1.0 - x), "C recompression fraction")
    r["x"] = xv
    rows.append(r)
    print(f"  x = {xv:.3f}: duty {r['duty (MW)']:.3f} MW, effectiveness {r['effectiveness']:.3f}, min dT {r['min dT (K)']:.2f} K, "
          f"hot outlet {r['hot outlet (C)']:.1f} C")

dp.TABLES.mkdir(parents=True, exist_ok=True)
dp.FIGURES.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.round(5).to_csv(dp.TABLES / "recuperator_checks.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.4))
for name, c in (("HTR", BLUE), ("LTR", ORANGE)):
    d = df[(df["case"] == "B part load") & (df["exchanger"] == name)].sort_values("flow factor hot")
    ax[0].plot(d["flow factor hot"], d["effectiveness"], color=c, lw=2, marker="o", ms=5, label=name)
    ax[1].plot(d["flow factor hot"], d["min dT (K)"], color=c, lw=2, marker="o", ms=5, label=name)
ax[0].set_ylabel("Effectiveness (-)")
ax[1].set_ylabel("Minimum temperature difference (K)")
for a in ax[:2]:
    a.set_xlabel("Flow relative to design (both sides)")
    a.axvline(1.0, color="#888888", lw=1, ls="--")
    a.legend(frameon=False, fontsize=9)
ax[0].set_title("Fixed UA, part load", fontsize=10)
ax[1].set_title("Fixed UA, part load", fontsize=10)
d = df[df["case"] == "C recompression fraction"]
ax[2].plot(d["x"], d["min dT (K)"], color=ORANGE, lw=2, marker="o", ms=5, label="LTR minimum dT (left)")
ax[2].axvline(x, color="#888888", lw=1, ls="--")
ax[2].set_xlabel("Recompression fraction x")
ax[2].set_ylabel("LTR minimum temperature difference (K)")
bx = ax[2].twinx()
bx.plot(d["x"], d["effectiveness"], color=BLUE, lw=1.8, ls="--", marker="s", ms=5, mfc="none", label="LTR effectiveness (right)")
bx.set_ylabel("LTR effectiveness (-)")
ax[2].set_title("LTR response to x (other states fixed)", fontsize=10)
ax[2].legend(handles=ax[2].get_lines() + bx.get_lines(), frameon=False, fontsize=8.5, loc="best")
for a in list(ax) + [bx]:
    a.spines["top"].set_visible(False)
for a in ax:
    a.grid(True, color="#dddddd", lw=0.6)
fig.suptitle("Off-design recuperator model at the design UA (dashed line = design point)", fontsize=10)
fig.tight_layout()
fig.savefig(dp.FIGURES / "recuperator_checks.png", dpi=170)
print(f"\nSaved: {dp.TABLES / 'recuperator_checks.csv'} and {dp.FIGURES / 'recuperator_checks.png'}")


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
