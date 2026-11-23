"""Checks of the off-design recuperator model against the design point.

A  Design: with the design UA and the design inlets and flows the model must return the design duty and outlet
   temperatures (it uses the same method, so the match should be to the solver tolerance).
B  Part load: both flows scaled together (inlet states kept), UA fixed: the effectiveness rises as the flow falls.
C  Recompression fraction: only the LTR cold flow (1 - x) m changes, hot flow and inlet states kept. This is a
   component response, not a cycle: in the cycle the inlet states move as well.
Output: tables/recuperator_checks.csv, figures/recuperator_checks.png.
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/check_recuperators.py" """
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
