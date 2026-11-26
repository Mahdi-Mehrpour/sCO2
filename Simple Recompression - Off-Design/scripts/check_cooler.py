"""Checks of the off-design dry air cooler model against the design point (design ambient 25 C, air warms by 10 K).

A  Design: design UA, air flow and CO2 inlet must give the design duty and T1 = 32 C.
B  Ambient temperature: air flow and UA fixed. T1 is the result; the slope dT1 / dT_ambient is the key number.
C  Air flow at two ambient temperatures (UA following the air flow, air side carrying 90 % of the resistance):
   how far more air lowers T1, and what the fan costs.
D  Fan control: the air flow needed to hold T1 = 32 C at each ambient temperature, and the fan power. Air flows above
   the assumed fan limit (1.3 times design) cannot be reached.
The CO2 inlet state and flow are the design values in every case; in the cycle they move as well (this is a component check).
Output: tables/cooler_checks.csv, figures/cooler_checks.png.
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/check_cooler.py" """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cooler as cl
import design_point as dp
from ua_scaling import ua_ratio

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": False})
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
K = dp.K
RSHARE_CO2 = 0.1                                       # placeholder: air side carries 90 % of the resistance
F_MAX = 1.3                                            # placeholder: highest air flow of the fans relative to design
T1_TARGET = dp.T1                                      # K

res, hs, x = dp.load_design()
C = dp.cooler_design(res, hs, x)
print(f"Design point: cooler duty {C['Q'] / 1e6:.3f} MW, CO2 {C['m_co2']:.2f} kg/s, air {C['m_air']:.0f} kg/s "
      f"(warms by {dp.AIR['dT']:.0f} K), UA {C['UA'] / 1e6:.4f} MW/K, fan {C['P_fan'] / 1e3:.0f} kW")


def run(label, T_amb_C=25.0, f_air=1.0, scale_ua=False):
    UA = C["UA"] * (ua_ratio(1.0, f_air, RSHARE_CO2) if scale_ua else 1.0)
    T_amb = T_amb_C + K
    r = cl.solve(UA, T_amb, C["m_air"] * f_air, C["co2_in"], C["m_co2"], C["dp"])
    P = cl.fan_power(C["m_air"] * f_air, C["m_air"], T_amb, C["P_fan"], C["T_air_in"])
    return {"case": label, "ambient (C)": T_amb_C, "air flow factor": f_air, "UA used (MW/K)": UA / 1e6,
            "duty (MW)": r["Q"] / 1e6, "T1 (C)": r["T1"] - K, "T1 - ambient (K)": r["T1"] - T_amb, "min dT (K)": r["min_dT"],
            "air outlet (C)": r["T_air_out"] - K, "fan power (kW)": P / 1e3}


rows = []
print("\nA. Design check")
r = run("A design")
rows.append(r)
print(f"  duty {r['duty (MW)']:.4f} MW (design {C['Q'] / 1e6:.4f}), T1 {r['T1 (C)']:.3f} C (design {C['T1'] - K:.3f}), "
      f"min dT {r['min dT (K)']:.2f} K, air outlet {r['air outlet (C)']:.1f} C, fan {r['fan power (kW)']:.0f} kW")

print("\nB. Ambient temperature (air flow, UA fixed)")
for t in (5, 10, 15, 20, 25, 30, 35, 40, 45):
    r = run("B ambient", T_amb_C=float(t))
    rows.append(r)
    print(f"  ambient {t:2d} C: T1 {r['T1 (C)']:.2f} C (T1 - ambient {r['T1 - ambient (K)']:.2f} K), duty {r['duty (MW)']:.2f} MW, "
          f"min dT {r['min dT (K)']:.2f} K")
b = pd.DataFrame([r for r in rows if r["case"] == "B ambient"]).set_index("ambient (C)")
print(f"  slope dT1/dT_ambient: {(b.loc[30.0, 'T1 (C)'] - b.loc[20.0, 'T1 (C)']) / 10.0:.2f} between 20 and 30 C, "
      f"{(b.loc[45.0, 'T1 (C)'] - b.loc[35.0, 'T1 (C)']) / 10.0:.2f} between 35 and 45 C")

print("\nC. Air flow x factor (UA following the air flow)")
for amb in (25.0, 35.0):
    for f in (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5):
        r = run(f"C air flow, ambient {amb:.0f} C", T_amb_C=amb, f_air=f, scale_ua=True)
        rows.append(r)
        print(f"  ambient {amb:.0f} C, air {f:.2f}: T1 {r['T1 (C)']:.2f} C, fan {r['fan power (kW)']:.0f} kW, duty {r['duty (MW)']:.2f} MW")

print(f"\nD. Fan control: air flow needed to hold T1 = {T1_TARGET - K:.0f} C (limit {F_MAX:.1f} x design)")
for t in (5, 10, 15, 20, 25, 26, 27, 28, 30, 33, 35, 40):
    lo, hi = 0.2, 3.0
    r_lo, r_hi = run("D", float(t), lo, True), run("D", float(t), hi, True)
    if r_hi["T1 (C)"] > T1_TARGET - K:
        f_need, note = float("nan"), "T1 cannot be held even at 3 x design air flow"
        r = r_hi
    elif r_lo["T1 (C)"] < T1_TARGET - K:
        f_need, note, r = lo, "T1 stays below the target at the lowest air flow", r_lo
    else:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if run("D", float(t), mid, True)["T1 (C)"] > T1_TARGET - K:
                lo = mid
            else:
                hi = mid
        f_need = hi
        r = run("D", float(t), f_need, True)
        note = "above the fan limit" if f_need > F_MAX else "ok"
    r = dict(r)
    r["case"], r["air flow factor"] = "D fan control", f_need
    if not np.isfinite(f_need) or f_need > 3.0 - 1e-9:
        r["fan power (kW)"] = float("nan")                  # no valid fan setting holds T1: do not plot a fan power
    rows.append(r)
    print(f"  ambient {t:2d} C: air flow factor {f_need:.2f}, T1 {r['T1 (C)']:.2f} C, fan {r['fan power (kW)']:.0f} kW ({note})")

dp.TABLES.mkdir(parents=True, exist_ok=True)
dp.FIGURES.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.round(5).to_csv(dp.TABLES / "cooler_checks.csv", index=False, encoding="utf-8-sig")

TC = 30.98
fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.4))
d = df[df["case"] == "B ambient"]
ax[0].plot(d["ambient (C)"], d["T1 (C)"], color=BLUE, lw=2, marker="o", ms=5)
ax[0].axhline(TC, color="black", lw=1.2, ls="--")
ax[0].text(ax[0].get_xlim()[0] + 0.5, TC + 0.4, "critical temperature", fontsize=8.5)
ax[0].plot([25.0], [T1_TARGET - K], marker="*", ms=14, color=ORANGE, ls="none", label="design (25 C, 32 C)")
ax[0].set_xlabel("Ambient temperature (C)")
ax[0].set_ylabel("Compressor inlet temperature T1 (C)")
ax[0].set_title("T1 against ambient (air flow, UA fixed)", fontsize=10)
ax[0].legend(frameon=False, fontsize=9, loc="upper left")
lines_c = []
for amb, c in ((25.0, BLUE), (35.0, ORANGE)):
    d = df[df["case"] == f"C air flow, ambient {amb:.0f} C"]
    lines_c += ax[1].plot(d["air flow factor"], d["T1 (C)"], color=c, lw=2, marker="o", ms=5, label=f"T1, ambient {amb:.0f} C (left)")
b1 = ax[1].twinx()
b1.plot(d["air flow factor"], d["fan power (kW)"], color=AQUA, lw=1.8, ls=":", marker="d", ms=5, label="fan power (right)")
ax[1].axhline(TC, color="black", lw=1.2, ls="--", label="_critical")
ax[1].set_xlabel("Air flow relative to design")
ax[1].set_ylabel("Compressor inlet temperature T1 (C)")
b1.set_ylabel("Fan power (kW)")
ax[1].set_title("Air flow (UA follows the air flow)", fontsize=10)
ax[1].legend(handles=lines_c + b1.get_lines(), frameon=False, fontsize=8, loc="upper right")
d = df[df["case"] == "D fan control"]
ax[2].plot(d["ambient (C)"], d["air flow factor"], color=BLUE, lw=2, marker="o", ms=5, label="air flow to hold T1 = 32 C (left)")
ax[2].axhline(F_MAX, color="#888888", lw=1, ls="--")
ax[2].text(ax[2].get_xlim()[0] + 0.5, F_MAX + 0.02, f"assumed fan limit {F_MAX:.1f}", fontsize=8.5, color="#555555")
b2 = ax[2].twinx()
b2.plot(d["ambient (C)"], d["fan power (kW)"], color=AQUA, lw=1.8, ls=":", marker="d", ms=5, label="fan power (right)")
ax[2].set_xlabel("Ambient temperature (C)")
ax[2].set_ylabel("Air flow relative to design")
b2.set_ylabel("Fan power (kW)")
ax[2].set_title("Fan control to hold T1 = 32 C", fontsize=10)
ax[2].legend(handles=ax[2].get_lines()[:1] + b2.get_lines(), frameon=False, fontsize=8, loc="upper left")
for a in list(ax) + [b1, b2]:
    a.spines["top"].set_visible(False)
for a in ax:
    a.grid(True, color="#dddddd", lw=0.6)
fig.suptitle("Off-design dry air cooler at the design UA (design ambient 25 C)", fontsize=10)
fig.tight_layout()
fig.savefig(dp.FIGURES / "cooler_checks.png", dpi=170)
print(f"\nSaved: {dp.TABLES / 'cooler_checks.csv'} and {dp.FIGURES / 'cooler_checks.png'}")
