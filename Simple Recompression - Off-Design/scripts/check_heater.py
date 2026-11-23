"""Checks of the off-design gas-heater model against the design point.

A  Design: design UA, design gas and CO2 flows and inlets must give the design duty, gas outlet and T7.
B  Gas flow: gas flow x factor (gas temperature, CO2 inlet and CO2 flow kept), once with the UA fixed and once with the
   UA following the gas flow (gas-side-dominated heater, assumed 90 % of the resistance on the gas side).
C  Gas temperature: gas inlet temperature varied, flows and UA fixed.
D  CO2 flow: CO2 flow x factor at the design gas, UA fixed: shows how the turbine inlet temperature T7 depends on the CO2
   flow (T7 is a result of the heater, not a free choice).
Output: tables/heater_checks.csv, figures/heater_checks.png.
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/check_heater.py" """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import design_point as dp
import heater as ht
from ua_scaling import ua_ratio

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": False})
BLUE, ORANGE, AQUA, STACK = "#2a78d6", "#eb6834", "#1baf7a", "#c0392b"
K = dp.K
RSHARE_GAS = 0.9                                                  # placeholder: gas side carries 90 % of the resistance
STACK_MIN = dp.EAF["T_stack_min"]

res, hs, x = dp.load_design()
H = dp.heater_design(res, hs)
print(f"Design point: CO2 {H['m_co2']:.2f} kg/s, gas {H['m_gas']:.2f} kg/s at {H['T_gas_in'] - K:.2f} C, heater UA {H['UA'] / 1e6:.4f} MW/K")


def run(label, f_gas=1.0, f_co2=1.0, dT_gas_in=0.0, scale_ua=False):
    UA = H["UA"] * (ua_ratio(f_gas, f_co2, RSHARE_GAS) if scale_ua else 1.0)
    T_in = H["T_gas_in"] + dT_gas_in
    r = ht.solve(UA, dp.GAS, T_in, H["m_gas"] * f_gas, H["co2_in"], H["m_co2"] * f_co2, H["dp"])
    return {"case": label, "gas flow factor": f_gas, "CO2 flow factor": f_co2, "gas inlet (C)": T_in - K,
            "UA used (MW/K)": UA / 1e6, "duty (MW)": r["Q"] / 1e6, "effectiveness": r["eps"], "min dT (K)": r["min_dT"],
            "stack (C)": r["T_gas_out"] - K, "T7 (C)": r["T_co2_out"] - K,
            "stack below limit": bool(r["T_gas_out"] < STACK_MIN)}


rows = []
print("\nA. Design check")
r = run("A design")
rows.append(r)
print(f"  duty {r['duty (MW)']:.4f} MW (design {H['Q'] / 1e6:.4f}), stack {r['stack (C)']:.3f} C (design {H['T_gas_out'] - K:.3f}), "
      f"T7 {r['T7 (C)']:.3f} C (design {H['T7'] - K:.3f}), min dT {r['min dT (K)']:.3f} K (design {H['min_dT']:.3f})")

print("\nB. Gas flow x factor (gas temperature, CO2 flow fixed)")
for f in (1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
    for sc in (False, True):
        r = run("B gas flow, UA " + ("scaled" if sc else "fixed"), f_gas=f, scale_ua=sc)
        rows.append(r)
        print(f"  gas {f:.1f} ({'UA scaled' if sc else 'UA fixed '}): duty {r['duty (MW)']:.2f} MW, stack {r['stack (C)']:.0f} C, "
              f"T7 {r['T7 (C)']:.0f} C, min dT {r['min dT (K)']:.1f} K" + ("  [stack below limit]" if r["stack below limit"] else ""))

print("\nC. Gas inlet temperature (flows, UA fixed)")
for dT in (-100.0, -50.0, 0.0, 50.0, 100.0):
    r = run("C gas temperature", dT_gas_in=dT)
    rows.append(r)
    print(f"  gas {r['gas inlet (C)']:.0f} C: duty {r['duty (MW)']:.2f} MW, stack {r['stack (C)']:.0f} C, T7 {r['T7 (C)']:.0f} C, "
          f"min dT {r['min dT (K)']:.1f} K" + ("  [stack below limit]" if r["stack below limit"] else ""))

print("\nD. CO2 flow x factor (design gas, UA fixed)")
for f in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
    r = run("D CO2 flow", f_co2=f)
    rows.append(r)
    print(f"  CO2 {f:.1f}: duty {r['duty (MW)']:.2f} MW, T7 {r['T7 (C)']:.0f} C, stack {r['stack (C)']:.0f} C, min dT {r['min dT (K)']:.1f} K"
          + ("  [stack below limit]" if r["stack below limit"] else ""))

dp.TABLES.mkdir(parents=True, exist_ok=True)
dp.FIGURES.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.round(5).to_csv(dp.TABLES / "heater_checks.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.4))
for lab, c, ls in (("B gas flow, UA fixed", BLUE, "-"), ("B gas flow, UA scaled", ORANGE, "--")):
    d = df[df["case"] == lab].sort_values("gas flow factor")
    ax[0].plot(d["gas flow factor"], d["duty (MW)"], color=c, lw=2, ls=ls, marker="o", ms=5, label=lab.replace("B gas flow, ", ""))
ax[0].set_xlabel("Gas flow relative to design")
ax[0].set_ylabel("Heat to the CO2 (MW)")
ax[0].set_title("Gas flow (gas temperature, CO2 flow fixed)", fontsize=10)
ax[0].legend(frameon=False, fontsize=9, title="UA", title_fontsize=9)
d = df[df["case"] == "C gas temperature"].sort_values("gas inlet (C)")
ax[1].plot(d["gas inlet (C)"], d["T7 (C)"], color=AQUA, lw=2, marker="o", ms=5, label="T7 (left)")
ax[1].set_xlabel("Gas inlet temperature (C)")
ax[1].set_ylabel("Turbine inlet temperature T7 (C)")
b1 = ax[1].twinx()
b1.plot(d["gas inlet (C)"], d["stack (C)"], color=STACK, lw=1.8, ls=":", marker="d", ms=6, label="stack T (right)")
b1.set_ylabel("Gas stack temperature (C)")
ax[1].set_title("Gas temperature (flows, UA fixed)", fontsize=10)
ax[1].legend(handles=ax[1].get_lines() + b1.get_lines(), frameon=False, fontsize=8.5, loc="upper left")
d = df[df["case"] == "D CO2 flow"].sort_values("CO2 flow factor")
ax[2].plot(d["CO2 flow factor"], d["T7 (C)"], color=AQUA, lw=2, marker="o", ms=5, label="T7 (left)")
ax[2].set_xlabel("CO2 flow relative to design")
ax[2].set_ylabel("Turbine inlet temperature T7 (C)")
b2 = ax[2].twinx()
b2.plot(d["CO2 flow factor"], d["min dT (K)"], color=ORANGE, lw=1.8, ls="--", marker="s", ms=5, mfc="none", label="min dT (right)")
b2.set_ylabel("Heater minimum temperature difference (K)")
ax[2].set_title("CO2 flow (design gas, UA fixed)", fontsize=10)
ax[2].legend(handles=ax[2].get_lines() + b2.get_lines(), frameon=False, fontsize=8.5, loc="upper right")
for a in list(ax) + [b1, b2]:
    a.spines["top"].set_visible(False)
for a in ax:
    a.grid(True, color="#dddddd", lw=0.6)
fig.suptitle("Off-design gas heater model at the design UA", fontsize=10)
fig.tight_layout()
fig.savefig(dp.FIGURES / "heater_checks.png", dpi=170)
print(f"\nSaved: {dp.TABLES / 'heater_checks.csv'} and {dp.FIGURES / 'heater_checks.png'}")
