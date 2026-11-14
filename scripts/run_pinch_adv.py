import numpy as np
import matplotlib.pyplot as plt
from sco2.cycle_adv import recompression_adv

# ----------------------------------------------------------------------
# 1) Pinch check of the default cycle (heat duty set by effectiveness only)
# ----------------------------------------------------------------------
res = recompression_adv()
print(f"Default cycle: eta = {res['eta']*100:.2f} %  (must still be 40.97 %)\n")

print(f"{'':4s} {'Q (kJ/kg)':>10s} {'cold end dT':>12s} {'hot end dT':>11s} "
      f"{'min dT (K)':>11s} {'at f':>6s} {'UA (kJ/kg.K)':>13s}")
for name in ("HTR", "LTR"):
    p = res["pinch"][name]
    print(f"{name:4s} {p['q']/1e3:10.2f} {p['dT_cold_end']:12.2f} "
          f"{p['dT_hot_end']:11.2f} {p['min_dT']:11.2f} {p['f_min']:6.2f} "
          f"{p['UA']/1e3:13.3f}")
    if p["min_dT"] <= 0:
        print(f"     WARNING: temperature cross in the {name}")
    elif p["min_dT"] < p["dT_cold_end"] - 0.5 and p["min_dT"] < p["dT_hot_end"] - 0.5:
        print(f"     NOTE: the pinch of the {name} is INSIDE the exchanger, "
              f"not at an end")

# Q-T diagrams
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name in zip(axes, ("HTR", "LTR")):
    p = res["pinch"][name]
    ax.plot(p["Q"] / 1e3, p["T_hot"] - 273.15, "r-", label="Hot side")
    ax.plot(p["Q"] / 1e3, p["T_cold"] - 273.15, "b-", label="Cold side")
    i = int(np.argmin(p["dT"]))
    ax.plot([p["Q"][i] / 1e3] * 2, [p["T_cold"][i] - 273.15, p["T_hot"][i] - 273.15],
            "k--", lw=1.5, label=f"min dT = {p['min_dT']:.1f} K")
    ax.set_xlabel("Heat transferred (kJ per kg of hot flow)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(name)
    ax.grid(True)
    ax.legend()
fig.tight_layout()
fig.savefig("figures/adv/recuperator_QT.png", dpi=150)
print("\nFigure saved: figures/adv/recuperator_QT.png")

# ----------------------------------------------------------------------
# 2) Effect of enforcing a minimum temperature difference
# ----------------------------------------------------------------------
dTs = [3, 5, 8, 10, 15, 20]
etas, rows = [], []
for d in dTs:
    r = recompression_adv(dT_pinch=d)
    etas.append(r["eta"] * 100)
    rows.append((d, r["eta"] * 100, r["q_htr"] / 1e3, r["q_ltr"] / 1e3,
                 r["pinch"]["HTR"]["min_dT"], r["pinch"]["LTR"]["min_dT"],
                 (r["pinch"]["HTR"]["UA"] + r["pinch"]["LTR"]["UA"]) / 1e3))

print("\nEnforced minimum temperature difference:")
print(f"{'dT_pinch':>8s} {'eta (%)':>8s} {'Q_HTR':>8s} {'Q_LTR':>8s} "
      f"{'minHTR':>7s} {'minLTR':>7s} {'UA tot':>8s}")
for row in rows:
    print(f"{row[0]:8.1f} {row[1]:8.2f} {row[2]:8.2f} {row[3]:8.2f} "
          f"{row[4]:7.2f} {row[5]:7.2f} {row[6]:8.3f}")

plt.figure()
plt.plot(dTs, etas, "o-", label="Pinch enforced")
plt.axhline(res["eta"] * 100, color="gray", ls="--", label="Effectiveness only")
plt.xlabel("Minimum temperature difference in recuperators (K)")
plt.ylabel("Thermal efficiency (%)")
plt.grid(True)
plt.legend()
plt.savefig("figures/adv/eta_vs_dT_pinch.png", dpi=150)
print("Figure saved: figures/adv/eta_vs_dT_pinch.png")