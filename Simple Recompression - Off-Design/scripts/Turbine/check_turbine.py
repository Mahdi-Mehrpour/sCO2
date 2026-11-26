"""Checks of the off-design radial turbine model (constant speed) against the design point and the supplied curve.

A  Design: the model must return the design flow, power and outlet state; prints C_s, nu, A_eff, K_t.
B  Efficiency curve: eta / eta_des from the formula against the table turbine_offdesign_efficiency_curve.csv.
C  Flow law: mass flow and efficiency against pressure ratio (inlet T and outlet p at design, inlet p varied), nozzle law
   and Stodola law side by side. Both pass through the design point and differ away from it.
D  Swallowing capacity: inlet pressure the turbine needs to pass a given mass flow at the design outlet pressure, for three
   inlet temperatures. This is the pressure the compressors must deliver, the matching condition of the cycle.
Output: tables/turbine_checks.csv, figures/turbine_checks.png.
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/Turbine/check_turbine.py" """
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                  # scripts/: design_point.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import design_point as dp
from turbine import NU_DES, RadialTurbine, eta_ratio

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": False})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#888888"
K = dp.K
ETA_T = 0.90                                                          # isentropic efficiency of the design study

res, hs, x = dp.load_design()
m0 = hs["m_co2"]
s7, s8 = res["states"][7], res["states"][8]
tn = RadialTurbine(s7["T"], s7["p"], s8["p"], m0, ETA_T, "nozzle")
ts = RadialTurbine(s7["T"], s7["p"], s8["p"], m0, ETA_T, "stodola")
d = tn.design
print(f"Design point: inlet {s7['T'] - K:.1f} C / {s7['p'] / 1e6:.3f} MPa, outlet {s8['p'] / 1e6:.3f} MPa, beta {tn.beta_des:.3f}, flow {m0:.2f} kg/s")

rows = []
print("\nA. Design check")
o = tn.expand(s7["T"], s7["p"], s8["p"])
print(f"  flow {o['m']:.3f} kg/s (design {m0:.3f}), power {o['power'] / 1e6:.4f} MW (cycle {m0 * res['w_t'] / 1e6:.4f}), "
      f"outlet T {o['T_out'] - K:.3f} C (cycle {s8['T'] - K:.3f}), eta {o['eta']:.4f}, nu {o['nu']:.4f}")
print(f"  isentropic enthalpy drop {d['dh_s'] / 1e3:.2f} kJ/kg, actual {d['w'] / 1e3:.2f} kJ/kg, spouting velocity C_s {d['Cs']:.1f} m/s, "
      f"outlet density {d['rho_out']:.2f} kg/m3, A_eff {d['A_eff'] * 1e3:.4f} x 1e-3 m2, K_t {d['K_t'] * 1e3:.4f} x 1e-3 m2")
print(f"  tip speed at nu = {NU_DES:.4f}: {NU_DES * d['Cs']:.1f} m/s")
rows.append({"case": "A design", "m (kg/s)": o["m"], "power (MW)": o["power"] / 1e6, "T out (C)": o["T_out"] - K, "eta": o["eta"]})

print("\nB. Efficiency curve against the supplied table")
tab = pd.read_csv(HERE / "turbine_offdesign_efficiency_curve.csv")
nu = NU_DES / tab["Cs_over_Cs_des"].to_numpy()
diff = eta_ratio(nu) - tab["eta_over_eta_des"].to_numpy()
print(f"  largest difference formula - table: {np.abs(diff).max():.5f} (rows: {len(tab)})")
print(f"  table nu column vs formula: largest difference {np.abs(nu - tab['nu_U_over_Cs'].to_numpy()).max():.5f}")

print("\nC. Pressure ratio at the design inlet temperature and outlet pressure (inlet pressure varied)")
pr = []
for f in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5):
    p_in = s7["p"] * f
    a, b = tn.expand(s7["T"], p_in, s8["p"]), ts.expand(s7["T"], p_in, s8["p"])
    r = {"case": "C pressure ratio", "p_in factor": f, "beta": a["beta"], "dh_s / design": a["dh_s"] / d["dh_s"], "eta / eta_des": a["eta"] / ETA_T,
         "m nozzle / design": a["m"] / m0, "m Stodola / design": b["m"] / m0, "power nozzle (MW)": a["power"] / 1e6}
    pr.append(r)
    rows.append(r)
    print(f"  p_in x {f:.1f} (beta {a['beta']:.2f}): dh_s {r['dh_s / design']:.2f}, eta/eta_des {r['eta / eta_des']:.3f}, "
          f"flow {r['m nozzle / design']:.3f} (nozzle) {r['m Stodola / design']:.3f} (Stodola), power {r['power nozzle (MW)']:.2f} MW")

print("\nD. Inlet pressure needed to pass a given flow (outlet pressure at design)")
dd = []
for dT in (-50.0, 0.0, 50.0):
    for f in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
        try:
            p = tn.inlet_pressure(m0 * f, s7["T"] + dT, s8["p"])
        except ValueError as e:
            p = float("nan")
        r = {"case": "D swallowing", "T_in (C)": s7["T"] + dT - K, "flow factor": f, "p_in needed (MPa)": p / 1e6}
        dd.append(r)
        rows.append(r)
    print("  inlet " + f"{s7['T'] + dT - K:.0f} C: " + ", ".join(f"{r['flow factor']:.1f}: {r['p_in needed (MPa)']:.2f} MPa" for r in dd if r["T_in (C)"] == s7["T"] + dT - K))

dp.TABLES.mkdir(parents=True, exist_ok=True)
dp.FIGURES.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).round(5).to_csv(dp.TABLES / "turbine_checks.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.4))
hh = np.linspace(0.5, 1.5, 100)
nu_h = NU_DES / np.sqrt(hh)
ax[0].plot(hh, eta_ratio(nu_h), color=BLUE, lw=2, label="formula, 2 nu sqrt(1 - nu^2)")
ax[0].plot(tab["dh_s_over_dh_s_des"], tab["eta_over_eta_des"], marker="o", ms=4, color=ORANGE, ls="none", label="supplied table")
ax[0].axvline(1.0, color=GREY, lw=1, ls="--")
ax[0].set_xlabel("Isentropic enthalpy drop / design")
ax[0].set_ylabel("Efficiency / design efficiency")
ax[0].set_title("Efficiency at constant speed", fontsize=10)
ax[0].legend(frameon=False, fontsize=8.5, loc="lower center")
c = pd.DataFrame(pr)
ax[1].plot(c["beta"], c["m nozzle / design"], color=BLUE, lw=2, marker="o", ms=5, label="nozzle law (Dyreby)")
ax[1].plot(c["beta"], c["m Stodola / design"], color=ORANGE, lw=1.8, ls="--", marker="s", ms=5, mfc="none", label="Stodola law")
ax[1].axvline(tn.beta_des, color=GREY, lw=1, ls="--")
ax[1].set_xlabel("Pressure ratio (inlet pressure varied)")
ax[1].set_ylabel("Mass flow / design")
ax[1].set_title("Flow law at the design inlet T and outlet p", fontsize=10)
ax[1].legend(frameon=False, fontsize=8.5, loc="upper left")
dd = pd.DataFrame(dd)
for dT, cc in ((-50.0, ORANGE), (0.0, BLUE), (50.0, AQUA)):
    q = dd[np.isclose(dd["T_in (C)"], s7["T"] + dT - K)]
    ax[2].plot(q["flow factor"], q["p_in needed (MPa)"], color=cc, lw=2, marker="o", ms=5, label=f"inlet {s7['T'] + dT - K:.0f} C")
ax[2].set_xlabel("Mass flow / design")
ax[2].set_ylabel("Inlet pressure needed (MPa)")
ax[2].set_title("Swallowing capacity (outlet pressure at design)", fontsize=10)
ax[2].legend(frameon=False, fontsize=8.5, loc="upper left")
for a in ax:
    a.grid(True, color="#dddddd", lw=0.6)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
fig.suptitle("Off-design radial turbine at constant speed (dashed line = design point)", fontsize=10)
fig.tight_layout()
fig.savefig(dp.FIGURES / "turbine_checks.png", dpi=170)
print(f"\nSaved: {dp.TABLES / 'turbine_checks.csv'} and {dp.FIGURES / 'turbine_checks.png'}")
