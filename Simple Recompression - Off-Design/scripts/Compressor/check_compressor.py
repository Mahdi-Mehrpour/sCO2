"""Checks of the off-design compressor models (Sandia correlations, Dyreby 2014) against the design point and the supplied map.

A  Design: both compressors must reproduce the design outlet pressure, outlet temperature and power of the cycle model;
   prints the sized hardware (tip speed, diameter, design speed, tip Mach number) and the stage efficiency of the recompressor.
B  Map: psi and eta / eta_des from the formulas against compressor_offdesign_map_SNL.csv (all speeds).
C  Flow at constant speed: outlet pressure, efficiency and flow coefficient against mass flow, speed ratios 0.8 to 1.1
   (inlet state and speed fixed), with the surge limit marked.
D  Speed needed to deliver the design outlet pressure for mass flows of 70 to 120 % of design.
E  Inlet temperature of the main compressor at design flow and design outlet pressure (p1 = 7.7 and 8.3 MPa): the speed
   needed and the efficiency. This is the density collapse of the T1 study seen from the compressor.
Output: tables/compressor_checks.csv, figures/compressor_checks.png.
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/Compressor/check_compressor.py" """
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                  # scripts/: design_point.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

import design_point as dp
from compressor import (FLUID, PHI_DES, PSI_DES, PHI_SURGE, CompressorStage, Recompressor, map_point, state_Tp, state_hp)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": False})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#888888"
K = dp.K
ETA_MC = 0.85

res, hs, x = dp.load_design()
m0 = hs["m_co2"]
st = res["states"]
m_mc, m_rc = (1.0 - x) * m0, x * m0
s1, s2, s10, s4 = st[1], st[2], st[10], st[4]
mc = CompressorStage.from_design(s1["T"], s1["p"], s2["p"], m_mc, ETA_MC)
rc = Recompressor(state_hp(s10["h"], s10["p"]), s4["p"], m_rc, s4["h"])
in1, in10 = state_hp(s1["h"], s1["p"]), state_hp(s10["h"], s10["p"])
rows = []

print(f"Design point: main compressor {m_mc:.2f} kg/s, recompressor {m_rc:.2f} kg/s (x = {x:.4f})")
print("\nA. Design check")
o = mc.operate(in1, m_mc, mc.N_des)
a1 = PropsSI("A", "H", in1["h"], "P", in1["p"], FLUID)
print(f"  main:  outlet {o['p_out'] / 1e6:.4f} MPa (cycle {s2['p'] / 1e6:.4f}), {o['T_out'] - K:.3f} C (cycle {s2['T'] - K:.3f}), "
      f"power {o['power'] / 1e6:.4f} MW (cycle {m_mc * res['w_mc'] / 1e6:.4f}), eta {o['eta']:.4f}, phi {o['phi']:.5f}, psi {o['psi']:.4f}")
print(f"         D {mc.D * 1e3:.1f} mm, N_des {mc.N_des:.0f} rpm, tip speed {mc.U_des:.1f} m/s, tip Mach number {mc.U_des / a1:.3f}, rho_in {in1['rho']:.1f} kg/m3")
rows.append({"case": "A main design", "p_out (MPa)": o["p_out"] / 1e6, "T_out (C)": o["T_out"] - K, "power (MW)": o["power"] / 1e6, "eta": o["eta"]})
r = rc.operate(in10, m_rc, rc.N_des)
a10 = PropsSI("A", "H", in10["h"], "P", in10["p"], FLUID)
print(f"  recompressor: outlet {r['p_out'] / 1e6:.4f} MPa (cycle {s4['p'] / 1e6:.4f}), {r['T_out'] - K:.3f} C (cycle {s4['T'] - K:.3f}), "
      f"power {r['power'] / 1e6:.4f} MW (cycle {m_rc * res['w_rc'] / 1e6:.4f})")
print(f"         stage efficiency {rc.eta_stage:.4f} (cycle model, single stage: {ETA_MC:.2f}), interstage pressure {rc.p_mid_des / 1e6:.3f} MPa, "
      f"pressure ratio {r['stage'][0]['p_out'] / in10['p']:.3f} and {r['p_out'] / r['p_mid']:.3f}")
for i, sg in enumerate(rc.stages, 1):
    ai = PropsSI("A", "H", r["stage"][i - 1]["state_out"]["h"] if i == 2 else in10["h"], "P", r["p_mid"] if i == 2 else in10["p"], FLUID)
    print(f"         stage {i}: D {sg.D * 1e3:.1f} mm, tip speed {sg.U_des:.1f} m/s, tip Mach number {sg.U_des / ai:.3f}")
print(f"         N_des {rc.N_des:.0f} rpm, rho_in {in10['rho']:.1f} kg/m3")
rows.append({"case": "A recompressor design", "p_out (MPa)": r["p_out"] / 1e6, "T_out (C)": r["T_out"] - K, "power (MW)": r["power"] / 1e6, "eta": rc.eta_stage})

print("\nB. Map formulas against the supplied table")
tab = pd.read_csv(HERE / "compressor_offdesign_map_SNL.csv")
mp = [map_point(a, b) for a, b in zip(tab["phi"], tab["N_over_Ndes"])]
dpsi = np.array([q["psi"] for q in mp]) - tab["psi_isentropic"].to_numpy()
deta = np.array([q["eta_ratio"] for q in mp]) - tab["eta_over_etades"].to_numpy()
print(f"  largest difference psi {np.abs(dpsi).max():.5f}, eta/eta_des {np.abs(deta).max():.5f} (rows: {len(tab)}); psi_des {PSI_DES:.4f}")

print("\nC. Outlet pressure against mass flow at constant speed (inlet state fixed)")
fl = np.linspace(0.6, 1.5, 19)
cr = []
for rN in (0.8, 0.9, 1.0, 1.1):
    for f in fl:
        a = mc.operate(in1, m_mc * f, rN * mc.N_des)
        b = rc.operate(in10, m_rc * f, rN * rc.N_des)
        cr.append({"case": "C flow", "N ratio": rN, "flow / design": f, "main p_out (MPa)": a["p_out"] / 1e6, "main eta": a["eta"],
                   "main phi": a["phi"], "main surge": a["surge"], "main overload": a["overload"],
                   "rc p_out (MPa)": b["p_out"] / 1e6, "rc eta": b["eta"], "rc surge": b["surge"], "rc overload": b["overload"]})
C = pd.DataFrame(cr)
for rN in (0.9, 1.0):
    q = C[(C["N ratio"] == rN)]
    ok = q[~q["main surge"]]
    print(f"  N ratio {rN}: main surge below flow {ok['flow / design'].min():.2f} of design; p_out at 0.8 / 1.0 / 1.2 flow = "
          + " / ".join(f"{q.loc[(q['flow / design'] - v).abs().idxmin(), 'main p_out (MPa)']:.2f}" for v in (0.8, 1.0, 1.2)) + " MPa")
rows += cr

print("\nD. Speed needed for the design outlet pressure")
dr = []
for f in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
    row = {"case": "D speed", "flow / design": f}
    for tag, mach, st_in, m, pt in (("main", mc, in1, m_mc, s2["p"]), ("rc", rc, in10, m_rc, s4["p"])):
        try:
            N = mach.speed_for(st_in, m * f, pt)
            q = mach.operate(st_in, m * f, N)
            row.update({f"{tag} N ratio": N / mach.N_des, f"{tag} eta": q["eta"], f"{tag} power (MW)": q["power"] / 1e6,
                        f"{tag} surge": q["surge"], f"{tag} overload": q["overload"], f"{tag} outside": q["outside"]})
        except ValueError:
            row.update({f"{tag} N ratio": np.nan})
    dr.append(row)
    print(f"  flow {f:.1f}: main N {row.get('main N ratio', np.nan):.3f} eta {row.get('main eta', np.nan):.3f}"
          f"{' SURGE' if row.get('main surge') else ''} | recompressor N {row.get('rc N ratio', np.nan):.3f} eta {row.get('rc eta', np.nan):.3f}"
          f"{' SURGE' if row.get('rc surge') else ''}")
rows += dr

print("\nE. Main compressor inlet temperature at design flow and design outlet pressure")
er = []
for p1 in (7.7e6, 8.3e6):
    for T in np.arange(30.0, 38.01, 1.0):
        s = state_Tp(T + K, p1)
        row = {"case": "E T1", "p1 (MPa)": p1 / 1e6, "T1 (C)": T, "rho (kg/m3)": s["rho"]}
        try:
            N = mc.speed_for(s, m_mc, s2["p"])
            q = mc.operate(s, m_mc, N)
            row.update({"N ratio": N / mc.N_des, "eta": q["eta"], "power (MW)": q["power"] / 1e6, "phi": q["phi"],
                        "surge": q["surge"], "overload": q["overload"], "outside": q["outside"]})
        except ValueError:
            row.update({"N ratio": np.nan})
        er.append(row)
E = pd.DataFrame(er)
print(E.round(3).to_string(index=False))
rows += er

pd.DataFrame(rows).to_csv(dp.TABLES / "compressor_checks.csv", index=False)

fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
pf = np.linspace(0.6, 1.7, 100) * PHI_DES
for rN, c in zip((0.7, 0.8, 0.9, 1.0, 1.1, 1.2), ("#9bb9e6", "#6e9ede", BLUE, "#0f3f7d", ORANGE, AQUA)):
    q = [map_point(f, rN) for f in pf]
    t = tab[tab["N_over_Ndes"] == rN]
    ax[0, 0].plot(pf / PHI_DES, [v["psi"] / PSI_DES for v in q], color=c, label=f"N/N$_{{des}}$ = {rN}")
    ax[0, 0].plot(t["phi_over_phides"], t["psi_over_psides"], ".", color=c, ms=3)
    ax[0, 1].plot(pf / PHI_DES, [v["eta_ratio"] for v in q], color=c)
    ax[0, 1].plot(t["phi_over_phides"], t["eta_over_etades"], ".", color=c, ms=3)
for a_, yl in ((ax[0, 0], r"$\psi\,/\,\psi_{des}$"), (ax[0, 1], r"$\eta\,/\,\eta_{des}$")):
    a_.set_xlabel(r"$\phi\,/\,\phi_{des}$"); a_.set_ylabel(yl); a_.set_ylim(0, 1.15); a_.set_xlim(0.6, 1.7)
    a_.grid(alpha=0.3); a_.axvline(PHI_SURGE / PHI_DES, color=GREY, ls="--", lw=1)
ax[0, 0].legend(fontsize=8, frameon=False)
ax[0, 0].set_title("Map: formula (lines) and supplied table (dots)")
ax[0, 1].set_title("Dashed: surge limit (at design speed)")

for rN, c in zip((0.8, 0.9, 1.0, 1.1), (BLUE, AQUA, ORANGE, "#0f3f7d")):
    q = C[C["N ratio"] == rN]
    ax[0, 2].plot(q["flow / design"], q["main p_out (MPa)"], color=c, label=f"main, N/N$_{{des}}$ = {rN}")
    qs = q[q["main surge"]]
    ax[0, 2].plot(qs["flow / design"], qs["main p_out (MPa)"], "o", color=c, ms=4, mfc="none")
    ax[0, 2].plot(q["flow / design"], q["rc p_out (MPa)"], color=c, ls=":")
ax[0, 2].axhline(s2["p"] / 1e6, color=GREY, lw=0.8)
ax[0, 2].set_ylim(0, 40); ax[0, 2].set_xlabel("mass flow / design"); ax[0, 2].set_ylabel("outlet pressure (MPa)")
ax[0, 2].set_title("Constant speed (solid: main, dotted: recompressor; circles: surge)")
ax[0, 2].legend(fontsize=8, frameon=False); ax[0, 2].grid(alpha=0.3)

D = pd.DataFrame(dr)
ax[1, 0].plot(D["flow / design"], D["main N ratio"], "o-", color=BLUE, label="main compressor")
ax[1, 0].plot(D["flow / design"], D["rc N ratio"], "s-", color=ORANGE, label="recompressor")
ax[1, 0].axhspan(0.6, 1.2, color=GREY, alpha=0.12, label="range of the data")
ax[1, 0].set_xlabel("mass flow / design"); ax[1, 0].set_ylabel("speed N / N$_{des}$"); ax[1, 0].set_title("Speed that holds the design outlet pressure")
ax[1, 0].legend(fontsize=8, frameon=False); ax[1, 0].grid(alpha=0.3)

for p1, c in ((7.7, BLUE), (8.3, ORANGE)):
    q = E[E["p1 (MPa)"] == p1]
    ax[1, 1].plot(q["T1 (C)"], q["N ratio"], "o-", color=c, label=f"p1 = {p1} MPa")
    ax[1, 2].plot(q["T1 (C)"], q["eta"], "o-", color=c, label=f"p1 = {p1} MPa")
ax[1, 1].axhspan(0.6, 1.2, color=GREY, alpha=0.12)
ax[1, 1].set_xlabel("main compressor inlet temperature (°C)"); ax[1, 1].set_ylabel("speed N / N$_{des}$")
ax[1, 1].set_title("Speed for design flow and 25 MPa"); ax[1, 1].legend(fontsize=8, frameon=False); ax[1, 1].grid(alpha=0.3)
ax[1, 2].set_xlabel("main compressor inlet temperature (°C)"); ax[1, 2].set_ylabel("efficiency")
ax[1, 2].set_title("Efficiency at that speed"); ax[1, 2].legend(fontsize=8, frameon=False); ax[1, 2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(dp.FIGURES / "compressor_checks.png", dpi=200)
print(f"\nWrote {dp.TABLES / 'compressor_checks.csv'} and {dp.FIGURES / 'compressor_checks.png'}")
