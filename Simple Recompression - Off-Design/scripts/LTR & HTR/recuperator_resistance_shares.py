"""PCHE recuperators (HTR and LTR): thermal-resistance shares (hot side / cold side / wall), segment by segment.

The design point is read from the cycle model (design_point.load_design), so the states, flows, duties and pressures are exactly
those of the design study. Each exchanger is split into the 30 equal-duty segments of the cycle model, CO2 properties are
evaluated at the local enthalpy and pressure (pressure linear), and for every segment the script computes the heat transfer
coefficient of both sides, the wall resistance, the three shares, the segment conductance UA_i and the channel length.
The geometry and the correlation constants are in recuperator_reference_data.csv (next to this script).

Default model: Ngo, Kato, Nikitin & Ishizuka (2007), Exp. Therm. Fluid Sci. 32, 560-570, zigzag correlation:
    Nu = 0.1696 Re^0.629 Pr^0.317,   f = 0.1924 Re^-0.091      valid 3500 < Re < 22000 and 0.75 < Pr < 2.2 (Eqs. 29 and 31)
    f is a FANNING factor (Ngo Eq. 23), so dp = 4 f (L / Dh) G^2 / (2 rho), L = length of the flow path (SI units throughout)
  Ngo measured a zigzag-FIN microchannel exchanger (Dh 1.09 mm); here it is applied to the continuous semicircular zigzag channels
  of a Heatric PCHE, whose geometry is Nikitin, Kato & Ngo (2006), Int. J. Refrig. 29, 807-814, hot side (d 1.9 mm, plate 1.63 mm,
  pitch 4.5 mm, 32.5 deg), Dh = 1.16 mm. Nikitin's own correlations cover only Re 2800-12100 and have no Pr term.
Alternative (--correlation gnielinski): straight channels, laminar Nu = 4.089 below Re 2300.

Off-design use, per segment and side:  h/h_des = (m/m_des)^a (mu_des/mu)^a (k/k_des) (Pr/Pr_des)^b   (a = 0.629, b = 0.317)
    1/UA_i = (1/UA_i,des) [ f_hot (h_des/h)_hot + f_cold (h_des/h)_cold + f_wall ]

Output: tables/HTR_resistance_shares[_<tag>].csv and tables/LTR_resistance_shares[_<tag>].csv (one row per segment).
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/LTR & HTR/recuperator_resistance_shares.py" """
import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                  # scripts/: design_point.py
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

import design_point as dp

ap = argparse.ArgumentParser()
ap.add_argument("--correlation", default="ngo_zigzag", choices=["ngo_zigzag", "gnielinski"])
ap.add_argument("--tag", default="", help="suffix of the output files")
args = ap.parse_args()
CORR = args.correlation

ref = pd.read_csv(HERE / "recuperator_reference_data.csv").set_index("parameter")["value"].astype(float)
d_ch, t_plate, k_wall, G_hot = ref["d_ch"], ref["t_plate"], ref["k_wall"], ref["G_hot"]
pitch_long, zz_angle = ref["pitch_long"], ref["zz_angle_deg"]
FM = ref["f_mult"] if CORR == "ngo_zigzag" else 4.0     # Gnielinski branch returns Fanning factors

A_ch = math.pi * d_ch ** 2 / 8
P_w = math.pi * d_ch / 2 + d_ch                        # wetted perimeter, also used as the heat-transfer perimeter
Dh = 4 * A_ch / P_w
R_wall = (t_plate - d_ch / 2) / (k_wall * P_w)         # K.m/W per channel pair per metre

res, hs, x = dp.load_design()
D = dp.recuperator_design(res, hs, x)
st = res["states"]
OUTLETS = {"HTR": (9, 6), "LTR": (10, 3)}              # (hot outlet state, cold outlet state)


def props(h, p):
    g = lambda q: CP.PropsSI(q, "H", h, "P", p, "CO2")
    return dict(T=g("T"), rho=g("D"), mu=g("V"), k=g("L"), cp=g("C"))


def h_and_f(G, pr):
    Re = G * Dh / pr["mu"]
    Pr = pr["cp"] * pr["mu"] / pr["k"]
    if CORR == "ngo_zigzag":
        Nu = ref["Nu_c"] * Re ** ref["Nu_a"] * Pr ** ref["Nu_b"]
        f = ref["f_c"] * Re ** ref["f_d"]
        re_ok, pr_ok = ref["Re_min"] <= Re <= ref["Re_max"], ref["Pr_min"] <= Pr <= ref["Pr_max"]
    else:
        if Re < 2300:
            Nu, f = 4.089, 15.78 / Re                # Fanning factor of the semicircular duct (Shah & London)
        else:
            fd = (0.79 * math.log(Re) - 1.64) ** -2
            Nu = (fd / 8) * (Re - 1000) * Pr / (1 + 12.7 * math.sqrt(fd / 8) * (Pr ** (2 / 3) - 1))
            f = fd / 4
        re_ok, pr_ok = (Re >= 2300 or Re < 2000), True
    return Nu * pr["k"] / Dh, f, Re, Pr, re_ok, pr_ok


for name in ("HTR", "LTR"):
    d = D[name]
    hi, ci = d["hot_in"], d["cold_in"]
    ho, co = st[OUTLETS[name][0]], st[OUTLETS[name][1]]
    mh, mc, Q = d["m_hot"], d["m_cold"], d["q"]
    N = dp.N_SEG
    n_ch = mh / (G_hot * A_ch)                         # channels per side
    Gh, Gc = mh / (n_ch * A_ch), mc / (n_ch * A_ch)
    fb = np.linspace(0.0, 1.0, N + 1)                  # 0 = cold end (hot outlet, cold inlet)
    hh, ph = ho["h"] + fb * (hi["h"] - ho["h"]), ho["p"] + fb * (hi["p"] - ho["p"])
    hc, pc = ci["h"] + fb * (co["h"] - ci["h"]), ci["p"] + fb * (co["p"] - ci["p"])
    Th = np.array([props(a, b)["T"] for a, b in zip(hh, ph)])
    Tc = np.array([props(a, b)["T"] for a, b in zip(hc, pc)])
    print(f"\n=== {name} ({CORR}) ===")
    print(f"Duty {Q / 1e6:.3f} MW (hot-side balance {mh * (hi['h'] - ho['h']) / 1e6:.3f}, cold-side {mc * (co['h'] - ci['h']) / 1e6:.3f}), "
          f"hot {Th[-1] - dp.K:.1f} -> {Th[0] - dp.K:.1f} C at {mh:.2f} kg/s, cold {Tc[0] - dp.K:.1f} -> {Tc[-1] - dp.K:.1f} C at {mc:.2f} kg/s")
    rows, L_tot, dp_h, dp_c = [], 0.0, 0.0, 0.0
    ua_log = ua_arith = 0.0
    for i in range(N):
        mh_, mc_ = props(0.5 * (hh[i] + hh[i + 1]), 0.5 * (ph[i] + ph[i + 1])), props(0.5 * (hc[i] + hc[i + 1]), 0.5 * (pc[i] + pc[i + 1]))
        hH, fH, ReH, PrH, reH, prH = h_and_f(Gh, mh_)
        hC, fC, ReC, PrC, reC, prC = h_and_f(Gc, mc_)
        RH, RC = 1 / (hH * P_w), 1 / (hC * P_w)
        R = RH + RC + R_wall
        dT1, dT2 = Th[i] - Tc[i], Th[i + 1] - Tc[i + 1]
        lmtd = dT1 if abs(dT2 - dT1) < 1e-6 else (dT2 - dT1) / math.log(dT2 / dT1)
        Q_i = Q / N
        UA_i = Q_i / lmtd
        ua_log += UA_i
        ua_arith += Q_i / (0.5 * (dT1 + dT2))
        L_seg = UA_i * R / n_ch                        # channel length of the segment, m
        L_tot += L_seg
        dp_h += FM * fH * (L_seg / Dh) * Gh ** 2 / (2 * mh_["rho"])
        dp_c += FM * fC * (L_seg / Dh) * Gc ** 2 / (2 * mc_["rho"])
        rows.append({"seg": i + 1, "f_from": fb[i], "f_to": fb[i + 1], "Q_i (W)": Q_i, "T_hot_mean (C)": 0.5 * (Th[i] + Th[i + 1]) - dp.K,
                     "T_cold_mean (C)": 0.5 * (Tc[i] + Tc[i + 1]) - dp.K, "UA_i (W/K)": UA_i, "L_seg (m)": L_seg,
                     "G_hot": Gh, "G_cold": Gc, "Re_hot": ReH, "Re_cold": ReC, "Pr_hot": PrH, "Pr_cold": PrC,
                     "h_hot (W/m2K)": hH, "h_cold (W/m2K)": hC, "f_hot": RH / R, "f_cold": RC / R, "f_wall": R_wall / R,
                     "rho_hot": mh_["rho"], "mu_hot": mh_["mu"], "k_hot": mh_["k"], "cp_hot": mh_["cp"],
                     "rho_cold": mc_["rho"], "mu_cold": mc_["mu"], "k_cold": mc_["k"], "cp_cold": mc_["cp"],
                     "Re_ok_hot": reH, "Pr_ok_hot": prH, "Re_ok_cold": reC, "Pr_ok_cold": prC})
    df = pd.DataFrame(rows)
    w = df["UA_i (W/K)"]
    print("seg  T_hot  T_cold   Re_h    Re_c   Pr_h  Pr_c   h_hot  h_cold  f_hot f_cold f_wall  UA_i(kW/K)")
    for r in rows:
        print(f"{r['seg']:3d} {r['T_hot_mean (C)']:6.1f} {r['T_cold_mean (C)']:7.1f} {r['Re_hot']:7.0f} {r['Re_cold']:7.0f} {r['Pr_hot']:5.2f} {r['Pr_cold']:5.2f} "
              f"{r['h_hot (W/m2K)']:7.0f} {r['h_cold (W/m2K)']:7.0f} {r['f_hot']:6.3f} {r['f_cold']:6.3f} {r['f_wall']:6.3f} {r['UA_i (W/K)'] / 1e3:9.1f}")
    wm = lambda c: float(np.average(df[c], weights=w))
    print(f"Shares (UA-weighted): f_hot {wm('f_hot'):.3f}, f_cold {wm('f_cold'):.3f}, f_wall {wm('f_wall'):.3f}; per-segment f_hot "
          f"{df['f_hot'].min():.2f}-{df['f_hot'].max():.2f}, f_cold {df['f_cold'].min():.2f}-{df['f_cold'].max():.2f}")
    print(f"UA: sum of segments (log-mean) {ua_log / 1e6:.4f} MW/K, arithmetic mean {ua_arith / 1e6:.4f} MW/K, design_point value {d['UA'] / 1e6:.4f} MW/K")
    print(f"Channels per side {n_ch:,.0f}, G hot / cold {Gh:.0f} / {Gc:.0f} kg/m2s, Dh {Dh * 1e3:.3f} mm; channel length {L_tot:.3f} m "
          f"(core ~{L_tot * math.cos(math.radians(zz_angle)):.3f} m), heat-transfer area per side {L_tot * n_ch * P_w:.0f} m2")
    print(f"Core friction dp (multiplier {FM:g}): hot {dp_h / 1e3:.1f} kPa ({dp_h / hi['p'] * 100:.2f} % of inlet; cycle assumes "
          f"{d['dp_hot'] * 100:.1f} %), cold {dp_c / 1e3:.1f} kPa ({dp_c / ci['p'] * 100:.2f} %; cycle assumes {d['dp_cold'] * 100:.1f} %) [headers excluded]")
    print(f"Re range hot {df['Re_hot'].min():,.0f}-{df['Re_hot'].max():,.0f}, cold {df['Re_cold'].min():,.0f}-{df['Re_cold'].max():,.0f}; "
          f"Pr range hot {df['Pr_hot'].min():.2f}-{df['Pr_hot'].max():.2f}, cold {df['Pr_cold'].min():.2f}-{df['Pr_cold'].max():.2f}")
    print(f"Segments outside the correlation range: hot Re {int((~df['Re_ok_hot']).sum())}, hot Pr {int((~df['Pr_ok_hot']).sum())}, "
          f"cold Re {int((~df['Re_ok_cold']).sum())}, cold Pr {int((~df['Pr_ok_cold']).sum())} (of {N})")
    out = dp.TABLES / f"{name}_resistance_shares{'_' + args.tag if args.tag else ''}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
