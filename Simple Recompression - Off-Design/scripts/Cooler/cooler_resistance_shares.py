"""Main cooler (sCO2 -> cooling water), straight-channel PCHE on both sides, fed by a wet cooling tower.

Design point only. The CO2 inlet state (state 10), outlet state (state 1), flow (1 - x) m and the pressure drop are read
from the cycle through design_point (no hard-coded design values). Geometry, correlation constants and the tower and
water-loop assumptions are in cooler_reference_data.csv (next to this script).

1) Cooler: water flow chosen so that the minimum CO2-water temperature difference equals pinch_min; then per-segment
   resistances and shares (CO2 / wall / water) on n_seg equal-duty segments with local properties.
2) Cooling tower: Merkel model sized at the design point (cooling_tower.py), with a check at other wet-bulb temperatures.

CO2-side correlations (--correlation):
  dang_hihara (default): Dang & Hihara (2004), Int. J. Refrig. 27, 736-747, Eqs. (21)-(27).
  pitla: Pitla, Groll & Ramadhyani (2002), Int. J. Refrig. 25, 887-895, Eqs. (1)-(5).
  gnielinski: bulk properties only.
Water side: Gnielinski (laminar Nu = 4.089 below Re 2300). The wall-temperature-dependent CO2 correlations iterate on the
CO2-side wall temperature of each segment.
Output: tables/cooler_resistance_shares[_<tag>].csv (one row per segment) and the printed summary.
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/Cooler/cooler_resistance_shares.py" """
import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # scripts folder: design_point
sys.path.insert(0, str(HERE))            # this folder first: cooling_tower

import pandas as pd
import CoolProp.CoolProp as CP

import design_point as dp
from cooling_tower import CoolingTower, merkel_number

ap = argparse.ArgumentParser()
ap.add_argument("--correlation", default="dang_hihara", choices=["dang_hihara", "pitla", "gnielinski"])
ap.add_argument("--tag", default="", help="suffix of the output file")
args = ap.parse_args()
CORR = args.correlation

ref = pd.read_csv(HERE / "cooler_reference_data.csv").set_index("parameter")["value"].astype(float)
d_ch, t_plate, k_wall, G_c_des = ref["d_ch"], ref["t_plate"], ref["k_wall"], ref["G_co2_des"]
F_MULT, N_SEG, PINCH_MIN = ref["f_mult"], int(ref["n_seg"]), ref["pinch_min"]
T_wb_des, approach, dT_loop = ref["T_wb_des"], ref["approach"], ref["dT_loop"]
LG_des, n_merkel, P_W = ref["LG_des"], ref["n_merkel"], ref["p_water"]

A_ch = math.pi * d_ch ** 2 / 8
P_w = math.pi * d_ch / 2 + d_ch
Dh = 4 * A_ch / P_w
R_wall = (t_plate - d_ch / 2) / (k_wall * P_w)          # K.m/W per channel pair per metre

res, hs, x = dp.load_design()
D = dp.water_cooler_design(res, hs, x)
m_c, Q = D["m_co2"], D["Q"]
h_in, p_in = D["co2_in"]["h"], D["co2_in"]["p"]
h_out, p_out = D["co2_out"]["h"], D["co2_out"]["p"]


def co2_ph(p, h):
    g = lambda q: CP.PropsSI(q, "P", p, "H", h, "CO2")
    return dict(T=g("T"), rho=g("D"), mu=g("V"), k=g("L"), cp=g("C"), h=h)


def co2_pT(p, T):
    g = lambda q: CP.PropsSI(q, "P", p, "T", T, "CO2")
    return dict(T=T, rho=g("D"), mu=g("V"), k=g("L"), cp=g("C"), h=g("H"))


def water(T):
    g = lambda q: CP.PropsSI(q, "P", P_W, "T", T, "Water")
    return dict(rho=g("D"), mu=g("V"), k=g("L"), cp=g("C"))


def gnielinski(Re, Pr, c=1.0):
    """Gnielinski with Filonenko friction; c = 1.0 (standard) or 1.07 (Pitla Eq. 2)."""
    if Re < 2300:
        return 4.089
    f = (0.79 * math.log(Re) - 1.64) ** -2
    return (f / 8) * (Re - 1000) * Pr / (c + 12.7 * math.sqrt(f / 8) * (Pr ** (2 / 3) - 1))


RHO_INLET = co2_ph(p_in, h_in)["rho"]


def h_co2(G, p, b, Tw):
    """CO2-side HTC for bulk state b and CO2-side wall temperature Tw (K)."""
    Re_b = G * Dh / b["mu"]
    if CORR == "gnielinski":
        return gnielinski(Re_b, b["cp"] * b["mu"] / b["k"]) * b["k"] / Dh
    w = co2_pT(p, Tw)
    if CORR == "pitla":
        Nu_b = gnielinski(Re_b, b["cp"] * b["mu"] / b["k"], 1.07)
        Re_w = w["rho"] * (G / RHO_INLET) * Dh / w["mu"]
        Nu_w = gnielinski(Re_w, w["cp"] * w["mu"] / w["k"], 1.07)
        return 0.5 * (Nu_b + Nu_w) * (w["k"] / b["k"]) * b["k"] / Dh
    cp_bar = (b["h"] - w["h"]) / (b["T"] - Tw) if abs(b["T"] - Tw) > 1e-4 else b["cp"]
    f_ = co2_pT(p, 0.5 * (b["T"] + Tw))
    if b["cp"] >= cp_bar:
        Pr = b["cp"] * b["mu"] / b["k"]
    elif b["mu"] / b["k"] >= f_["mu"] / f_["k"]:
        Pr = cp_bar * b["mu"] / b["k"]
    else:
        Pr = cp_bar * f_["mu"] / f_["k"]
    Re_f = G * Dh / f_["mu"]
    ff = (1.82 * math.log10(Re_f) - 1.64) ** -2
    Nu = (ff / 8) * (Re_b - 1000) * Pr / (1.07 + 12.7 * math.sqrt(ff / 8) * (Pr ** (2 / 3) - 1))
    return Nu * f_["k"] / Dh


n_ch = m_c / (G_c_des * A_ch)
G_c = m_c / (n_ch * A_ch)
T_w_in = T_wb_des + approach + dT_loop


def profile(m_w):
    """CO2 and water temperatures (C) at the segment boundaries (i = 0 cold end) for a water flow."""
    pts = []
    for i in range(N_SEG + 1):
        fr = i / N_SEG
        h = h_out + fr * (h_in - h_out)
        p = p_out + fr * (p_in - p_out)
        pts.append((h, p, co2_ph(p, h)["T"] - 273.15, T_w_in + m_c * (h - h_out) / (m_w * 4180.0)))
    return pts


lo, hi = Q / (4180 * 60), Q / (4180 * 0.5)               # larger water flow -> larger pinch
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if min(pt[2] - pt[3] for pt in profile(mid)) < PINCH_MIN:
        lo = mid
    else:
        hi = mid
m_w = hi
pts = profile(m_w)
T_w_out = pts[-1][3]
G_w = m_w / (n_ch * A_ch)

rows, L_tot, dp_c, dp_w, warn = [], 0.0, 0.0, 0.0, 0
for i in range(N_SEG):
    a, b_ = pts[i], pts[i + 1]
    hm, pm = 0.5 * (a[0] + b_[0]), 0.5 * (a[1] + b_[1])
    bulk = co2_ph(pm, hm)
    Tb = bulk["T"]
    Twat = 0.5 * (a[3] + b_[3]) + 273.15
    wp = water(Twat)
    Re_w = G_w * Dh / wp["mu"]
    Pr_w = wp["cp"] * wp["mu"] / wp["k"]
    h_w = gnielinski(Re_w, Pr_w) * wp["k"] / Dh
    R_w = 1 / (h_w * P_w)
    Twall = 0.5 * (Tb + Twat)
    for _ in range(30):
        h_c = h_co2(G_c, pm, bulk, Twall)
        R_c = 1 / (h_c * P_w)
        R = R_c + R_wall + R_w
        Twall_new = Tb - (Tb - Twat) * R_c / R
        if abs(Twall_new - Twall) < 1e-3:
            break
        Twall = 0.5 * (Twall + Twall_new)
    dT1, dT2 = a[2] - a[3], b_[2] - b_[3]
    lmtd = (dT2 - dT1) / math.log(dT2 / dT1) if abs(dT2 - dT1) > 1e-6 else dT1
    UA_i = (Q / N_SEG) / lmtd
    L_seg = UA_i * R / n_ch
    L_tot += L_seg
    Re_c = G_c * Dh / bulk["mu"]
    fc = (0.79 * math.log(Re_c) - 1.64) ** -2 / 4 if Re_c > 2300 else 16 / Re_c
    fw = (0.79 * math.log(Re_w) - 1.64) ** -2 / 4 if Re_w > 2300 else 15.78 / Re_w
    dp_c += F_MULT * fc * (L_seg / Dh) * G_c ** 2 / (2 * bulk["rho"])
    dp_w += F_MULT * fw * (L_seg / Dh) * G_w ** 2 / (2 * wp["rho"])
    warn += Re_w < 2300 or Re_c < 2300
    rows.append({"seg": i + 1, "f_from": i / N_SEG, "f_to": (i + 1) / N_SEG, "Q_i (W)": Q / N_SEG,
                 "T_co2_mean (C)": Tb - 273.15, "T_water_mean (C)": Twat - 273.15, "T_wall_co2 (C)": Twall - 273.15,
                 "dT_min_seg (K)": min(dT1, dT2), "UA_i (W/K)": UA_i, "L_seg (m)": L_seg,
                 "f_co2": R_c / R, "f_wall": R_wall / R, "f_water": R_w / R, "h_co2 (W/m2K)": h_c, "h_water (W/m2K)": h_w,
                 "Re_co2": Re_c, "Re_water": Re_w, "cp_co2 (J/kgK)": bulk["cp"], "rho_co2": bulk["rho"],
                 "mu_co2": bulk["mu"], "k_co2": bulk["k"], "rho_water": wp["rho"], "mu_water": wp["mu"],
                 "k_water": wp["k"], "cp_water": wp["cp"]})

df = pd.DataFrame(rows)
out = dp.TABLES / f"cooler_resistance_shares{'_' + args.tag if args.tag else ''}.csv"
dp.TABLES.mkdir(exist_ok=True)
df.to_csv(out, index=False)

UA = df["UA_i (W/K)"].sum()
wavg = lambda k: float((df[k] * df["UA_i (W/K)"]).sum() / UA)
i_pk = int(df["cp_co2 (J/kgK)"].idxmax())
print(f"=== MAIN COOLER ({CORR}) ===")
print(f"Design from the cycle: duty {Q / 1e6:.3f} MW, CO2 {m_c:.2f} kg/s, state 10 {D['co2_in']['T'] - dp.K:.1f} C "
      f"{p_in / 1e6:.3f} MPa -> state 1 {D['T1'] - dp.K:.1f} C {p_out / 1e6:.3f} MPa")
print(f"UA {UA / 1e6:.3f} MW/K, water {m_w:.1f} kg/s ({T_w_in:.1f} -> {T_w_out:.1f} C), minimum dT {PINCH_MIN} K by design")
print(f"Channels per side {n_ch:,.0f}, G CO2/water {G_c:.0f}/{G_w:.0f} kg/m2s, Dh {Dh * 1e3:.3f} mm")
print(f"Shares (UA-weighted): CO2 {wavg('f_co2'):.2f}, wall {wavg('f_wall'):.2f}, water {wavg('f_water'):.2f}")
print(f"cp peak in segment {i_pk + 1} (CO2 {df['T_co2_mean (C)'][i_pk]:.2f} C): h_co2 {df['h_co2 (W/m2K)'][i_pk]:.0f}, "
      f"CO2 share {df['f_co2'][i_pk]:.2f}, water share {df['f_water'][i_pk]:.2f}")
print(f"Hot end: CO2 share {df['f_co2'].iloc[-1]:.2f}, water share {df['f_water'].iloc[-1]:.2f}; "
      f"CO2 outlet single-phase check: T1 {D['T1'] - dp.K:.1f} C at {p_out / 1e6:.3f} MPa (critical 31.0 C / 7.38 MPa)")
print(f"Channel length {L_tot:.3f} m; friction dp CO2 {dp_c / 1e3:.1f} kPa ({dp_c / p_in * 100:.2f} % of inlet, "
      f"cycle assumes {D['dp'] * 100:.1f} %), water {dp_w / 1e3:.1f} kPa; {int(warn)} segments with Re < 2300")

tower = CoolingTower(T_wb_des, T_w_in - dT_loop, T_w_out - dT_loop, m_w, LG_des, n_merkel)
print("\n=== COOLING TOWER (Merkel) ===")
print(f"Design: T_wb {T_wb_des:.1f} C, approach {approach:.1f} K, range {T_w_out - T_w_in:.1f} K, water {m_w:.1f} kg/s, "
      f"air {tower.m_air_des:.1f} kg/s (L/G {LG_des}), Me {tower.Me_des:.3f}, c {tower.c:.3f}")
for Twb in (10.0, 15.0, 20.0, 25.0, 28.0):
    Tc = tower.cold_water(Twb, m_w, tower.m_air_des, T_w_out - dT_loop)
    print(f"  same flows and hot water: T_wb {Twb:4.1f} C -> cold water {Tc:.2f} C (approach {Tc - Twb:.2f} K)")
print(f"\nSaved: {out}")
