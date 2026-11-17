# sCO2 EAF waste-heat unit: economic and exergoeconomic results

## Sizes
W_T 8.690 MW | W_C 1.788 MW | W_el,gross 8.346 | motor 1.901 | tower 0.121 | gas fan 0.127 | **W_net 6.197 MW**
| HX | Q (MW) | effective LMTD (K) | UA (kW/K) |
|---|---|---|---|
| HTR | 7.348 | 16.03 | 458.3 |
| LTR | 15.927 | 27.23 | 585.0 |
| MH | 11.565 | 91.06 | 127.0 |
| BH | 7.443 | 56.22 | 132.4 |
| cool | 12.105 | 14.29 | 847.2 |

## Purchased equipment cost and cost rates
| Component | PEC (k$) | band (k$) | Z_dot ($/h) | Source |
|---|---|---|---|---|
| Turbine | 3,221 | 2,190-4,863 | 53.5 | [W19] radial turbine, W_sh |
| Gearbox+generator | 910 | 737-1,119 | 15.1 | [W19] gearbox (W_T) + generator (W_e) |
| Compressor+motor | 2,624 | 1,575-3,884 | 43.6 | [W19] IG compressor + synchronous motor |
| HTR | 1,688 | 1,181-2,194 | 28.1 | [WDS16] 2500 $/(kW/K), end-point UA |
| LTR | 2,154 | 1,508-2,800 | 35.8 | [WDS16] 2500 $/(kW/K), end-point UA |
| Main heater | 935 | 748-1,871 | 15.5 | [WDS16] 5000 $/(kW/K), fin tube |
| Branch heater | 975 | 780-1,950 | 16.2 | [WDS16] 5000 $/(kW/K), fin tube |
| Cooler | 2,121 | 1,485-2,758 | 35.3 | [WDS16] 1700 $/(kW/K), shell & tube |
| Cooling tower | 575 | 403-748 | 9.6 | [P19] modified Zanker (band ASSUMED, not in source) |
| Makeup water | | | 31.9 | 172,137 m3/yr |

PEC 15.20 M$ | **TCI 19.76 M$** (3,189 $/kWe net; [WDS16] FOAK benchmark 1717-2019 $/kWe)
Cross-check [W19] recuperator: HTR 1,297 k$, LTR 1,559 k$

## Economic indicators (capital band in brackets)
- LCOE: **50.1 $/MWh** [38.5-67.8]
- NPV: **15.05 M$** [6.15 to 20.92]
- IRR: **13.4 %** [7.5-20.8]
- Simple payback: **6.59 yr** [4.52-9.86]
- Discounted payback: **8.28 yr** [5.29-14.39]

## SPECO
Unit cost of electricity c_el = 12.74 $/GJ = **45.9 $/MWh** (levelized, no degradation). Balance residual -1.14e-13 $/h.
| Component | E_F (MW) | E_P (MW) | E_D (MW) | eps | c_F ($/GJ) | c_P ($/GJ) | Z_dot ($/h) | C_D ($/h) | f (%) | r (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| Turbine | 9.111 | 8.690 | 0.420 | 0.954 | 9.58 | 11.76 | 53.5 | 14.5 | 78.7 | 22.7 |
| Gearbox+generator | 8.690 | 8.346 | 0.344 | 0.960 | 11.76 | 12.74 | 15.1 | 14.6 | 50.9 | 8.4 |
| Compressor+motor | 1.901 | 1.554 | 0.347 | 0.817 | 12.74 | 23.39 | 43.6 | 15.9 | 73.3 | 83.5 |
| HTR | 3.948 | 3.787 | 0.161 | 0.959 | 9.58 | 12.05 | 28.1 | 5.5 | 83.5 | 25.7 |
| LTR | 5.679 | 4.767 | 0.912 | 0.839 | 9.58 | 13.50 | 35.8 | 31.5 | 53.2 | 40.9 |
| Main heater | 7.605 | 6.861 | 0.744 | 0.902 | 0.14 | 0.78 | 15.5 | 0.4 | 97.6 | 459.7 |
| Branch heater | 3.390 | 2.642 | 0.748 | 0.779 | 0.14 | 1.88 | 16.2 | 0.4 | 97.7 | 1243.5 |

Mixer E_D 0.0678 MW; cooler exergy loss 0.805 MW (cost charged to the product).

| Stream | C ($/h) | c ($/GJ) |
|---|---|---|
| 1 | 560.0 | 11.92 |
| 2 | 690.8 | 13.14 |
| 3 | 747.5 | 13.25 |
| 5 | 940.4 | 11.91 |
| 6 | 1104.6 | 11.93 |
| 7 | 1124.0 | 9.58 |
| 8 | 809.8 | 9.58 |
| 9 | 673.6 | 9.58 |
| 10 | 477.7 | 9.58 |
| 11 | 192.9 | 8.45 |
| gas stack (g3) | 0.3 | |