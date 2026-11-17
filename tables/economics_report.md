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
| Turbine | 3,221 | 2,190-4,863 | 66.3 | [W19] radial turbine, W_sh |
| Gearbox+generator | 910 | 737-1,119 | 18.7 | [W19] gearbox (W_T) + generator (W_e) |
| Compressor+motor | 2,624 | 1,575-3,884 | 54.0 | [W19] IG compressor + synchronous motor |
| HTR | 1,688 | 1,181-2,194 | 34.7 | [WDS16] 2500 $/(kW/K), end-point UA |
| LTR | 2,154 | 1,508-2,800 | 44.3 | [WDS16] 2500 $/(kW/K), end-point UA |
| Main heater | 935 | 748-1,871 | 19.2 | [WDS16] 5000 $/(kW/K), fin tube |
| Branch heater | 975 | 780-1,950 | 20.1 | [WDS16] 5000 $/(kW/K), fin tube |
| Cooler | 2,121 | 1,485-2,758 | 43.7 | [WDS16] 1700 $/(kW/K), shell & tube |
| Cooling tower | 575 | 403-748 | 11.8 | [P19] modified Zanker (band ASSUMED, not in source) |
| Makeup water | | | 31.0 | 172,137 m3/yr |

PEC 15.20 M$ | **TCI 19.76 M$** (3,189 $/kWe net; [WDS16] FOAK benchmark 1717-2019 $/kWe)
Cross-check [W19] recuperator: HTR 1,297 k$, LTR 1,559 k$

## Economic indicators (capital band in brackets)
- LCOE: **60.1 $/MWh** [45.3-82.5]
- NPV: **7.98 M$** [-1.00 to 13.89]
- IRR: **13.4 %** [7.5-20.8]
- Simple payback: **6.59 yr** [4.52-9.86]
- Discounted payback: **9.97 yr** [5.89-inf]

## SPECO
Unit cost of electricity c_el = 15.40 $/GJ = **55.4 $/MWh** (levelized, no degradation). Balance residual 0.00e+00 $/h.
| Component | E_F (MW) | E_P (MW) | E_D (MW) | eps | c_F ($/GJ) | c_P ($/GJ) | Z_dot ($/h) | C_D ($/h) | f (%) | r (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| Turbine | 9.111 | 8.690 | 0.420 | 0.954 | 11.52 | 14.19 | 66.3 | 17.4 | 79.2 | 23.2 |
| Gearbox+generator | 8.690 | 8.346 | 0.344 | 0.960 | 14.19 | 15.40 | 18.7 | 17.6 | 51.6 | 8.5 |
| Compressor+motor | 1.901 | 1.554 | 0.347 | 0.817 | 15.40 | 28.50 | 54.0 | 19.2 | 73.7 | 85.0 |
| HTR | 3.948 | 3.787 | 0.161 | 0.959 | 11.52 | 14.55 | 34.7 | 6.7 | 83.9 | 26.4 |
| LTR | 5.679 | 4.767 | 0.912 | 0.839 | 11.52 | 16.30 | 44.3 | 37.8 | 54.0 | 41.6 |
| Main heater | 7.605 | 6.861 | 0.744 | 0.902 | 0.17 | 0.97 | 19.2 | 0.5 | 97.7 | 470.7 |
| Branch heater | 3.390 | 2.642 | 0.748 | 0.779 | 0.17 | 2.33 | 20.1 | 0.5 | 97.8 | 1273.3 |

Mixer E_D 0.0678 MW; cooler exergy loss 0.805 MW (cost charged to the product).

| Stream | C ($/h) | c ($/GJ) |
|---|---|---|
| 1 | 667.6 | 14.21 |
| 2 | 827.0 | 15.73 |
| 3 | 897.3 | 15.91 |
| 5 | 1128.9 | 14.29 |
| 6 | 1327.3 | 14.33 |
| 7 | 1351.2 | 11.52 |
| 8 | 973.4 | 11.52 |
| 9 | 809.8 | 11.52 |
| 10 | 574.3 | 11.52 |
| 11 | 231.6 | 10.15 |
| gas stack (g3) | 0.3 | |