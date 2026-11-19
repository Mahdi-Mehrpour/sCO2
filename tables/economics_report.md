# sCO2 EAF waste-heat unit: economic and exergoeconomic results

## Sizes
W_T 11.709 MW | W_C 2.409 MW | W_el,gross 11.245 | motor 2.561 | tower 0.205 | gas fan 0.246 | **W_net 8.233 MW**
| HX | Q (MW) | effective LMTD (K) | UA (kW/K) |
|---|---|---|---|
| HTR | 8.496 | 13.56 | 626.5 |
| LTR | 18.652 | 44.80 | 416.3 |
| MH | 15.232 | 60.09 | 253.5 |
| BH | 14.592 | 73.45 | 198.7 |
| cool | 20.523 | 17.81 | 1152.1 |

## Purchased equipment cost and cost rates
| Component | PEC (k$) | band (k$) | Z_dot ($/h) | Source |
|---|---|---|---|---|
| Turbine | 4,088 | 2,780-6,173 | 84.7 | [W19] radial turbine, W_sh |
| Gearbox+generator | 1,028 | 832-1,264 | 21.3 | [W19] gearbox (W_T) + generator (W_e) |
| Compressor+motor | 2,990 | 1,794-4,426 | 61.9 | [W19] IG compressor + synchronous motor |
| HTR | 2,307 | 1,615-2,999 | 47.8 | [WDS16] 2500 $/(kW/K), end-point UA |
| LTR | 1,533 | 1,073-1,993 | 31.8 | [WDS16] 2500 $/(kW/K), end-point UA |
| Main heater | 1,867 | 1,493-3,734 | 38.7 | [WDS16] 5000 $/(kW/K), fin tube |
| Branch heater | 1,463 | 1,171-2,926 | 30.3 | [WDS16] 5000 $/(kW/K), fin tube |
| Cooler | 2,885 | 2,020-3,751 | 59.8 | [WDS16] 1700 $/(kW/K), shell & tube |
| Cooling tower | 898 | 628-1,167 | 18.6 | [P19] modified Zanker (band ASSUMED, not in source) |
| Makeup water | | | 52.6 | 291,847 m3/yr |

PEC 19.06 M$ | **TCI 24.78 M$** (3,010 $/kWe net; [WDS16] FOAK benchmark 1717-2019 $/kWe)
Cross-check [W19] recuperator: HTR 1,642 k$, LTR 1,206 k$

## Economic indicators (capital band in brackets)
- LCOE: **58.8 $/MWh** [45.2-81.5]
- NPV: **11.27 M$** [-0.79 to 18.54]
- IRR: **14.1 %** [7.7-21.5]
- Simple payback: **6.31 yr** [4.37-9.68]
- Discounted payback: **9.36 yr** [5.65-inf]

## SPECO
Unit cost of electricity c_el = 15.07 $/GJ = **54.2 $/MWh** (levelized, no degradation). Balance residual 0.00e+00 $/h.
| Component | E_F (MW) | E_P (MW) | E_D (MW) | eps | c_F ($/GJ) | c_P ($/GJ) | Z_dot ($/h) | C_D ($/h) | f (%) | r (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| Turbine | 12.276 | 11.709 | 0.566 | 0.954 | 11.40 | 13.97 | 84.7 | 23.2 | 78.5 | 22.5 |
| Gearbox+generator | 11.709 | 11.245 | 0.464 | 0.960 | 13.97 | 15.07 | 21.3 | 23.3 | 47.7 | 7.9 |
| Compressor+motor | 2.561 | 2.094 | 0.467 | 0.817 | 15.07 | 26.65 | 61.9 | 25.4 | 71.0 | 76.9 |
| HTR | 4.620 | 4.440 | 0.180 | 0.961 | 11.40 | 14.86 | 47.8 | 7.4 | 86.6 | 30.3 |
| LTR | 7.438 | 5.793 | 1.645 | 0.779 | 11.40 | 16.17 | 31.8 | 67.6 | 32.0 | 41.8 |
| Main heater | 9.695 | 9.049 | 0.646 | 0.933 | 0.21 | 1.41 | 38.7 | 0.5 | 98.7 | 565.4 |
| Branch heater | 6.669 | 4.992 | 1.677 | 0.749 | 0.21 | 1.97 | 30.3 | 1.3 | 95.9 | 826.8 |

Mixer E_D 0.0363 MW; cooler exergy loss 1.998 MW (cost charged to the product).

| Stream | C ($/h) | c ($/GJ) |
|---|---|---|
| 1 | 945.9 | 14.95 |
| 2 | 1146.7 | 16.19 |
| 3 | 1031.0 | 16.18 |
| 5 | 1519.3 | 13.87 |
| 6 | 1756.7 | 14.00 |
| 7 | 1802.8 | 11.40 |
| 8 | 1298.8 | 11.40 |
| 9 | 1109.1 | 11.40 |
| 10 | 803.8 | 11.40 |
| 11 | 488.3 | 10.63 |
| gas stack (g3) | 0.8 | |

## Validity warnings
- gearbox: SP=11.71 outside [W19] range 4-10