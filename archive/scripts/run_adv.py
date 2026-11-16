import numpy as np
import matplotlib.pyplot as plt
from sco2.cycle_rc import recompression_brayton      # old cycle, for comparison only
from sco2.cycle_adv import recompression_adv
from sco2.plotting_adv import plot_ts_adv

ideal = recompression_brayton()
adv = recompression_adv()

print("States of the advanced cycle:")
for k, s in adv["states"].items():
    print(f"  {k:2d}: T={s['T']-273.15:7.1f} C  p={s['p']/1e6:6.3f} MPa  h={s['h']/1e3:7.1f} kJ/kg")

print(f"Recompression without pressure drops: eta = {ideal['eta']*100:.2f} %")
print(f"Advanced cycle (with pressure drops): eta = {adv['eta']*100:.2f} %")
print(f"Turbine pressure ratio = {adv['pr_turbine']:.3f} (compressor pr = 2.6)")

# Check 1: energy balance
bal = adv["q_in"] - adv["w_net"] - adv["q_out"]
print(f"Energy balance check = {bal:.4f} J/kg (should be ~0)")

# Check 2: with all drops at zero, the result must equal the old cycle
zero = recompression_adv(dp_heater=0, dp_cooler=0, dp_ltr_cold=0,
                         dp_ltr_hot=0, dp_htr_cold=0, dp_htr_hot=0)
print(f"Zero-drop check: {zero['eta']*100:.2f} % vs {ideal['eta']*100:.2f} % (should match)")

# Effect of the size of the pressure drops (scale factor on all default drops)
scales = np.linspace(0, 3, 13)
etas = []
for sc in scales:
    r = recompression_adv(dp_heater=0.02*sc, dp_cooler=0.01*sc,
                          dp_ltr_cold=0.01*sc, dp_ltr_hot=0.01*sc,
                          dp_htr_cold=0.01*sc, dp_htr_hot=0.01*sc)
    etas.append(r["eta"] * 100)
plt.figure()
plt.plot(scales, etas, "o-")
plt.xlabel("Pressure-drop scale (1 = default drops)")
plt.ylabel("Thermal efficiency (%)")
plt.grid(True)
plt.savefig("figures/adv/eta_vs_dp_scale.png", dpi=150)
print("Figure saved: figures/adv/eta_vs_dp_scale.png")

plot_ts_adv(adv, save_path="figures/adv/ts_diagram_adv.png")
print("Figure saved: figures/adv/ts_diagram_adv.png")
