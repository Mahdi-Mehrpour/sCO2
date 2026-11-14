import numpy as np
import matplotlib.pyplot as plt
from sco2.cycle import recuperated_brayton

# Run the base case with default parameters
res = recuperated_brayton()

# Print the six cycle states (T in C, p in MPa, h in kJ/kg)
print("States:")
for k, s in res["states"].items():
    print(f"  {k}: T={s['T']-273.15:7.1f} C  p={s['p']/1e6:5.2f} MPa  h={s['h']/1e3:7.1f} kJ/kg")

# Print net specific work and thermal efficiency
print(f"w_net = {res['w_net']/1e3:.1f} kJ/kg")
print(f"eta   = {res['eta']*100:.2f} %")

# Effect of pressure ratio on thermal efficiency
prs = np.linspace(1.5, 4.0, 15)   # 15 pressure ratios between 1.5 and 4.0
etas = [recuperated_brayton(pr=p)["eta"] * 100 for p in prs]

# Plot and save the figure
plt.plot(prs, etas, "o-")
plt.xlabel("Pressure ratio")
plt.ylabel("Thermal efficiency (%)")
plt.grid(True)
plt.savefig("figures/eta_vs_pr.png", dpi=150)
print("Figure saved: figures/eta_vs_pr.png")