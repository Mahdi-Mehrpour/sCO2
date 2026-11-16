import numpy as np
import matplotlib.pyplot as plt
from sco2.cycle import recuperated_brayton
from sco2.cycle_rc import recompression_brayton

simple = recuperated_brayton()
rc = recompression_brayton()

print("Recompression cycle states:")
for k, s in rc["states"].items():
    print(f"  {k:2d}: T={s['T']-273.15:7.1f} C  p={s['p']/1e6:5.2f} MPa  h={s['h']/1e3:7.1f} kJ/kg")
print(f"Simple recuperated: eta = {simple['eta']*100:.2f} %")
print(f"Recompression (x={rc['x']}): eta = {rc['eta']*100:.2f} %")

# Energy balance check: q_in - w_net - q_out should be ~0
bal = rc["q_in"] - rc["w_net"] - rc["q_out"]
print(f"Energy balance check = {bal:.4f} J/kg (should be ~0)")

# Effect of the recompression fraction
xs = np.linspace(0.0, 0.5, 11)
etas = [recompression_brayton(x=v)["eta"] * 100 for v in xs]
plt.plot(xs, etas, "o-", label="Recompression")
plt.axhline(simple["eta"] * 100, color="gray", ls="--", label="Simple recuperated")
plt.xlabel("Recompression fraction x")
plt.ylabel("Thermal efficiency (%)")
plt.grid(True)
plt.legend()
plt.savefig("figures/rc/eta_vs_x.png", dpi=150)
print("Figure saved: figures/rc/eta_vs_x.png")
