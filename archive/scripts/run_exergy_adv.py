import matplotlib.pyplot as plt
from sco2.cycle_adv import recompression_adv
from sco2.exergy_adv import exergy_analysis_adv

res = recompression_adv()
ex = exergy_analysis_adv(res)

print(f"Heat source temperature = {ex['T_source']-273.15:.1f} C")
print(f"Exergy of heat input    = {ex['ex_heat']/1e3:.2f} kJ/kg")

print(f"Net work                = {res['w_net']/1e3:.2f} kJ/kg")
print(f"Thermal efficiency      = {res['eta']*100:.2f} %")
print(f"Exergy efficiency       = {ex['eta_ex']*100:.2f} %")
print()
print(f"{'Component':18s} {'kJ/kg':>8s} {'% of heat exergy':>18s}")
for k, v in ex["destruction"].items():
    print(f"{k:18s} {v/1e3:8.2f} {v/ex['ex_heat']*100:18.2f}")

# Check: exergy in = work out + all destruction (must be ~0)
bal = ex["ex_heat"] - res["w_net"] - ex["total"]
print(f"\nExergy balance check = {bal:.6f} J/kg (should be ~0)")

# Bar chart of destruction, sorted from largest to smallest
items = sorted(ex["destruction"].items(), key=lambda kv: kv[1], reverse=True)
names = [k for k, _ in items]
vals = [v / ex["ex_heat"] * 100 for _, v in items]
plt.figure(figsize=(8, 5))
plt.bar(names, vals)
plt.ylabel("Exergy destruction (% of heat exergy)")
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("figures/adv/exergy_destruction.png", dpi=150)
print("Figure saved: figures/adv/exergy_destruction.png")
