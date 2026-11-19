import json
from pathlib import Path

import matplotlib.pyplot as plt

from sco2.heat_source_adv import EAF
from sco2.parasitics_splitflow import PARASITIC
from sco2.plant_splitflow import evaluate_plant, DRIVE
from sco2.schematic_splitflow import draw_schematic_splitflow

# Editable text in the SVG file (useful for polishing in draw.io or Inkscape)
plt.rcParams["svg.fonttype"] = "none"

if DRIVE != "motor":
    raise SystemExit("The schematic draws a motor-driven compressor; "
                     f"DRIVE is '{DRIVE}'.")

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"], check_pinch=False)
res, hs, par, cp = P["res"], P["hs"], P["par"], P["cooler"]

gb, gen, mot = (PARASITIC["eta_gearbox"], PARASITIC["eta_generator"],
                PARASITIC["eta_motor"])
elec = {"gas_dp": PARASITIC["gas_dp_total"],
        "P_gen": par["W_turbine"] * gb * gen,
        "P_motor": par["W_compressor"] / (gb * mot),
        "fan_tower": cp["P_aux"], "fan_gas": par["fan_gas"],
        "loss_gb": par["loss_gearbox"], "loss_gen": par["loss_generator"],
        "loss_mot": par["loss_motor"], "net_shaft": par["net_shaft"],
        "net_el": P["net_el"], "eta_el": P["eta_el"], "makeup": cp["flow"]}

check = (elec["P_gen"] - elec["P_motor"] - elec["fan_tower"]
         - elec["fan_gas"] - elec["net_el"])
print(f"Net electrical power: {P['net_el'] / 1e6:.4f} MW")
print(f"Generator {elec['P_gen'] / 1e6:.4f} - motor {elec['P_motor'] / 1e6:.4f} "
      f"- tower {elec['fan_tower'] / 1e6:.4f} - gas fan {elec['fan_gas'] / 1e6:.4f} "
      f"MW; check = {check / 1e6:.6f} MW (should be 0)")

out = Path("figures")
out.mkdir(exist_ok=True)
fig, ax = draw_schematic_splitflow(res, hs, EAF, elec=elec)
for ext in ("png", "svg", "pdf"):
    fig.savefig(out / f"schematic.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved: {out}/schematic.{ext}")