import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt

from sco2.cycle_adv import splitflow_adv
from sco2.heat_source_adv import EAF, GasStream, couple_cascade
from sco2.schematic_splitflow import draw_schematic_splitflow

# Editable text in the SVG file (useful for polishing in draw.io or Inkscape)
plt.rcParams["svg.fonttype"] = "none"

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
P = {k: v.default for k, v in inspect.signature(splitflow_adv).parameters.items()}
P.update({"y": opt["y"], "T_b": opt["T_b"], "pr": opt["pr"], "check_pinch": False})

gas = GasStream(EAF["X"], EAF["p"], T0=298.15)
res = splitflow_adv(**P)
hs = couple_cascade(res, gas, EAF["T_in"], EAF["m_gas"],
                    EAF["dT_pinch_heater"], EAF["dT_pinch_heater"],
                    EAF["T_stack_min"])

out = Path("figures")
out.mkdir(parents=True, exist_ok=True)
fig, ax = draw_schematic_splitflow(res, hs, EAF)
for ext in ("png", "svg", "pdf"):
    fig.savefig(out / f"schematic.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved: {out}/schematic.{ext}")