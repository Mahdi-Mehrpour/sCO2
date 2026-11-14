from sco2.cycle_rc import recompression_brayton
from sco2.plotting import plot_ts_rc

# Recompression cycle with default inputs
res = recompression_brayton()
plot_ts_rc(res, save_path="figures/rc/ts_diagram_rc.png")
print("Figure saved: figures/rc/ts_diagram_rc.png")

