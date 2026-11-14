from sco2.cycle import recuperated_brayton
from sco2.plotting import plot_ts

# Base case with default inputs
res = recuperated_brayton()
plot_ts(res, save_path="figures/simple/ts_diagram.png")
print("Figure saved: figures/simple/ts_diagram.png")
