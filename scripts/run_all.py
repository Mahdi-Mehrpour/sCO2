"""Regenerate every table and figure of the current design, in order.
  python scripts/run_all.py          fast steps (seconds)
  python scripts/run_all.py --all    also the gas temperature sensitivity (minutes)
The design optimisation (run_splitflow_opt_adv.py) is not part of it."""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

FAST = [("Plant performance", "scripts/run_plant_splitflow.py"),
        ("Exergy analysis, states and figures", "scripts/run_exergy_splitflow.py"),
        ("Cooling options", "scripts/run_cooling_splitflow.py"),
        ("Schematic", "scripts/run_schematic_splitflow.py"),
        ("Assumptions", "scripts/make_assumptions_splitflow.py")]
SLOW = [("Gas temperature sensitivity", "scripts/run_sensitivity_gas.py")]

steps = FAST + (SLOW if "--all" in sys.argv else [])
for name, script in steps:
    t0 = time.time()
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED: {name} ({script})\n{r.stderr[-1500:]}")
        sys.exit(1)
    print(f"OK  {name:40s} {time.time() - t0:7.1f} s")

files = sorted(Path("tables").glob("[0-9][0-9]_*.csv"))
with pd.ExcelWriter("tables/tables.xlsx") as writer:
    for f in files:
        df = pd.read_csv(f)
        sheet = f.stem[:31]
        df.to_excel(writer, sheet_name=sheet, index=False)
        ws = writer.sheets[sheet]
        for i, col in enumerate(df.columns):
            width = max(len(str(col)), *(len(str(v)) for v in df[col])) + 2
            ws.column_dimensions[chr(65 + i)].width = min(width, 60)
print(f"Excel: tables/tables.xlsx ({len(files)} sheets)")