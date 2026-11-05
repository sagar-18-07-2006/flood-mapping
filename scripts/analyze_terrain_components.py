import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

BASE = Path("data/processed")
TERRAIN = BASE / "terrain"
OUT = BASE / "terrain_susceptibility"

layers = {
    "HAND": TERRAIN / "hand.tif",
    "Slope": TERRAIN / "slope_degrees.tif",
    "TWI": TERRAIN / "twi.tif",
    "Flow accumulation": TERRAIN / "flow_accumulation_cells.tif",
}

sar_path = OUT / "sar_candidate_fraction_30m.tif"

with rasterio.open(sar_path) as ds:
    sar = ds.read(1)
    sar_valid = np.isfinite(sar) & (sar != ds.nodata)

records = []

for name, path in layers.items():
    with rasterio.open(path) as ds:
        values = ds.read(1).astype("float64")
        valid = np.isfinite(values) & (values != ds.nodata)

    mask = valid & sar_valid

    if name == "Flow accumulation":
        values = np.log1p(values)

    values = values[mask]
    candidate_fraction = sar[mask]

    # Five equal-count terrain groups.
    groups = pd.qcut(
        values,
        q=5,
        labels=False,
        duplicates="drop"
    )

    for group in np.unique(groups):
        selected = groups == group
        fractions = candidate_fraction[selected]

        records.append({
            "terrain_variable": name,
            "quintile": int(group) + 1,
            "cell_count": int(selected.sum()),
            "mean_sar_candidate_fraction": float(
                fractions.mean()
            ),
            "candidate_presence_percent": float(
                100 * np.mean(fractions > 0)
            ),
        })

result = pd.DataFrame(records)

output = OUT / "terrain_component_sar_comparison.csv"
result.to_csv(output, index=False)

print(result.to_string(index=False))
print("\nSaved:", output)
