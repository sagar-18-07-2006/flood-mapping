import numpy as np
import rasterio
import pandas as pd
from pathlib import Path

BASE = Path("data/processed")
OUT = BASE / "terrain_susceptibility"

files = {
    "hand": BASE / "terrain/hand.tif",
    "sar": OUT / "sar_candidate_fraction_30m.tif",
    "slope": BASE / "terrain/slope_degrees.tif",
    "twi": BASE / "terrain/twi.tif",
    "flow": BASE / "terrain/flow_accumulation_cells.tif",
}

data = {}

for name, path in files.items():
    with rasterio.open(path) as ds:
        data[name] = ds.read(1).astype("float64")
        if name == "hand":
            transform = ds.transform
            crs = ds.crs

hand = data["hand"]
sar = data["sar"]

valid = (
    np.isfinite(hand) &
    np.isfinite(sar) &
    (hand != -9999) &
    (sar != -9999) &
    (sar >= 0)
)

# Candidate cells are cells with any SAR candidate fraction.
candidate = valid & (sar > 0)

# HAND quintile boundaries.
hand_limits = np.percentile(hand[valid], [20, 40, 60, 80])

print("HAND quintile boundaries:", hand_limits)

# Compare Q1 and Q5 candidate populations.
q1 = candidate & (hand <= hand_limits[0])
q5 = candidate & (hand > hand_limits[3])

for label, mask in [
    ("Low HAND candidate cells", q1),
    ("High HAND candidate cells", q5),
]:
    rows, cols = np.where(mask)

    print(f"\n========== {label} ==========")
    print("Cell count:", len(rows))

    if len(rows) == 0:
        continue

    print("HAND median:", np.median(hand[mask]))
    print("HAND mean:", np.mean(hand[mask]))
    print("SAR fraction median:", np.median(sar[mask]))

    print("Sample locations (UTM EPSG:32643):")

    for r, c in zip(rows[:10], cols[:10]):
        x, y = rasterio.transform.xy(transform, r, c)
        print(
            f"Easting={x:.1f}, Northing={y:.1f}, "
            f"HAND={hand[r,c]:.1f}, "
            f"SAR fraction={sar[r,c]:.3f}"
        )

# Additional diagnostic: high-HAND candidate terrain properties.
print("\n========== HIGH-HAND CANDIDATE TERRAIN ==========")

for name in ["slope", "twi", "flow"]:
    arr = data[name]
    mask = q5 & np.isfinite(arr) & (arr != -9999)

    if mask.any():
        print(
            name,
            "median =", round(float(np.median(arr[mask])), 3),
            "mean =", round(float(np.mean(arr[mask])), 3)
        )

print("\nDiagnostic complete.")
