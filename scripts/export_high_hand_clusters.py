import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy import ndimage
from rasterio.transform import xy

BASE = Path("data/processed")
OUT = BASE / "terrain_susceptibility"

hand_path = BASE / "terrain/hand.tif"
sar_path = OUT / "sar_candidate_fraction_30m.tif"

with rasterio.open(hand_path) as ds:
    hand = ds.read(1)
    transform = ds.transform
    hand_valid = hand != ds.nodata

with rasterio.open(sar_path) as ds:
    sar = ds.read(1)
    sar_valid = np.isfinite(sar) & (sar != ds.nodata)

valid = hand_valid & sar_valid & (sar >= 0)

boundary = np.percentile(hand[valid], 80)

candidate = valid & (hand > boundary) & (sar > 0)

labels, count = ndimage.label(
    candidate,
    structure=np.ones((3, 3), dtype=int)
)

records = []

for cluster_id in range(1, count + 1):
    rows, cols = np.where(labels == cluster_id)

    if len(rows) < 50:
        continue

    center_row = float(rows.mean())
    center_col = float(cols.mean())

    x, y = xy(transform, center_row, center_col)

    records.append({
        "cluster_id": cluster_id,
        "cells": len(rows),
        "area_km2": len(rows) * 900 / 1e6,
        "center_easting": x,
        "center_northing": y,
        "median_hand_m": float(np.median(hand[rows, cols])),
        "median_sar_fraction": float(np.median(sar[rows, cols])),
    })

result = pd.DataFrame(records)
result = result.sort_values("cells", ascending=False).head(20)

output = OUT / "high_hand_cluster_locations.csv"
result.to_csv(output, index=False)

print(result.to_string(index=False))
print("\nSaved:", output)
