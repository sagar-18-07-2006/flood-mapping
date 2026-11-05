from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt

BASE = Path("data/processed")
TERRAIN = BASE / "terrain/aligned"
FLOOD = BASE / "flood_detection"

hand_path = TERRAIN / "hand_10m.tif"
streams_path = TERRAIN / "streams_10m.tif"
candidate_path = FLOOD / "flood_candidate_no_osm_water.tif"

output_dir = FLOOD / "terrain_relationship"
output_dir.mkdir(parents=True, exist_ok=True)


def read_raster(path):
    with rasterio.open(path) as ds:
        return ds.read(1, masked=True), ds


hand, hand_ds = read_raster(hand_path)
streams, streams_ds = read_raster(streams_path)
candidate, candidate_ds = read_raster(candidate_path)

# Confirm that all three rasters share the same grid.
for ds in [streams_ds, candidate_ds]:
    assert ds.crs == hand_ds.crs, "CRS mismatch"
    assert ds.transform == hand_ds.transform, "Transform mismatch"
    assert ds.width == hand_ds.width and ds.height == hand_ds.height, \
        "Dimension mismatch"

hand_data = hand.filled(np.nan).astype("float32")
stream_data = streams.filled(0)
candidate_data = candidate.filled(0)

valid = (
    np.isfinite(hand_data)
    & (hand_data >= 0)
    & ~np.ma.getmaskarray(candidate)
)

flood = (candidate_data > 0) & valid
non_flood = (candidate_data == 0) & valid

# Streams are treated as cells with positive values.
stream_mask = (
    (stream_data > 0)
    & ~np.ma.getmaskarray(streams)
)

if not stream_mask.any():
    raise RuntimeError("No stream cells found. Inspect streams_10m.tif.")

# Euclidean distance to nearest stream cell, in metres.
distance = distance_transform_edt(
    ~stream_mask,
    sampling=(10, 10)
).astype("float32")

print("\n========== OVERALL SUMMARY ==========")
print("Valid comparison pixels:", int(valid.sum()))
print("SAR flood candidate pixels:", int(flood.sum()))
print("Other valid pixels:", int(non_flood.sum()))
print("Candidate area (km²):", round(flood.sum() * 0.0001, 3))
print("Stream cells:", int(stream_mask.sum()))

def describe(label, values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        print(label, ": no valid values")
        return

    print(f"\n{label}")
    print("Count:", values.size)
    print("Mean:", round(float(np.mean(values)), 2))
    print("Median:", round(float(np.median(values)), 2))
    print("25th percentile:", round(float(np.percentile(values, 25)), 2))
    print("75th percentile:", round(float(np.percentile(values, 75)), 2))
    print("90th percentile:", round(float(np.percentile(values, 90)), 2))

describe("HAND: SAR flood candidates (metres)", hand_data[flood])
describe("HAND: other valid pixels (metres)", hand_data[non_flood])

describe("Distance to streams: flood candidates (metres)", distance[flood])
describe("Distance to streams: other valid pixels (metres)", distance[non_flood])

# HAND elevation bands.
hand_bins = [0, 2, 5, 10, 20, 50, 100, 250, np.inf]
hand_labels = [
    "0-2", "2-5", "5-10", "10-20",
    "20-50", "50-100", "100-250", "250+"
]

hand_band = pd.cut(
    hand_data[valid],
    bins=hand_bins,
    labels=hand_labels,
    right=False
)

band_table = pd.DataFrame({
    "HAND_band_m": hand_band,
    "flood_candidate": flood[valid]
}).groupby("HAND_band_m", observed=False).agg(
    total_pixels=("flood_candidate", "size"),
    candidate_pixels=("flood_candidate", "sum")
).reset_index()

band_table["candidate_percent"] = (
    100 * band_table["candidate_pixels"]
    / band_table["total_pixels"].replace(0, np.nan)
)

band_table["candidate_area_km2"] = (
    band_table["candidate_pixels"] * 0.0001
)

band_table.to_csv(output_dir / "hand_band_statistics.csv", index=False)

# Distance bands.
distance_bins = [0, 30, 60, 100, 250, 500, 1000, np.inf]
distance_labels = [
    "0-30", "30-60", "60-100", "100-250",
    "250-500", "500-1000", "1000+"
]

distance_band = pd.cut(
    distance[valid],
    bins=distance_bins,
    labels=distance_labels,
    right=False
)

distance_table = pd.DataFrame({
    "distance_band_m": distance_band,
    "flood_candidate": flood[valid]
}).groupby("distance_band_m", observed=False).agg(
    total_pixels=("flood_candidate", "size"),
    candidate_pixels=("flood_candidate", "sum")
).reset_index()

distance_table["candidate_percent"] = (
    100 * distance_table["candidate_pixels"]
    / distance_table["total_pixels"].replace(0, np.nan)
)

distance_table["candidate_area_km2"] = (
    distance_table["candidate_pixels"] * 0.0001
)

distance_table.to_csv(
    output_dir / "stream_distance_statistics.csv",
    index=False
)

print("\n========== HAND BAND STATISTICS ==========")
print(band_table.to_string(index=False))

print("\n========== STREAM DISTANCE STATISTICS ==========")
print(distance_table.to_string(index=False))

print("\nCSV files saved in:", output_dir)
