
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd

from rasterio.features import geometry_mask
from shapely.geometry import mapping
from scipy.ndimage import distance_transform_edt

BASE = Path("data/processed")
OUT = BASE / "cluster_investigation/cluster_2239"

CLUSTER_PATH = (
    BASE / "terrain_susceptibility/"
    "high_hand_clusters_osm.gpkg"
)

HAND_PATH = BASE / "terrain/hand.tif"
DEM_PATH = BASE / "dem_bengaluru_utm43.tif"
D8_PATH = BASE / "terrain/d8_pointer.tif"
FLOW_PATH = BASE / "terrain/flow_accumulation_cells.tif"
STREAM_PATH = BASE / "terrain/streams.tif"

# WhiteboxTools D8 pointer convention:
# E=1, NE=2, N=4, NW=8,
# W=16, SW=32, S=64, SE=128.

DIRECTIONS = {
    1: (0, 1),       # East
    2: (1, 1),       # Southeast
    4: (1, 0),       # South
    8: (1, -1),      # Southwest
    16: (0, -1),     # West
    32: (-1, -1),    # Northwest
    64: (-1, 0),     # North
    128: (-1, 1),    # Northeast
}
# --------------------------------------------------
# 1. Load cluster
# --------------------------------------------------

clusters = gpd.read_file(
    CLUSTER_PATH,
    layer="high_hand_clusters",
    engine="pyogrio"
)

cluster = clusters.loc[
    clusters["cluster_id"] == 2239
].geometry.iloc[0]

# --------------------------------------------------
# 2. Read terrain rasters
# --------------------------------------------------

paths = [
    HAND_PATH,
    DEM_PATH,
    D8_PATH,
    FLOW_PATH,
    STREAM_PATH
]

datasets = [rasterio.open(p) for p in paths]

hand_ds, dem_ds, d8_ds, flow_ds, stream_ds = datasets

for ds in datasets[1:]:
    if (
        ds.shape != hand_ds.shape
        or ds.transform != hand_ds.transform
        or ds.crs != hand_ds.crs
    ):
        raise ValueError("Terrain grids do not match")

hand = hand_ds.read(1)
dem = dem_ds.read(1)
d8 = d8_ds.read(1)
flow = flow_ds.read(1)
streams = stream_ds.read(1)

transform = hand_ds.transform

# --------------------------------------------------
# 3. Validity masks
# --------------------------------------------------

def valid_mask(array, ds):
    mask = np.isfinite(array.astype("float64"))

    if ds.nodata is not None:
        mask &= array != ds.nodata

    return mask


hand_valid = valid_mask(hand, hand_ds)
dem_valid = valid_mask(dem, dem_ds)
d8_valid = valid_mask(d8, d8_ds)
flow_valid = valid_mask(flow, flow_ds)
stream_valid = valid_mask(streams, stream_ds)

stream_mask = stream_valid & (streams > 0)

# --------------------------------------------------
# 4. Find high-HAND cluster pixels
# --------------------------------------------------

inside = geometry_mask(
    [mapping(cluster)],
    out_shape=hand.shape,
    transform=transform,
    invert=True,
    all_touched=False
)

candidate_pixels = (
    inside
    & hand_valid
    & d8_valid
    & (hand > 400)
)

rows, cols = np.where(candidate_pixels)

print("High-HAND pixels:", len(rows))

if len(rows) == 0:
    raise ValueError("No high-HAND pixels found")

# --------------------------------------------------
# 5. Select representative pixels
# --------------------------------------------------

# Sample up to 10 pixels spread across the cluster.
indices = np.linspace(
    0, len(rows) - 1,
    min(10, len(rows)),
    dtype=int
)

selected = [
    (int(rows[i]), int(cols[i]))
    for i in indices
]

# --------------------------------------------------
# 6. Trace D8 flow paths
# --------------------------------------------------

height, width = hand.shape
results = []

for sample_id, (start_r, start_c) in enumerate(selected, 1):

    r, c = start_r, start_c
    visited = set()
    steps = 0
    status = "unknown"

    start_hand = float(hand[r, c])
    start_dem = float(dem[r, c])

    while steps < 100000:

        if (r, c) in visited:
            status = "cycle_detected"
            break

        visited.add((r, c))

        if stream_mask[r, c]:
            status = "reached_stream"
            break

        if not d8_valid[r, c]:
            status = "invalid_pointer"
            break

        pointer = int(d8[r, c])

        if pointer not in DIRECTIONS:
            status = "terminal_or_invalid_direction"
            break

        dr, dc = DIRECTIONS[pointer]

        nr = r + dr
        nc = c + dc

        if (
            nr < 0 or nr >= height
            or nc < 0 or nc >= width
        ):
            status = "left_raster"
            break

        r, c = nr, nc
        steps += 1

    else:
        status = "step_limit"

    results.append({
        "sample_id": sample_id,
        "start_row": start_r,
        "start_col": start_c,
        "start_hand_m": start_hand,
        "start_dem_m": start_dem,
        "end_row": r,
        "end_col": c,
        "end_dem_m": (
            float(dem[r, c]) if dem_valid[r, c] else np.nan
        ),
        "end_hand_m": (
            float(hand[r, c]) if hand_valid[r, c] else np.nan
        ),
        "end_flow_accumulation": (
            float(flow[r, c]) if flow_valid[r, c] else np.nan
        ),
        "reached_stream": bool(stream_mask[r, c]),
        "steps": steps,
        "status": status
    })

    print(
        f"Sample {sample_id}: "
        f"status={status}, steps={steps}, "
        f"start HAND={start_hand:.2f} m"
    )

# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

result = pd.DataFrame(results)

output = OUT / "cluster_2239_d8_trace.csv"

result.to_csv(output, index=False)

print("\nD8 TRACE RESULTS")
print(result.to_string(index=False))

print("\nSaved:", output)

for ds in datasets:
    ds.close()

