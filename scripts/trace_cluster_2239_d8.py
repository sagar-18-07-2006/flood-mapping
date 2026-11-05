
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
}Saved: data/processed/terrain/hydroconditioning_difference.tif
(.venv) sm@fedora:~/Downloads/flood_mapping_bengaluru$ nano scripts/investigate_hydroconditioning.py
(.venv) sm@fedora:~/Downloads/flood_mapping_bengaluru$ python scripts/investigate_hydroconditioning.py

WHOLE STUDY AREA
Count: 2761988
Median: 0.0
Mean: 1.3024992
95th percentile: 0.9795009
99th percentile: 15.184454
Maximum: 830.4429

CLUSTER 2239
Count: 1184
Median: 0.0
Mean: 0.4070684
95th percentile: 2.4230115
99th percentile: 6.7402673
Maximum: 9.85313

STREAM CELLS
Count: 53445
Median: 15.961211
Mean: 61.785442
95th percentile: 342.62634
99th percentile: 616.57135
Maximum: 830.4429
STREAM CELLS INSIDE CLUSTER No valid pixels

TOP 20 ELEVATION REDUCTIONS
 row  col         x_utm        y_utm  original_dem  hydro_dem  reduction_m  is_stream  inside_cluster_2239
1289 1642 809380.418454 1.422341e+06    838.777039   8.334187   830.442871       True                False
1292 1645 809470.418454 1.422251e+06    837.407776   7.044187   830.363586       True                False
1291 1644 809440.418454 1.422281e+06    837.580994   7.474187   830.106812       True                False
1290 1643 809410.418454 1.422311e+06    837.394775   7.904187   829.490601       True                False
1289 1641 809350.418454 1.422341e+06    837.674622   8.764187   828.910461       True                False
1288 1640 809320.418454 1.422371e+06    837.627197   9.194187   828.432983       True                False
1287 1639 809290.418454 1.422401e+06    837.625732   9.624187   828.001526       True                False
1286 1638 809260.418454 1.422431e+06    836.948547  10.054187   826.894348       True                False
1285 1638 809260.418454 1.422461e+06    836.787537  10.484187   826.303345       True                False
1284 1637 809230.418454 1.422491e+06    837.136169  10.914187   826.221985       True                False
1283 1636 809200.418454 1.422521e+06    836.760254  11.344187   825.416077       True                False
1282 1635 809170.418454 1.422551e+06    836.564819  11.774187   824.790649       True                False
1279 1634 809140.418454 1.422641e+06    837.748108  13.064187   824.683899       True                False
1280 1635 809170.418454 1.422611e+06    837.063599  12.634187   824.429382       True                False
1277 1634 809140.418454 1.422701e+06    838.277344  13.924187   824.353149       True                False
1281 1635 809170.418454 1.422581e+06    836.388000  12.204187   824.183838       True                False
1278 1634 809140.418454 1.422671e+06    837.369690  13.494187   823.875488       True                False
1276 1633 809110.418454 1.422731e+06    837.658264  14.354187   823.304077       True                False
1275 1632 809080.418454 1.422761e+06    837.167053  14.784187   822.382874       True                False
1270 1631 809050.418454 1.422911e+06    839.309875  16.934187   822.375671       True                False

TOP 20 WITH DISTANCE TO EDGE
 reduction_m  is_stream  inside_cluster_2239  distance_to_raster_edge_m         x_utm        y_utm
  830.442871       True                False               62643.735521 809380.418454 1.422341e+06
  830.363586       True                False               62770.109128 809470.418454 1.422251e+06
  830.106812       True                False               62727.984186 809440.418454 1.422281e+06
  829.490601       True                False               62685.859650 809410.418454 1.422311e+06
  828.910461       True                False               62620.147716 809350.418454 1.422341e+06
  828.432983       True                False               62578.022500 809320.418454 1.422371e+06
  828.001526       True                False               62535.897691 809290.418454 1.422401e+06
  826.894348       True                False               62493.773290 809260.418454 1.422431e+06
  826.303345       True                False               62475.243097 809260.418454 1.422461e+06
  826.221985       True                False               62433.121018 809230.418454 1.422491e+06
  825.416077       True                False               62390.999351 809200.418454 1.422521e+06
  824.790649       True                False               62348.878097 809170.418454 1.422551e+06
  824.683899       True                False               62269.739039 809140.418454 1.422641e+06
  824.429382       True                False               62311.856015 809170.418454 1.422611e+06
  824.353149       True                False               62232.756648 809140.418454 1.422701e+06
  824.183838       True                False               62330.362585 809170.418454 1.422581e+06
  823.875488       True                False               62251.243361 809140.418454 1.422671e+06
  823.304077       True                False               62190.643991 809110.418454 1.422731e+06
  822.382874       True                False               62148.531761 809080.418454 1.422761e+06
  822.375671       True                False               62032.586598 809050.418454 1.422911e+06

D8 DESTINATION ELEVATION REDUCTIONS
Sample 1: original=860.06 m, hydro=431.88 m, reduction=428.18 m, stream=True
Sample 2: original=859.59 m, hydro=424.14 m, reduction=435.45 m, stream=True
Sample 3: original=859.34 m, hydro=418.12 m, reduction=441.21 m, stream=True
Sample 4: original=858.72 m, hydro=403.50 m, reduction=455.21 m, stream=True
Sample 5: original=858.72 m, hydro=403.50 m, reduction=455.21 m, stream=True
Sample 6: original=858.38 m, hydro=399.20 m, reduction=459.18 m, stream=True
Sample 7: original=858.20 m, hydro=394.90 m, reduction=463.30 m, stream=True
Sample 8: original=857.80 m, hydro=385.44 m, reduction=472.35 m, stream=True
Sample 9: original=857.51 m, hydro=377.27 m, reduction=480.24 m, stream=True
Sample 10: original=856.97 m, hydro=364.80 m, reduction=492.16 m, stream=True

Saved: data/processed/cluster_investigation/cluster_2239/largest_hydroconditioning_reductions.csv
(.venv) sm@fedora:~/Downloads/flood_mapping_bengaluru$ 


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

