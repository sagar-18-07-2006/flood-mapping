
import numpy as np
import pandas as pd
import rasterio

from pathlib import Path
from scipy.ndimage import distance_transform_edt

BASE = Path("data/processed")
OUT = BASE / "cluster_investigation/cluster_2239"

HAND_PATH = BASE / "terrain/aligned/hand_10m.tif"
STREAM_PATH = BASE / "terrain/aligned/streams_10m.tif"

CLUSTER_PATH = (
    BASE / "terrain_susceptibility/"
    "high_hand_clusters_osm.gpkg"
)

import geopandas as gpd
from rasterio.features import geometry_mask
from shapely.geometry import mapping

# ------------------------------------------------
# 1. Load cluster geometry
# ------------------------------------------------

clusters = gpd.read_file(
    CLUSTER_PATH,
    layer="high_hand_clusters",
    engine="pyogrio"
)

cluster = clusters.loc[
    clusters["cluster_id"] == 2239
].geometry.iloc[0]

# ------------------------------------------------
# 2. Read HAND and stream rasters
# ------------------------------------------------

with rasterio.open(HAND_PATH) as hsrc, \
     rasterio.open(STREAM_PATH) as ssrc:

    if (
        hsrc.crs != ssrc.crs
        or hsrc.transform != ssrc.transform
        or hsrc.shape != ssrc.shape
    ):
        raise ValueError("HAND and stream grids do not match")

    hand = hsrc.read(1).astype("float32")
    streams = ssrc.read(1)

    hand_nodata = hsrc.nodata
    stream_nodata = ssrc.nodata

    transform = hsrc.transform
    resolution = hsrc.res

    print("Raster shape:", hand.shape)
    print("Resolution:", resolution)
    print("HAND NoData:", hand_nodata)
    print("Stream NoData:", stream_nodata)

# ------------------------------------------------
# 3. Create independent validity masks
# ------------------------------------------------

hand_valid = np.isfinite(hand)

if hand_nodata is not None:
    hand_valid &= hand != hand_nodata

stream_valid = np.isfinite(streams.astype("float32"))

if stream_nodata is not None:
    stream_valid &= streams != stream_nodata

stream_mask = stream_valid & (streams > 0)

if not stream_mask.any():
    raise ValueError("No valid stream cells")

# ------------------------------------------------
# 4. Rasterize cluster
# ------------------------------------------------

inside = geometry_mask(
    [mapping(cluster)],
    out_shape=hand.shape,
    transform=transform,
    invert=True,
    all_touched=False
)

# ------------------------------------------------
# 5. Calculate distance to nearest stream
# ------------------------------------------------

distance = distance_transform_edt(
    ~stream_mask,
    sampling=(
        abs(resolution[1]),
        abs(resolution[0])
    )
)

# ------------------------------------------------
# 6. HAND statistics on stream cells
# ------------------------------------------------

stream_hand = hand[
    stream_mask & hand_valid
]

print("\nHAND VALUES ON DERIVED STREAM CELLS")

if len(stream_hand):
    print("Count:", len(stream_hand))
    print("Minimum:", np.min(stream_hand))
    print("Median:", np.median(stream_hand))
    print("Mean:", np.mean(stream_hand))
    print("95th percentile:", np.percentile(stream_hand, 95))
else:
    print("No jointly valid stream/HAND cells")

# ------------------------------------------------
# 7. HAND by distance from streams
# ------------------------------------------------

bands = [
    (0, 10),
    (10, 30),
    (30, 60),
    (60, 100),
    (100, 250),
    (250, 500),
    (500, 1000),
]

records = []

for lower, upper in bands:

    mask = (
        hand_valid
        & (distance >= lower)
        & (distance < upper)
    )

    values = hand[mask]

    if len(values) == 0:
        continue

    records.append({
        "distance_lower_m": lower,
        "distance_upper_m": upper,
        "pixels": len(values),
        "hand_median": float(np.median(values)),
        "hand_mean": float(np.mean(values)),
        "hand_p25": float(np.percentile(values, 25)),
        "hand_p75": float(np.percentile(values, 75)),
    })

df = pd.DataFrame(records)

print("\nHAND BY DISTANCE TO STREAMS")
print(df.to_string(index=False))

df.to_csv(
    OUT / "hand_by_stream_distance.csv",
    index=False
)

# ------------------------------------------------
# 8. HAND by distance, only inside cluster
# ------------------------------------------------

records = []

for lower, upper in bands:

    mask = (
        inside
        & hand_valid
        & (distance >= lower)
        & (distance < upper)
    )

    values = hand[mask]

    if len(values) == 0:
        continue

    records.append({
        "distance_lower_m": lower,
        "distance_upper_m": upper,
        "pixels": len(values),
        "hand_median": float(np.median(values)),
        "hand_mean": float(np.mean(values)),
        "hand_min": float(np.min(values)),
        "hand_max": float(np.max(values)),
    })

cluster_df = pd.DataFrame(records)

print("\nCLUSTER 2239: HAND BY DISTANCE TO STREAMS")
print(cluster_df.to_string(index=False))

cluster_df.to_csv(
    OUT / "cluster_2239_hand_by_stream_distance.csv",
    index=False
)

# ------------------------------------------------
# 9. Compare HAND inside and outside cluster
# ------------------------------------------------

inside_values = hand[inside & hand_valid]
outside_values = hand[~inside & hand_valid]

print("\nHAND COMPARISON")
print(
    "Cluster median:",
    np.median(inside_values)
)
print(
    "Outside median:",
    np.median(outside_values)
)

print("\nSaved diagnostics in:", OUT)
