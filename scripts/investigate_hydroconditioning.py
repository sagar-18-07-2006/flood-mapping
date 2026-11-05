
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from scipy.ndimage import distance_transform_edt

BASE = Path("data/processed")
TERRAIN = BASE / "terrain"
OUT = BASE / "cluster_investigation/cluster_2239"

ORIGINAL = BASE / "dem_bengaluru_utm43.tif"
HYDRO = TERRAIN / "dem_hydroconditioned.tif"
STREAMS = TERRAIN / "streams.tif"

CLUSTERS = (
    BASE / "terrain_susceptibility/"
    "high_hand_clusters_osm.gpkg"
)

TRACE = OUT / "cluster_2239_d8_trace.csv"

# --------------------------------------------------
# 1. Read rasters
# --------------------------------------------------

with rasterio.open(ORIGINAL) as src:
    original = src.read(1)
    profile = src.profile.copy()
    transform = src.transform
    nodata_original = src.nodata
    crs = src.crs

with rasterio.open(HYDRO) as src:
    hydro = src.read(1)
    nodata_hydro = src.nodata

with rasterio.open(STREAMS) as src:
    streams = src.read(1)
    nodata_stream = src.nodata

assert original.shape == hydro.shape == streams.shape

valid = (
    np.isfinite(original)
    & np.isfinite(hydro)
    & (original != nodata_original)
    & (hydro != nodata_hydro)
)

reduction = np.full(original.shape, np.nan, dtype="float32")
reduction[valid] = original[valid] - hydro[valid]

stream_mask = (
    (streams > 0)
    & (streams != nodata_stream)
    & valid
)

# --------------------------------------------------
# 2. Load cluster 2239
# --------------------------------------------------

gdf = gpd.read_file(CLUSTERS, engine="pyogrio")

cluster = gdf.loc[
    gdf["cluster_id"] == 2239
].geometry.iloc[0]

inside = geometry_mask(
    [mapping(cluster)],
    out_shape=original.shape,
    transform=transform,
    invert=True
)

# --------------------------------------------------
# 3. Compare elevation reduction
# --------------------------------------------------

def report(name, mask):
    values = reduction[mask & valid]

    if len(values) == 0:
        print(name, "No valid pixels")
        return

    print(f"\n{name}")
    print("Count:", len(values))
    print("Median:", np.median(values))
    print("Mean:", np.mean(values))
    print("95th percentile:", np.percentile(values, 95))
    print("99th percentile:", np.percentile(values, 99))
    print("Maximum:", np.max(values))

report("WHOLE STUDY AREA", np.ones(valid.shape, dtype=bool))
report("CLUSTER 2239", inside)
report("STREAM CELLS", stream_mask)
report("STREAM CELLS INSIDE CLUSTER", stream_mask & inside)

# --------------------------------------------------
# 4. Inspect largest reductions
# --------------------------------------------------

rows, cols = np.where(valid & (reduction > 100))

records = []

for r, c in zip(rows, cols):
    x, y = rasterio.transform.xy(transform, r, c)

    records.append({
        "row": int(r),
        "col": int(c),
        "x_utm": x,
        "y_utm": y,
        "original_dem": float(original[r, c]),
        "hydro_dem": float(hydro[r, c]),
        "reduction_m": float(reduction[r, c]),
        "is_stream": bool(stream_mask[r, c]),
        "inside_cluster_2239": bool(inside[r, c])
    })

top = pd.DataFrame(records).sort_values(
    "reduction_m", ascending=False
)

print("\nTOP 20 ELEVATION REDUCTIONS")
print(top.head(20).to_string(index=False))

top.to_csv(
    OUT / "largest_hydroconditioning_reductions.csv",
    index=False
)

# --------------------------------------------------
# 5. Distance from raster boundary
# --------------------------------------------------

distance_to_edge = distance_transform_edt(
    valid,
    sampling=(
        abs(transform.e),
        abs(transform.a)
    )
)

top["distance_to_raster_edge_m"] = [
    distance_to_edge[int(r), int(c)]
    for r, c in zip(top["row"], top["col"])
]

print("\nTOP 20 WITH DISTANCE TO EDGE")
print(
    top.head(20)[
        ["reduction_m", "is_stream",
         "inside_cluster_2239",
         "distance_to_raster_edge_m",
         "x_utm", "y_utm"]
    ].to_string(index=False)
)

# --------------------------------------------------
# 6. Check the exact D8 stream destinations
# --------------------------------------------------

trace = pd.read_csv(TRACE)

print("\nD8 DESTINATION ELEVATION REDUCTIONS")

for _, item in trace.iterrows():
    r = int(item.end_row)
    c = int(item.end_col)

    print(
        f"Sample {int(item.sample_id)}: "
        f"original={original[r,c]:.2f} m, "
        f"hydro={hydro[r,c]:.2f} m, "
        f"reduction={reduction[r,c]:.2f} m, "
        f"stream={stream_mask[r,c]}"
    )

print("\nSaved:", OUT / "largest_hydroconditioning_reductions.csv")
