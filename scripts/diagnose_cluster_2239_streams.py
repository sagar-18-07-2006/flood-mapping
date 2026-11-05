
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd

from scipy.ndimage import distance_transform_edt
from rasterio.features import geometry_mask
from shapely.geometry import mapping


BASE = Path("data/processed")
OUT = BASE / "cluster_investigation/cluster_2239"

STREAM_PATH = BASE / "terrain/aligned/streams_10m.tif"
HAND_PATH = BASE / "terrain/aligned/hand_10m.tif"

CLUSTER_PATH = (
    BASE / "terrain_susceptibility/"
    "high_hand_clusters_osm.gpkg"
)

HYDRO_PATH = BASE / "bengaluru_hydrography.gpkg"

# --------------------------------------------------
# 1. Load cluster
# --------------------------------------------------

clusters = gpd.read_file(
    CLUSTER_PATH,
    layer="high_hand_clusters",
    engine="pyogrio"
)

selected = clusters[
    clusters["cluster_id"] == 2239
]

cluster = selected.geometry.iloc[0]
crs = clusters.crs

# --------------------------------------------------
# 2. Read full stream raster
# --------------------------------------------------

with rasterio.open(STREAM_PATH) as src:
    streams = src.read(1)
    transform = src.transform
    nodata = src.nodata
    resolution = src.res

    print("Stream raster:", src.shape)
    print("Resolution:", resolution)
    print("NoData:", nodata)

    values, counts = np.unique(
        streams, return_counts=True
    )

    print("\nStream raster unique values:")
    for value, count in zip(values, counts):
        print(value, count)

    valid_stream = np.isfinite(
        streams.astype("float32")
    )

    if nodata is not None:
        valid_stream &= streams != nodata

    stream_mask = valid_stream & (streams > 0)

    print("\nTotal derived stream pixels:", stream_mask.sum())

    if not stream_mask.any():
        raise ValueError("No positive stream pixels found")

    # Distance from every pixel to nearest derived stream.
    distance = distance_transform_edt(
        ~stream_mask,
        sampling=(abs(resolution[1]), abs(resolution[0]))
    )

    # --------------------------------------------------
    # 3. Rasterize cluster geometry
    # --------------------------------------------------

    inside = geometry_mask(
        [mapping(cluster)],
        out_shape=streams.shape,
        transform=transform,
        invert=True,
        all_touched=False
    )

    values_inside = distance[inside]

    print("\nCluster distance to DEM-derived streams:")
    print("Minimum:", values_inside.min())
    print("Median:", np.median(values_inside))
    print("Mean:", np.mean(values_inside))
    print("Maximum:", values_inside.max())

    # --------------------------------------------------
    # 4. Distance bands
    # --------------------------------------------------

    bands = [
        (0, 30),
        (30, 60),
        (60, 100),
        (100, 250),
        (250, 500),
        (500, 1000),
        (1000, np.inf)
    ]

    records = []

    for lower, upper in bands:
        count = int(
            ((values_inside >= lower)
             & (values_inside < upper)).sum()
        )

        records.append({
            "distance_lower_m": lower,
            "distance_upper_m": upper,
            "pixels": count,
            "percentage": 100 * count / len(values_inside)
        })

    df = pd.DataFrame(records)

    print("\nDistance distribution:")
    print(df.to_string(index=False))

    df.to_csv(
        OUT / "cluster_2239_dem_stream_distance.csv",
        index=False
    )

# --------------------------------------------------
# 5. Inspect OSM waterway types
# --------------------------------------------------

waterways = gpd.read_file(
    HYDRO_PATH,
    layer="waterways",
    engine="pyogrio"
).to_crs(crs)

intersecting = waterways[
    waterways.geometry.intersects(cluster)
].copy()

print("\nOSM waterways inside/intersecting cluster:")

columns = [
    col for col in [
        "name", "waterway", "type", "fclass"
    ]
    if col in intersecting.columns
]

print(intersecting[columns].to_string(index=False))

intersecting.to_file(
    OUT / "cluster_2239_osm_waterways.gpkg",
    layer="intersecting_waterways",
    driver="GPKG",
    engine="pyogrio"
)

print("\nSaved diagnostic outputs in:", OUT)

