
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from scipy.ndimage import distance_transform_edt
from rasterio.features import geometry_mask
from shapely.geometry import mapping

BASE = Path("data/processed")
OUT = BASE / "cluster_investigation/cluster_2239"

CLUSTER_PATH = (
    BASE / "terrain_susceptibility/"
    "high_hand_clusters_osm.gpkg"
)

DEM_PATH = BASE / "dem_bengaluru_utm43.tif"
HAND_PATH = BASE / "terrain/hand.tif"
STREAM_PATH = BASE / "terrain/streams.tif"

HYDRO_PATH = BASE / "bengaluru_hydrography.gpkg"

# --------------------------------------------------
# 1. Load cluster geometry
# --------------------------------------------------

clusters = gpd.read_file(
    CLUSTER_PATH,
    layer="high_hand_clusters",
    engine="pyogrio"
)

cluster = clusters.loc[
    clusters["cluster_id"] == 2239
].geometry.iloc[0]

crs = clusters.crs

# --------------------------------------------------
# 2. Read native 30m rasters
# --------------------------------------------------

with rasterio.open(DEM_PATH) as dem_ds, \
     rasterio.open(HAND_PATH) as hand_ds, \
     rasterio.open(STREAM_PATH) as stream_ds:

    if not (
        dem_ds.shape == hand_ds.shape == stream_ds.shape
        and dem_ds.transform == hand_ds.transform == stream_ds.transform
        and dem_ds.crs == hand_ds.crs == stream_ds.crs
    ):
        raise ValueError("Native terrain grids do not match")

    if dem_ds.crs != crs:
        raise ValueError("CRS mismatch")

    dem = dem_ds.read(1).astype("float32")
    hand = hand_ds.read(1).astype("float32")
    streams = stream_ds.read(1)

    transform = dem_ds.transform
    resolution = dem_ds.res

    dem_valid = np.isfinite(dem)
    hand_valid = np.isfinite(hand)
    stream_valid = np.isfinite(streams.astype("float32"))

    if dem_ds.nodata is not None:
        dem_valid &= dem != dem_ds.nodata

    if hand_ds.nodata is not None:
        hand_valid &= hand != hand_ds.nodata

    if stream_ds.nodata is not None:
        stream_valid &= streams != stream_ds.nodata

    stream_mask = stream_valid & (streams > 0)

    inside = geometry_mask(
        [mapping(cluster)],
        out_shape=dem.shape,
        transform=transform,
        invert=True,
        all_touched=False
    )

    # --------------------------------------------------
    # 3. Cluster DEM and HAND values
    # --------------------------------------------------

    cluster_dem = dem[inside & dem_valid]
    cluster_hand = hand[inside & hand_valid]

    print("\nCLUSTER NATIVE TERRAIN")
    print("DEM median:", np.median(cluster_dem))
    print("HAND median:", np.median(cluster_hand))
    print("DEM - HAND median:", np.median(cluster_dem - hand[inside & dem_valid & hand_valid]))

    # --------------------------------------------------
    # 4. Distance to nearest native stream cell
    # --------------------------------------------------

    distance = distance_transform_edt(
        ~stream_mask,
        sampling=(
            abs(resolution[1]),
            abs(resolution[0])
        )
    )

    # For each cluster pixel, locate its nearest stream.
    nearest_distance = distance[inside]

    print("\nNEAREST STREAM DISTANCE")
    print("Minimum:", nearest_distance.min())
    print("Median:", np.median(nearest_distance))
    print("Maximum:", nearest_distance.max())

    # --------------------------------------------------
    # 5. Inspect stream elevations
    # --------------------------------------------------

    stream_dem = dem[stream_mask & dem_valid]
    stream_hand = hand[stream_mask & hand_valid]

    print("\nALL NATIVE STREAM CELLS")
    print("Count:", len(stream_dem))
    print("DEM median:", np.median(stream_dem))
    print("DEM mean:", np.mean(stream_dem))
    print("HAND median:", np.median(stream_hand))

    # Inspect stream pixels within 500m of cluster.
    nearby_streams = (
        stream_mask
        & (distance <= 500)
        & dem_valid
        & hand_valid
    )

    nearby_dem = dem[nearby_streams]
    nearby_hand = hand[nearby_streams]

    print("\nSTREAM CELLS WITHIN 500M")
    print("Count:", len(nearby_dem))
    print("DEM median:", np.median(nearby_dem))
    print("DEM mean:", np.mean(nearby_dem))
    print("HAND median:", np.median(nearby_hand))

    # --------------------------------------------------
    # 6. Compare elevation and HAND by cluster pixel
    # --------------------------------------------------

    joint = inside & dem_valid & hand_valid

    difference = dem[joint] - hand[joint]

    print("\nDEM MINUS HAND INSIDE CLUSTER")
    print("Minimum:", np.min(difference))
    print("Median:", np.median(difference))
    print("Mean:", np.mean(difference))
    print("Maximum:", np.max(difference))

    # Save diagnostic raster statistics.
    pd.DataFrame([
        {
            "region": "cluster",
            "dem_median": np.median(cluster_dem),
            "hand_median": np.median(cluster_hand),
            "dem_minus_hand_median": np.median(difference)
        },
        {
            "region": "all_stream_cells",
            "dem_median": np.median(stream_dem),
            "hand_median": np.median(stream_hand),
            "dem_minus_hand_median": np.median(
                stream_dem[:min(len(stream_dem), len(stream_hand))]
                - stream_hand[:min(len(stream_dem), len(stream_hand))]
            )
        },
        {
            "region": "streams_within_500m",
            "dem_median": np.median(nearby_dem),
            "hand_median": np.median(nearby_hand),
            "dem_minus_hand_median": np.median(
                nearby_dem - nearby_hand
            )
        }
    ]).to_csv(
        OUT / "cluster_2239_drainage_elevation_summary.csv",
        index=False
    )

# --------------------------------------------------
# 7. Sample only the river portion inside cluster
# --------------------------------------------------

waterways = gpd.read_file(
    HYDRO_PATH,
    layer="waterways",
    engine="pyogrio"
).to_crs(crs)

named_river = waterways[
    waterways["name"].astype(str).str.contains(
        "Dakshina Pinakini",
        case=False,
        na=False
    )
]

records = []

with rasterio.open(DEM_PATH) as src:

    for _, feature in named_river.iterrows():

        intersection = feature.geometry.intersection(cluster)

        if intersection.is_empty:
            continue

        if intersection.geom_type == "LineString":
            lines = [intersection]
        elif intersection.geom_type == "MultiLineString":
            lines = list(intersection.geoms)
        else:
            continue

        for line in lines:

            distances = np.arange(0, line.length, 10)

            for d in distances:

                point = line.interpolate(float(d))

                value = next(
                    src.sample([(point.x, point.y)])
                )[0]

                if src.nodata is None or value != src.nodata:
                    records.append({
                        "river": feature.get("name"),
                        "distance_along_intersection_m": d,
                        "dem_elevation_m": float(value)
                    })

river_df = pd.DataFrame(records)

print("\nDAKSHINA PINAKINI INSIDE CLUSTER")

if not river_df.empty:
    print(river_df["dem_elevation_m"].describe())
else:
    print("No river samples found inside cluster")

river_df.to_csv(
    OUT / "cluster_2239_river_inside_elevation.csv",
    index=False
)

print("\nSaved outputs in:", OUT)
