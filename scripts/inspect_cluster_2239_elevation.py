
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import pandas as pd

from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import mapping

BASE = Path("data/processed")
OUT = BASE / "cluster_investigation/cluster_2239"

CLUSTER_PATH = (
    BASE / "terrain_susceptibility/"
    "high_hand_clusters_osm.gpkg"
)

HYDRO_PATH = BASE / "bengaluru_hydrography.gpkg"

DEM_PATH = BASE / "dem_bengaluru_utm43.tif"

FILLED_PATH = (
    BASE / "terrain/dem_hydroconditioned.tif"
)

HAND_PATH = BASE / "terrain/hand.tif"

# ------------------------------------------------
# 1. Load cluster
# ------------------------------------------------

clusters = gpd.read_file(
    CLUSTER_PATH,
    layer="high_hand_clusters",
    engine="pyogrio"
)

cluster = clusters.loc[
    clusters["cluster_id"] == 2239
].geometry.iloc[0]

crs = clusters.crs

# ------------------------------------------------
# 2. Load OSM waterways
# ------------------------------------------------

hydro = gpd.read_file(
    HYDRO_PATH,
    layer="waterways",
    engine="pyogrio"
).to_crs(crs)

rivers = hydro[
    hydro["name"].astype(str).str.contains(
        "Dakshina Pinakini",
        case=False,
        na=False
    )
]

intersecting = rivers[
    rivers.geometry.intersects(cluster)
]

print("Named Dakshina Pinakini features:", len(rivers))
print("Intersecting named river features:", len(intersecting))

# ------------------------------------------------
# 3. Extract raster statistics
# ------------------------------------------------

paths = {
    "Original DEM": DEM_PATH,
    "Hydroconditioned DEM": FILLED_PATH,
    "HAND": HAND_PATH
}

rows = []

for name, path in paths.items():

    with rasterio.open(path) as src:

        geom = cluster

        if src.crs != crs:
            raise ValueError(f"CRS mismatch: {name}")

        window = from_bounds(
            *geom.bounds,
            transform=src.transform
        ).round_offsets().round_lengths()

        window = window.intersection(
            rasterio.windows.Window(
                0, 0, src.width, src.height
            )
        )

        array = src.read(
            1,
            window=window,
            masked=True
        ).astype("float64")

        transform = src.window_transform(window)

        mask = geometry_mask(
            [mapping(geom)],
            out_shape=array.shape,
            transform=transform,
            invert=True,
            all_touched=False
        )

        values = array.compressed()
        valid = mask & ~np.ma.getmaskarray(array)
        values_inside = array.data[valid]

        rows.append({
            "layer": name,
            "pixels": len(values_inside),
            "minimum": np.min(values_inside),
            "median": np.median(values_inside),
            "mean": np.mean(values_inside),
            "maximum": np.max(values_inside)
        })

result = pd.DataFrame(rows)

print("\nELEVATION STATISTICS")
print(result.to_string(index=False))

result.to_csv(
    OUT / "cluster_2239_elevation_statistics.csv",
    index=False
)

# ------------------------------------------------
# 4. Sample original DEM along river geometry
# ------------------------------------------------

river_rows = []

with rasterio.open(DEM_PATH) as src:

    for _, feature in intersecting.iterrows():

        line = feature.geometry

        # Sample approximately every 30 m along river.
        distance_values = np.arange(
            0,
            line.length,
            30
        )

        for distance in distance_values:

            point = line.interpolate(float(distance))

            value = next(
                src.sample([(point.x, point.y)])
            )[0]

            if src.nodata is None or value != src.nodata:
                river_rows.append({
                    "name": feature.get("name"),
                    "distance_along_river_m": distance,
                    "dem_elevation_m": float(value)
                })

river_df = pd.DataFrame(river_rows)

if not river_df.empty:
    print("\nRIVER ELEVATION SAMPLES")
    print(river_df.describe().to_string())

    river_df.to_csv(
        OUT / "cluster_2239_river_elevation_samples.csv",
        index=False
    )

print("\nSaved outputs in:", OUT)
