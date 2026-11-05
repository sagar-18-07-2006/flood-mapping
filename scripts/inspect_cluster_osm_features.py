import geopandas as gpd
import pandas as pd
from pathlib import Path

BASE = Path("data/processed")
OUT = BASE / "terrain_susceptibility"

CLUSTERS = OUT / "high_hand_clusters_osm.gpkg"
HYDRO = BASE / "bengaluru_hydrography.gpkg"

clusters = gpd.read_file(
    CLUSTERS,
    layer="high_hand_clusters",
    engine="pyogrio"
)

water = gpd.read_file(
    HYDRO,
    layer="waterbodies",
    engine="pyogrio"
)

waterways = gpd.read_file(
    HYDRO,
    layer="waterways",
    engine="pyogrio"
)

water = water.to_crs(clusters.crs)
waterways = waterways.to_crs(clusters.crs)

results = []

for _, cluster in clusters.iterrows():
    cid = cluster["cluster_id"]
    geom = cluster.geometry

    for label, layer in [
        ("waterbody", water),
        ("waterway", waterways),
    ]:
        possible = layer.sindex.query(geom)

        for idx in possible:
            feature = layer.iloc[idx]

            if not geom.intersects(feature.geometry):
                continue

            intersection = geom.intersection(feature.geometry)

            if intersection.is_empty:
                continue

            record = {
                "cluster_id": cid,
                "feature_type": label,
                "intersection_area_m2": (
                    intersection.area if label == "waterbody" else None
                ),
                "intersection_length_m": (
                    intersection.length if label == "waterway" else None
                ),
            }

            # Preserve useful OSM attributes.
            for column in [
                "name", "water", "waterway", "natural",
                "landuse", "type", "fclass", "osm_id"
            ]:
                if column in layer.columns:
                    record[column] = feature[column]

            results.append(record)

result = pd.DataFrame(results)

output = OUT / "cluster_osm_intersecting_features.csv"
result.to_csv(output, index=False)

print("Total intersecting features:", len(result))

if not result.empty:
    print("\nFeature counts by type:")
    print(
        result.groupby("feature_type").size().to_string()
    )

    print("\nOSM attributes:")
    print(result.head(40).to_string(index=False))

print("\nSaved:", output)
