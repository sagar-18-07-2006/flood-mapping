import geopandas as gpd
import pandas as pd
from pathlib import Path

BASE = Path("data/processed")
OUT = BASE / "terrain_susceptibility"

clusters = gpd.read_file(
    OUT / "high_hand_clusters_osm.gpkg",
    layer="high_hand_clusters",
    engine="pyogrio"
)

hydro_file = BASE / "bengaluru_hydrography.gpkg"

water = gpd.read_file(
    hydro_file, layer="waterbodies", engine="pyogrio"
).to_crs(clusters.crs)

waterways = gpd.read_file(
    hydro_file, layer="waterways", engine="pyogrio"
).to_crs(clusters.crs)

records = []

for _, cluster in clusters.iterrows():
    geom = cluster.geometry
    cid = cluster.cluster_id

    # Candidate waterbody features intersecting this cluster.
    indices = water.sindex.query(geom, predicate="intersects")
    water_parts = []

    for idx in indices:
        intersection = geom.intersection(
            water.geometry.iloc[idx]
        )
        if not intersection.is_empty:
            water_parts.append(intersection)

    # Union prevents double-counting overlapping waterbody features.
    if water_parts:
        water_union = water_parts[0]
        for part in water_parts[1:]:
            water_union = water_union.union(part)

        water_area = water_union.area
    else:
        water_area = 0.0

    # Waterway lengths inside cluster.
    indices = waterways.sindex.query(
        geom, predicate="intersects"
    )

    line_records = []

    for idx in indices:
        feature = waterways.iloc[idx]
        intersection = geom.intersection(feature.geometry)

        if intersection.is_empty:
            continue

        line_records.append({
            "length_m": intersection.length,
            "name": feature.get("name"),
            "waterway": feature.get("waterway"),
        })

    total_line_length = sum(
        item["length_m"] for item in line_records
    )

    records.append({
        "cluster_id": cid,
        "cluster_area_km2": geom.area / 1e6,
        "waterbody_overlap_km2": water_area / 1e6,
        "waterbody_overlap_percent": (
             100 * water_area / geom.area
            if geom.area > 0 else 0
            ),
        "waterway_length_inside_m": total_line_length,
        "waterway_feature_count": len(line_records),
        "waterway_names": "; ".join(sorted({
            str(x["name"])
            for x in line_records if x["name"]
        })),
        "waterway_types": "; ".join(sorted({
            str(x["waterway"])
            for x in line_records if x["waterway"]
        })),
    })

result = pd.DataFrame(records).sort_values(
    "waterbody_overlap_percent",
    ascending=False
)

output = OUT / "cluster_hydrography_overlap.csv"
result.to_csv(output, index=False)

print(result.to_string(index=False))
print("\nSaved:", output)
