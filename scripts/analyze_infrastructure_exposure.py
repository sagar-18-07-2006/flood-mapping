from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import shapes
from shapely.geometry import shape, box
from shapely.strtree import STRtree

BASE = Path("data/processed")
INFRA = BASE / "infrastructure"
FLOOD = BASE / "flood_detection"
OUT = INFRA / "exposure"

OUT.mkdir(parents=True, exist_ok=True)

BUILDINGS = INFRA / "bengaluru_buildings.gpkg"
ROADS = INFRA / "bengaluru_roads.gpkg"
CANDIDATES = FLOOD / "flood_candidate_no_osm_water.tif"

CHUNK_SIZE = 10000

# ---------------------------------------------------------
# 1. Read the original full-resolution candidate raster
# ---------------------------------------------------------

print("Reading SAR candidate raster...")

with rasterio.open(CANDIDATES) as src:
    candidate = src.read(1)
    transform = src.transform
    crs = src.crs
    bounds = src.bounds
    nodata = src.nodata

valid = np.isfinite(candidate)

if nodata is not None:
    valid &= candidate != nodata

flood_mask = valid & (candidate == 1)

pixel_area = abs(transform.a * transform.e)
flood_pixels = int(flood_mask.sum())

study_area = box(
    bounds.left, bounds.bottom,
    bounds.right, bounds.top
)

print("\n========== FLOOD BASELINE ==========")
print("CRS:", crs)
print("Raster dimensions:", candidate.shape[1], "x", candidate.shape[0])
print("Candidate pixels:", flood_pixels)
print("Pixel area (m²):", pixel_area)
print("Candidate area (km²):", round(
    flood_pixels * pixel_area / 1e6, 4
))

# ---------------------------------------------------------
# 2. Polygonize without removing small candidate patches
# ---------------------------------------------------------

print("\nPolygonizing all candidate pixels...")

flood_geometries = [
    shape(geom)
    for geom, value in shapes(
        flood_mask.astype("uint8"),
        mask=flood_mask,
        transform=transform,
        connectivity=8
    )
    if value == 1
]

print("Candidate polygons:", len(flood_geometries))

flood_array = np.asarray(flood_geometries, dtype=object)

# Build a spatial tree once and reuse it for both datasets.
print("Building flood polygon spatial index...")
tree = STRtree(flood_array)

# ---------------------------------------------------------
# 3. Read and clip infrastructure to study area
# ---------------------------------------------------------

bbox = (
    bounds.left,
    bounds.bottom,
    bounds.right,
    bounds.top
)

def load_infrastructure(path, layer, label):
    print(f"\nReading {label}...")

    gdf = gpd.read_file(
        path,
        layer=layer,
        engine="pyogrio",
        bbox=bbox
    )

    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    # Exact clipping to the raster's rectangular study extent.
    intersects = shapely.intersects(
        gdf.geometry.array,
        study_area
    )
    gdf = gdf.loc[intersects].copy()

    clipped = shapely.intersection(
        gdf.geometry.array,
        study_area
    )

    gdf.geometry = gpd.GeoSeries(
        clipped,
        index=gdf.index,
        crs=crs
    )

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    # Invalid source geometries are repaired without simplifying them.
    invalid = ~shapely.is_valid(gdf.geometry.array)

    if invalid.any():
        print("Repairing invalid geometries:", int(invalid.sum()))
        repaired = shapely.make_valid(gdf.geometry.array[invalid])
        gdf.loc[invalid, "geometry"] = repaired

    gdf = gdf.reset_index(drop=True)

    print(f"{label} inside study area:", len(gdf))
    return gdf


buildings = load_infrastructure(
    BUILDINGS, "buildings", "Buildings"
)

roads = load_infrastructure(
    ROADS, "roads", "Roads"
)

# ---------------------------------------------------------
# 4. Exact indexed intersections in manageable chunks
# ---------------------------------------------------------

def calculate_exposure(gdf, label, measurement):
    """
    measurement:
        'area'   -> unioned overlap area per feature
        'length' -> unioned overlap length per feature

    Spatial index finds candidate pairs.
    Exact intersections are calculated only for those pairs.
    Intersection pieces are unioned per original feature to
    avoid double-counting.
    """

    print(f"\n========== {label.upper()} EXPOSURE ==========")

    exposed_indices = []
    exposed_measurements = []

    total = len(gdf)

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)

        chunk = gdf.geometry.iloc[start:end].array

        # Exact predicate, accelerated by the STRtree.
        pairs = tree.query(
            chunk,
            predicate="intersects"
        )

        if pairs.shape[1] == 0:
            print(f"Processed {end:,}/{total:,}")
            continue

        # Calculate exact geometry intersections for matching pairs.
        pieces = shapely.intersection(
            chunk.take(pairs[0]),
            flood_array.take(pairs[1])
        )

        # Process each original infrastructure feature once.
        unique_local = np.unique(pairs[0])

        for local_index in unique_local:
            selected = pieces[pairs[0] == local_index]

            # Union overlapping pieces before measuring.
            merged = shapely.union_all(selected)

            if measurement == "area":
                value = shapely.area(merged)
            else:
                value = shapely.length(merged)

            if value > 0:
                exposed_indices.append(start + int(local_index))
                exposed_measurements.append(float(value))

        print(
            f"Processed {end:,}/{total:,} "
            f"| matched features: {len(exposed_indices):,}"
        )

    result = gdf.iloc[exposed_indices].copy()

    column = (
        "intersection_area_m2"
        if measurement == "area"
        else "exposed_length_m"
    )

    result[column] = exposed_measurements

    if measurement == "area":
        result["feature_area_m2"] = result.geometry.area
        result["overlap_percent"] = (
            100 * result[column]
            / result["feature_area_m2"].replace(0, np.nan)
        ).clip(upper=100)

    else:
        result["total_length_m"] = result.geometry.length

    print("Exposed features:", len(result))

    return result


exposed_buildings = calculate_exposure(
    buildings, "Buildings", "area"
)

exposed_roads = calculate_exposure(
    roads, "Roads", "length"
)

# ---------------------------------------------------------
# 5. Save exact exposure geometries
# ---------------------------------------------------------

print("\nSaving GIS layers...")

building_output = OUT / "potentially_exposed_buildings.gpkg"
road_output = OUT / "potentially_exposed_roads.gpkg"

for path in [building_output, road_output]:
    if path.exists():
        path.unlink()

if not exposed_buildings.empty:
    exposed_buildings.to_file(
        building_output,
        layer="exposed_buildings",
        driver="GPKG"
    )

if not exposed_roads.empty:
    exposed_roads.to_file(
        road_output,
        layer="exposed_roads",
        driver="GPKG"
    )

# ---------------------------------------------------------
# 6. Category statistics
# ---------------------------------------------------------

if "building" in exposed_buildings.columns:
    building_summary = (
        exposed_buildings.assign(
            building_type=exposed_buildings["building"].fillna("unspecified")
        )
        .groupby("building_type")
        .agg(
            exposed_features=("osm_id", "size"),
            unique_osm_ids=("osm_id", "nunique"),
            overlap_area_m2=("intersection_area_m2", "sum")
        )
        .reset_index()
    )
else:
    building_summary = pd.DataFrame()

building_summary.to_csv(
    OUT / "building_exposure_by_type.csv",
    index=False
)

if "highway" in exposed_roads.columns:
    road_summary = (
        exposed_roads.assign(
            road_type=exposed_roads["highway"].fillna("unspecified")
        )
        .groupby("road_type")
        .agg(
            exposed_segments=("osm_id", "size"),
            unique_osm_ids=("osm_id", "nunique"),
            exposed_length_m=("exposed_length_m", "sum")
        )
        .reset_index()
    )
    road_summary["exposed_length_km"] = (
        road_summary["exposed_length_m"] / 1000
    )
else:
    road_summary = pd.DataFrame()

road_summary.to_csv(
    OUT / "road_exposure_by_type.csv",
    index=False
)

# ---------------------------------------------------------
# 7. Overall summary
# ---------------------------------------------------------

summary = pd.DataFrame([
    ("SAR candidate pixels", flood_pixels),
    ("SAR candidate area (km²)",
     flood_pixels * pixel_area / 1e6),
    ("Buildings in study area", len(buildings)),
    ("Potentially exposed buildings", len(exposed_buildings)),
    ("Road segments in study area", len(roads)),
    ("Potentially exposed road segments", len(exposed_roads)),
    ("Potentially exposed road length (km)",
     exposed_roads["exposed_length_m"].sum() / 1000
     if not exposed_roads.empty else 0),
], columns=["metric", "value"])

summary.to_csv(
    OUT / "exposure_summary.csv",
    index=False
)

print("\n========== FINAL EXPOSURE SUMMARY ==========")
print(summary.to_string(index=False))

print("\nBUILDING EXPOSURE BY TYPE")
print(building_summary.to_string(index=False))

print("\nROAD EXPOSURE BY TYPE")
print(road_summary.to_string(index=False))

print("\nAll outputs saved to:", OUT)
print("Infrastructure exposure analysis complete.")
