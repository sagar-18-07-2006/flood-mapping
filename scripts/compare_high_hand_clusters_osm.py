import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from pathlib import Path
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

BASE = Path("data/processed")
OUT = BASE / "terrain_susceptibility"

HAND_FILE = BASE / "terrain/hand.tif"
SAR_FILE = OUT / "sar_candidate_fraction_30m.tif"
OSM_FILE = BASE / "bengaluru_hydrography.gpkg"
ROAD_FILE = BASE / "infrastructure/bengaluru_roads.gpkg"

# Read rasters.
with rasterio.open(HAND_FILE) as ds:
    hand = ds.read(1)
    transform = ds.transform
    crs = ds.crs
    hand_nodata = ds.nodata

with rasterio.open(SAR_FILE) as ds:
    sar = ds.read(1)
    sar_nodata = ds.nodata

valid = (
    np.isfinite(hand)
    & (hand != hand_nodata)
    & np.isfinite(sar)
    & (sar != sar_nodata)
    & (sar >= 0)
)

boundary = np.percentile(hand[valid], 80)
candidate = valid & (hand > boundary) & (sar > 0)

labels, count = ndimage.label(
    candidate,
    structure=np.ones((3, 3), dtype=int)
)

sizes = np.bincount(labels.ravel())
top_ids = np.argsort(sizes[1:])[::-1][:20] + 1

# Create one geometry per cluster.
records = []

for cluster_id in top_ids:
    mask = labels == cluster_id

    geometries = [
        shape(geom)
        for geom, value in shapes(
            mask.astype("uint8"),
            mask=mask,
            transform=transform
        )
        if value == 1
    ]

    geometry = unary_union(geometries)
    rows, cols = np.where(mask)

    records.append({
        "cluster_id": int(cluster_id),
        "cells": len(rows),
        "area_km2": len(rows) * 900 / 1e6,
        "median_hand_m": float(np.median(hand[mask])),
        "median_sar_fraction": float(np.median(sar[mask])),
        "geometry": geometry
    })

clusters = gpd.GeoDataFrame(records, crs=crs)

assert len(clusters) == 20
assert clusters["cluster_id"].is_unique

# Load OSM.
water = gpd.read_file(
    OSM_FILE, layer="waterbodies", engine="pyogrio"
).to_crs(crs)

waterways = gpd.read_file(
    OSM_FILE, layer="waterways", engine="pyogrio"
).to_crs(crs)

roads = gpd.read_file(
    ROAD_FILE, layer="roads", engine="pyogrio"
).to_crs(crs)

def clean(gdf):
    return gdf[
        gdf.geometry.notna() & ~gdf.geometry.is_empty
    ].copy()

water = clean(water)
waterways = clean(waterways)
roads = clean(roads)

# Calculate centroid distances and actual polygon intersections.
centroids = clusters[["cluster_id"]].copy()
centroids["geometry"] = clusters.geometry.centroid
centroids = gpd.GeoDataFrame(centroids, crs=crs)

def add_proximity(source, target, name):
    if target.empty:
        clusters[f"{name}_centroid_m"] = np.nan
        clusters[f"{name}_intersects"] = False
        return

    # Centroid-to-nearest-feature distance.
    nearest = gpd.sjoin_nearest(
        centroids,
        target[["geometry"]],
        how="left",
        distance_col="distance_m"
    )

    distances = nearest.groupby("cluster_id")[
        "distance_m"
    ].min()

    clusters[f"{name}_centroid_m"] = (
        clusters["cluster_id"].map(distances)
    )

    # Actual cluster polygon intersection.
    tree = target.sindex

    intersects = []

    for geom in clusters.geometry:
        possible = tree.query(geom)
        intersects.append(
            any(geom.intersects(target.geometry.iloc[i])
                for i in possible)
        )

    clusters[f"{name}_intersects"] = intersects


add_proximity(water, water, "waterbody")
add_proximity(waterways, waterways, "waterway")
add_proximity(roads, roads, "road")

# Export.
csv = OUT / "high_hand_clusters_osm.csv"
gpkg = OUT / "high_hand_clusters_osm.gpkg"

clusters.drop(columns="geometry").to_csv(csv, index=False)

clusters.to_file(
    gpkg,
    layer="high_hand_clusters",
    driver="GPKG",
    engine="pyogrio"
)

print("\n========== CLUSTER / OSM COMPARISON ==========")
print(clusters.drop(columns="geometry").to_string(index=False))

print("\nUnique clusters:", clusters.cluster_id.nunique())
print("Rows:", len(clusters))
print("Saved:", csv)
print("Saved:", gpkg)
