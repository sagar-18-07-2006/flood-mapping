import numpy as np
import geopandas as gpd
import rasterio

from pathlib import Path
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

BASE = Path("data/processed")
GPKG = BASE / "bengaluru_hydrography.gpkg"

# Load mapped waterways
waterways = gpd.read_file(GPKG, layer="waterways")

# Use all mapped waterways, including urban drains
print("OSM waterways:", len(waterways))

thresholds = {
    1000: BASE / "terrain" / "streams.tif",
    3000: BASE / "terrain_3000" / "streams.tif",
    5000: BASE / "terrain_5000" / "streams.tif",
    10000: BASE / "terrain_10000" / "streams.tif",
}

# Use the DEM as the spatial reference
with rasterio.open(BASE / "dem_bengaluru_utm43.tif") as dem:
    transform = dem.transform
    crs = dem.crs
    shape = (dem.height, dem.width)
    dem_valid = dem.read(1, masked=True).mask == False
    pixel_x = abs(transform.a)
    pixel_y = abs(transform.e)

if waterways.crs != crs:
    waterways = waterways.to_crs(crs)

# Rasterize OSM waterways onto the DEM grid
osm_mask = rasterize(
    [(geom, 1) for geom in waterways.geometry if geom is not None and not geom.is_empty],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=True,
).astype(bool)

# Distance from every pixel to the nearest OSM waterway
distance = distance_transform_edt(
    ~osm_mask,
    sampling=(pixel_y, pixel_x)
)

print("\nOSM mapped waterway pixels:", int(osm_mask.sum()))
print("\n===== DRAINAGE VALIDATION =====")

for threshold, path in thresholds.items():
    if not path.exists():
        print(f"\nThreshold {threshold}: Missing {path}")
        continue

    with rasterio.open(path) as src:
        streams = src.read(1, masked=True).filled(0) != 0

    streams &= dem_valid
    total = int(streams.sum())

    print(f"\nThreshold: {threshold}")
    print("DEM stream pixels:", total)

    if total == 0:
        continue

    for radius in (30, 60, 90):
        matched = int(np.count_nonzero(streams & (distance <= radius)))
        percentage = matched / total * 100
        print(
            f"Within {radius:2d} m of OSM waterways: "
            f"{matched:,} pixels ({percentage:.2f}%)"
        )
