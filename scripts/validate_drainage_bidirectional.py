import numpy as np
import geopandas as gpd
import rasterio

from pathlib import Path
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

BASE = Path("data/processed")
GPKG = BASE / "bengaluru_hydrography.gpkg"

waterways = gpd.read_file(GPKG, layer="waterways")

thresholds = {
    1000: BASE / "terrain" / "streams.tif",
    3000: BASE / "terrain_3000" / "streams.tif",
    5000: BASE / "terrain_5000" / "streams.tif",
    10000: BASE / "terrain_10000" / "streams.tif",
}

with rasterio.open(BASE / "dem_bengaluru_utm43.tif") as dem:
    shape = (dem.height, dem.width)
    transform = dem.transform
    crs = dem.crs
    valid = dem.read(1, masked=True).mask == False
    px = abs(transform.a)
    py = abs(transform.e)

waterways = waterways.to_crs(crs)

osm_mask = rasterize(
    [(g, 1) for g in waterways.geometry if g is not None and not g.is_empty],
    out_shape=shape,
    transform=transform,
    fill=0,
    all_touched=True,
).astype(bool)

osm_mask &= valid
osm_total = int(osm_mask.sum())

print("OSM reference pixels:", osm_total)

for threshold, path in thresholds.items():
    with rasterio.open(path) as src:
        dem_streams = src.read(1, masked=True).filled(0) != 0

    dem_streams &= valid
    dem_total = int(dem_streams.sum())

    # Distance from each DEM pixel to the nearest OSM waterway
    dist_to_osm = distance_transform_edt(
        ~osm_mask, sampling=(py, px)
    )

    # Distance from each OSM pixel to the nearest DEM stream
    dist_to_dem = distance_transform_edt(
        ~dem_streams, sampling=(py, px)
    )

    print(f"\n===== THRESHOLD {threshold} =====")
    print("DEM stream pixels:", dem_total)

    for radius in (30, 60, 90):
        dem_match = int(
            np.count_nonzero(dem_streams & (dist_to_osm <= radius))
        )
        osm_match = int(
            np.count_nonzero(osm_mask & (dist_to_dem <= radius))
        )

        precision = dem_match / dem_total if dem_total else 0
        recall = osm_match / osm_total if osm_total else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        print(
            f"{radius} m | "
            f"DEM→OSM: {precision*100:.2f}% | "
            f"OSM→DEM: {recall*100:.2f}% | "
            f"F1-style: {f1*100:.2f}%"
        )
