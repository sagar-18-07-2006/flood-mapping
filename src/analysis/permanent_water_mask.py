
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


ROOT = Path("data/processed")
VECTOR = ROOT / "bengaluru_hydrography.gpkg"
LAYER = "lakes_ponds_reservoirs"

REFERENCE = (
    ROOT / "sentinel1/aligned/S1A_20220812_bengaluru.tif"
)

OUTPUT_DIR = ROOT / "flood_detection"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading OSM waterbody polygons...")
    water = gpd.read_file(
        VECTOR,
        layer=LAYER,
        engine="pyogrio"
    )

    if water.crs is None:
        raise ValueError("Waterbody layer has no CRS.")

    print("Waterbody features:", len(water))
    print("Vector CRS:", water.crs)

    with rasterio.open(REFERENCE) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        shape = (ref.height, ref.width)
        raster_crs = ref.crs

    if water.crs != raster_crs:
        water = water.to_crs(raster_crs)

    # Exclude empty or invalid geometries.
    water = water[
        water.geometry.notna()
        & ~water.geometry.is_empty
    ].copy()

    water["geometry"] = water.geometry.make_valid()

    # Rasterize mapped waterbody polygons onto the exact SAR grid.
    shapes = (
        (geom, 1)
        for geom in water.geometry
        if geom is not None and not geom.is_empty
    )

    mask = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8"
    )

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=None,
        compress="deflate",
        tiled=True
    )

    mask_path = OUTPUT_DIR / "osm_waterbody_mask.tif"

    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, "OSM mapped waterbody mask")

    water_pixels = int(np.count_nonzero(mask == 1))
    total_pixels = mask.size

    print("\nWaterbody mask created:", mask_path)
    print("Raster dimensions:", shape[1], "x", shape[0])
    print("Mapped waterbody pixels:", water_pixels)
    print(
        "Mapped waterbody percentage:",
        round(100 * water_pixels / total_pixels, 3)
    )

    # Apply the exclusion mask to baseline and threshold experiments.
    candidate_files = [
        OUTPUT_DIR / "flood_candidate_sar.tif"
    ]

    candidate_files.extend(
        sorted(OUTPUT_DIR.glob("threshold_*db/flood_candidate_sar.tif"))
    )

    for candidate_path in candidate_files:
        if not candidate_path.exists():
            continue

        with rasterio.open(candidate_path) as src:
            candidate = src.read(1)
            candidate_profile = src.profile.copy()
            nodata = src.nodata

        if candidate.shape != mask.shape:
            raise ValueError(
                f"Grid dimensions differ: {candidate_path}"
            )

        # Preserve invalid pixels; remove candidates inside OSM waterbodies.
        valid = np.ones(candidate.shape, dtype=bool)
        if nodata is not None:
            valid &= candidate != nodata

        before = int(np.count_nonzero((candidate == 1) & valid))

        excluded = (mask == 1) & valid
        result = candidate.copy()
        result[excluded] = 0

        after = int(np.count_nonzero((result == 1) & valid))
        removed = before - after

        output_path = candidate_path.parent / (
            "flood_candidate_no_osm_water.tif"
        )

        with rasterio.open(
            output_path,
            "w",
            **candidate_profile
        ) as dst:
            dst.write(result, 1)

        print(f"\n{candidate_path.parent.name}")
        print("Candidates before:", before)
        print("Candidates excluded:", removed)
        print("Candidates after:", after)
        print("Saved:", output_path)


if __name__ == "__main__":
    main()

