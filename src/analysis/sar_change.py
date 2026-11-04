
from pathlib import Path
import argparse

import numpy as np
import rasterio
from rasterio.enums import Resampling


def read_band(path, band):
    with rasterio.open(path) as ds:
        return ds.read(band).astype("float32"), ds.profile.copy()


def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-1 pre/post flood backscatter change analysis"
    )
    parser.add_argument("--pre", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--vv-threshold-db",
        type=float,
        default=-3.0,
        help="VV decrease threshold in dB"
    )
    parser.add_argument(
        "--vh-threshold-db",
        type=float,
        default=-3.0,
        help="VH decrease threshold in dB"
    )
    args = parser.parse_args()

    pre_path = Path(args.pre)
    post_path = Path(args.post)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify matching raster grids.
    with rasterio.open(pre_path) as pre_ds, rasterio.open(post_path) as post_ds:
        checks = {
            "CRS": pre_ds.crs == post_ds.crs,
            "dimensions": (
                pre_ds.width == post_ds.width
                and pre_ds.height == post_ds.height
            ),
            "transform": pre_ds.transform == post_ds.transform,
            "band count": pre_ds.count >= 2 and post_ds.count >= 2,
        }

        print("Grid verification:")
        for name, passed in checks.items():
            print(f"  {name}: {'OK' if passed else 'FAILED'}")

        if not all(checks.values()):
            raise ValueError(
                "Pre/post rasters are not aligned. Stop and align them first."
            )

        profile = pre_ds.profile.copy()
        pre_vv = pre_ds.read(1).astype("float32")
        pre_vh = pre_ds.read(2).astype("float32")
        post_vv = post_ds.read(1).astype("float32")
        post_vh = post_ds.read(2).astype("float32")

        pre_nodata = pre_ds.nodata
        post_nodata = post_ds.nodata

    # Valid data mask.
    valid = (
        np.isfinite(pre_vv)
        & np.isfinite(pre_vh)
        & np.isfinite(post_vv)
        & np.isfinite(post_vh)
        & (pre_vv > 0)
        & (pre_vh > 0)
        & (post_vv > 0)
        & (post_vh > 0)
    )

    for arr, nodata in [
        (pre_vv, pre_nodata),
        (pre_vh, pre_nodata),
        (post_vv, post_nodata),
        (post_vh, post_nodata),
    ]:
        if nodata is not None:
            valid &= arr != nodata

    if valid.sum() == 0:
        raise ValueError("No valid overlapping pixels found.")

    # Sigma0 is linear power. Convert to dB before calculating differences.
    def to_db(arr):
        result = np.full(arr.shape, np.nan, dtype="float32")
        result[valid] = 10.0 * np.log10(arr[valid])
        return result

    pre_vv_db = to_db(pre_vv)
    post_vv_db = to_db(post_vv)
    pre_vh_db = to_db(pre_vh)
    post_vh_db = to_db(post_vh)

    vv_change = post_vv_db - pre_vv_db
    vh_change = post_vh_db - pre_vh_db

    # Candidate rule: decrease in both polarizations.
    candidate = (
        valid
        & (vv_change <= args.vv_threshold_db)
        & (vh_change <= args.vh_threshold_db)
    )

    # Write outputs.
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=-9999.0,
        compress="deflate",
        tiled=True,
        BIGTIFF="YES",
    )

    outputs = {
        "vv_change_db.tif": vv_change,
        "vh_change_db.tif": vh_change,
    }

    for filename, array in outputs.items():
        result = np.where(valid, array, -9999.0).astype("float32")
        with rasterio.open(out_dir / filename, "w", **profile) as dst:
            dst.write(result, 1)

    mask_profile = profile.copy()
    mask_profile.update(dtype="uint8", nodata=255)

    result = np.where(valid, candidate.astype("uint8"), 255).astype("uint8")
    with rasterio.open(out_dir / "flood_candidate_sar.tif", "w", **mask_profile) as dst:
        dst.write(result, 1)

    # Summary statistics.
    print("\nSAR change analysis complete")
    print("Valid pixels:", int(valid.sum()))
    print("Candidate pixels:", int(candidate.sum()))
    print("Candidate percentage:", round(100 * candidate.sum() / valid.sum(), 3))
    print("VV change median (dB):", float(np.nanmedian(vv_change[valid])))
    print("VH change median (dB):", float(np.nanmedian(vh_change[valid])))
    print("Outputs:", out_dir)


if __name__ == "__main__":
    main()
