"""Mosaic DEM GeoTIFF tiles, reproject to UTM 43N, and clip to a buffered AOI."""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds, Window
from rasterio.transform import array_bounds


def prepare(inputs, output, bounds, target_crs="EPSG:32643", resolution=30):
    srcs = [rasterio.open(p) for p in inputs]
    try:
        if any(ds.crs is None for ds in srcs):
            raise ValueError("Every input DEM tile must have a CRS.")
        if any(ds.crs != srcs[0].crs for ds in srcs):
            raise ValueError("All DEM tiles must use the same CRS before mosaicking.")

        mosaic, mosaic_transform = merge(srcs)
        src_crs = srcs[0].crs
        src_height, src_width = mosaic.shape[1:]
        src_left, src_bottom, src_right, src_top = array_bounds(
            src_height, src_width, mosaic_transform
        )
        transform, width, height = calculate_default_transform(
            src_crs, target_crs, src_width, src_height,
            src_left, src_bottom, src_right, src_top,
            resolution=resolution
        )

        dest = np.full((height, width), -9999.0, dtype="float32")
        reproject(
            source=mosaic[0], destination=dest,
            src_transform=mosaic_transform, src_crs=src_crs,
            src_nodata=srcs[0].nodata,
            dst_transform=transform, dst_crs=target_crs,
            dst_nodata=-9999.0, resampling=Resampling.bilinear
        )

        west, south, east, north = bounds
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", target_crs, west, south, east, north
        )
        win = from_bounds(left, bottom, right, top, transform)
        col0 = max(0, int(np.floor(win.col_off)))
        row0 = max(0, int(np.floor(win.row_off)))
        col1 = min(width, int(np.ceil(win.col_off + win.width)))
        row1 = min(height, int(np.ceil(win.row_off + win.height)))
        cropped = dest[row0:row1, col0:col1]
        if cropped.size == 0:
            raise ValueError("AOI does not overlap the supplied DEM tiles.")

        out_transform = rasterio.windows.transform(
            Window(col0, row0, col1 - col0, row1 - row0), transform
        )
        profile = {
            "driver": "GTiff", "height": cropped.shape[0],
            "width": cropped.shape[1], "count": 1, "dtype": "float32",
            "crs": target_crs, "transform": out_transform,
            "nodata": -9999.0, "compress": "deflate"
        }
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(cropped.astype("float32"), 1)

        print(f"Prepared DEM: {output}")
        print(f"CRS: {target_crs}; target pixel size: {resolution} m")
        print(f"Processing bounds (WGS84): {bounds}")
    finally:
        for ds in srcs:
            ds.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Input DEM GeoTIFF tiles.")
    p.add_argument("--output", default="data/processed/dem_bengaluru_utm43.tif")
    # 0.05 degree buffer around the project's current approximate AOI.
    p.add_argument("--bounds", nargs=4, type=float,
                   default=[77.40, 12.75, 77.85, 13.20],
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--crs", default="EPSG:32643")
    p.add_argument("--resolution", type=float, default=30)
    a = p.parse_args()
    prepare(a.inputs, a.output, a.bounds, a.crs, a.resolution)
