import rasterio
import numpy as np
from pathlib import Path

folder = Path("data/processed/terrain")

layers = [
    "dem_hydroconditioned.tif",
    "slope_degrees.tif",
    "flow_accumulation_cells.tif",
    "hand.tif",
    "twi.tif",
]

for name in layers:
    with rasterio.open(folder / name) as src:
        a = src.read(1, masked=True).compressed()

    print(f"\n===== {name} =====")
    print("Min:", np.min(a))
    print("P25:", np.percentile(a, 25))
    print("Median:", np.percentile(a, 50))
    print("P75:", np.percentile(a, 75))
    print("P90:", np.percentile(a, 90))
    print("P95:", np.percentile(a, 95))
    print("P99:", np.percentile(a, 99))
    print("Max:", np.max(a))

# Correct stream-network statistics
dem_path = Path("data/processed/dem_bengaluru_utm43.tif")

with rasterio.open(dem_path) as dem_src:
    dem = dem_src.read(1, masked=True)
    dem_valid_pixels = dem.count()

    pixel_area = abs(
        dem_src.transform.a * dem_src.transform.e
        - dem_src.transform.b * dem_src.transform.d
    )

with rasterio.open(folder / "streams.tif") as src:
    streams = src.read(1, masked=True)

    stream_mask = (
        ~np.ma.getmaskarray(streams)
        & (streams.filled(0) != 0)
    )

    stream_pixels = np.count_nonzero(stream_mask)

print("\n===== CORRECTED STREAM NETWORK =====")
print("Valid DEM pixels:", dem_valid_pixels)
print("Stream pixels:", stream_pixels)

print("Stream density (%):",
      round(stream_pixels / dem_valid_pixels * 100, 3))

print("Stream raster area (km2):",
      round(stream_pixels * pixel_area / 1e6, 3))

print("DEM area (km2):",
      round(dem_valid_pixels * pixel_area / 1e6, 3))