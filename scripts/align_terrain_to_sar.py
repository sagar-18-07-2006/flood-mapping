from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

reference_path = Path(
    "data/processed/sentinel1/aligned/S1A_20220812_bengaluru.tif"
)

hand_path = Path("data/processed/terrain/hand.tif")
streams_path = Path("data/processed/terrain/streams.tif")

output_dir = Path("data/processed/terrain/aligned")
output_dir.mkdir(parents=True, exist_ok=True)

outputs = [
    (hand_path, output_dir / "hand_10m.tif", Resampling.bilinear),
    (streams_path, output_dir / "streams_10m.tif", Resampling.nearest),
]

with rasterio.open(reference_path) as ref:
    target_crs = ref.crs
    target_transform = ref.transform
    target_width = ref.width
    target_height = ref.height

    print("Reference grid:")
    print("CRS:", target_crs)
    print("Dimensions:", target_width, "x", target_height)
    print("Transform:", target_transform)

    for source_path, output_path, resampling in outputs:
        print(f"\nProcessing: {source_path}")

        with rasterio.open(source_path) as src:
            destination = np.full(
                (target_height, target_width),
                -9999,
                dtype=np.float32
            )

            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=target_transform,
                dst_crs=target_crs,
                dst_nodata=-9999,
                resampling=resampling,
                init_dest_nodata=True,
            )

            profile = {
                "driver": "GTiff",
                "height": target_height,
                "width": target_width,
                "count": 1,
                "dtype": "float32",
                "crs": target_crs,
                "transform": target_transform,
                "nodata": -9999,
                "compress": "DEFLATE",
                "tiled": True,
            }

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(destination, 1)

            valid = destination[destination != -9999]

            print("Output:", output_path)
            print("Resampling:", resampling.name)
            print("Valid pixels:", valid.size)

            if valid.size:
                print("Minimum:", float(valid.min()))
                print("Maximum:", float(valid.max()))
                print("Unique values (first 15):",
                      np.unique(valid)[:15])

print("\nTerrain alignment complete.")
