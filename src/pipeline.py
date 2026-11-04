import argparse
from pathlib import Path
import numpy as np

from .config import CONFIG, ROOT
from .io_utils import write_raster, demo_profile
from .flood_detection import detect_change, apply_terrain_filter

def run_demo():
    out_dir = ROOT / CONFIG["paths"]["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    h, w = 300, 300

    before = -12 + rng.normal(0, 1.5, (h, w)).astype("float32")
    after = before.copy()

    yy, xx = np.ogrid[:h, :w]
    flood_region = (
        ((xx - 165) ** 2 / 55**2) +
        ((yy - 155) ** 2 / 75**2)
    ) < 1
    after[flood_region] -= 6.0

    mask, change = detect_change(
        before,
        after,
        threshold_db=CONFIG["processing"]["flood_threshold_db"],
        min_pixels=CONFIG["processing"]["min_connected_pixels"],
    )

    slope = 2 + 3 * (yy / h) + np.zeros((h, w))
    final_mask = apply_terrain_filter(
        mask,
        slope,
        CONFIG["processing"]["max_slope_degrees"],
    )

    profile = demo_profile(w, h)
    write_raster(
        out_dir / "demo_sar_change_db.tif",
        change,
        profile,
        dtype="float32",
        nodata=-9999,
    )
    write_raster(
        out_dir / "demo_flood_mask.tif",
        final_mask,
        {**profile, "dtype": "uint8"},
        dtype="uint8",
        nodata=0,
    )

    print(f"Demo complete. Flood candidate pixels: {int(final_mask.sum())}")
    print(f"Outputs: {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Bengaluru Flood Mapping pipeline")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
