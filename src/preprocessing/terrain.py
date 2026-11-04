"""Terrain and hydrology derivatives using WhiteboxTools.

Input DEM must be a clipped, projected GeoTIFF in a metric CRS (e.g. EPSG:32643).
The DEM should cover the study area plus a hydrological buffer.
"""
from pathlib import Path
import argparse

import rasterio
from whitebox import WhiteboxTools


def _run(wbt: WhiteboxTools, tool: str, **kwargs) -> None:
    args = []
    for key, value in kwargs.items():
        if isinstance(value, (str, Path)):
            args.append(f'--{key}="{value}"')
        else:
            args.append(f"--{key}={value}")
    code = wbt.run_tool(tool, args)
    if code != 0:
        raise RuntimeError(f"WhiteboxTools operation failed: {tool} (exit {code})")


def validate_dem(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"DEM not found: {path}")
    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise ValueError("DEM has no CRS. Georeference it before processing.")
        if not ds.crs.is_projected:
            raise ValueError(
                f"DEM CRS is {ds.crs}; reproject to a metric projected CRS first "
                "(for Bengaluru, use EPSG:32643)."
            )
        if ds.width < 3 or ds.height < 3:
            raise ValueError("DEM is too small for terrain derivatives.")


def process_dem(dem_path: str, output_dir: str, stream_threshold: int = 1000) -> None:
    dem = Path(dem_path).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    validate_dem(dem)

    wbt = WhiteboxTools()
    wbt.set_working_dir(str(out))
    wbt.verbose = True

    dem_hydro = out / "dem_hydroconditioned.tif"
    slope = out / "slope_degrees.tif"
    aspect = out / "aspect_degrees.tif"
    pointer = out / "d8_pointer.tif"
    accumulation = out / "flow_accumulation_cells.tif"
    sca = out / "specific_contributing_area.tif"
    streams = out / "streams.tif"
    twi = out / "twi.tif"
    hand = out / "hand.tif"

    # Terrain derivatives use the original DEM. Flow routing uses a conditioned copy.
    _run(wbt, "slope", dem=dem, output=slope, zfactor=1.0, units="degrees")
    _run(wbt, "aspect", dem=dem, output=aspect)
    _run(wbt, "breach_depressions", dem=dem, output=dem_hydro)
    _run(wbt, "d8_pointer", dem=dem_hydro, output=pointer)
    _run(wbt, "d8_flow_accumulation", input=dem_hydro,
         output=accumulation, out_type="cells")
    _run(wbt, "d8_flow_accumulation", input=dem_hydro,
         output=sca, out_type="specific contributing area")
    _run(wbt, "extract_streams", flow_accum=accumulation,
         threshold=stream_threshold, output=streams)
    _run(wbt, "wetness_index", sca=sca, slope=slope, output=twi)
    _run(wbt, "elevation_above_stream", dem=dem_hydro,
         streams=streams, output=hand)

    print("\nTerrain processing completed.")
    print(f"Outputs written to: {out}")
    print("Inspect the drainage network and tune stream threshold before analysis.")


def main():
    parser = argparse.ArgumentParser(description="Generate DEM terrain/hydrology layers.")
    parser.add_argument("--dem", required=True, help="Projected DEM GeoTIFF (metres).")
    parser.add_argument("--output-dir", default="data/processed/terrain")
    parser.add_argument("--stream-threshold", type=int, default=1000,
                        help="D8 contributing-cell threshold for stream extraction.")
    args = parser.parse_args()
    process_dem(args.dem, args.output_dir, args.stream_threshold)


if __name__ == "__main__":
    main()
