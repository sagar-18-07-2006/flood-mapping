# DEM workflow — Bengaluru

## 1. Obtain DEM tiles

Use Copernicus DEM GLO-30 where access is available. It is a **digital surface model (DSM)**, so vegetation and buildings may affect elevation. For urban drainage interpretation, compare with a bare-earth terrain product such as FABDEM or authoritative local elevation data where available.

Official information:
- https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM
- https://documentation.dataspace.copernicus.eu/Data/Others/CCM.html

Keep downloaded original tiles in `data/raw/dem/`. Download every tile intersecting the buffered processing bounds.

## 2. Install WhiteboxTools

```bash
pip install whitebox
```

The Python wrapper also needs the WhiteboxTools executable. Check the package installation instructions if the executable is not bundled; configure its directory with `wbt.set_whitebox_dir(...)` if installed separately.

## 3. Mosaic, reproject and clip

From the project root, activate `.venv`, then run (replace filenames with the actual downloaded tiles):

```bash
python scripts/prepare_dem.py \
  --inputs data/raw/dem/tile_1.tif data/raw/dem/tile_2.tif \
  --output data/processed/dem_bengaluru_utm43.tif
```

The default processing extent adds a small buffer around the project's approximate Bengaluru AOI. The output is reprojected to UTM zone 43N (metres) at a nominal 30 m pixel size.

## 4. Generate terrain and hydrology rasters

```bash
python -m src.preprocessing.terrain \
  --dem data/processed/dem_bengaluru_utm43.tif \
  --output-dir data/processed/terrain \
  --stream-threshold 1000
```

Outputs:
- `slope_degrees.tif`
- `aspect_degrees.tif`
- `dem_hydroconditioned.tif`
- `d8_pointer.tif`
- `flow_accumulation_cells.tif`
- `specific_contributing_area.tif`
- `streams.tif`
- `twi.tif`
- `hand.tif`

## Scientific cautions

- Preserve the original DEM. Breaching depressions modifies elevations to create drainage paths; inspect changes around lakes, reservoirs, roads and embankments.
- The stream threshold is only a starting value. Inspect extracted streams and test several thresholds.
- HAND is elevation above the extracted stream network, not a direct flood-depth estimate.
- TWI depends on flow routing, contributing area, slope and flat-cell treatment.
- A 30 m DEM cannot resolve Bengaluru's storm drains, culverts or many road embankments.
- Terrain derivatives describe drainage and topographic susceptibility. They do not independently predict a specific flood event.
