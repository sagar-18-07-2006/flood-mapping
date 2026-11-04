# Bengaluru Flood Area Mapping

GIS + Remote Sensing + AI prototype for flood extent mapping.

## Pipeline
Sentinel-1 SAR -> preprocessing -> flood/change detection -> DEM/slope filtering
-> OSM exposure analysis -> GeoTIFF/GeoJSON outputs -> web GIS visualization.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --demo
pytest
```

The demo creates synthetic before/after SAR data and runs the detection pipeline.
Replace the synthetic inputs with real Sentinel-1 data in the next phase.

## Initial AOI
Bengaluru, Karnataka. The bounding box in `config.yaml` is an approximate starting AOI.
Use the final project AOI before producing scientific results.

## Data sources planned
- Sentinel-1 GRD: Copernicus Data Space / Sentinel Hub
- DEM: Copernicus DEM or authoritative equivalent
- Rainfall: IMD / ERA5-Land / GPM
- River/lake levels: authoritative local sources and/or GloFAS
- OSM: Geofabrik / Overpass

A simple backscatter threshold is only a baseline detector and must be validated before
being treated as an operational flood product.
