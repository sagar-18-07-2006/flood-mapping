import geopandas as gpd
from pathlib import Path

source = Path("data/processed/bengaluru_hydrography.geojson")
output = Path("data/processed/bengaluru_hydrography.gpkg")

gdf = gpd.read_file(source)

print("Input features:", len(gdf))
print("Input CRS:", gdf.crs)

# Keep only relevant waterway lines
waterway_types = [
    "river", "stream", "canal", "drain", "ditch"
]

waterways = gdf[
    gdf["waterway"].isin(waterway_types)
    & gdf.geom_type.isin(["LineString", "MultiLineString"])
].copy()

# Keep mapped water-body polygons
waterbodies = gdf[
    (
        (gdf["natural"] == "water")
        | gdf["water"].notna()
    )
    & gdf.geom_type.isin(["Polygon", "MultiPolygon"])
].copy()

# Water infrastructure points and lines
infrastructure_types = [
    "weir", "dam", "waterfall", "sluice_gate", "lock_gate"
]

infrastructure = gdf[
    gdf["waterway"].isin(infrastructure_types)
].copy()

# Reproject all layers to match the DEM
target_crs = "EPSG:32643"

layers = {
    "waterways": waterways,
    "waterbodies": waterbodies,
    "water_infrastructure": infrastructure,
}

for name, layer in layers.items():
    layer = layer.to_crs(target_crs)
    layer = layer[layer.geometry.notna() & ~layer.geometry.is_empty]

    if name == "waterways":
        layer = layer.explode(index_parts=False)
    elif name == "waterbodies":
        layer = layer.explode(index_parts=False)

    layers[name] = layer

# Write separate layers into one GeoPackage
if output.exists():
    output.unlink()

for name, layer in layers.items():
    layer.to_file(
        output,
        layer=name,
        driver="GPKG",
        mode="w" if name == "waterways" else "a",
    )
    print(f"{name}: {len(layer)} features")

print("\nSaved:", output)
print("Target CRS:", target_crs)
