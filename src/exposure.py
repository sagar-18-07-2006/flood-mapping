import geopandas as gpd

def reproject_to(gdf, crs):
    if gdf.crs is None:
        raise ValueError("Input vector has no CRS.")
    return gdf.to_crs(crs)

def affected_features(flood_gdf, feature_gdf):
    flood = flood_gdf.to_crs(feature_gdf.crs)
    return gpd.sjoin(
        feature_gdf,
        flood[["geometry"]],
        predicate="intersects",
        how="inner",
    )
