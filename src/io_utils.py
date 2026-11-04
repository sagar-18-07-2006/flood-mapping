from pathlib import Path
import rasterio
from rasterio.transform import from_bounds

def read_raster(path):
    with rasterio.open(path) as src:
        return (
            src.read(1),
            src.profile.copy(),
            src.transform,
            src.crs,
            src.nodata,
        )

def write_raster(path, array, profile, dtype=None, nodata=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        compress="deflate",
        nodata=nodata,
    )
    if dtype:
        p["dtype"] = dtype
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array, 1)

def demo_profile(width=300, height=300, bounds=(77.45, 12.80, 77.80, 13.15)):
    return {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": width,
        "height": height,
        "crs": "EPSG:4326",
        "transform": from_bounds(*bounds, width, height),
    }
