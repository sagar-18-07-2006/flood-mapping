from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

BASE = Path("data/processed")
TERRAIN = BASE / "terrain"
FLOOD = BASE / "flood_detection"
OUT = BASE / "terrain_susceptibility"

OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "hand": TERRAIN / "hand.tif",
    "slope": TERRAIN / "slope_degrees.tif",
    "twi": TERRAIN / "twi.tif",
    "flow": TERRAIN / "flow_accumulation_cells.tif",
}

SAR_FILE = FLOOD / "flood_candidate_no_osm_water.tif"

# --------------------------------------------------
# 1. Read terrain layers and verify their grids
# --------------------------------------------------

arrays = {}
profile = None
reference = None

for name, path in FILES.items():
    with rasterio.open(path) as ds:
        if reference is None:
            reference = ds
            profile = ds.profile.copy()
            reference_transform = ds.transform
            reference_crs = ds.crs
            height, width = ds.height, ds.width
        else:
            assert ds.crs == reference_crs, f"{name}: CRS mismatch"
            assert ds.transform == reference_transform, \
                f"{name}: transform mismatch"
            assert ds.width == width and ds.height == height, \
                f"{name}: dimension mismatch"

        arr = ds.read(1).astype("float32")
        valid = np.isfinite(arr)

        if ds.nodata is not None:
            valid &= arr != ds.nodata

        arrays[name] = (arr, valid)

print("Terrain grid:", width, "x", height)
print("CRS:", reference_crs)
print("Resolution:", reference.res)

# --------------------------------------------------
# 2. Common valid-data mask
# --------------------------------------------------

valid = np.ones((height, width), dtype=bool)

for arr, mask in arrays.values():
    valid &= mask

hand, _ = arrays["hand"]
slope, _ = arrays["slope"]
twi, _ = arrays["twi"]
flow, _ = arrays["flow"]

# Physically meaningful screening constraints.
valid &= (hand >= 0)
valid &= (slope >= 0)
valid &= (twi > 0)
valid &= (flow >= 1)

print("Common valid terrain pixels:", int(valid.sum()))

# --------------------------------------------------
# 3. Percentile normalization
# --------------------------------------------------

def normalize_percentile(values, mask, reverse=False):
    sample = values[mask]

    low, high = np.percentile(sample, [5, 95])

    if high <= low:
        raise ValueError("Invalid percentile range")

    score = np.clip(
        (values - low) / (high - low),
        0,
        1
    )

    if reverse:
        score = 1 - score

    print(
        "Percentile limits:",
        round(float(low), 4),
        round(float(high), 4)
    )

    return score.astype("float32")


print("\nNormalizing HAND...")
hand_score = normalize_percentile(
    hand, valid, reverse=True
)

print("\nNormalizing slope...")
slope_score = normalize_percentile(
    slope, valid, reverse=True
)

print("\nNormalizing TWI...")
twi_score = normalize_percentile(
    twi, valid, reverse=False
)

print("\nNormalizing log flow accumulation...")
log_flow = np.log1p(flow)

flow_score = normalize_percentile(
    log_flow, valid, reverse=False
)

# --------------------------------------------------
# 4. Equal-weight exploratory index
# --------------------------------------------------

index = np.full((height, width), -9999, dtype="float32")

index[valid] = (
    hand_score[valid]
    + slope_score[valid]
    + twi_score[valid]
    + flow_score[valid]
) / 4

# Quintile classes based on the valid study-area pixels.
class_limits = np.percentile(
    index[valid],
    [20, 40, 60, 80]
)

susceptibility = np.zeros((height, width), dtype="uint8")

susceptibility[valid & (index <= class_limits[0])] = 1
susceptibility[valid & (index > class_limits[0]) &
               (index <= class_limits[1])] = 2
susceptibility[valid & (index > class_limits[1]) &
               (index <= class_limits[2])] = 3
susceptibility[valid & (index > class_limits[2]) &
               (index <= class_limits[3])] = 4
susceptibility[valid & (index > class_limits[3])] = 5

# --------------------------------------------------
# 5. Write outputs
# --------------------------------------------------

def save_raster(name, data, dtype, nodata):
    output_profile = profile.copy()

    # Remove inherited block settings from the source raster.
    output_profile.pop("blockxsize", None)
    output_profile.pop("blockysize", None)

    output_profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="DEFLATE",
        tiled=True,
        blockxsize=256,
        blockysize=256
    )

    path = OUT / name

    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(data.astype(dtype), 1)

    print("Saved:", path)


save_raster(
    "terrain_susceptibility_index.tif",
    index,
    "float32",
    -9999
)

save_raster(
    "terrain_susceptibility_classes.tif",
    susceptibility,
    "uint8",
    0
)

# --------------------------------------------------
# 6. Aggregate SAR candidates to native 30 m grid
# --------------------------------------------------

# --------------------------------------------------
# 6. Aggregate SAR candidates to native 30 m grid
# --------------------------------------------------

print("\nAggregating SAR candidates to terrain grid...")

with rasterio.open(SAR_FILE) as sar:
    sar_data = sar.read(1).astype("float32")

    sar_valid = np.isfinite(sar_data)

    if sar.nodata is not None:
        sar_valid &= sar_data != sar.nodata

    sar_valid &= (sar_data == 0) | (sar_data == 1)

    # Reproject candidate values and valid-data coverage separately.
    numerator_source = np.where(
        sar_valid, sar_data, 0
    ).astype("float32")

    denominator_source = sar_valid.astype("float32")

    numerator = np.zeros(
        (height, width), dtype="float32"
    )

    denominator = np.zeros(
        (height, width), dtype="float32"
    )

    for source, destination in [
        (numerator_source, numerator),
        (denominator_source, denominator),
    ]:
        reproject(
            source=source,
            destination=destination,
            src_transform=sar.transform,
            src_crs=sar.crs,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            resampling=Resampling.average
        )

    candidate_fraction = np.full(
        (height, width), -9999, dtype="float32"
    )

    available = denominator > 0

    candidate_fraction[available] = (
        numerator[available] / denominator[available]
    )

save_raster(
    "sar_candidate_fraction_30m.tif",
    candidate_fraction,
    "float32",
    -9999
)

# --------------------------------------------------
# 7. Compare SAR occurrence by susceptibility class
# --------------------------------------------------

comparison_mask = valid & available

records = []

for cls in range(1, 6):
    selected = comparison_mask & (susceptibility == cls)

    fractions = candidate_fraction[selected]

    if fractions.size == 0:
        continue

    records.append({
        "susceptibility_class": cls,
        "terrain_pixels": int(selected.sum()),
        "mean_sar_candidate_fraction": float(fractions.mean()),
        "pixels_with_any_candidate": int(
            np.sum(fractions > 0)
        ),
        "candidate_presence_percent": float(
            100 * np.mean(fractions > 0)
        ),
    })

comparison = pd.DataFrame(records)

comparison.to_csv(
    OUT / "sar_by_susceptibility_class.csv",
    index=False
)

print("\n========== SUSCEPTIBILITY CLASS LIMITS ==========")
print(class_limits)

print("\n========== SAR COMPARISON ==========")
print(comparison.to_string(index=False))

print("\nAll outputs saved in:", OUT)
print("Terrain susceptibility analysis complete.")
