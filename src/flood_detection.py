import numpy as np
from .raster_utils import db_ratio, clean_mask

def detect_change(before_db, after_db, threshold_db=-2.0, min_pixels=25):
    # Baseline: inundation often causes a decrease in SAR backscatter.
    # Pixels with change <= threshold are marked as candidate flooded pixels.
    change = db_ratio(after_db, before_db)
    candidate = np.isfinite(change) & (change <= threshold_db)
    mask = clean_mask(candidate, min_pixels=min_pixels)
    return mask.astype("uint8"), change.astype("float32")

def apply_terrain_filter(mask, slope_degrees, max_slope=8.0):
    valid = np.isfinite(slope_degrees) & (slope_degrees <= max_slope)
    return (mask.astype(bool) & valid).astype("uint8")
