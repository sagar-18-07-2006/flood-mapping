import numpy as np
from scipy.ndimage import binary_opening, binary_closing, label

def db_ratio(after_db, before_db):
    return after_db - before_db

def remove_small_components(mask, min_pixels=25):
    labeled, n = label(mask.astype(bool))
    if n == 0:
        return mask.astype(bool)
    counts = np.bincount(labeled.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    return keep[labeled]

def clean_mask(mask, min_pixels=25):
    x = binary_opening(mask)
    x = binary_closing(x)
    return remove_small_components(x, min_pixels)
