import numpy as np
from src.flood_detection import detect_change

def test_detects_strong_backscatter_drop():
    before = np.full((20, 20), -10.0, dtype=float)
    after = before.copy()
    after[5:15, 5:15] -= 5.0

    mask, change = detect_change(
        before, after, threshold_db=-2.0, min_pixels=5
    )

    assert mask.sum() > 0
    assert change[10, 10] <= -5.0
