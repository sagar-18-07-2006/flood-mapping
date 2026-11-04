
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("data/processed/flood_detection")

files = [
    ("VV change (dB)", base / "vv_change_db.tif"),
    ("VH change (dB)", base / "vh_change_db.tif"),
    ("SAR flood candidates", base / "flood_candidate_sar.tif"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (title, path) in zip(axes, files):
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True)

    if "candidates" in title:
        image = ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    else:
        image = ax.imshow(
            arr,
            cmap="RdYlBu_r",
            vmin=-10,
            vmax=10
        )
        plt.colorbar(image, ax=ax, shrink=0.7, label="Change (dB)")

    ax.set_title(title)
    ax.set_axis_off()

plt.tight_layout()

output = "data/processed/flood_detection/sar_change_overview.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
print("Saved:", output)
