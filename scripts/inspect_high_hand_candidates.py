import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

BASE = Path("data/processed")
TERRAIN = BASE / "terrain"
OUT = BASE / "terrain_susceptibility"
OUT.mkdir(exist_ok=True)

paths = {
    "HAND": TERRAIN / "hand.tif",
    "Slope": TERRAIN / "slope_degrees.tif",
    "TWI": TERRAIN / "twi.tif",
    "Flow accumulation": TERRAIN / "flow_accumulation_cells.tif",
    "SAR fraction": OUT / "sar_candidate_fraction_30m.tif",
}

data = {}
valid_masks = {}

for name, path in paths.items():
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float64")
        mask = np.isfinite(arr)

        if ds.nodata is not None:
            mask &= arr != ds.nodata

        if name == "HAND":
            transform = ds.transform
            profile = ds.profile.copy()

        if name == "Flow accumulation":
            mask &= arr >= 1

        if name == "TWI":
            mask &= arr > 0

        if name == "SAR fraction":
            mask &= (arr >= 0) & (arr <= 1)

        data[name] = arr
        valid_masks[name] = mask

# Common valid mask for analysis.
valid = np.logical_and.reduce(list(valid_masks.values()))

hand = data["HAND"]
sar = data["SAR fraction"]

limits = np.percentile(hand[valid], [20, 40, 60, 80])

high_hand = valid & (hand > limits[3])
high_candidates = high_hand & (sar > 0)

print("High HAND boundary:", limits[3])
print("High HAND valid cells:", int(high_hand.sum()))
print("High HAND candidate cells:", int(high_candidates.sum()))
print("High HAND candidate percentage:",
      round(100 * high_candidates.sum() / high_hand.sum(), 3))

for name in ["HAND", "Slope", "TWI", "Flow accumulation"]:
    arr = data[name]
    mask = high_candidates & valid_masks[name]

    print(
        name,
        "median:", float(np.median(arr[mask])),
        "mean:", float(np.mean(arr[mask])),
        "valid samples:", int(mask.sum())
    )

# Identify connected high-HAND candidate clusters.
labels, count = ndimage.label(
    high_candidates,
    structure=np.ones((3, 3), dtype=int)
)

sizes = np.bincount(labels.ravel())[1:]
largest = np.sort(sizes)[::-1][:10]

print("\nLargest connected high-HAND candidate clusters:")
for i, size in enumerate(largest, 1):
    print(f"{i}. {size} cells ({size * 900 / 1e6:.3f} km²)")

# Plot four diagnostic panels.
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

plots = [
    ("HAND", "HAND (m)", "viridis"),
    ("SAR fraction", "SAR candidate fraction", "Reds"),
    ("Slope", "Slope (degrees)", "terrain"),
    ("Flow accumulation", "log10(flow accumulation + 1)", "Blues"),
]

for ax, (name, label, cmap) in zip(axes.ravel(), plots):
    arr = data[name].copy()
    arr[~valid_masks[name]] = np.nan

    if name == "Flow accumulation":
        arr = np.log10(arr + 1)

    image = ax.imshow(arr, cmap=cmap)
    ax.set_title(label)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046)

fig.suptitle(
    "Bengaluru Terrain and SAR Candidate Diagnostic",
    fontsize=15
)
fig.tight_layout()

output = OUT / "high_hand_diagnostic.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nSaved:", output)
