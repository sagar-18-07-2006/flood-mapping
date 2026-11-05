from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

BASE = Path("data/processed")
TERRAIN = BASE / "terrain/aligned"
FLOOD = BASE / "flood_detection"

hand_path = TERRAIN / "hand_10m.tif"
streams_path = TERRAIN / "streams_10m.tif"
candidate_path = FLOOD / "flood_candidate_no_osm_water.tif"

out_dir = FLOOD / "terrain_relationship"
out_dir.mkdir(parents=True, exist_ok=True)

# Read at reduced resolution for plotting to limit memory use.
scale = 5

with rasterio.open(hand_path) as ds:
    hand = ds.read(
        1,
        out_shape=(ds.height // scale, ds.width // scale),
        masked=True
    )
    transform = ds.transform * ds.transform.scale(
        ds.width / hand.shape[1],
        ds.height / hand.shape[0]
    )
    bounds = rasterio.transform.array_bounds(
        hand.shape[0], hand.shape[1], transform
    )

with rasterio.open(streams_path) as ds:
    streams = ds.read(
        1,
        out_shape=(ds.height // scale, ds.width // scale),
        masked=True,
        resampling=rasterio.enums.Resampling.nearest
    )

with rasterio.open(candidate_path) as ds:
    candidates = ds.read(
        1,
        out_shape=(ds.height // scale, ds.width // scale),
        masked=True,
        resampling=rasterio.enums.Resampling.nearest
    )

hand_data = hand.filled(np.nan)
stream_data = streams.filled(0)
candidate_data = candidates.filled(0)

stream_mask = (stream_data > 0) & ~np.ma.getmaskarray(streams)

# Approximate distance at the reduced plotting resolution.
distance = distance_transform_edt(
    ~stream_mask,
    sampling=(10 * scale, 10 * scale)
)

left, bottom, right, top = bounds
extent = [left / 1000, right / 1000, bottom / 1000, top / 1000]

fig, axes = plt.subplots(1, 3, figsize=(19, 7))

im0 = axes[0].imshow(
    hand_data,
    extent=extent,
    origin="upper",
    cmap="terrain",
    vmin=0,
    vmax=100
)
axes[0].set_title("HAND elevation (m)")
fig.colorbar(im0, ax=axes[0], shrink=0.75, label="Metres")

im1 = axes[1].imshow(
    distance,
    extent=extent,
    origin="upper",
    cmap="viridis",
    vmin=0,
    vmax=1000
)
axes[1].set_title("Distance to DEM streams (m)")
fig.colorbar(im1, ax=axes[1], shrink=0.75, label="Metres")

candidate_plot = np.ma.masked_where(
    candidate_data <= 0,
    candidate_data
)
axes[2].imshow(
    candidate_plot,
    extent=extent,
    origin="upper",
    cmap="Blues",
    vmin=1,
    vmax=1
)
axes[2].set_title("SAR flood candidates")
axes[2].text(
    0.02, 0.02,
    "After OSM waterbody exclusion",
    transform=axes[2].transAxes,
    color="black",
    fontsize=9,
    bbox={"facecolor": "white", "alpha": 0.8}
)

for ax in axes:
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")

fig.suptitle(
    "Bengaluru: SAR Flood Candidates and Terrain Context",
    fontsize=15
)
fig.tight_layout()

output = out_dir / "terrain_flood_comparison.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print("Saved:", output)
