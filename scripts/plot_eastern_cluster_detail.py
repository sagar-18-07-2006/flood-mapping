import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.plot import show
from matplotlib.colors import Normalize

BASE = Path("data/processed")
OUT = BASE / "terrain_susceptibility"

clusters = gpd.read_file(
    OUT / "high_hand_clusters_osm.gpkg",
    layer="high_hand_clusters",
    engine="pyogrio"
)

hydro = BASE / "bengaluru_hydrography.gpkg"

water = gpd.read_file(
    hydro, layer="waterbodies", engine="pyogrio"
).to_crs(clusters.crs)

waterways = gpd.read_file(
    hydro, layer="waterways", engine="pyogrio"
).to_crs(clusters.crs)

hand_path = BASE / "terrain" / "hand.tif"

# Eastern corridor bounding box in EPSG:32643
xmin, xmax = 798000, 805000
ymin, ymax = 1433000, 1460000

fig, ax = plt.subplots(figsize=(13, 12))

with rasterio.open(hand_path) as src:
    window = rasterio.windows.from_bounds(
        xmin, ymin, xmax, ymax,
        transform=src.transform
    )
    data = src.read(1, window=window, masked=True)
    transform = src.window_transform(window)

    show(
        data,
        transform=transform,
        ax=ax,
        cmap="terrain",
        norm=Normalize(0, 200)
    )

clusters.plot(
    ax=ax,
    facecolor="none",
    edgecolor="red",
    linewidth=1.5
)

waterways.clip(
    (xmin, ymin, xmax, ymax)
).plot(
    ax=ax,
    color="blue",
    linewidth=0.8
)

water.clip(
    (xmin, ymin, xmax, ymax)
).plot(
    ax=ax,
    color="cyan",
    edgecolor="darkblue",
    alpha=0.6
)

selected = clusters.cx[xmin:xmax, ymin:ymax]

for _, row in selected.iterrows():
    c = row.geometry.centroid
    ax.annotate(
        str(int(row.cluster_id)),
        (c.x, c.y),
        fontsize=9,
        color="black",
        bbox=dict(
            facecolor="white",
            alpha=0.7,
            edgecolor="none"
        )
    )

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_title("Eastern Corridor: High-HAND SAR Candidate Clusters")
ax.set_xlabel("UTM Easting (m)")
ax.set_ylabel("UTM Northing (m)")

plt.tight_layout()

output = OUT / "eastern_cluster_detail.png"
plt.savefig(output, dpi=250, bbox_inches="tight")
plt.close()

print("Saved:", output)