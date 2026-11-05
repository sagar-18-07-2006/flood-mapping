import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.plot import show

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

fig, ax = plt.subplots(figsize=(14, 12))

with rasterio.open(hand_path) as src:
    show(
        src,
        ax=ax,
        cmap="terrain",
        vmin=0,
        vmax=200,
        title="High-HAND SAR Candidate Clusters"
    )

waterways.plot(
    ax=ax,
    color="blue",
    linewidth=0.7,
    label="OSM waterways"
)

water.plot(
    ax=ax,
    color="cyan",
    edgecolor="darkblue",
    alpha=0.5,
    label="OSM waterbodies"
)

clusters.boundary.plot(
    ax=ax,
    color="red",
    linewidth=1.2,
    label="SAR candidate clusters"
)

for _, row in clusters.iterrows():
    centroid = row.geometry.centroid
    ax.annotate(
        str(int(row["cluster_id"])),
        (centroid.x, centroid.y),
        fontsize=7,
        color="black"
    )

ax.legend()
ax.set_xlabel("UTM Easting (m)")
ax.set_ylabel("UTM Northing (m)")
ax.set_title("Bengaluru: High-HAND SAR Candidate Investigation")

plt.tight_layout()

output = OUT / "high_hand_clusters_osm_overlay.png"
plt.savefig(output, dpi=250, bbox_inches="tight")
plt.close()

print("Saved:", output)