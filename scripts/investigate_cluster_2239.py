
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from matplotlib.colors import Normalize


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

BASE = Path("data/processed")

PRE_PATH = (
    BASE / "sentinel1/aligned/S1A_20220812_bengaluru.tif"
)

POST_PATH = (
    BASE / "sentinel1/aligned/S1A_20220905_bengaluru.tif"
)

HAND_PATH = BASE / "terrain/aligned/hand_10m.tif"

STREAM_PATH = BASE / "terrain/aligned/streams_10m.tif"

CLUSTER_PATH = (
    BASE / "terrain_susceptibility/"
    "high_hand_clusters_osm.gpkg"
)

HYDRO_PATH = BASE / "bengaluru_hydrography.gpkg"

OUT = BASE / "cluster_investigation/cluster_2239"
OUT.mkdir(parents=True, exist_ok=True)

CLUSTER_ID = 2239
BUFFER_M = 500


# ---------------------------------------------------------
# 2. LOAD CLUSTER GEOMETRY
# ---------------------------------------------------------

clusters = gpd.read_file(
    CLUSTER_PATH,
    layer="high_hand_clusters",
    engine="pyogrio"
)

selected = clusters[
    clusters["cluster_id"] == CLUSTER_ID
]

if len(selected) != 1:
    raise ValueError(
        f"Expected exactly one cluster {CLUSTER_ID}, "
        f"found {len(selected)}"
    )

cluster = selected.geometry.iloc[0]
cluster_crs = clusters.crs

if cluster.is_empty:
    raise ValueError("Cluster geometry is empty")

print("Cluster:", CLUSTER_ID)
print("Cluster area (km2):", cluster.area / 1e6)
print("Cluster CRS:", cluster_crs)


# ---------------------------------------------------------
# 3. LOAD OSM HYDROGRAPHY
# ---------------------------------------------------------

water = gpd.read_file(
    HYDRO_PATH,
    layer="waterbodies",
    engine="pyogrio"
).to_crs(cluster_crs)

waterways = gpd.read_file(
    HYDRO_PATH,
    layer="waterways",
    engine="pyogrio"
).to_crs(cluster_crs)


# ---------------------------------------------------------
# 4. READ AND VERIFY RASTER ALIGNMENT
# ---------------------------------------------------------

raster_paths = [
    PRE_PATH,
    POST_PATH,
    HAND_PATH,
    STREAM_PATH
]

datasets = []

for path in raster_paths:
    if not path.exists():
        raise FileNotFoundError(path)

    datasets.append(rasterio.open(path))

reference = datasets[0]

for ds in datasets[1:]:
    if (
        ds.crs != reference.crs
        or ds.transform != reference.transform
        or ds.width != reference.width
        or ds.height != reference.height
    ):
        raise ValueError(
            f"Raster grid mismatch: {ds.name}"
        )

if reference.count < 2:
    raise ValueError("Expected VV and VH bands")

if reference.crs != cluster_crs:
    raise ValueError(
        "Cluster and raster CRS do not match"
    )

print("\nRaster CRS:", reference.crs)
print("Raster resolution:", reference.res)
print("Raster dimensions:", reference.width, reference.height)


# ---------------------------------------------------------
# 5. CREATE LOCAL ANALYSIS WINDOW
# ---------------------------------------------------------

# Include a 500 m surrounding area for comparison.
analysis_geometry = cluster.buffer(BUFFER_M)

xmin, ymin, xmax, ymax = analysis_geometry.bounds

window = from_bounds(
    xmin, ymin, xmax, ymax,
    transform=reference.transform
)

window = window.round_offsets().round_lengths()

# Keep window within raster boundaries.
window = window.intersection(
    rasterio.windows.Window(
        0, 0,
        reference.width,
        reference.height
    )
)

if window.width <= 0 or window.height <= 0:
    raise ValueError("Analysis window is outside raster")

transform = reference.window_transform(window)

# Exact map extent for plotting.
left = transform.c
top = transform.f
right = left + window.width * transform.a
bottom = top + window.height * transform.e

extent = [left, right, bottom, top]


# ---------------------------------------------------------
# 6. READ SENTINEL-1 PRE AND POST DATA
# ---------------------------------------------------------

pre = datasets[0]
post = datasets[1]
hand_ds = datasets[2]
stream_ds = datasets[3]

# SNAP output bands:
# Band 1 = Sigma0 VV (linear)
# Band 2 = Sigma0 VH (linear)

pre_vv = pre.read(1, window=window).astype("float32")
pre_vh = pre.read(2, window=window).astype("float32")

post_vv = post.read(1, window=window).astype("float32")
post_vh = post.read(2, window=window).astype("float32")

hand = hand_ds.read(1, window=window).astype("float32")
streams = stream_ds.read(1, window=window)

def valid_sigma(ds, array):
    nodata = ds.nodata
    valid = np.isfinite(array) & (array > 0)

    if nodata is not None:
        valid &= array != nodata

    return valid



# Independent validity masks: do not require a pixel to be a stream.

sar_valid = (
    valid_sigma(pre, pre_vv)
    & valid_sigma(pre, pre_vh)
    & valid_sigma(post, post_vv)
    & valid_sigma(post, post_vh)
)

hand_valid = np.isfinite(hand)

if hand_ds.nodata is not None:
    hand_valid &= hand != hand_ds.nodata

stream_valid = np.isfinite(streams.astype("float32"))

if stream_ds.nodata is not None and stream_ds.nodata != 0:
    stream_valid &= streams != stream_ds.nodata

# Joint mask only when comparing SAR and HAND together.
sar_hand_valid = sar_valid & hand_valid





# ---------------------------------------------------------
# 7. CONVERT SIGMA0 LINEAR TO dB
# ---------------------------------------------------------

def to_db(array):
    result = np.full(array.shape, np.nan, dtype="float32")

    good = np.isfinite(array) & (array > 0)

    result[good] = 10.0 * np.log10(array[good])

    return result


pre_vv_db = to_db(pre_vv)
post_vv_db = to_db(post_vv)

pre_vh_db = to_db(pre_vh)
post_vh_db = to_db(post_vh)


# ---------------------------------------------------------
# 8. CALCULATE BACKSCATTER CHANGE
# ---------------------------------------------------------

vv_change = post_vv_db - pre_vv_db
vh_change = post_vh_db - pre_vh_db

for array in [
    pre_vv_db, post_vv_db,
    pre_vh_db, post_vh_db,
    vv_change, vh_change
]:
    array[~sar_valid] = np.nan

hand[~hand_valid] = np.nan

# ---------------------------------------------------------
# 9. CREATE CLUSTER AND SURROUNDING RING MASKS
# ---------------------------------------------------------

inside = geometry_mask(
    [mapping(cluster)],
    out_shape=pre_vv.shape,
    transform=transform,
    invert=True,
    all_touched=False
)

outside = geometry_mask(
    [mapping(analysis_geometry)],
    out_shape=pre_vv.shape,
    transform=transform,
    invert=True,
    all_touched=False
)

ring = outside & ~inside

inside_sar = inside & sar_valid
ring_sar = ring & sar_valid

inside_hand = inside & hand_valid
ring_hand = ring & hand_valid

inside_joint = inside & sar_hand_valid
ring_joint = ring & sar_hand_valid

print("\nMASK DIAGNOSTICS")
print("Pixels inside cluster:", int(inside.sum()))
print("Valid SAR pixels inside:", int(inside_sar.sum()))
print("Valid HAND pixels inside:", int(inside_hand.sum()))
print("Joint valid pixels inside:", int(inside_joint.sum()))
print("Valid SAR pixels in ring:", int(ring_sar.sum()))
# ---------------------------------------------------------
# 10. STATISTICAL SUMMARY
# ---------------------------------------------------------

def summarize(name, array, mask):
    values = array[mask]
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "region": name,
            "valid_pixels": 0
        }

    return {
        "region": name,
        "valid_pixels": len(values),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90))
    }

statistics = []

variables = {
    "pre_VV_dB": (pre_vv_db, inside_sar, ring_sar),
    "post_VV_dB": (post_vv_db, inside_sar, ring_sar),
    "VV_change_dB": (vv_change, inside_sar, ring_sar),
    "pre_VH_dB": (pre_vh_db, inside_sar, ring_sar),
    "post_VH_dB": (post_vh_db, inside_sar, ring_sar),
    "VH_change_dB": (vh_change, inside_sar, ring_sar),
    "HAND_m": (hand, inside_hand, ring_hand),
}

for variable_name, (array, cluster_mask, ring_mask) in variables.items():

    statistics.append({
        "variable": variable_name,
        **summarize("cluster_2239", array, cluster_mask)
    })

    statistics.append({
        "variable": variable_name,
        **summarize("surrounding_500m_ring", array, ring_mask)
    })

statistics_df = pd.DataFrame(statistics)

statistics_path = OUT / "cluster_2239_statistics.csv"
statistics_df.to_csv(statistics_path, index=False)

print("\nSTATISTICS")
print(statistics_df.to_string(index=False))

# ---------------------------------------------------------
# 11. CANDIDATE THRESHOLD DIAGNOSTICS
# ---------------------------------------------------------

candidate_results = []

for threshold in [-2, -3, -4, -5]:
    candidate = (
    sar_valid
    & (vv_change <= threshold)
    & (vh_change <= threshold)
)

    selected_pixels = candidate & inside

    count = int(selected_pixels.sum())

    candidate_results.append({
        "threshold_db": threshold,
        "candidate_pixels_inside_cluster": count,
        "candidate_area_km2": count * 100 / 1e6,
        "cluster_area_km2": cluster.area / 1e6,
        "candidate_percent_of_cluster": (
            100 * count / inside.sum()
            if inside.sum() else 0
        )
    })

candidate_df = pd.DataFrame(candidate_results)

candidate_path = OUT / "cluster_2239_thresholds.csv"
candidate_df.to_csv(candidate_path, index=False)

print("\nTHRESHOLD SENSITIVITY")
print(candidate_df.to_string(index=False))


# ---------------------------------------------------------
# 12. HAND AND STREAM DIAGNOSTICS
# ---------------------------------------------------------

hand_values = hand[inside_hand]

if len(hand_values):
    print("\nHAND INSIDE CLUSTER")
    print("Median:", np.median(hand_values))
    print("Mean:", np.mean(hand_values))
    print("Minimum:", np.min(hand_values))
    print("Maximum:", np.max(hand_values))

# Stream raster may be binary or contain other positive values.
stream_valid = np.isfinite(streams.astype("float32"))

if stream_ds.nodata is not None:
    stream_valid &= streams != stream_ds.nodata

stream_pixels = (
    inside
    & stream_valid
    & (streams > 0)
)

print(
    "\nSTREAM PIXELS INSIDE CLUSTER:",
    int(stream_pixels.sum())
)


# OSM feature intersections and lengths.
water_intersections = water[
    water.geometry.intersects(cluster)
]

waterway_intersections = waterways[
    waterways.geometry.intersects(cluster)
]

print(
    "OSM waterbody features intersecting:",
    len(water_intersections)
)

print(
    "OSM waterway features intersecting:",
    len(waterway_intersections)
)

if len(waterway_intersections):
    lengths = [
        cluster.intersection(geom).length
        for geom in waterway_intersections.geometry
    ]

    print(
        "Total OSM waterway intersection length (m):",
        sum(lengths)
    )


# ---------------------------------------------------------
# 13. PLOT BACKSCATTER AND TERRAIN COMPARISON
# ---------------------------------------------------------

fig, axes = plt.subplots(
    2, 3,
    figsize=(18, 13)
)

panels = [
    (pre_vv_db, "Pre-event VV (dB)", "gray", -25, 0),
    (post_vv_db, "Flood-period VV (dB)", "gray", -25, 0),
    (vv_change, "VV change (dB)", "RdBu_r", -10, 10),
    (pre_vh_db, "Pre-event VH (dB)", "gray", -35, -5),
    (post_vh_db, "Flood-period VH (dB)", "gray", -35, -5),
    (vh_change, "VH change (dB)", "RdBu_r", -10, 10),
]

for ax, (array, title, cmap, vmin, vmax) in zip(
    axes.flat, panels
):
    image = ax.imshow(
        array,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)
    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")

    selected.boundary.plot(
        ax=ax,
        color="yellow",
        linewidth=1.5
    )

    plt.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

fig.suptitle(
    "Cluster 2239: Sentinel-1 Backscatter Comparison",
    fontsize=16
)

plt.tight_layout()

plot_path = OUT / "cluster_2239_sar_comparison.png"

plt.savefig(
    plot_path,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 14. HAND + STREAM + OSM OVERLAY MAP
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 12))

hand_image = ax.imshow(
    hand,
    extent=extent,
    origin="upper",
    cmap="terrain",
    vmin=0,
    vmax=600
)

plt.colorbar(
    hand_image,
    ax=ax,
    label="HAND (m)"
)

# Derived stream pixels
stream_display = np.where(
    stream_valid & (streams > 0),
    1,
    np.nan
)

ax.imshow(
    stream_display,
    extent=extent,
    origin="upper",
    cmap="Blues",
    alpha=0.7
)

water.clip((xmin, ymin, xmax, ymax)).plot(
    ax=ax,
    facecolor="cyan",
    edgecolor="darkblue",
    alpha=0.5
)

waterways.clip((xmin, ymin, xmax, ymax)).plot(
    ax=ax,
    color="blue",
    linewidth=1.0
)

selected.boundary.plot(
    ax=ax,
    color="red",
    linewidth=2
)

ax.set_title(
    "Cluster 2239: HAND, Derived Streams and OSM Hydrography"
)

ax.set_xlabel("UTM Easting (m)")
ax.set_ylabel("UTM Northing (m)")

plt.tight_layout()

overlay_path = OUT / "cluster_2239_hand_drainage.png"

plt.savefig(
    overlay_path,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 15. CLOSE RASTERS AND REPORT
# ---------------------------------------------------------

for ds in datasets:
    ds.close()

print("\nINVESTIGATION COMPLETE")
print("Output directory:", OUT)
print("Statistics:", statistics_path)
print("Thresholds:", candidate_path)
print("SAR comparison:", plot_path)
print("HAND/drainage overlay:", overlay_path)
