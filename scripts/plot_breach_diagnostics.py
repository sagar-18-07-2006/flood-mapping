import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

original_path = "data/processed/dem_bengaluru_utm43.tif"
breach_path = "data/processed/terrain_3000/dem_hydroconditioned.tif"

out = Path("data/experiments/breach_diagnostics/breach_overview.png")

if out.exists():
    raise FileExistsError(f"Refusing to overwrite: {out}")

with rasterio.open(original_path) as src:
    dem = src.read(1).astype("float32")
    bounds = src.bounds
    transform = src.transform

with rasterio.open(breach_path) as src:
    hydro = src.read(1).astype("float32")

if dem.shape != hydro.shape:
    raise ValueError("Raster shapes differ")

drop = dem - hydro

extent = [
    bounds.left, bounds.right,
    bounds.bottom, bounds.top
]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

plots = [
    (dem, "Original DEM (m)", "terrain", None),
    (hydro, "Breached DEM (m)", "terrain", None),
    (drop, "Elevation reduction (m)", "magma", (0, 450)),
    (np.ma.masked_where(drop <= 100, drop),
     "Severe breach: reduction >100 m", "inferno", (100, 450))
]

for ax, (data, title, cmap, limits) in zip(axes.flat, plots):
    kwargs = {}
    if limits:
        kwargs.update(vmin=limits[0], vmax=limits[1])

    im = ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap=cmap,
        **kwargs
    )

    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.colorbar(im, ax=ax, shrink=0.8)

# Mark the known extreme internal breach locations.
for ax in axes.flat:
    for r, c in [(522, 1351), (536, 1359), (549, 1362)]:
        x, y = rasterio.transform.xy(transform, r, c)
        ax.scatter(x, y, marker="x", color="cyan", s=35)

fig.suptitle(
    "Bengaluru DEM — BreachDepressions Diagnostic",
    fontsize=15
)

plt.tight_layout()
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()

print("Diagnostic map saved:", out)
