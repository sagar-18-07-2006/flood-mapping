import rasterio
import numpy as np
from scipy.ndimage import label, find_objects

original_path = "data/processed/dem_bengaluru_utm43.tif"
breach_path = "data/processed/terrain_3000/dem_hydroconditioned.tif"

with rasterio.open(original_path) as src:
    dem = src.read(1)

with rasterio.open(breach_path) as src:
    hydro = src.read(1)

reduction = dem.astype(np.float64) - hydro.astype(np.float64)
modified = reduction > 0.01

print("Modified cells:", int(modified.sum()), flush=True)

labels, count = label(
    modified,
    structure=np.ones((3, 3), dtype=np.uint8)
)

print("Total components:", count, flush=True)

objects = find_objects(labels)

components = []

for cid, slc in enumerate(objects, start=1):
    if slc is None:
        continue

    region = labels[slc] == cid
    values = reduction[slc][region]

    components.append((
        cid,
        int(region.sum()),
        slc[0].start,
        slc[0].stop - 1,
        slc[1].start,
        slc[1].stop - 1,
        float(values.max()),
        float(np.median(values))
    ))

components.sort(key=lambda x: x[1], reverse=True)

print("\nTOP 15 COMPONENTS")

for x in components[:15]:
    print(
        f"ID={x[0]} cells={x[1]} "
        f"rows={x[2]}-{x[3]} "
        f"cols={x[4]}-{x[5]} "
        f"max_drop={x[6]:.2f}m "
        f"median_drop={x[7]:.2f}m"
    )

print("\nKNOWN LOCATIONS")

for r, c in [(522, 1351), (536, 1359),
             (549, 1362), (1292, 1645)]:
    print(
        f"row={r}, col={c}, component={labels[r,c]}, "
        f"reduction={reduction[r,c]:.2f}m"
    )

print("\nRIGHT BOUNDARY")

ids = np.unique(labels[:, -1])
ids = ids[ids > 0]

print("Components touching right boundary:", len(ids))
print("Largest boundary components:")

for cid in ids[:20]:
    print("Component:", int(cid))

