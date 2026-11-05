import rasterio
import numpy as np
from scipy.ndimage import label, find_objects

with rasterio.open("data/processed/dem_bengaluru_utm43.tif") as s:
    dem = s.read(1)

with rasterio.open(
    "data/processed/terrain_3000/dem_hydroconditioned.tif"
) as s:
    hydro = s.read(1)

drop = dem.astype(np.float64) - hydro.astype(np.float64)

print("=" * 65)
print("SEVERE BREACH ANALYSIS")
print("=" * 65)

for threshold in [10, 50, 100, 200, 400]:
    mask = drop > threshold

    labels, n = label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )

    print(f"\nReduction > {threshold} m")
    print("Cells:", int(mask.sum()))
    print("Components:", n)

    if mask.any():
        r, c = np.where(mask)
        print("Bounding rows:", int(r.min()), "-", int(r.max()))
        print("Bounding cols:", int(c.min()), "-", int(c.max()))

        boundary = (
            mask[0, :].sum() + mask[-1, :].sum()
            + mask[:, 0].sum() + mask[:, -1].sum()
        )
        print("Boundary intersections:", int(boundary))

        idx = np.argmax(drop)
        rr, cc = np.unravel_index(idx, drop.shape)
        print(
            "Maximum drop location:",
            rr, cc,
            "drop:", round(float(drop[rr, cc]), 2)
        )

        print("Largest components:")
        objects = find_objects(labels)
        sizes = np.bincount(labels.ravel())
        top = np.argsort(sizes[1:])[-5:][::-1] + 1

        for cid in top:
            if sizes[cid] == 0:
                continue
            slc = objects[cid - 1]
            print(
                f" ID={cid}, cells={sizes[cid]}, "
                f"rows={slc[0].start}-{slc[0].stop-1}, "
                f"cols={slc[1].start}-{slc[1].stop-1}"
            )

print("\n" + "=" * 65)
print("COMPONENT 45 SEVERE REDUCTION DISTRIBUTION")
print("=" * 65)

labels, _ = label(
    drop > 0.01,
    structure=np.ones((3, 3), dtype=np.uint8)
)

m = labels == 45
values = drop[m]

for q in [50, 75, 90, 95, 99, 99.9, 100]:
    print(f"P{q}: {np.percentile(values, q):.2f} m")

print("Component 45 cells:", int(m.sum()))

