
import rasterio
import numpy as np

base = "data/processed/terrain/"

with rasterio.open(base + "dem_hydroconditioned.tif") as ds:
    dem = ds.read(1)
    nd = ds.nodata

with rasterio.open(base + "d8_pointer.tif") as ds:
    ptr = ds.read(1)
    ptr_nd = ds.nodata

# WhiteboxTools default D8 directions
directions = {
    1: (0, 1, "E"),
    2: (1, 1, "SE"),
    4: (1, 0, "S"),
    8: (1, -1, "SW"),
    16: (0, -1, "W"),
    32: (-1, -1, "NW"),
    64: (-1, 0, "N"),
    128: (-1, 1, "NE"),
}

samples = [
    (522, 1351),
    (536, 1359),
    (549, 1362),
    (574, 1377),
    (583, 1382),
    (593, 1385),
    (615, 1385),
    (634, 1388),
    (656, 1380),
]

h, w = dem.shape

for r, c in samples:
    code = int(ptr[r, c])

    print("\n", "=" * 55)
    print(f"Cell: row={r}, col={c}")
    print(f"Elevation: {dem[r,c]:.3f}")
    print(f"D8 code: {code}")

    if code not in directions:
        print("Invalid or terminal direction")
        continue

    dr, dc, name = directions[code]
    nr, nc = r + dr, c + dc

    if not (0 <= nr < h and 0 <= nc < w):
        print("Direction leaves raster")
        continue

    print(
        f"Pointer target: {name}, "
        f"elevation={dem[nr,nc]:.3f}, "
        f"difference={dem[nr,nc] - dem[r,c]:.3f} m"
    )

    neighbours = []

    for d, (dr, dc, label) in directions.items():
        rr, cc = r + dr, c + dc

        if 0 <= rr < h and 0 <= cc < w:
            z = dem[rr, cc]

            if np.isfinite(z) and z != nd:
                neighbours.append((z, label, d))

    print("Lowest neighbours:")
    for z, label, d in sorted(neighbours)[:3]:
        print(f"  {label}: {z:.3f} m (code {d})")

