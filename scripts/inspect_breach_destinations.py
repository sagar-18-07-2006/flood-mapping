
import rasterio
import pandas as pd
import numpy as np

BASE = "data/processed/terrain/"

def read(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds.nodata, ds.transform

original, original_nd, transform = read(
    "data/processed/dem_bengaluru_utm43.tif"
)
hydro, hydro_nd, _ = read(BASE + "dem_hydroconditioned.tif")
d8, d8_nd, _ = read(BASE + "d8_pointer.tif")
flow, flow_nd, _ = read(BASE + "flow_accumulation_cells.tif")
streams, streams_nd, _ = read(BASE + "streams.tif")

trace = pd.read_csv(
    "data/processed/cluster_investigation/"
    "cluster_2239/cluster_2239_d8_trace.csv"
)

directions = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}

h, w = hydro.shape

# --------------------------------------------------
# Inspect each destination's local neighbourhood
# --------------------------------------------------

for _, item in trace.iterrows():
    r = int(item.end_row)
    c = int(item.end_col)

    print("\n" + "=" * 65)
    print("SAMPLE", int(item.sample_id))
    print("Destination:", r, c)
    print("Original elevation:", original[r, c])
    print("Hydro elevation:", hydro[r, c])
    print("Reduction:", original[r, c] - hydro[r, c])
    print("D8 pointer:", d8[r, c])
    print("Flow accumulation:", flow[r, c])

    print("\nLOCAL 3x3 ORIGINAL DEM")
    print(original[r-1:r+2, c-1:c+2])

    print("\nLOCAL 3x3 HYDRO DEM")
    print(hydro[r-1:r+2, c-1:c+2])

    print("\nLOCAL 3x3 REDUCTION")
    print(
        original[r-1:r+2, c-1:c+2]
        - hydro[r-1:r+2, c-1:c+2]
    )

# --------------------------------------------------
# Continue D8 trace beyond the first stream cell
# --------------------------------------------------

for _, item in trace.iterrows():
    r = int(item.end_row)
    c = int(item.end_col)

    print("\n" + "=" * 65)
    print("DOWNSTREAM TRACE FROM SAMPLE", int(item.sample_id))

    visited = set()

    for step in range(100):
        if (r, c) in visited:
            print("Cycle detected")
            break

        visited.add((r, c))

        print(
            f"Step {step}: row={r}, col={c}, "
            f"original={original[r,c]:.2f}, "
            f"hydro={hydro[r,c]:.2f}, "
            f"reduction={original[r,c]-hydro[r,c]:.2f}, "
            f"flow={flow[r,c]}, "
            f"stream={streams[r,c]}, "
            f"D8={d8[r,c]}"
        )

        code = int(d8[r, c])

        if code not in directions:
            print("Terminal or invalid D8 code")
            break

        dr, dc = directions[code]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < h and 0 <= nc < w):
            print("Raster boundary reached")
            break

        r, c = nr, nc

print("\nInspection complete.")
