
import rasterio
import pandas as pd
import numpy as np

BASE = "data/processed/terrain/"

files = {
    "dem": BASE + "dem_hydroconditioned.tif",
    "hand": BASE + "hand.tif",
    "d8": BASE + "d8_pointer.tif",
    "streams": BASE + "streams.tif",
    "flow": BASE + "flow_accumulation_cells.tif",
}

arrays = {}
for name, path in files.items():
    with rasterio.open(path) as ds:
        arrays[name] = ds.read(1)
        arrays[name + "_nodata"] = ds.nodata

dem = arrays["dem"]
hand = arrays["hand"]
d8 = arrays["d8"]
streams = arrays["streams"]
flow = arrays["flow"]

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

df = pd.read_csv(
    "data/processed/cluster_investigation/"
    "cluster_2239/cluster_2239_d8_trace.csv"
)

results = []

for _, sample in df.iterrows():
    r = int(sample.start_row)
    c = int(sample.start_col)

    start_dem = float(dem[r, c])
    start_hand = float(hand[r, c])
    start_flow = float(flow[r, c])

    path = [(r, c)]
    visited = {(r, c)}
    status = "unknown"

    for step in range(100000):
        if streams[r, c] > 0:
            status = "reached_stream"
            break

        code = int(d8[r, c])

        if code not in directions:
            status = "invalid_pointer"
            break

        dr, dc = directions[code]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < dem.shape[0] and
                0 <= nc < dem.shape[1]):
            status = "left_raster"
            break

        if (nr, nc) in visited:
            status = "cycle"
            break

        visited.add((nr, nc))
        r, c = nr, nc
        path.append((r, c))

    if status != "reached_stream":
        print("Sample", int(sample.sample_id), status)
        continue

    stream_dem = float(dem[r, c])
    stream_hand = float(hand[r, c])
    stream_flow = float(flow[r, c])

    results.append({
        "sample": int(sample.sample_id),
        "steps": len(path) - 1,
        "start_dem": start_dem,
        "stream_dem": stream_dem,
        "start_hand": start_hand,
        "expected_dem_difference": start_dem - stream_dem,
        "hand_minus_dem_difference": (
            start_hand - (start_dem - stream_dem)
        ),
        "start_flow": start_flow,
        "stream_flow": stream_flow,
        "stream_hand": stream_hand,
        "flow_non_decreasing": all(
            flow[path[i + 1]] >= flow[path[i]]
            for i in range(len(path) - 1)
        ),
    })

result = pd.DataFrame(results)

print("\nPIXEL-LEVEL HAND VALIDATION")
print(result.to_string(index=False))

output = (
    "data/processed/cluster_investigation/"
    "cluster_2239/hand_pixel_validation.csv"
)

result.to_csv(output, index=False)
print("\nSaved:", output)
