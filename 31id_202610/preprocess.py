#%% preprocess the data from citius camera

from pathlib import Path

import numpy as np
import h5py

scan = "S0006"

data_root = Path("/home/beams/FENGLING.ZHANG/Downloads")

data = "scan_00006.h5"

#%% read the summed citius data
# scan_00006.h5 is written by sum_dwells.py: each scan point is the sum of the
# raw Citius frames recorded during that dwell (~435 frames per point).
# gzip-compressed, so plain h5py reads it (no hdf5plugin needed).

h5_path = data_root / data

with h5py.File(h5_path, "r") as f:
    entry = f["entry"]
    print(f"attributes of /entry in {h5_path.name}:")
    for key, value in entry.attrs.items():
        print(f"  {key} = {value}")

    g = entry["data"]
    print(f"datasets in /entry/data: {list(g.keys())}")
    patterns = g["data"][()]                # (n_points, rows, cols) summed counts, float32
    frame_count = g["frame_count"][()]      # raw frames summed into each point
    frame_start = g["frame_start"][()]      # global raw-frame index, inclusive
    frame_end = g["frame_end"][()]          # global raw-frame index, exclusive
    file_boundary = g["file_boundary"][()]  # 1 if the point spans a shard boundary
    edge_clipped = g["edge_clipped"][()]    # 1 if cut off by the start/end of acquisition
    point_index = g["point_index"][()]      # index among all detected dwells

total_counts = patterns.sum(axis=(1, 2), dtype=np.float64)

print(f"patterns: shape {patterns.shape}, dtype {patterns.dtype}, "
      f"{patterns.nbytes / 1e6:.0f} MB")
print(f"total counts per point: min {total_counts.min():.3e}, "
      f"max {total_counts.max():.3e}, median {np.median(total_counts):.3e}")
print(f"frames per point: {frame_count.min()} to {frame_count.max()}")
print(f"points spanning a file boundary: {int(file_boundary.sum())} of {len(file_boundary)}")
print(f"points clipped at acquisition edges: {int(edge_clipped.sum())}")
