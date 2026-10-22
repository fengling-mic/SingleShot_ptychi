#%% diagnostics: visualize the camera trigger signal and the scan grid for S0006
# Reads whatever camera_data shard files + position_data are available locally;
# does not write/modify the raw data. Plots are saved next to this script.

from pathlib import Path

import numpy as np
import h5py
import hdf5plugin  # noqa: F401  (registers the lz4 filter needed for frame_data)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

scan = "S0006"
data_root = Path("/home/beams/FENGLING.ZHANG/Downloads")
camera_data = "S00006_master.h5"
position_data = "scan_00006.dat"

out_dir = Path(__file__).parent / "diagnostics"
out_dir.mkdir(parents=True, exist_ok=True)

#%% ---------------- camera trigger signal ----------------
# The master file only holds external links; the pixel data + per-frame
# frame_valid/frame_ids/train_ids live in the sibling S0000N_data_NNNNNN.h5
# shard files. Use whichever shards are physically present.
shard_paths = sorted(data_root.glob(f"{camera_data.split('_master')[0]}_data_*.h5"))
print(f"found {len(shard_paths)} shard files: {[p.name for p in shard_paths]}")

per_frame_sum = []
valid_all = []
n_frames_per_shard = []
train_ids = []
for p in shard_paths:
    with h5py.File(p, "r") as f:
        ds = f["entry/data/frame_data"]
        n = ds.shape[0]
        tid = int(np.asarray(f["entry/data/train_ids"][()]).reshape(-1)[0])
        valid = np.asarray(f["entry/data/frame_valid"][()]).reshape(-1)
        chunk = 250
        sums = np.empty(n, dtype=np.float64)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sums[start:end] = ds[start:end, 0, :, :].sum(axis=(1, 2))
        per_frame_sum.append(sums)
        valid_all.append(valid)
        n_frames_per_shard.append(n)
        train_ids.append(tid)
    print(f"{p.name}: train_id={tid} n_frames={n} n_invalid={(valid == 0).sum()}")

per_frame_sum = np.concatenate(per_frame_sum) if per_frame_sum else np.array([])
valid_all = np.concatenate(valid_all) if valid_all else np.array([])
print(f"total frames read: {len(per_frame_sum)}, total invalid: {(valid_all == 0).sum()}")

# Run-length encode frame_valid to see candidate "frames per position" burst lengths.
if len(valid_all) > 1:
    changes = np.flatnonzero(np.diff(valid_all) != 0) + 1
    run_starts = np.concatenate(([0], changes))
    run_ends = np.concatenate((changes, [len(valid_all)]))
    run_values = valid_all[run_starts]
    run_lengths = run_ends - run_starts
    valid_runs = run_lengths[run_values == 1]
    invalid_runs = run_lengths[run_values == 0]
    print(f"valid runs: n={len(valid_runs)}, median={np.median(valid_runs) if len(valid_runs) else 'n/a'}, "
          f"first 20={valid_runs[:20]}")
    print(f"invalid runs: n={len(invalid_runs)}, median={np.median(invalid_runs) if len(invalid_runs) else 'n/a'}, "
          f"first 20={invalid_runs[:20]}")

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

ax = axes[0]
ax.plot(per_frame_sum, lw=0.6)
boundary = 0
ymax = per_frame_sum.max() if len(per_frame_sum) else 1.0
for n, tid in zip(n_frames_per_shard, train_ids):
    ax.axvline(boundary, color="gray", ls="--", lw=0.6)
    ax.text(boundary + 5, ymax * 0.97, f"train {tid}", fontsize=7, rotation=90, va="top")
    boundary += n
ax.set_xlabel("frame index (global, across available shards)")
ax.set_ylabel("total counts / frame")
ax.set_title(f"{scan}: per-frame total counts ({len(per_frame_sum)} frames, "
             f"{len(shard_paths)} shards available)")

ax = axes[1]
ax.step(np.arange(len(valid_all)), valid_all, where="post", lw=0.6)
boundary = 0
for n, tid in zip(n_frames_per_shard, train_ids):
    ax.axvline(boundary, color="gray", ls="--", lw=0.6)
    boundary += n
ax.set_ylim(-0.2, 1.3)
ax.set_xlabel("frame index (global)")
ax.set_ylabel("frame_valid")
ax.set_title("frame_valid flag, full available range (candidate trigger signal)")

ax = axes[2]
zoom_n = min(1000, len(valid_all))
ax.step(np.arange(zoom_n), valid_all[:zoom_n], where="post", lw=1.0)
ax.set_ylim(-0.2, 1.3)
ax.set_xlabel("frame index (global)")
ax.set_ylabel("frame_valid")
ax.set_title(f"frame_valid, zoom on first {zoom_n} frames")

fig.tight_layout()
fig.savefig(out_dir / "trigger_signal.png", dpi=150)
print(f"saved {out_dir / 'trigger_signal.png'}")

#%% ---------------- scan grid (commanded + measured positions) ----------------
dat_path = data_root / position_data
with open(dat_path) as fh:
    lines = fh.readlines()
header_line = lines[0].strip()
cols = lines[1].split()
pos_data = np.array([[float(x) for x in line.split()] for line in lines[2:] if line.strip()])
col_idx = {name: i for i, name in enumerate(cols)}
target_x = pos_data[:, col_idx["Target_x"]]
target_y = pos_data[:, col_idx["Target_y"]]
avg_x = pos_data[:, col_idx["Average_x_st_fzp"]]
avg_y = pos_data[:, col_idx["Average_y_st_fzp"]]

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.plot(target_x, target_y, "-", lw=0.5, color="tab:blue", alpha=0.6, label="commanded (Target)")
ax2.scatter(target_x, target_y, s=8, color="tab:blue")
ax2.scatter(avg_x, avg_y, s=6, color="tab:orange", alpha=0.6, label="measured (Average_*_st_fzp)")
ax2.set_xlabel("x (st_fzp stage units)")
ax2.set_ylabel("y (st_fzp stage units)")
ax2.set_aspect("equal")
ax2.set_title(f"{scan} scan grid: {len(pos_data)} positions\n{header_line}")
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(out_dir / "scan_grid.png", dpi=150)
print(f"saved {out_dir / 'scan_grid.png'}")
