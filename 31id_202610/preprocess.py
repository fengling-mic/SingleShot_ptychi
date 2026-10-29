#%% preprocess the data from citius camera

from pathlib import Path
from utils import use_inline_backend
use_inline_backend()             # Plot Viewer-compatible backend, in its own cell

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider

# In the VS Code interactive window (IPython) the slider needs the ipympl widget
# backend; a plain `python preprocess.py` keeps the default TkAgg window.
try:
    _shell = get_ipython().__class__.__name__  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
except NameError:
    _shell = None
if _shell == "ZMQInteractiveShell":
    get_ipython().run_line_magic("matplotlib", "widget")  # noqa: F821  # pyright: ignore[reportUndefinedVariable]

scan = "S0006"

data_root = Path("/home/beams/FENGLING.ZHANG/Downloads")

data = "scan_00006.h5"
position_data = "scan_00006.dat"

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

#%% read the scan positions
# The .h5 holds no positions; they come from the scan's .dat file, one row per
# scan point (DataPoint 0..N-1). Line 1 is a "Scan 00006, lsamrot_encoder ..."
# header, line 2 the column names. Positions are assumed to be in um.

pos = np.genfromtxt(data_root / position_data, skip_header=1, names=True)
assert len(pos) == len(patterns), f"{len(pos)} positions vs {len(patterns)} patterns"
assert np.array_equal(pos["DataPoint"].astype(int), point_index), "DataPoint != point_index"

pos_x = pos["Average_x_st_fzp"]    # measured
pos_y = pos["Average_y_st_fzp"]
target_x = pos["Target_x"]         # commanded
target_y = pos["Target_y"]

print(f"{len(pos)} positions: x [{pos_x.min():.3f}, {pos_x.max():.3f}] um, "
      f"y [{pos_y.min():.3f}, {pos_y.max():.3f}] um")
print(f"measured - commanded rms: x {np.std(pos_x - target_x) * 1e3:.1f} nm, "
      f"y {np.std(pos_y - target_y) * 1e3:.1f} nm")

#%% plot the results. show scan grids, and corresponding diffraction patterns.
# Left: scan grid (measured positions coloured by total counts, commanded as grey
# crosses, red circle = current point). Right: diffraction pattern of the current
# point on a fixed log scale. Drag the slider, or use the left/right arrow keys.

n_points = len(patterns)
is_boundary = file_boundary.astype(bool)

# Free the arrow keys from the toolbar's back/forward navigation.
plt.rcParams["keymap.back"] = [k for k in plt.rcParams["keymap.back"] if k != "left"]
plt.rcParams["keymap.forward"] = [k for k in plt.rcParams["keymap.forward"] if k != "right"]

fig, (ax_grid, ax_dp) = plt.subplots(1, 2, figsize=(11, 7))
fig.subplots_adjust(left=0.07, right=0.95, bottom=0.14, top=0.9, wspace=0.3)

# --- scan grid
ax_grid.plot(target_x, target_y, "+", color="0.7", ms=5, zorder=1, label="commanded")
ax_grid.plot(pos_x, pos_y, "-", color="0.6", lw=0.4, alpha=0.6, zorder=1)
sc = ax_grid.scatter(pos_x, pos_y, c=total_counts, cmap="viridis", s=18, zorder=2,
                     label="measured")
ax_grid.scatter(pos_x[is_boundary], pos_y[is_boundary], s=60, facecolors="none",
                edgecolors="tab:orange", linewidths=1.0, zorder=3,
                label="spans file boundary")
(marker,) = ax_grid.plot([pos_x[0]], [pos_y[0]], "o", mfc="none", mec="red", ms=16,
                         mew=2, zorder=4, label="current point")
fig.colorbar(sc, ax=ax_grid, label="total counts")
ax_grid.set_aspect("equal")
ax_grid.set_xlabel("x [um]")
ax_grid.set_ylabel("y [um]")
ax_grid.set_title(f"{scan} scan grid ({n_points} points)")
ax_grid.legend(loc="upper left", fontsize=8, framealpha=0.8)

# --- diffraction pattern
dp_cmap = plt.get_cmap("magma").copy()
dp_cmap.set_bad("black")           # zero-count pixels (log of 0)
im = ax_dp.imshow(patterns[0], cmap=dp_cmap, interpolation="nearest",
                  norm=LogNorm(vmin=1, vmax=float(patterns.max())))
fig.colorbar(im, ax=ax_dp, label="counts")
ax_dp.set_xlabel("detector column [px]")
ax_dp.set_ylabel("detector row [px]")


def dp_title(i):
    line1 = f"point {i}: {frame_count[i]} frames, {total_counts[i]:.3e} counts"
    if is_boundary[i]:
        line1 += " (spans file boundary)"
    return f"{line1}\nx = {pos_x[i]:.3f} um, y = {pos_y[i]:.3f} um"


ax_dp.set_title(dp_title(0), fontsize=10)

# --- slider
ax_slider = fig.add_axes([0.15, 0.04, 0.7, 0.03])
slider = Slider(ax_slider, "point", 0, n_points - 1, valinit=0, valstep=1, valfmt="%d")


def update(val):
    i = int(val)
    im.set_data(patterns[i])
    marker.set_data([pos_x[i]], [pos_y[i]])
    ax_dp.set_title(dp_title(i), fontsize=10)
    fig.canvas.draw_idle()


def on_key(event):
    if event.key == "right":
        slider.set_val(min(int(slider.val) + 1, n_points - 1))
    elif event.key == "left":
        slider.set_val(max(int(slider.val) - 1, 0))


slider.on_changed(update)
fig.canvas.mpl_connect("key_press_event", on_key)

#%%
plt.show()
