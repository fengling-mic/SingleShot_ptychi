"""Turn a raw 4-ID-D flyscan into the dp/para pair ptychi_reconstruction_4idd.py reads.

The master file (``scan_XXXXXX_master.hdf``) is metadata only; the payload lives in the
two files it names under ``entry/instrument/bluesky/metadata/detectors_file_relative_path``:

    DefaultSample/eiger/scan_XXXXXX.h5       entry/data/data   (n_frames, 1028, 1062) int32
    DefaultSample/pos_stream/scan_XXXXXX.h5  entry/data/data   (n_samples, 24)        int32

This writes, into ``<RESULT_DIR>/data_roi{ROI}_Ndp{N}_us{U}_{dp,para}.hdf5``:

    dp.hdf5     dp      (n_kept, 256, 256) float32, I0-normalized, gzip
    para.hdf5   lambda  (1,)      float64   m
                dx      (1,)      float64   m, object-plane pixel size
                ppY     (n_kept,) float64   m
                ppX     (n_kept,) float64   m

Same algorithm as the beamline's 4idd_data_preprocessing_flyscan_v2.py, minus the PyQt
tuning GUI and minus the full-stack read (that one allocates ~16 GB for a 3766-frame scan).

Run with an interpreter that has hdf5plugin -- the Eiger data uses the bitshuffle filter:

    /home/beams/FENGLING.ZHANG/.conda/envs/python_env/bin/python preprocess_4idd_flyscan.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import h5py
import hdf5plugin  # noqa: F401  -- registers the bitshuffle filter the Eiger files use

# process_position_stream is the beamline's own reducer for the interferometer stream;
# import it rather than reimplementing the trigger-group logic.
sys.path.append("/net/micdata/data2/4IDD/2026_Sep")
from process_flyscan import process_position_stream


#%% ---------------------------------------------------------------- what to process

SCANS = [521]
SOURCE_DIR = "/net/micdata/data2/4IDD/2026_Sep/DefaultSample"
RESULT_DIR = "/net/micdata/data2/4IDD/2026_Sep/results/ML_recon/fly{:03d}"


#%% ---------------------------------------------------------------- geometry

# ENERGY is the value fly196..fly394 were processed with. Scan 520's master file records
# 6.2121 keV; keeping 6.208 makes `lambda` and `dx` bit-identical to the earlier scans so
# fly520 stays directly comparable to them.
ENERGY = 6.208                   # keV
DET_SAMPLE_DIST = 1.91           # m
DET_PIXEL_SIZE = 75e-6           # m, Eiger
DET_NPIXEL = 256                 # detector crop, before upsampling

# ROI center on the raw 1028x1062 frame. Recovered by matching fly394's saved dp against
# raw scan 394 (correlation 0.99998); scan 520's direct beam sits within 4 px of it.
CEN_X = 708
CEN_Y = 375

UPSAMPLE = 1
ROI = 0                          # label only -- goes into the file name

LAMBDA = 1.23984193e-9 / ENERGY
DX = LAMBDA * DET_SAMPLE_DIST / DET_PIXEL_SIZE / DET_NPIXEL
DATA_NAME = f"data_roi{ROI:d}_Ndp{DET_NPIXEL * UPSAMPLE}_us{UPSAMPLE}"


#%% ---------------------------------------------------------------- frame selection

# The scanner's final exposure is a shutter-close frame (its I0-normalized sum runs ~12%
# above its neighbours), so it is trimmed. fly394 was processed the same way: 3765 frames
# kept out of 3766. No MAD-based outlier rejection.
N_DROP_START = 0
N_DROP_END = 1

# Frames whose I0 falls below this are dropped too -- the beam dips periodically during
# the scan and those exposures are not worth normalizing. This is an absolute counter
# value, so it is specific to one scan's I0 scale (scan 520 sits at a median of 8206;
# scan 394 ran at 27000). Set to None to disable.
I0_MIN = 7740

# Dither added to the positions to break the raster grid's periodicity. The beamline
# script uses the same amplitude but leaves it unseeded; seeding makes reruns identical.
POS_JITTER_M = 5e-9
RANDOM_SEED = 0

# Frames per h5py read. Only the ROI window is fetched, so peak memory is roughly
# BATCH * 256 * 256 on top of the output stack.
BATCH = 100

# Eiger flags bad/gap pixels with out-of-range values; both ends are zeroed.
COUNT_MAX = 65530


#%% ---------------------------------------------------------------- loading


def load_positions(scan_num, base_path=SOURCE_DIR):
    """Per-frame (ppY, ppX) in metres and I0, from the interferometer stream.

    The detector skips the first trigger, so the leading position group is dropped to
    line the two up. Positions are stored negated, matching the convention the existing
    fly* files use.
    """
    i0s, xs, ys, _, _ = process_position_stream(scan_num, base_path)

    ppX = -xs[1:]
    ppY = -ys[1:]
    i0 = i0s[1:]

    positions = np.column_stack((ppY, ppX)) * 1e-9
    rng = np.random.default_rng(RANDOM_SEED)
    positions += rng.normal(0, POS_JITTER_M, positions.shape)

    return positions, i0


def load_patterns(scan_num, base_path=SOURCE_DIR, det_npixel=DET_NPIXEL,
                  cen_x=CEN_X, cen_y=CEN_Y, batch=BATCH):
    """The ROI-cropped Eiger stack as float32, read in batches.

    h5py fetches only the ROI bytes, so the whole 1028x1062 stack is never resident.
    """
    h5_path = os.path.join(base_path, "eiger", f"scan_{scan_num:06d}.h5")

    index_y = slice(int(cen_y - det_npixel // 2), int(cen_y + (det_npixel + 1) // 2))
    index_x = slice(int(cen_x - det_npixel // 2), int(cen_x + (det_npixel + 1) // 2))

    with h5py.File(h5_path, "r") as f:
        dset = f["entry/data/data"]
        n = dset.shape[0]
        print(f"  eiger {dset.shape} -> crop rows {index_y.start}:{index_y.stop}, "
              f"cols {index_x.start}:{index_x.stop}")

        dp = np.empty((n, det_npixel, det_npixel), dtype=np.float32)
        for i in range(0, n, batch):
            j = min(i + batch, n)
            chunk = dset[i:j, index_y, index_x]
            chunk[(chunk < 0) | (chunk > COUNT_MAX)] = 0
            dp[i:j] = chunk

    return dp


def keep_mask(i0, n_drop_start=N_DROP_START, n_drop_end=N_DROP_END, i0_min=I0_MIN):
    """Boolean keep-mask: trim the ends, then drop anything below the I0 floor."""
    n = i0.size
    keep = np.ones(n, dtype=bool)
    if n_drop_start > 0:
        keep[:n_drop_start] = False
    if n_drop_end > 0:
        keep[n - n_drop_end:] = False
    if i0_min is not None:
        low = i0 < i0_min
        print(f"  I0 < {i0_min}: {int(low.sum())} frames "
              f"({int((low & keep).sum())} not already trimmed)")
        keep &= ~low
    return keep


#%% ---------------------------------------------------------------- diagnostics


def plot_i0_dpsum(i0, dp_sum, pos, keep, scan_num, save_path):
    """Three-panel record of what was kept: I0 trace, sum(dp) trace, position scatter."""
    idx = np.arange(i0.size)
    x_um = pos[:, 1] * 1e6
    y_um = pos[:, 0] * 1e6
    dropped = ~keep

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0],
                          left=0.06, right=0.97, top=0.93, bottom=0.07,
                          hspace=0.28, wspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    for ax, values, label, color in ((ax0, i0, "I0", "C0"),
                                     (ax1, dp_sum, "sum(dp)", "C2")):
        ax.plot(idx[keep], values[keep], ".-", color=color,
                label=f"{label} (kept, {keep.sum()})")
        if dropped.any():
            ax.plot(idx[dropped], values[dropped], "rx", ms=8,
                    label=f"dropped ({dropped.sum()})")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    ax1.set_xlabel("frame index")

    for ax, values, label in ((ax2, i0, "I0"), (ax3, dp_sum, "sum(dp)")):
        sc = ax.scatter(x_um[keep], y_um[keep], c=values[keep], cmap="viridis", s=18)
        fig.colorbar(sc, ax=ax, label=f"{label} (kept)")
        if dropped.any():
            ax.scatter(x_um[dropped], y_um[dropped], c="red", marker="x", s=40,
                       label=f"dropped ({dropped.sum()})")
            ax.legend(loc="best")
        ax.set_xlabel("ppX (um)")
        ax.set_ylabel("ppY (um)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"scan {scan_num} -- kept {keep.sum()}/{i0.size} "
                 f"(N_DROP_START={N_DROP_START}, N_DROP_END={N_DROP_END}, "
                 f"I0_MIN={I0_MIN})")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


#%% ---------------------------------------------------------------- driver


def process_scan(scan_num, result_dir=RESULT_DIR):
    scan_dir = result_dir.format(scan_num)
    para_path = os.path.join(scan_dir, DATA_NAME + "_para.hdf5")
    if os.path.isfile(para_path):
        print(f"Scan {scan_num:03d} has already been processed ({para_path}).")
        print("To reprocess, delete the existing output files first.")
        return

    print(f"Processing scan {scan_num} -> {scan_dir}")
    os.makedirs(scan_dir, exist_ok=True)

    pos, i0 = load_positions(scan_num)
    dp = load_patterns(scan_num)

    if UPSAMPLE > 1:
        from scipy.ndimage import zoom
        print(f"  upsampling by {UPSAMPLE}")
        dp = zoom(dp, (1, UPSAMPLE, UPSAMPLE))
        dp -= dp.min()

    # The three arrays can differ by a frame at the tail; trim to the common length.
    n = min(dp.shape[0], pos.shape[0], i0.shape[0])
    print(f"  dp={dp.shape}, pos={pos.shape}, i0={i0.shape} -> n={n}")
    dp, pos, i0 = dp[:n], pos[:n], i0[:n]

    dp_sum = dp.sum(axis=(1, 2))
    keep = keep_mask(i0)
    plot_i0_dpsum(i0, dp_sum, pos, keep, scan_num,
                  os.path.join(scan_dir, DATA_NAME + "_i0_dpsum.png"))
    print(f"  keeping {keep.sum()}/{n} frames")

    dp, pos, i0 = dp[keep], pos[keep], i0[keep]
    if np.any(i0 <= 0):
        raise ValueError(f"scan {scan_num}: {int((i0 <= 0).sum())} kept frames have I0 <= 0")

    dp = dp / i0[:, None, None] * 1e5

    print(f"  writing {DATA_NAME}_dp.hdf5 ({dp.shape})")
    with h5py.File(os.path.join(scan_dir, DATA_NAME + "_dp.hdf5"), "w") as f:
        f.create_dataset("dp", data=dp, dtype="float32", compression="gzip")

    print(f"  writing {DATA_NAME}_para.hdf5 (lambda={LAMBDA:.6e}, dx={DX:.6e})")
    with h5py.File(para_path, "w") as f:
        f.create_dataset("lambda", data=[LAMBDA], dtype="float64")
        f.create_dataset("dx", data=[DX], dtype="float64")
        f.create_dataset("ppY", data=pos[:, 0], dtype="float64")
        f.create_dataset("ppX", data=pos[:, 1], dtype="float64")


def main():
    for scan_num in SCANS:
        process_scan(scan_num)


if __name__ == "__main__":
    main()
