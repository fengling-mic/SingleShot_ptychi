# Preprocessing script for the visible lightdataset
r"""Turn a laser-lab TIFF raster scan into the dp/para pair the pty-chi recon scripts read.

Input (one folder per scan under date_Laserlab/):

    <scan_name>/frame_00000.tiff ...   (n, 256, 256) uint8, one frame per scan point
    <scan_name>/positions.csv          given_x,given_y,readout_x,readout_y   (mm)
    <scan_name>_bg/frame_*.tiff        dark frames -- averaged and subtracted

This writes, into ``preproc/<scan_name>/data_roi0_Ndp{N}_{dp,para}.h5``, the same layout
as the 12-ID preprocessor (2A_12ide_preprocess.py, save_results):

    dp.h5     dp                 (n, N, N) float32   dark-subtracted counts, gzip
              det_pixel_mask     (N, N)    bool      all True (valid)
    para.h5   energy             (1,) float32  "keV" = 1.24e-9 / wavelength, so readers
                                               doing 1.24e-9 / energy get the wavelength back
              detector_distance  (1,) float32  m
              exposure_time_s    (1,) float32  s
              ppX, ppY           (n,) float64  m   attrs: pos_status, pos_conv_fac, use_trigger_diff
              det_pixel_size     (1,) float64  m   extra -- the readers hardcode the Pilatus pixel
              lambda             (1,) float64  m   extra -- for beamPropagation.py
              dx                 (1,) float64  m   extra -- object-plane pixel size

No flip, transpose, fftshift, threshold or normalisation: the beam stays at the array
centre, as in the 12-ID files.

Run from PowerShell (Git Bash's Python cannot see the micdata share):

    & .\pty-chi\.venv\Scripts\python.exe "\\micdata\data3\fengling\date_Laserlab\preproc\preprocess_laserlab.py"
"""

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

##########################################################################
# common parameters to change
##########################################################################

data_root  = Path(r"\\micdata\data3\fengling\date_Laserlab")  # /mnt/micdata3/fengling/date_Laserlab on Linux
scan_name  = "20261107_s1805_8mmzp_2ms_area2"
bg_name    = scan_name + "_bg"
scan_dir   = data_root / scan_name
bg_dir     = data_root / bg_name
result_dir = data_root / "preproc" / scan_name

wavelength      = 780e-9   # m
dist            = 100e-3   # m, sample-detector distance
det_pixel       = 5.5e-6   # m, camera pixel of the 256x256 frames
exposure_time_s = 2e-3     # s, from the scan name "_2ms"

cen_y, cen_x = 128, 128    # beam center (pix) on the raw 256x256 frame
det_Npixel   = 256
det_xwidth   = det_Npixel // 2
dataName     = f"data_roi0_Ndp{det_Npixel:d}"

pos_cols = ("readout_x", "readout_y")  # encoder readback; ("given_x", "given_y") for commanded
conv_fac = {
    "ok": {"ppX": 1e-3, "ppY": 1e-3},  # mm -> m
}
use_trigger_diff = False   # kept for parity with the 12-ID para files; there is no trigger stream here

dx = wavelength * dist / (det_Npixel * det_pixel)  # object-plane pixel size (m)

##########################################################################
# functions
##########################################################################

def read_frames(folder):
    """Stack every frame_*.tif* in folder as float32. Names are zero-padded, so sorted = frame order."""
    files = sorted(folder.glob("frame_*.tif*"))
    if not files:
        raise FileNotFoundError(f"No frame_*.tif* files in {folder}")
    print(f"  {folder.name}: {len(files)} frames")
    return np.stack([tifffile.imread(f) for f in files]).astype(np.float32)


def read_positions(csv_path):
    """Per-frame [x, y] in mm, picked by column name from positions.csv."""
    table = np.genfromtxt(csv_path, delimiter=",", names=True)
    return np.column_stack([table[c] for c in pos_cols])


def save_results(result_dir, dp, pos):
    """Convert positions to metres, check them, and write the dp/para pair in the 12-ID layout."""
    pos = np.array(pos, dtype=np.float64)
    pos[:, 0] *= conv_fac["ok"]["ppX"]
    pos[:, 1] *= conv_fac["ok"]["ppY"]

    if len(pos) != len(dp):
        raise ValueError(f"{len(pos)} positions but {len(dp)} diffraction patterns")

    # Position quality check
    step_norms = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    large_jumps = np.where(step_norms > 5 * np.median(step_norms))[0]
    if len(large_jumps) > 0:
        print(f"WARNING: {len(large_jumps)} large position jumps at indices: {large_jumps}")

    print(f"Final scan points: {pos.shape[0]}")
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save diffraction patterns
    with h5py.File(result_dir / (dataName + "_dp.h5"), "w") as f:
        f.create_dataset("dp", dtype="float32", data=dp, compression="gzip")
        f.create_dataset("det_pixel_mask", dtype=bool, data=np.ones(dp.shape[1:], dtype=bool),
                         compression="gzip")

    # Save scan parameters and positions
    with h5py.File(result_dir / (dataName + "_para.h5"), "w") as f:
        f.create_dataset("energy",            shape=(1,), dtype="float32", data=1.24e-9 / wavelength)
        f.create_dataset("detector_distance", shape=(1,), dtype="float32", data=dist)
        f.create_dataset("exposure_time_s",   shape=(1,), dtype="float32", data=exposure_time_s)

        dsetX = f.create_dataset("ppX", dtype="float64", data=pos[:, 0])
        dsetY = f.create_dataset("ppY", dtype="float64", data=pos[:, 1])
        for dset, key in ((dsetX, "ppX"), (dsetY, "ppY")):
            dset.attrs["pos_status"] = "ok"
            dset.attrs["pos_conv_fac"] = conv_fac["ok"][key]
            dset.attrs["use_trigger_diff"] = use_trigger_diff

        # Extras, not in the 12-ID files: the readers hardcode the Pilatus pixel (172 um),
        # so the camera geometry has to travel with the data.
        f.create_dataset("det_pixel_size", shape=(1,), dtype="float64", data=det_pixel)
        f.create_dataset("lambda",         shape=(1,), dtype="float64", data=wavelength)
        f.create_dataset("dx",             shape=(1,), dtype="float64", data=dx)

    return pos


def plot_dpsum(dp, pos, save_path):
    """sum(dp) per frame, the same sum on the scan positions, and the log10 mean pattern."""
    dp_sum = dp.sum(axis=(1, 2))
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(17, 5))

    ax0.plot(dp_sum, ".-")
    ax0.set_xlabel("frame index")
    ax0.set_ylabel("sum(dp)")
    ax0.grid(True, alpha=0.3)

    sc = ax1.scatter(pos[:, 0] * 1e3, pos[:, 1] * 1e3, c=dp_sum, cmap="viridis", s=30)
    fig.colorbar(sc, ax=ax1, label="sum(dp)")
    ax1.set_xlabel("ppX (mm)")
    ax1.set_ylabel("ppY (mm)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, alpha=0.3)

    im = ax2.imshow(np.log10(dp.mean(axis=0) + 1e-2), cmap="magma")
    fig.colorbar(im, ax=ax2, label="log10 mean(dp)")
    ax2.set_title(f"mean pattern, {dp.shape[1]}x{dp.shape[2]}")

    fig.suptitle(f"{scan_name} -> {dataName}  ({len(dp)} frames, dark = mean of {bg_name})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


#%%#########################################################################
# main
##########################################################################

para_path = result_dir / (dataName + "_para.h5")
if para_path.exists():
    print(f"{scan_name} already processed ({para_path}) -- delete output to reprocess")
else:
    print(f"Processing {scan_dir} -> {result_dir}")
    frames = read_frames(scan_dir)
    dark = read_frames(bg_dir).mean(axis=0)
    pos = read_positions(scan_dir / "positions.csv")
    print(f"  frames {frames.shape}, dark mean {dark.mean():.3f}, positions {pos.shape}")

    saturated = frames >= 255
    if saturated.any():
        print(f"  WARNING: {int(saturated.sum())} saturated pixels (255) in frames "
              f"{np.unique(np.nonzero(saturated)[0])} -- left as is")

    # Dark subtraction only; negatives from read noise are clipped like 2A does
    dp = frames - dark
    dp[dp < 0] = 0

    # Crop to detector ROI
    n_y, n_x = dp.shape[1:]
    if not (det_xwidth <= cen_y <= n_y - det_xwidth and det_xwidth <= cen_x <= n_x - det_xwidth):
        raise ValueError(f"ROI {det_Npixel} around ({cen_y}, {cen_x}) does not fit in {n_y}x{n_x} frames")
    dp = dp[:, cen_y-det_xwidth:cen_y+det_xwidth, cen_x-det_xwidth:cen_x+det_xwidth]
    print(f"  dp {dp.shape}, mean sum/frame {dp.sum(axis=(1, 2)).mean():.0f}")

    pos = save_results(result_dir, dp, pos)
    plot_dpsum(dp, pos, result_dir / (dataName + "_dpsum.png"))
    print(f"  lambda={wavelength:.4e} m, dist={dist} m, det_pixel={det_pixel:.3e} m -> dx={dx:.4e} m")
    print(f"Done: {result_dir}")
