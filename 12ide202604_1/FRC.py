#%% Fourier Ring Correlation: step-scan vs. single-frame ptychography, at frame 62's area
#
# Two curves, both computed over the same frame-62-sized area, as one two-group split:
#   group 1 = dir1 (gaussian noise model, the most-iterated of the 3 step-scan recons)
#   group 2 = dir2 (poisson) and dir3 (the original piezo_test LSQML recon dir1/dir2 and
#             the RPI run were bootstrapped from), registered to each other and
#             complex-averaged together
# dir1/dir2/dir3 all reconstruct the IDENTICAL raw S0019 diffraction data (confirmed via
# their pear_params.json), so group1-vs-group2 is a proper independent-estimate FRC split:
#
#   1. dir1 crop vs single-frame reconstruction
#   2. dir1 crop vs (dir2+dir3) combined crop     <- step-scan's own resolution ceiling:
#      what the two independent step-scan groups agree to, for curve 1 to be read against.
#
# dir3's positions_px[62] is frame 62's own scan position (dir3 is frame 62's own
# `path_to_init_recon`, see its rpi_params.json); dir1/dir2's positions_px[62] are each
# reconstruction's own independent re-estimate of the identical physical point. Each
# object's phase ramp/offset (gauge freedom: O -> O*exp(i(k.r+c)) is invisible to the
# data) is removed before any comparison, and every pair is cross-correlated to register
# before computing FRC, since independent reconstructions are never pixel-perfect aligned.

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation

from utils import (
    crop_at_position,
    fourier_ring_correlation,
    frc_crossing_px,
    frc_half_bit_threshold,
    plot_frc,
    plot_frc_rings,
    remove_phase_ramp,
)

#%% FRC from multiple step scans
step_recon_dir1 = Path("/mnt/micdata3/fengling/2026_03_results") / "ptychi_recons" / "S0019" / "Ndp1024_LSQML_c100_m0.25_gaussian_p5_opr1_pc1_f_ul20_initprobe_fromzpOSA_GS_eswonj_stepsize" / "recon_Niter15500.h5"
step_recon_dir2 = Path("/mnt/micdata3/fengling/2026_03_results") / "ptychi_recons" / "S0019" / "Ndp1024_LSQML_c100_m0.25_poisson_p5_opr1_pc1_f_ul20_goodinit" / "recon_Niter300.h5"
step_recon_dir3 = Path("/mnt/micdata2/12IDC/2026_Data/2026_3/01_piezo_test") / "ptychi_recons" / "S0019" / "Ndp1024_LSQML_c20_m0.5_gaussian_p10_cp_mm_opr3_ic_pc1_f_ul2" / "recon_Niter400.h5"

#%% FRC from single-frame ptychography
singleframe_recon_dir = Path("/mnt/micdata3/fengling/2026_03_results") / "singleframe_recons" / "S0019" / "frame62_RPI_Adam_Nobj92_R0.5_relax100_0-1000_p10_opr1_prel200_blrelax" / "recon_Niter4000.h5"
frame_index = 62   # must match rpi_params.json's "frame_index" next to singleframe_recon_dir


#%% ---------------------------------------------------------------- helpers

def load_step_recon(path):
    """(object [2D, slice axis dropped], positions_px, obj_pixel_size_m)."""
    with h5py.File(path, "r") as f:
        obj = f["object"][()]
        positions_px = f["positions_px"][()].astype(np.float64)
        pixel_size_m = float(f["obj_pixel_size_m"][()])
    if obj.ndim == 3:
        obj = obj[0]
    return obj, positions_px, pixel_size_m


def align_pair(img_a, img_b, upsample_factor=20):
    """Cross-correlate on amplitude and Fourier-shift `img_b` onto `img_a`'s grid.
    Returns (img_a, aligned_b, shift_rc).

    (skimage's `error` return value is dropped -- under the default
    `normalization="phase"` it mixes a phase-normalized correlation peak with an
    amplitude-based error formula and saturates near 1.0 regardless of fit
    quality; not meaningful here.)
    """
    shift, _error, _ = phase_cross_correlation(np.abs(img_a), np.abs(img_b),
                                               upsample_factor=upsample_factor)
    aligned_b = np.fft.ifft2(fourier_shift(np.fft.fft2(img_b), shift))
    print(f"    registration shift (row, col) = ({shift[0]:+.2f}, {shift[1]:+.2f}) px")
    return img_a, aligned_b, shift


def crop_deramped(path, frame_index, out_shape):
    """Load a step-scan recon, crop `out_shape` centered on ITS OWN
    `positions_px[frame_index]`, and remove the phase ramp/offset.
    Returns (corrected_crop, pixel_size_m)."""
    obj, positions_px, pixel_size_m = load_step_recon(path)
    position_rc = positions_px[frame_index]
    crop, center_rc = crop_at_position(obj, position_rc, out_shape)
    corrected, ramp, offset = remove_phase_ramp(crop)
    print(f"  {path.parent.name}:\n"
          f"    frame {frame_index} position (row, col) = ({position_rc[0]:.2f}, "
          f"{position_rc[1]:.2f}) px -> crop center {center_rc} in {obj.shape} object\n"
          f"    pixel {pixel_size_m * 1e9:.4f} nm, ramp removed ({ramp[0]:+.2e}, "
          f"{ramp[1]:+.2e}) cyc/px, offset {offset:+.3f} rad")
    return corrected, pixel_size_m


#%% ---------------------------------------------------------------- load everything

print("=== loading the single-frame reconstruction and the 3 step-scan crops at its area ===")

with h5py.File(singleframe_recon_dir, "r") as f:
    sf_object = f["object"][()]
    sf_pixel_size_m = float(f["obj_pixel_size_m"][()])
sf_object, sf_ramp, sf_offset = remove_phase_ramp(sf_object)
print(f"  single-frame: {sf_object.shape}, pixel {sf_pixel_size_m * 1e9:.4f} nm, "
      f"ramp removed ({sf_ramp[0]:+.2e}, {sf_ramp[1]:+.2e}) cyc/px, offset {sf_offset:+.3f} rad")

dir1_crop, dir1_pixel_size_m = crop_deramped(step_recon_dir1, frame_index, sf_object.shape)
dir2_crop, dir2_pixel_size_m = crop_deramped(step_recon_dir2, frame_index, sf_object.shape)
dir3_crop, dir3_pixel_size_m = crop_deramped(step_recon_dir3, frame_index, sf_object.shape)
step_pixel_size_m = dir1_pixel_size_m
print(f"  step-scan pixel sizes (dir1/dir2/dir3): "
      f"{dir1_pixel_size_m * 1e9:.4f} / {dir2_pixel_size_m * 1e9:.4f} / "
      f"{dir3_pixel_size_m * 1e9:.4f} nm")


#%% ---------------------------------------------------------------- group 2 = dir2 + dir3

print("registering dir2 vs dir3, to combine them into the 'group 2' reference:")
_, dir3_aligned, _shift = align_pair(dir2_crop, dir3_crop)
combined_23 = 0.5 * (dir2_crop + dir3_aligned)

#%% sanity check: do all three groups show the same feature, before trusting the FRC?
fig, axes = plt.subplots(2, 3, figsize=(11.5, 8))
for col, (img_amp, img_phase, title) in enumerate((
    (np.abs(dir1_crop), np.angle(dir1_crop), "dir1 (group 1)"),
    (np.abs(combined_23), np.angle(combined_23), "dir2+dir3 (group 2)"),
    (np.abs(sf_object), np.angle(sf_object), "single-frame"),
)):
    axes[0, col].imshow(img_amp, cmap="gray")
    axes[0, col].set_title(f"{title}\n|O|", fontsize=9)
    axes[1, col].imshow(img_phase, cmap="gray")
    axes[1, col].set_title("phase", fontsize=9)
for ax in axes.ravel():
    ax.set_xticks([]), ax.set_yticks([])
fig.suptitle(f"frame {frame_index} area: the three groups being compared")
plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------- the FRC curves

print("computing the FRC curves:")

print("  1) dir1 vs single-frame:")
_, sf_aligned_1, _ = align_pair(dir1_crop, sf_object)
freq1, frc1, n1 = fourier_ring_correlation(np.angle(dir1_crop), np.angle(sf_aligned_1),
                                           pixel_size_m=sf_pixel_size_m)

print("  2) dir1 vs dir2+dir3 (step-scan resolution ceiling):")
_, combined_23_aligned, _ = align_pair(dir1_crop, combined_23)
freq2, frc2, n2 = fourier_ring_correlation(np.angle(dir1_crop), np.angle(combined_23_aligned),
                                           pixel_size_m=step_pixel_size_m)

half_bit = frc_half_bit_threshold(n1)   # same crop size for both -> same n_per_ring

print("FRC half-bit crossings:")
plot_frc(
    freq1, [frc1, frc2], threshold=half_bit,
    labels=["single-frame", "step-scan"],
    title=f"FRC at frame {frame_index}'s area (object phase)",
)
plt.show()

#%% FRC rings: where each curve's resolution cutoff sits in Fourier space, on dir1's
# own power spectrum (dir1 is the common image in both curves above).
r1 = frc_crossing_px(freq1, frc1, half_bit, dir1_crop.shape, sf_pixel_size_m)
r2 = frc_crossing_px(freq2, frc2, half_bit, dir1_crop.shape, step_pixel_size_m)
plot_frc_rings(
    np.angle(dir1_crop), [r1, r2], labels=["single-frame", "step-scan"],
    colors=["C0", "C1"], log_floor=1e-10, title=f"FRC rings at frame {frame_index}'s area (dir1 power spectrum)",
)
plt.show()

#%%
