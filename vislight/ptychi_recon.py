#%%
# Pty-Chi (LSQML) reconstruction of a laser-lab (780 nm) raster scan, preprocessed by
# vislight/Preprocess.py into the 12-ID dp/para layout.

from utils import use_inline_backend
use_inline_backend()             # Plot Viewer-compatible backend, in its own cell

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # This makes GPU N appear as GPU 0 to CuPy

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import median_filter

from utils import center_crop_or_pad, show_ptychogram, make_probe, hermite_secondary_modes

import ptychi.api as api
from ptychi.api.options.base import OptimizationPlan
from ptychi.api.task import PtychographyTask
from ptychi.utils import (
    add_additional_opr_probe_modes_to_probe,
    generate_initial_opr_mode_weights,
    get_default_complex_dtype,
    get_suggested_object_size,
    rescale_probe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


#%% ---------------------------------------------------------------- paths

scan = "20261107_s1805_8mmzp_2ms_area2"

# On Linux the same share is /mnt/micdata3; here on Windows it is \\micdata\data3.
data_root = Path(r"\\micdata\data3\fengling\date_Laserlab")
# data_root = Path("/mnt/micdata3/fengling/date_Laserlab")

dp_file = data_root / "preproc" / scan / "data_roi0_Ndp256_dp.h5"
para_file = data_root / "preproc" / scan / "data_roi0_Ndp256_para.h5"

out_dir = data_root / "ptychi_recons"
recon_dir_suffix = ""            # appended to the folder name, e.g. "_v2" or "_pos"

#%% ---------------------------------------------------------------- read the parameters from the file

with h5py.File(para_file, "r") as f:
    print(f"keys in {para_file.name}: {list(f.keys())}")
    ppx = np.asarray(f["ppX"][()], dtype=np.float64).squeeze()   # m
    ppy = np.asarray(f["ppY"][()], dtype=np.float64).squeeze()   # m
    if "lambda" in f:
        wavelength_m = float(np.asarray(f["lambda"][()]).squeeze())
    else:
        wavelength_m = 1.24e-9 / float(np.asarray(f["energy"][()]).squeeze())
    detector_distance_m = float(np.asarray(f["detector_distance"][()]).squeeze())
    det_pixel_m = float(np.asarray(f["det_pixel_size"][()]).squeeze())
    exposure_time_s = float(np.asarray(f["exposure_time_s"][()]).squeeze())
    file_pixel_size_m = float(np.asarray(f["dx"][()]).squeeze())
print(f"lambda {wavelength_m * 1e9:.1f} nm, distance {detector_distance_m * 1e3:.1f} mm, "
      f"camera pixel {det_pixel_m * 1e6:.2f} um, exposure {exposure_time_s * 1e3:.1f} ms, "
      f"{len(ppx)} positions")


#%% ---------------------------------------------------------------- geometry & knobs

n_dp = 256                          # detector crop (the preprocessed frames are 256 x 256)
# The sample sits at the front focal plane of a Fourier lens and the camera at the
# back one, so the camera records the exact Fourier transform of the exit wave: far
# field, with the lens focal length in place of the detector distance.
det_dist_m = detector_distance_m
far_field = True

pixel_size_m = wavelength_m * det_dist_m / (n_dp * det_pixel_m)

# Background offset [counts], fracPy's params.backgroundOffset: subtracted from every
# pixel of every frame, negatives clipped to 0 -- posit(ptychogram - backgroundOffset)
# in fracPy's BaseEngine._checkMISC. fracPy re-applies it at every engine start (it
# edits the shared ptychogram in place), so chained engines subtract it again each
# time; here it is applied once. 0 disables it.
#
# Preprocess.py subtracts the mean dark and clips the negatives, so the read noise
# leaves a floor of ~0.2 counts/px over the whole camera (58% zeros, the rest in
# 1/9-count steps from the 9-frame dark average) -- ~13k counts a frame, ~20% of the
# total, which the model would fit as high-angle scattering. The real signal reaches
# that floor at r ~ 70 px. Counts kept (all / central beam r < 16 px / beyond r = 110 px):
#   0.5 -> 58% / 96% / 26%
#   1.0 -> 41% / 92% /  4%
background_offset = 0

n_probe_modes = 1                # mixed-state modes
n_opr_modes = 1                  # variable probe (OPR); 1 disables it

# Initial probe: make_probe + Gerchberg-Saxton against the data (initial-probe cell).
probe_init_type = "disk"         # "disk" or "gaussian"
# Beam size on the sample. 20261107_s1805_8mmzp_2ms_area2: the autocorrelation of the
# mean pattern falls to 10% at r ~ 2.6 mm, a disk of ~3 mm -- fracPy's
# entrancePupilDiameter for this scan. (20261110_5ti30au_zp_2ms: ~1.5 mm.)
probe_diameter_m = 3e-3
probe_speckle = 0.0              # a laser beam has no diffuser; > 0 mottles the disk
probe_speckle_grain_m = None     # delivered grain [m], only used with probe_speckle > 0
probe_speckle_phase_only = False
probe_speckle_seed = 0           # change for a different realization
gs_iterations = 200              # GS iterations fitting the probe's far field to the data; 0 keeps the plain disk

object_padding_px = 100          # extra object buffer around the scan bounding box

num_epochs = 1000
save_freq_iterations = 500       # write a recon_Niter*.h5 snapshot every N epochs
batch_size = 50                  # the number of scan positions (196 in total)
batching_mode = api.BatchingModes.COMPACT
noise_model = api.NoiseModels.GAUSSIAN  # camera ADU, not photon counts
momentum_gain = 0.25

probe_start = 100                  # epoch at which the probe starts updating
opr_start = None                  # None disables OPR weight optimization
position_start = None             # None disables position correction
orthogonalization_stride = 5
position_update_limit_px = 2.0    # per update; the scan step is ~5.4 px
# Per-position beam intensity ("ic"). Pty-Chi applies the OPR weights in the forward
# model only when there are >= 2 OPR modes (forward_models.get_unique_probes), so it
# needs n_opr_modes >= 2 and opr_start set -- with one OPR mode the fitted weights
# would never be used.
optimize_intensity_variation = False

# fracPy flip switches
flip_dp_x = False
# fracPy's preprocess_780nm.py flips every frame (np.flipud); vislight/Preprocess.py does
# not. Its ptychogram equals flipud(dp) for both laser-lab scans, and nothing else in
# the geometry differs (same FFT sign and patch convention), so flip here to match.
flip_dp_y = True
flip_positions_x = False
flip_positions_y = False
swap_position_axes = False       # True if stored positions are (x, y) not (y, x)

device = api.Devices.GPU if torch.cuda.is_available() else api.Devices.CPU
dtype = api.Dtypes.FLOAT32
random_seed = 123

print(f"pixel size = {pixel_size_m * 1e6:.3f} um, FOV = {n_dp * pixel_size_m * 1e3:.2f} mm")


#%% ---------------------------------------------------------------- load diffraction patterns

with h5py.File(dp_file, "r") as f:
    print(f"keys in {dp_file.name}: {list(f.keys())}")
    patterns = f["dp"][()]
    det_pixel_mask = np.asarray(f["det_pixel_mask"][()], dtype=bool)
print(f"raw ptychogram: {patterns.shape} {patterns.dtype}")

patterns = center_crop_or_pad(patterns, n_dp)
det_pixel_mask = center_crop_or_pad(det_pixel_mask, n_dp)
if flip_dp_x:
    patterns = patterns[..., :, ::-1]
    det_pixel_mask = det_pixel_mask[:, ::-1]
if flip_dp_y:
    patterns = patterns[..., ::-1, :]
    det_pixel_mask = det_pixel_mask[::-1, :]
patterns = np.ascontiguousarray(patterns, dtype=np.float32)
np.clip(patterns, 0, None, out=patterns)

print(f"ptychogram: {patterns.shape}, total counts {patterns.sum():.3e}")

# Detector mask, built BEFORE the background offset is subtracted: otherwise every
# pixel the offset zeroes in all frames would be taken for dead, and "no signal" would
# become "no measurement". Pixels kept out of the likelihood:
#
#   file  -- det_pixel_mask from Preprocess.py (all valid for this camera);
#   dead  -- zero in every frame AND surrounded by signal (5x5 median of the frame sum
#            > 0): pixels hot in the dark frames, wiped out by the dark subtraction (20
#            on 20261110_5ti30au). Zero alone is not enough: on a dim scan most of the
#            camera reads zero in every frame because there is no light there, and
#            masking those (60015 px, 92%, on 20261107_s1805) left the probe free to
#            grow a high-angle halo, so the loss climbed after epoch ~200;
#   stuck -- the identical count in all frames, so no noise: a defect, not photons.
frame_sum = patterns.sum(axis=0)
dead_pixels = (frame_sum == 0) & (median_filter(frame_sum, size=5) > 0)
stuck_pixels = (patterns.std(axis=0) == 0) & (frame_sum > 0)
valid_pixel_mask = det_pixel_mask & ~(dead_pixels | stuck_pixels)
print(
    f"detector mask: {(~det_pixel_mask).sum()} from file + {dead_pixels.sum()} dead + "
    f"{stuck_pixels.sum()} stuck -> {100 * valid_pixel_mask.mean():.2f}% valid, "
    f"{100 * frame_sum[valid_pixel_mask].sum() / frame_sum.sum():.3f}% of counts kept"
)

# Background offset (see background_offset above), then check what it removed.
_r = np.hypot(*(np.indices((n_dp, n_dp)) - n_dp // 2))
_core, _outer = _r < 16, _r >= 110
mean_before = frame_sum / len(patterns)
sum_before = patterns.sum(axis=(1, 2))
if background_offset > 0:
    patterns -= background_offset
    np.clip(patterns, 0, None, out=patterns)
mean_after = patterns.mean(axis=0)
sum_after = patterns.sum(axis=(1, 2))
core_after = patterns[:, _core].sum(axis=1)
print(
    f"background offset {background_offset} counts: "
    f"{100 * sum_after.sum() / sum_before.sum():.1f}% of all "
    f"counts kept, {100 * mean_after[_core].sum() / mean_before[_core].sum():.2f}% inside "
    f"r < 16 px, {100 * mean_after[_outer].sum() / mean_before[_outer].sum():.1f}% beyond r = 110 px"
)

# show_ptychogram(patterns, interactive=True)

# Check: mean pattern before/after the offset, its radial profile, and the per-frame
# sums (the floor's jitter dominates the frame sum before the offset).
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
_norm = LogNorm(vmin=1e-2, vmax=float(mean_before.max()))
axes[0].imshow(np.maximum(mean_before, 1e-2), cmap="inferno", norm=_norm)
axes[0].set_title("mean pattern, before offset (log)", fontsize=9)
axes[1].imshow(np.maximum(mean_after, 1e-2), cmap="inferno", norm=_norm)
axes[1].set_title(f"mean pattern, offset {background_offset} (log)", fontsize=9)
_rbin = np.arange(0, int(_r.max()) + 1)
_ri = np.digitize(_r.ravel(), _rbin) - 1
_count = np.maximum(np.bincount(_ri, minlength=len(_rbin)), 1)
axes[2].semilogy(_rbin, np.bincount(_ri, mean_before.ravel(), len(_rbin)) / _count, label="before")
axes[2].semilogy(_rbin, np.bincount(_ri, mean_after.ravel(), len(_rbin)) / _count, label="after")
if background_offset > 0:
    axes[2].axhline(background_offset, color="gray", ls="--", lw=0.8, label="background_offset")
axes[2].set_xlabel("radius [px]"), axes[2].set_ylabel("mean counts / px")
axes[2].set_title("radial profile of the mean pattern", fontsize=9)
axes[2].legend(fontsize=8)
axes[3].plot(sum_before, ".-", lw=0.5, ms=3, label="frame sum, before")
axes[3].plot(sum_after, ".-", lw=0.5, ms=3, label="frame sum, after")
axes[3].plot(core_after, ".-", lw=0.5, ms=3, label="r < 16 px, after")
axes[3].set_xlabel("frame index"), axes[3].set_ylabel("counts")
axes[3].set_title("per-frame counts", fontsize=9)
axes[3].legend(fontsize=8)
plt.tight_layout()
plt.show()

#%% ---------------------------------------------------------------- positions

# Measured positions from the para file. They form a perfect 14 x 14 serpentine grid
# (0.3 mm = ~5.4 px step), which invites raster-grid pathology; position correction
# is the only remedy available here.
positions_px = np.stack((ppy, ppx), axis=-1) / pixel_size_m    # m -> object pixels, (y, x)

if swap_position_axes:
    positions_px = positions_px[:, ::-1].copy()
if flip_positions_y:
    positions_px[:, 0] = -positions_px[:, 0]
if flip_positions_x:
    positions_px[:, 1] = -positions_px[:, 1]

# Pty-Chi maps position (0, 0) to the center of the object buffer
# (object_options.determine_position_origin_coords_by), so keep the scan centered.
positions_px = positions_px - positions_px.mean(axis=0, keepdims=True)

assert len(positions_px) == len(patterns), (
    f"{len(patterns)} patterns vs {len(positions_px)} positions"
)
print(
    f"{len(positions_px)} positions: "
    f"y [{positions_px[:, 0].min():.1f}, {positions_px[:, 0].max():.1f}] px, "
    f"x [{positions_px[:, 1].min():.1f}, {positions_px[:, 1].max():.1f}] px"
)

plt.figure(figsize=(3, 3))
plt.plot(positions_px[:, 1], positions_px[:, 0], ".-", lw=0.3, ms=2)
plt.gca().set_aspect("equal")
plt.title("scan positions [px]")
plt.show()


#%% ---------------------------------------------------------------- initial probe

probe_init = make_probe(
    n_dp, probe_init_type, diameter=probe_diameter_m, pixel_size_m=pixel_size_m,
    speckle=probe_speckle, speckle_grain_m=probe_speckle_grain_m,
    speckle_phase_only=probe_speckle_phase_only, seed=probe_speckle_seed,
    verbose=True,
)

# make_probe draws a flat-phase disc on the sample plane. With the sample at the
# front focal plane of the Fourier lens the camera sees the beam's exact Fourier
# transform, so the disc's far field is its Airy pattern -- the right kind of thing
# (the measured beam is a compact central spot), but its width and rings follow the
# idealized edge, not the real beam's profile.
#
# Fix it with Gerchberg-Saxton against the measured illumination: a low percentile
# across positions keeps what every frame has in common (the beam) and suppresses the
# position-dependent object scatter. Alternate imposing that amplitude in the far
# field with make_probe's own real-space envelope, so the probe keeps the footprint
# you set -- only its far-field amplitude is corrected, from your own data. The
# percentile is taken after the background offset is subtracted, so the read-noise
# floor is not built into the probe. Scale is irrelevant here, rescale_probe follows.
#
# Note this fixes the far-field AMPLITUDE only. The far-field PHASE stays arbitrary --
# GS converges to some field consistent with both amplitudes, not to the beam's
# actual phase, which only a reconstruction recovers.
_pct10 = np.percentile(patterns, 10, axis=0)
_ff_amp = np.sqrt(np.fft.ifftshift(_pct10))
_measured = np.fft.ifftshift(valid_pixel_mask)   # no measurement at dead pixels
_p = np.asarray(probe_init, dtype=np.complex64)
_flat = _p.reshape(-1, n_dp, n_dp)
for _i in range(_flat.shape[0]):
    _mode = _flat[_i]
    _envelope = np.abs(_mode) > 0.01 * np.abs(_mode).max()
    for _ in range(gs_iterations):
        _far = np.fft.fft2(_mode)
        _far = np.where(_measured, _ff_amp * np.exp(1j * np.angle(_far)), _far)
        _mode = np.fft.ifft2(_far) * _envelope
    _flat[_i] = _mode
probe_init = _flat.reshape(_p.shape)

probe = torch.as_tensor(probe_init, dtype=get_default_complex_dtype())

# Check: mode 0 in real space, and its far field next to the measured beam.
_m0 = probe_init.reshape(-1, n_dp, n_dp)[0]
_amp0 = np.abs(_m0)
_far0 = np.fft.fftshift(np.abs(np.fft.fft2(_m0)) ** 2)
_far0 *= _pct10[valid_pixel_mask].sum() / _far0[valid_pixel_mask].sum()
_mismatch = np.linalg.norm((np.sqrt(_far0) - np.sqrt(_pct10))[valid_pixel_mask]) / np.linalg.norm(
    np.sqrt(_pct10)[valid_pixel_mask]
)
print(f"synthesized {probe_init_type} probe, diameter {probe_diameter_m * 1e3:.2f} mm, "
      + (f"far field matched to the data in {gs_iterations} GS iterations " if gs_iterations
         else "plain disk, no GS fit ")
      + f"(far-field amplitude mismatch to the data {_mismatch:.3f})")

_ext = np.array([-0.5, 0.5, 0.5, -0.5]) * n_dp * pixel_size_m * 1e3
_zoom_mm = 1.5 * probe_diameter_m * 1e3
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(_amp0, cmap="inferno", extent=_ext)
axes[0].set_title(f"initial probe |mode 0| ({probe_init_type} + GS)", fontsize=9)
axes[1].imshow(np.where(_amp0 > 0.05 * _amp0.max(), np.angle(_m0), np.nan),
               cmap="twilight", extent=_ext)
axes[1].set_title("phase where |p| > 5% of max", fontsize=9)
for ax in axes[:2]:
    ax.set_xlim(-_zoom_mm, _zoom_mm)
    ax.set_ylim(_zoom_mm, -_zoom_mm)
    ax.set_xlabel("x [mm]"), ax.set_ylabel("y [mm]")
_norm = LogNorm(vmin=max(float(_pct10[_pct10 > 0].min()), 1e-1), vmax=float(_pct10.max()))
axes[2].imshow(np.where(valid_pixel_mask, _pct10, np.nan), cmap="inferno", norm=_norm)
axes[2].set_title("measured beam far field (10th percentile)", fontsize=9)
axes[3].imshow(_far0, cmap="inferno", norm=_norm)
axes[3].set_title("|FFT(mode 0)|^2", fontsize=9)
for ax in axes[2:]:
    ax.set_xlim(n_dp // 2 - 60, n_dp // 2 + 60)
    ax.set_ylim(n_dp // 2 + 60, n_dp // 2 - 60)
plt.tight_layout()
plt.show()

# Incoherent modes: keep what we have, fill the rest with Hermite modes.
if probe.shape[1] > n_probe_modes:
    probe = probe[:, :n_probe_modes]
elif probe.shape[1] < n_probe_modes:
    padded = torch.zeros(
        (probe.shape[0], n_probe_modes, *probe.shape[-2:]), dtype=get_default_complex_dtype()
    )
    padded[:, : probe.shape[1]] = probe
    # secondary_mode_energy is the energy of EACH secondary mode, not the total:
    # mode 0 gets 1 - (n_probe_modes - 1) * s. hermite_secondary_modes is Pty-Chi's
    # orthogonalize_initial_probe with the modes walked in square order.
    probe = hermite_secondary_modes(padded, secondary_mode_energy=0.05)

# OPR modes.
if probe.shape[0] > n_opr_modes:
    probe = probe[:n_opr_modes]
elif probe.shape[0] < n_opr_modes:
    probe = add_additional_opr_probe_modes_to_probe(probe, n_opr_modes - probe.shape[0])


fig, axes = plt.subplots(1, probe.shape[1], figsize=(3 * probe.shape[1], 3))
for i, ax in enumerate(np.atleast_1d(axes)):
    ax.imshow(np.abs(probe[0, i].numpy()), cmap="inferno")
    ax.set_title(f"mode {i}")
    ax.set_xticks([]), ax.set_yticks([])
plt.show()

#%%
probe = torch.as_tensor(rescale_probe(probe, patterns), dtype=get_default_complex_dtype())
opr_weights = generate_initial_opr_mode_weights(len(positions_px), probe.shape[0], probe=probe)
print(f"probe: {tuple(probe.shape)} (n_opr, n_modes, h, w)")

#%% ---------------------------------------------------------------- initial object

object_shape = get_suggested_object_size(positions_px, probe.shape[-2:], extra=object_padding_px)
obj = torch.ones((1, *object_shape), dtype=get_default_complex_dtype())  # (n_slices, h, w)

print(f"object buffer: {tuple(obj.shape)}")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(np.abs(obj[0].numpy()), cmap="gray")
axes[0].set_title("initial object magnitude")
axes[1].imshow(np.angle(obj[0].numpy()), cmap="gray")
axes[1].set_title("initial object phase")
for ax in axes:
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()

#%% ---------------------------------------------------------------- options

options = api.LSQMLOptions()

# --- data / geometry ---
options.data_options.wavelength_m = wavelength_m
options.data_options.free_space_propagation_distance_m = np.inf if far_field else det_dist_m
# Measured patterns have DC at the center; the far-field forward model does not
# shift after the FFT, so the data must be pre-shifted. Near-field involves no
# Fraunhofer FFT, so it must not be shifted.
options.data_options.fft_shift = far_field
options.data_options.save_data_on_device = True    # 196 x 256^2 fits easily in VRAM

# --- reconstructor ---
options.reconstructor_options.num_epochs = num_epochs
options.reconstructor_options.batch_size = batch_size
options.reconstructor_options.batching_mode = batching_mode
options.reconstructor_options.noise_model = noise_model
options.reconstructor_options.momentum_acceleration_gain = momentum_gain
options.reconstructor_options.default_device = device
options.reconstructor_options.default_dtype = dtype
options.reconstructor_options.random_seed = random_seed
options.reconstructor_options.rescale_probe_intensity_in_first_epoch = True

# --- object ---
options.object_options.optimizable = True
options.object_options.optimizer = api.Optimizers.SGD
options.object_options.step_size = 0.2
options.object_options.pixel_size_m = pixel_size_m
options.object_options.build_preconditioner_with_all_modes = True
options.object_options.determine_position_origin_coords_by = (
    api.ObjectPosOriginCoordsMethods.SUPPORT
)
# fracPy object constraints (all off):
# options.object_options.l2_norm_constraint.enabled = True
# options.object_options.l2_norm_constraint.weight = 1e-3
# options.object_options.smoothness_constraint.enabled = True
# options.object_options.smoothness_constraint.alpha = 0.05

# object constrain in the Fourier space
options.object_options.fourier_support_constraint.enabled = False
options.object_options.fourier_support_constraint.optimization_plan = (
    OptimizationPlan(start=0, stride=1)
)
# Absolute cutoff in FFT bins of the object buffer. Takes priority over the ratio
# below; set it to None to fall back to the ratio. Note the object buffer is sized
# per scan, so a fixed bin radius drifts in physical frequency between scans.
options.object_options.fourier_support_constraint.radius_px = 100
# Used only when radius_px is None: ko / kp in cycles/m against the probe cutoff.
options.object_options.fourier_support_constraint.radius_ratio_to_probe = 0.5

# --- probe ---
options.probe_options.optimizable = True
options.probe_options.optimizer = api.Optimizers.SGD
options.probe_options.step_size = 0.2
options.probe_options.optimization_plan = OptimizationPlan(start=probe_start)
options.probe_options.orthogonalize_incoherent_modes.enabled = n_probe_modes > 1
options.probe_options.orthogonalize_incoherent_modes.optimization_plan = OptimizationPlan(
    stride=orthogonalization_stride
)
options.probe_options.orthogonalize_incoherent_modes.method = api.OrthogonalizationMethods.SVD
options.probe_options.orthogonalize_opr_modes.enabled = n_opr_modes > 1
options.probe_options.power_constraint.enabled = True      # fracPy probePowerCorrectionSwitch
options.probe_options.center_constraint.enabled = True     # fracPy comStabilizationSwitch

# probe constrain in the real space
options.probe_options.support_constraint.enabled = False
options.probe_options.support_constraint.fixed_probe_support = (api.ProbeSupportMethods.ELLIPSE)
options.probe_options.support_constraint.threshold = 1e-3
options.probe_options.support_constraint.fixed_probe_support_params = [
    n_dp / 2, n_dp / 2,       # center row, center column
    probe_diameter_m / pixel_size_m / 2,
    probe_diameter_m / pixel_size_m / 2,  # radius in pixels
]

# probe constrain in the Fourier space
options.probe_options.fourier_support_constraint.enabled = True
options.probe_options.fourier_support_constraint.optimization_plan = (
    OptimizationPlan(start=probe_start, stride=1)
)
# In far field the probe's FFT bins are camera pixels. The beam's own far field (10th
# percentile across positions) holds 95% of its power within ~41-50 px and the signal
# reaches the read-noise floor at ~70 px, so 64 px keeps the beam and keeps the flat
# floor out of the probe. (The farthest bin is ~181 px from DC at n_dp = 256.)
options.probe_options.fourier_support_constraint.radius_px = 64

# --- probe positions (fracPy pcPIE) ---
if position_start is None:
    options.probe_position_options.optimizable = False
else:
    options.probe_position_options.optimizable = True
    options.probe_position_options.optimizer = api.Optimizers.SGD
    options.probe_position_options.step_size = 0.3
    options.probe_position_options.optimization_plan = OptimizationPlan(start=position_start)
    options.probe_position_options.constrain_position_mean = True
    options.probe_position_options.correction_options.correction_type = (
        api.PositionCorrectionTypes.GRADIENT
    )
    options.probe_position_options.correction_options.differentiation_method = (
        api.ImageGradientMethods.FOURIER_DIFFERENTIATION
    )
    options.probe_position_options.correction_options.update_magnitude_limit = (
        position_update_limit_px
    )
    options.probe_position_options.correction_options.clip_update_magnitude_by_mad = True
    options.probe_position_options.momentum_acceleration_gain = 0.5

# --- OPR mode weights (variable probe) ---
# With a single OPR mode Pty-Chi never applies the OPR weights in the forward model
# (forward_models.get_unique_probes), so intensity correction would fit unused weights.
assert not optimize_intensity_variation or (n_opr_modes > 1 and opr_start is not None), (
    "optimize_intensity_variation needs n_opr_modes >= 2 and opr_start set"
)
if n_opr_modes > 1 and opr_start is not None:
    options.opr_mode_weight_options.optimizable = True
    options.opr_mode_weight_options.optimize_eigenmode_weights = True
    options.opr_mode_weight_options.optimize_intensity_variation = optimize_intensity_variation
    options.opr_mode_weight_options.optimization_plan = OptimizationPlan(start=opr_start)
    options.opr_mode_weight_options.update_relaxation = 0.1
else:
    options.opr_mode_weight_options.optimizable = False

print(
    f"schedule: object 0-, probe {probe_start}-, OPR {opr_start}-, positions {position_start}-, "
    f"{num_epochs} epochs on {device}"
)


#%% ---------------------------------------------------------------- build task

task = PtychographyTask(
    options,
    diffraction_data=patterns,
    object_data=obj,
    probe_data=probe,
    probe_position_y_px=positions_px[:, 0],
    probe_position_x_px=positions_px[:, 1],
    opr_mode_weights_data=opr_weights,
    # Same detector layout as `patterns`; Pty-Chi fft-shifts the mask alongside the
    # data under the same fft_shift flag (io_handles.PtychographyDataset).
    valid_pixel_mask=valid_pixel_mask,
)


#%% ---------------------------------------------------------------- run & save output in the PEAR/beamline layout
from utils import make_recon_dir_name, save_initial_conditions, save_reconstruction

recon_dir = out_dir / scan / make_recon_dir_name(recon_dir_suffix)
save_initial_conditions(recon_dir)

# Like the beamline batch script, run in chunks of save_freq_iterations and drop a
# recon_Niter*.h5 snapshot after each one. State persists between task.run() calls,
# so this is identical to one long run — and the intermediate files let you pick up
# a reconstruction that was stopped early.
epochs_done = 0
while epochs_done < num_epochs:
    chunk = min(save_freq_iterations, num_epochs - epochs_done)
    task.run(chunk)
    epochs_done += chunk
    save_reconstruction(task, recon_dir, epochs_done)


#%% ---------------------------------------------------------------- inspect

recon_obj = task.get_data_to_cpu("object", as_numpy=True)[0]
recon_probe = task.get_data_to_cpu("probe", as_numpy=True)[0]
recon_pos = task.get_data_to_cpu("probe_positions", as_numpy=True)
loss_table = task.reconstructor.loss_tracker.table

fig, axes = plt.subplots(1, 3, figsize=(15, 7))
axes[0].imshow(np.angle(recon_obj), cmap="gray")
axes[0].set_title("object phase")
axes[1].imshow(np.abs(recon_obj), cmap="gray")
axes[1].set_title("object magnitude")
axes[2].imshow(np.abs(recon_probe[0]), cmap="inferno")
axes[2].set_title("probe mode 0")
for ax in axes:
    ax.set_xticks([]), ax.set_yticks([])
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 3))
plt.semilogy(loss_table["epoch"], loss_table["loss"])
plt.xlabel("epoch"), plt.ylabel("loss")
plt.tight_layout()
plt.show()

if position_start is not None:
    plt.figure(figsize=(4, 4))
    plt.plot(positions_px[:, 1], positions_px[:, 0], ".", ms=3, label="initial")
    plt.plot(recon_pos[:, 1], recon_pos[:, 0], ".", ms=3, label="corrected")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.title("position correction")
    plt.show()


#%% ---------------------------------------------------------------- save

# The run loop above already wrote recon_Niter{save_freq_iterations}.h5 ...
# recon_Niter{num_epochs}.h5. This cell stands on its own: run it any time -- after a
# hand-issued task.run(n), or after interrupting the loop -- and it snapshots the
# current state under the number of epochs actually completed.
save_reconstruction(task, recon_dir)

# Extra, not part of the beamline layout: the full loss table including reg_loss.
task.reconstructor.loss_tracker.table.to_csv(recon_dir / "loss.csv", index=False)
print(f"results in {recon_dir}")

# %%
