#%%
# Pty-Chi (LSQML) reconstruction of the 12-ID-C Siemens star scan.

from utils import use_inline_backend
use_inline_backend()             # Plot Viewer-compatible backend, in its own cell

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # This makes GPU N appear as GPU 0 to CuPy

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from utils import center_crop_or_pad,show_ptychogram,make_randomized_zoneplate_probe,hermite_secondary_modes

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

scan = "S0019"
scan_initialGuess = "S0019"  # scan used to generate the initial guess (positions + probe)
# On the beamline (Linux) the shares are /mnt/micdata2 and /mnt/micdata3;
# here on Windows the same shares are \\micdata\data2 and \\micdata\data3.
data_root = Path("/mnt/micdata2/12IDC/2026_Data/2026_3/01_piezo_test")
# data_root = Path(r"\\micdata\data2\12IDC\2026_Data\2026_3\01_piezo_test")

dp_file = data_root / "preproc" / scan / "data_roi0_Ndp1024_dp.hdf5"
para_file = data_root / "preproc" / scan / "data_roi0_Ndp1024_para.hdf5"

# Previous Pty-Chi recon used as the source of positions and of the probe guess
# (same file the fracPy script read). Set to None to start from the para file
# positions and a synthesized probe instead.
init_recon_file = (
    data_root / "ptychi_recons" / scan_initialGuess
    / "Ndp1024_LSQML_c20_m0.5_gaussian_p10_cp_mm_opr3_ic_pc1_f_ul2" / "recon_Niter400.h5"
)

# init_recon_file = None

out_dir = Path("/mnt/micdata3/fengling/2026_03_results") / "ptychi_recons"
# out_dir = Path(r"\\micdata\data3\fengling\2026_03_results") / "ptychi_recons"
recon_dir_suffix = "_initprobe_fromzpOSA_edgek200"            # appended to the folder name, e.g. "_v2" or "_pos"

#%% ---------------------------------------------------------------- read the parameters from the file
with h5py.File(para_file, "r") as f:
    print(f"keys in {para_file.name}: {list(f.keys())}")
    ppx = np.asarray(f["ppX"][()], dtype=np.float64).squeeze()
    ppy = np.asarray(f["ppY"][()], dtype=np.float64).squeeze()
    energy_kev = np.asarray(f["energy"][()], dtype=np.float32).squeeze()
    detector_distance_m = np.asarray(f["detector_distance"][()], dtype=np.float32).squeeze()
    exposure_time_ms = np.asarray(f["exposure_time_s"][()], dtype=np.float32).squeeze()


#%% ---------------------------------------------------------------- geometry & knobs

n_dp = 1024                          # detector crop
wavelength_m = 1.24e-9 / energy_kev # trasfer from keV to meters         
det_pixel_m = 172e-6                # fracPy exampleData.dxd
det_dist_m = detector_distance_m    # fracPy exampleData.zo
far_field = True                    # fracPy propagatorType 'Fraunhofer'

# fracPy fixed dxo = dxp = 1.76e-8; that is exactly the far-field sampling below.
pixel_size_m = wavelength_m * det_dist_m / (n_dp * det_pixel_m)

n_probe_modes = 5                # fracPy reconstruction.npsm, mix-state modes
n_opr_modes = 1                  # variable probe (OPR); 1 disables it
probe_focal_spot_diameter_m = 3.5e-6   # desired illuminated footprint at the sample (2*Rmax)
probe_diameter_m = probe_focal_spot_diameter_m  # feeds the real-space support constraint below
probe_outer_zone_width_m = 50e-9       # finest zone of the (randomized) zone plate
probe_zp_central_stop = 0.15           # fraction of the simulated zone plate's radius blocked
probe_zp_seed = 0                      # change for a different speckle realization

# Rotation of the incoherent-mode basis. The reference's modes 1 and 2 are
# two-lobe modes split along the diagonals (dipole axes -55.7 and +38.1 deg) and
# mode 3 is four-part on the axes (quadrupole at -1.2 deg); 0.0 would put the
# dipoles on the axes and the quadrupole on the diagonals instead.
#
# Those are SVD eigenmodes of the converged beam, not Hermite functions, so no
# single angle fits all three. Sweeping against the reference, the mean axis
# error is 4.3 deg at 36, 3.7 at 38, 4.0 at 40, 6.7 at 45 -- 38 is the best
# compromise, trading ~5 deg on the quadrupole for ~7 on the dipoles.
probe_mode_rotation_deg = 38.0

object_padding_px = 100          # extra object buffer around the scan bounding box

num_epochs = 1500
save_freq_iterations = 500       # write a recon_Niter*.h5 snapshot every N epochs
batch_size = 100                 # the number of scan positions
batching_mode = api.BatchingModes.COMPACT
noise_model = api.NoiseModels.GAUSSIAN  # fracPy params.noiseModel
momentum_gain = 0.25              # fracPy params.momentumAcceleration

probe_start = 10                  # epoch at which the probe starts updating
opr_start = None                   # None disables OPR weight optimization
position_start = 10              # None disables position correction (fracPy pcPIE)
orthogonalization_stride = 5     # fracPy params.orthogonalizationFrequency
position_update_limit_px = 20.0
optimize_intensity_variation = False   # per-position beam intensity ("ic")

# fracPy flip switches
flip_dp_x = False
flip_dp_y = False
flip_positions_x = False
flip_positions_y = False
swap_position_axes = False       # True if stored positions are (x, y) not (y, x)

device = api.Devices.GPU if torch.cuda.is_available() else api.Devices.CPU
dtype = api.Dtypes.FLOAT32
random_seed = 123

print(f"pixel size = {pixel_size_m * 1e9:.3f} nm, FOV = {n_dp * pixel_size_m * 1e6:.2f} um")


#%% ---------------------------------------------------------------- load diffraction patterns

with h5py.File(dp_file, "r") as f:
    print(f"keys in {dp_file.name}: {list(f.keys())}")
    patterns = f["dp"][()]
print(f"raw ptychogram: {patterns.shape}")

patterns = center_crop_or_pad(patterns, n_dp)
if flip_dp_x:
    patterns = patterns[..., :, ::-1]
if flip_dp_y:
    patterns = patterns[..., ::-1, :]
patterns = np.ascontiguousarray(patterns, dtype=np.float32)
np.clip(patterns, 0, None, out=patterns)

print(f"ptychogram: {patterns.shape}, total counts {patterns.sum():.3e}")

# Detector mask. Two classes of pixel carry no measurement and must be kept out of
# the likelihood instead of being fit as data:
#
#   dead  -- the Pilatus module gaps, a hard zero in every frame (21% of the array,
#            including ~4300 px inside r=128 where 98% of the counts land). Left
#            unmasked these assert "no scattered intensity here", which is false.
#   stuck -- pixels reporting the identical count in all frames, so no Poisson
#            variation: a detector defect, not photons. In S0019 this is
#            (921, 532) and (921, 533) at 5021 counts/frame each -- 4.5x the
#            brightest genuine pixel in a frame, sitting at r=409 where the beam
#            has no real signal. The probe grows a spurious high-angle lobe
#            trying to explain them.
frame_sum = patterns.sum(axis=0)
dead_pixels = frame_sum == 0
stuck_pixels = (patterns.std(axis=0) == 0) & (frame_sum > 0)
valid_pixel_mask = ~(dead_pixels | stuck_pixels)
print(
    f"detector mask: {dead_pixels.sum()} dead + {stuck_pixels.sum()} stuck "
    f"-> {100 * valid_pixel_mask.mean():.2f}% valid, "
    f"{100 * frame_sum[valid_pixel_mask].sum() / frame_sum.sum():.3f}% of counts kept"
)

# show_ptychogram(patterns, interactive=True)

#%% ---------------------------------------------------------------- positions + prior probe/object

prior = {}
if init_recon_file is not None:
    with h5py.File(init_recon_file, "r") as f:
        print(f"keys in {init_recon_file.name}: {list(f.keys())}")
        # prior["object"] = np.asarray(f["object"][()]).view(np.complex64)
        # prior["probe"] = np.asarray(f["probe"][()]).view(np.complex64)
        prior["positions_px"] = np.asarray(f["positions_px"][()], dtype=np.float64)

if "positions_px" in prior:
    positions_px = prior["positions_px"].copy()
else:
    with h5py.File(para_file, "r") as f:
        print(f"keys in {para_file.name}: {list(f.keys())}")
        ppx = np.asarray(f["ppX"][()], dtype=np.float64).squeeze()
        ppy = np.asarray(f["ppY"][()], dtype=np.float64).squeeze()
    positions_px = np.stack((ppy, ppx), axis=-1) / pixel_size_m

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

plt.figure(figsize=(2, 2))
plt.plot(positions_px[:, 1], positions_px[:, 0], ".-", lw=0.3, ms=2)
plt.gca().set_aspect("equal")
plt.title("scan positions [px]")
plt.show()


#%% ---------------------------------------------------------------- initial probe

if "probe" in prior:
    probe = center_crop_or_pad(prior["probe"], n_dp)
    # Pty-Chi wants exactly (n_opr, n_modes, h, w); fold any extra leading axes
    # (e.g. a wavelength axis) into the OPR axis.
    probe = probe.reshape((-1,) + probe.shape[-3:])
    probe = torch.as_tensor(np.ascontiguousarray(probe), dtype=get_default_complex_dtype())
    print(f"probe from {init_recon_file.name}: {tuple(probe.shape)}")
else:
    probe_init = make_randomized_zoneplate_probe(
        n_dp, pixel_size_m, wavelength_m,
        probe_outer_zone_width_m, probe_focal_spot_diameter_m,
        central_stop=probe_zp_central_stop, seed=probe_zp_seed,
    )
    # The physical simulation gets the probe's real-space footprint, grain, and
    # confinement right, but it is still a random realization -- its far-field
    # ENERGY DISTRIBUTION has no reason to match the real optic's. Measured:
    # this simulated probe's far field peaks at DC, with 60% of its power
    # inside detector radius 20 px, where the real beam (recon_Niter400.h5,
    # same method) instead shows a clean annulus peaking at r~36 with ~0% near
    # DC -- the hallmark of a real zone plate + central beamstop. Left alone,
    # the reconstructor has no way to explain that mismatch except by
    # corrupting the object before the probe is even allowed to update
    # (probe_start=100) -- not a resolution problem, a "can't see the spokes
    # at all" problem.
    #
    # Fix it with Gerchberg-Saxton against the measured illumination: a low
    # percentile across positions keeps what every frame has in common (the
    # beam) and suppresses the position-dependent object scatter. Alternate
    # imposing that amplitude in the far field with the simulated probe's own
    # real-space envelope, so the probe keeps the footprint/grain already
    # validated -- only its far-field amplitude is corrected, from the data.
    #
    # Note this fixes the far-field AMPLITUDE only. The far-field PHASE stays
    # arbitrary -- GS converges to some field consistent with both amplitudes,
    # not to the real beam's actual speckle phase, which only a reconstruction
    # recovers.
    #
    # The target amplitude is gap-filled across dead/stuck detector pixels
    # (module gaps) before use, instead of leaving those rows/columns
    # unconstrained in every GS iteration. Leaving them free lets the
    # iteration settle on an arbitrary value there each time that need not
    # agree with its now tightly-constrained neighbors -- confirmed to
    # produce grid-aligned streak artifacts in the far field, exactly at the
    # detector's module-gap rows/columns, that the real probe does not have.
    # Filling the gaps first (normalized convolution: blur data*mask and
    # mask separately, divide) and imposing the result everywhere removes
    # the streaks while keeping the same ring match.
    gs_iterations = 200
    _percentile_10 = np.percentile(patterns, 10, axis=0)
    _mask = valid_pixel_mask.astype(np.float32)
    _num = gaussian_filter(_percentile_10 * _mask, sigma=3.0)
    _den = gaussian_filter(_mask, sigma=3.0)
    _gap_filled = np.where(valid_pixel_mask, _percentile_10, _num / np.maximum(_den, 1e-6))
    _ff_amp = np.sqrt(np.fft.ifftshift(_gap_filled))
    _p = np.asarray(probe_init, dtype=np.complex64)
    _flat = _p.reshape(-1, n_dp, n_dp)
    for _i in range(_flat.shape[0]):
        _mode = _flat[_i]
        _envelope = np.abs(_mode) > 0.01 * np.abs(_mode).max()
        for _ in range(gs_iterations):
            _far = np.fft.fft2(_mode)
            _far = _ff_amp * np.exp(1j * np.angle(_far))   # imposed everywhere, no free region
            _mode = np.fft.ifft2(_far) * _envelope
        _flat[_i] = _mode
    probe_init = _flat.reshape(_p.shape)

    probe = torch.as_tensor(probe_init, dtype=get_default_complex_dtype())
    print(f"synthesized randomized-zone-plate probe: focal spot "
          f"{probe_focal_spot_diameter_m * 1e6:.2f} um, outer zone "
          f"{probe_outer_zone_width_m * 1e9:.0f} nm, "
          f"far field matched to the data in {gs_iterations} GS iterations")

# Incoherent modes: keep what we have, fill the rest with Hermite modes.
if probe.shape[1] > n_probe_modes:
    probe = probe[:, :n_probe_modes]
elif probe.shape[1] < n_probe_modes:
    padded = torch.zeros(
        (probe.shape[0], n_probe_modes, *probe.shape[-2:]), dtype=get_default_complex_dtype()
    )
    padded[:, : probe.shape[1]] = probe
    # secondary_mode_energy is the energy of EACH secondary mode, not the total:
    # mode 0 gets 1 - (n_probe_modes - 1) * s. At 5 modes, 0.0575 puts 77% in
    # mode 0 and 5.75% in each of the other four, matching the reference recon's
    # 77.6% mode-0 share. hermite_secondary_modes is Pty-Chi's
    # orthogonalize_initial_probe with the modes walked in square order, so the
    # ladder is dipole, dipole, quadrupole like the reference, and with the
    # basis rotated onto the diagonals.
    probe = hermite_secondary_modes(
        padded, secondary_mode_energy=0.05, rotation_deg=probe_mode_rotation_deg
    )

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

probe = torch.as_tensor(rescale_probe(probe, patterns), dtype=get_default_complex_dtype())
opr_weights = generate_initial_opr_mode_weights(len(positions_px), probe.shape[0], probe=probe)
print(f"probe: {tuple(probe.shape)} (n_opr, n_modes, h, w)")

#%% ---------------------------------------------------------------- initial object

object_shape = get_suggested_object_size(positions_px, probe.shape[-2:], extra=object_padding_px)
obj = torch.ones((1, *object_shape), dtype=get_default_complex_dtype())  # (n_slices, h, w)
# obj = torch.full((1, *object_shape), 1j, dtype=get_default_complex_dtype())

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
# Same detector layout as `patterns`; Pty-Chi fft-shifts the mask alongside the
# data under the same fft_shift flag (io_handles.PtychographyDataset).
options.data_options.valid_pixel_mask = valid_pixel_mask
options.data_options.save_data_on_device = False   # True is faster if it fits in VRAM

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
options.object_options.step_size = 1.0
options.object_options.pixel_size_m = pixel_size_m
options.object_options.build_preconditioner_with_all_modes = True
options.object_options.determine_position_origin_coords_by = (
    api.ObjectPosOriginCoordsMethods.SUPPORT
)
# fracPy object constraints (all off in the source script):
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
options.probe_options.step_size = 0.4
options.probe_options.optimization_plan = OptimizationPlan(start=probe_start)
options.probe_options.orthogonalize_incoherent_modes.enabled = n_probe_modes > 1
options.probe_options.orthogonalize_incoherent_modes.optimization_plan = OptimizationPlan(
    stride=orthogonalization_stride
)
options.probe_options.orthogonalize_incoherent_modes.method = api.OrthogonalizationMethods.SVD
options.probe_options.orthogonalize_opr_modes.enabled = n_opr_modes > 1
options.probe_options.power_constraint.enabled = True      # fracPy probePowerCorrectionSwitch
options.probe_options.center_constraint.enabled = True     # fracPy comStabilizationSwitch


# probe constrain in the Fourier space
options.probe_options.fourier_support_constraint.enabled = True
options.probe_options.fourier_support_constraint.optimization_plan = (
    OptimizationPlan(start=probe_start, stride=1)
)
options.probe_options.fourier_support_constraint.radius_px = 200

# --- probe positions (fracPy pcPIE) ---
if position_start is None:
    options.probe_position_options.optimizable = False
else:
    options.probe_position_options.optimizable = False
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
)


#%% ---------------------------------------------------------------- run & save output in the PEAR/beamline layout
from utils import make_recon_dir_name, save_initial_conditions, save_reconstruction

# Re-derived here because use_simulated_data rewrites n_dp and batch_size after the
# cell that first set it; for a real scan this is the same path.
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
# axes[0].imshow(np.angle(recon_obj[380:480, 480:580]), cmap="gray")
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

# probe_init_det = np.fft.fftshift(np.fft.fft2(probe_init[0,0,:,:]))
# plt.imshow(np.abs(probe_init_det))
# recon_probe_det = np.fft.fftshift(np.fft.fft2(recon_probe[0,:,:]))
# plt.imshow(np.abs(recon_probe_det))

# init_recon_file = (
#     data_root / "ptychi_recons" / scan_initialGuess
#     / "Ndp1024_LSQML_c20_m0.5_gaussian_p10_cp_mm_opr3_ic_pc1_f_ul2" / "recon_Niter400.h5"
# )
# with h5py.File(init_recon_file, "r") as f:
#     prior["probe"] = np.asarray(f["probe"][()]).view(np.complex64)
# probe = center_crop_or_pad(prior["probe"], n_dp)
# probe_det = np.fft.fftshift(np.fft.fft2(probe[0,0,:,:]))
# plt.imshow(np.abs(probe_det))


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
