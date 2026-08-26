#%%
# Pty-Chi (LSQML) reconstruction of the 12-ID-C Siemens star scan.
# Port of fracPy_Argonne/Argonne_work/reconstruction_20260726_seimenStar.py
#
# fracPy runs a chain of engines with hand-tuned betas (mPIE -> multiPIE -> pcPIE).
# LSQML solves the object/probe step sizes analytically, so instead of beta staging
# the schedule below is expressed with per-parameter OptimizationPlan(start=...):
#
#   epoch 0        object only                      (fracPy: mPIE, betaProbe ~ 0)
#   probe_start    probe modes released             (fracPy: multiPIE)
#   opr_start      OPR modes + intensity variation
#   position_start position correction on           (fracPy: pcPIE)
#
# fracPy -> Pty-Chi parameter map:
#   exampleData.ptychogram         -> data_options.data                (n, N, N) intensities
#   exampleData.encoder / dxo      -> probe_position_{y,x}_px
#   exampleData.wavelength         -> data_options.wavelength_m
#   exampleData.dxo / dxp          -> object_options.pixel_size_m
#   propagatorType 'Fraunhofer'    -> free_space_propagation_distance_m = inf
#   reconstruction.npsm            -> probe axis 1 (incoherent modes)
#   params.orthogonalizationSwitch -> probe_options.orthogonalize_incoherent_modes
#   params.momentumAcceleration    -> reconstructor_options.momentum_acceleration_gain
#   params.positionCorrectionSwitch-> probe_position_options.optimizable

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # This makes GPU N appear as GPU 0 to CuPy

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

import ptychi.api as api
from ptychi.api.options.base import OptimizationPlan
from ptychi.api.task import PtychographyTask
from ptychi.utils import (
    add_additional_opr_probe_modes_to_probe,
    generate_initial_opr_mode_weights,
    get_default_complex_dtype,
    get_suggested_object_size,
    orthogonalize_initial_probe,
    rescale_probe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


#%% ---------------------------------------------------------------- paths

scan = "S0206"
scan_initialGuess = "S0206"  # scan used to generate the initial guess (positions + probe)
data_root = Path("/mnt/micdata2/12IDC/2026_Data/2026_2/01_ptycho")

dp_file = data_root / "preproc" / scan / "data_roi0_Ndp1024_dp.hdf5"
para_file = data_root / "preproc" / scan / "data_roi0_Ndp1024_para.hdf5"

# Previous Pty-Chi recon used as the source of positions and of the probe guess
# (same file the fracPy script read). Set to None to start from the para file
# positions and a synthesized probe instead.
init_recon_file = (
    data_root / "ptychi_recons" / scan_initialGuess
    / "Ndp512_LSQML_c150_m0.5_gaussian_p10_cp_mm_opr3_ic_pc1_f_ul2" / "recon_Niter500.h5"
)

out_dir = Path(__file__).parent / "recon_out" / scan

# Set True to skip the beamline files and run the whole pipeline on synthetic
# data (useful to debug the structure without the data share).
use_simulated_data = False


#%% ---------------------------------------------------------------- geometry & knobs

n_dp = 512                       # detector crop, fracPy cropSize
wavelength_m = 0.155e-9          # fracPy exampleData.wavelength
det_pixel_m = 172e-6             # fracPy exampleData.dxd
det_dist_m = 10.0                # fracPy exampleData.zo
far_field = True                 # fracPy propagatorType 'Fraunhofer'

# fracPy fixed dxo = dxp = 1.76e-8; that is exactly the far-field sampling below.
pixel_size_m = wavelength_m * det_dist_m / (n_dp * det_pixel_m)

n_probe_modes = 5                # fracPy reconstruction.npsm
n_opr_modes = 1                  # variable probe (OPR); 1 disables it
probe_diameter_m = 1.0e-6        # only used when no probe comes from init_recon_file
object_padding_px = 100          # extra object buffer around the scan bounding box

num_epochs = 500
batch_size = 200                 # the number of scan positions
batching_mode = api.BatchingModes.COMPACT
noise_model = api.NoiseModels.POISSON
momentum_gain = 0.25              # fracPy params.momentumAcceleration

probe_start = 1                  # epoch at which the probe starts updating
opr_start = 10                   # None disables OPR weight optimization
position_start = 10              # None disables position correction (fracPy pcPIE)
orthogonalization_stride = 5    # fracPy params.orthogonalizationFrequency
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


#%% ---------------------------------------------------------------- helpers


def center_crop_or_pad(arr, size):
    """Center-crop (or zero-pad) the last two axes of `arr` to (size, size)."""
    out = arr
    for axis in (-2, -1):
        n = out.shape[axis]
        if n > size:
            start = (n - size) // 2
            out = np.take(out, np.arange(start, start + size), axis=axis)
        elif n < size:
            pad = [(0, 0)] * out.ndim
            before = (size - n) // 2
            pad[axis] = (before, size - n - before)
            out = np.pad(out, pad)
    return out


def make_disk_probe(size, diameter_px):
    """Soft-edged disk, used when there is no probe to inherit."""
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2
    r = np.hypot(yy, xx)
    edge = max(0.1 * diameter_px / 2, 1.0)
    disk = 0.5 * (1 - np.tanh((r - diameter_px / 2) / edge))
    return disk.astype(np.complex64)[None, None]  # (n_opr, n_modes, h, w)


def siemens_star(shape, n_spokes=24):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    theta = np.arctan2(yy - shape[0] / 2, xx - shape[1] / 2)
    r = np.hypot(yy - shape[0] / 2, xx - shape[1] / 2)
    spokes = (np.cos(n_spokes * theta) > 0) & (r < 0.45 * min(shape))
    return ((1 - 0.15 * spokes) * np.exp(1j * 0.8 * spokes)).astype(np.complex64)


#%% ---------------------------------------------------------------- load diffraction patterns

if use_simulated_data:
    # Small synthetic far-field dataset: 8x8 jittered grid over a Siemens star.
    n_dp = 64
    pixel_size_m = wavelength_m * det_dist_m / (n_dp * det_pixel_m)
    n_probe_modes, n_opr_modes, batch_size, object_padding_px = 2, 2, 16, 8

    rng = np.random.default_rng(0)
    grid = (np.arange(8) - 3.5) * 12.0
    gy, gx = np.meshgrid(grid, grid, indexing="ij")
    positions_px = np.stack([gy.ravel(), gx.ravel()], -1) + rng.normal(0, 0.5, (64, 2))

    sim_shape = get_suggested_object_size(positions_px, (n_dp, n_dp), extra=object_padding_px)
    sim_obj = siemens_star(sim_shape)
    sim_probe = make_disk_probe(n_dp, 40)[0, 0]

    patterns = np.empty((len(positions_px), n_dp, n_dp), dtype=np.float32)
    for i, (py, px) in enumerate(positions_px):
        y0 = int(round(sim_shape[0] / 2 + py - n_dp / 2))
        x0 = int(round(sim_shape[1] / 2 + px - n_dp / 2))
        psi = sim_obj[y0 : y0 + n_dp, x0 : x0 + n_dp] * sim_probe
        patterns[i] = np.abs(np.fft.fftshift(np.fft.fft2(psi, norm="ortho"))) ** 2
    patterns = rng.poisson(patterns / patterns.max() * 1e4).astype(np.float32)
    prior = {}
else:
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


#%% ---------------------------------------------------------------- positions + prior probe/object

if not use_simulated_data:
    prior = {}
    if init_recon_file is not None:
        with h5py.File(init_recon_file, "r") as f:
            print(f"keys in {init_recon_file.name}: {list(f.keys())}")
            prior["object"] = np.asarray(f["object"][()]).view(np.complex64)
            prior["probe"] = np.asarray(f["probe"][()]).view(np.complex64)
            prior["positions_px"] = np.asarray(f["positions_px"][()], dtype=np.float64)

    if "positions_px" in prior:
        positions_px = prior["positions_px"].copy()
    else:
        # fracPy read ppX/ppY (meters) from the para file; Pty-Chi wants pixels.
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

plt.figure(figsize=(4, 4))
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
    probe = torch.as_tensor(
        make_disk_probe(n_dp, probe_diameter_m / pixel_size_m),
        dtype=get_default_complex_dtype(),
    )
    print(f"synthesized disk probe, diameter {probe_diameter_m * 1e6:.2f} um")

# Incoherent modes: keep what we have, fill the rest with Hermite modes.
if probe.shape[1] > n_probe_modes:
    probe = probe[:, :n_probe_modes]
elif probe.shape[1] < n_probe_modes:
    padded = torch.zeros(
        (probe.shape[0], n_probe_modes, *probe.shape[-2:]), dtype=get_default_complex_dtype()
    )
    padded[:, : probe.shape[1]] = probe
    probe = orthogonalize_initial_probe(padded, secondary_mode_energy=0.02)

# OPR modes.
if probe.shape[0] > n_opr_modes:
    probe = probe[:n_opr_modes]
elif probe.shape[0] < n_opr_modes:
    probe = add_additional_opr_probe_modes_to_probe(probe, n_opr_modes - probe.shape[0])

probe = torch.as_tensor(rescale_probe(probe, patterns), dtype=get_default_complex_dtype())
opr_weights = generate_initial_opr_mode_weights(len(positions_px), probe.shape[0], probe=probe)
print(f"probe: {tuple(probe.shape)} (n_opr, n_modes, h, w)")

fig, axes = plt.subplots(1, probe.shape[1], figsize=(3 * probe.shape[1], 3))
for i, ax in enumerate(np.atleast_1d(axes)):
    ax.imshow(np.abs(probe[0, i].numpy()), cmap="inferno")
    ax.set_title(f"mode {i}")
    ax.set_xticks([]), ax.set_yticks([])
plt.show()


#%% ---------------------------------------------------------------- initial object

object_shape = get_suggested_object_size(positions_px, probe.shape[-2:], extra=object_padding_px)
obj = torch.ones((1, *object_shape), dtype=get_default_complex_dtype())  # (n_slices, h, w)

# fracPy 0726 started flat; the 0728 variant pasted the prior object in. Uncomment to do that.
# if "object" in prior:
#     prior_obj = prior["object"].reshape((-1,) + prior["object"].shape[-2:])[0]
#     prior_obj = center_crop_or_pad(prior_obj, min(object_shape))
#     y0 = (object_shape[0] - prior_obj.shape[-2]) // 2
#     x0 = (object_shape[1] - prior_obj.shape[-1]) // 2
#     obj[0, y0 : y0 + prior_obj.shape[-2], x0 : x0 + prior_obj.shape[-1]] = torch.as_tensor(
#         prior_obj, dtype=get_default_complex_dtype()
#     )

print(f"object buffer: {tuple(obj.shape)}")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(np.abs(obj[0].numpy()), cmap="inferno")
axes[0].set_title("initial object magnitude")
axes[1].imshow(np.angle(obj[0].numpy()), cmap="inferno")
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

# --- probe ---
options.probe_options.optimizable = True
options.probe_options.optimizer = api.Optimizers.SGD
options.probe_options.step_size = 1.0
options.probe_options.optimization_plan = OptimizationPlan(start=probe_start)
options.probe_options.orthogonalize_incoherent_modes.enabled = n_probe_modes > 1
options.probe_options.orthogonalize_incoherent_modes.optimization_plan = OptimizationPlan(
    stride=orthogonalization_stride
)
options.probe_options.orthogonalize_incoherent_modes.method = api.OrthogonalizationMethods.SVD
options.probe_options.orthogonalize_opr_modes.enabled = n_opr_modes > 1
options.probe_options.power_constraint.enabled = False      # fracPy probePowerCorrectionSwitch
options.probe_options.center_constraint.enabled = False     # fracPy comStabilizationSwitch

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


#%% ---------------------------------------------------------------- run

# task.run() runs the full num_epochs. For debugging, call task.run(10) repeatedly
# instead: state persists between calls, so you can inspect after each chunk.
task.run()


#%% ---------------------------------------------------------------- inspect

recon_obj = task.get_data_to_cpu("object", as_numpy=True)[0]
recon_probe = task.get_data_to_cpu("probe", as_numpy=True)[0]
recon_pos = task.get_data_to_cpu("probe_positions", as_numpy=True)
loss_table = task.reconstructor.loss_tracker.table

fig, axes = plt.subplots(1, 3, figsize=(15, 7))
axes[0].imshow(np.angle(recon_obj[380:480, 480:580]), cmap="gray")
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

# Same layout as the beamline recon_Niter*.h5 files, so these results can be fed
# back in through init_recon_file.
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f"recon_Niter{num_epochs}.h5"

with h5py.File(out_file, "w") as f:
    f.create_dataset("object", data=task.get_data_to_cpu("object", as_numpy=True))
    f.create_dataset("probe", data=task.get_data_to_cpu("probe", as_numpy=True))
    f.create_dataset("positions_px", data=recon_pos)
    f.create_dataset("opr_mode_weights", data=task.get_data_to_cpu("opr_mode_weights", as_numpy=True))
    f.attrs["pixel_size_m"] = pixel_size_m
    f.attrs["wavelength_m"] = wavelength_m
    f.attrs["detector_distance_m"] = det_dist_m
    f.attrs["num_epochs"] = num_epochs

loss_table.to_csv(out_dir / "loss.csv", index=False)
print(f"saved {out_file}")

# %%
# reconstruct the RPI data