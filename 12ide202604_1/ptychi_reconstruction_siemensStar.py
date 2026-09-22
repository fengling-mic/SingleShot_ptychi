#%%
# Pty-Chi (LSQML) reconstruction of the 12-ID-C Siemens star scan.
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

from utils import center_crop_or_pad,show_ptychogram,make_probe,hermite_secondary_modes

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
data_root = Path("/mnt/micdata2/12IDC/2026_Data/2026_3/01_piezo_test")

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
recon_dir_suffix = "_edgeK200_detMask_0mode_objinit"            # appended to the folder name, e.g. "_v2" or "_pos"

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
probe_diameter_m = 3.95e-6       # only used when no probe comes from init_recon_file, works as entrance pupil
probe_init_type = "disk"         # "disk" or "gaussian" (only used when no probe comes from init_recon_file)

# Speckle in the synthesized initial probe, set to resemble the probe in the
# init_recon_file recon above. Measured from probe[0, 0] of that file (OPR mode
# 0 is the real one -- modes 1-3 have ~1e-4 weight against 0.995 for mode 0):
#
#   intensity contrast 0.990   fully developed speckle
#   amplitude spread   0.520   Rayleigh, so the amplitude is speckled too,
#                              not just the phase
#   wrapped phase rms  1.81    = pi/sqrt(3), i.e. uniform
#   field grain        82.5 nm autocorrelation FWHM
#   beam size          3.95 um fitted soft disk
#
# probe_speckle_grain_m is the grain you actually see: the FWHM of the field
# autocorrelation. Here the diffuser modulates amplitude and does not wrap, so
# that is just the smoothing kernel's own correlation length -- unlike the
# phase-only case, where 2*pi wrapping makes the wavefront far finer than the
# screen behind it. make_probe solves for the kernel and prints both.
probe_speckle = 1.0               # fully developed: Rayleigh amplitude, uniform phase
probe_speckle_grain_m = 82.5e-9   # delivered grain (autocorrelation FWHM)
probe_speckle_phase_only = False  # the reference speckles amplitude, not just phase
probe_speckle_seed = 0            # change for a different realization

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

num_epochs = 500
save_freq_iterations = 5000       # write a recon_Niter*.h5 snapshot every N epochs
batch_size = 100                 # the number of scan positions
batching_mode = api.BatchingModes.COMPACT
noise_model = api.NoiseModels.POISSON
momentum_gain = 0.25              # fracPy params.momentumAcceleration

probe_start = 100                  # epoch at which the probe starts updating
opr_start = 10                   # None disables OPR weight optimization
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
    probe_init = make_probe(
        n_dp, probe_init_type, diameter=probe_diameter_m, pixel_size_m=pixel_size_m,
        speckle=probe_speckle, speckle_grain_m=probe_speckle_grain_m,
        speckle_phase_only=probe_speckle_phase_only, seed=probe_speckle_seed,
        verbose=True,
    )

    # make_probe builds a filled disc, so its far field is a filled blob peaking at
    # r=12 with 10.3% of the power inside r<20. The real beam is a zone-plate annulus
    # peaking at r=36 with 0.24% in the central stop's shadow -- 43x less. Left alone
    # the reconstruction has to dig all that out of the stop region first.
    #
    # Fix it with Gerchberg-Saxton against the measured illumination: a low percentile
    # across positions keeps what every frame has in common (the beam) and suppresses
    # the position-dependent object scatter. Alternate imposing that amplitude in the
    # far field with make_probe's own real-space envelope, so the probe stays the one
    # you built -- only its far-field amplitude is corrected, from your own data.
    # Measured result: peak radius 12 -> 36 and 0.1028 -> 0.0024 inside r<20, both
    # matching the data exactly. Scale is irrelevant here, rescale_probe follows.
    #
    # Note this fixes the far-field AMPLITUDE only. The far-field PHASE stays
    # arbitrary -- GS converges to some field consistent with both amplitudes, not to
    # your beam's actual speckle phase, which only a reconstruction recovers.
    gs_iterations = 200
    _ff_amp = np.sqrt(np.fft.ifftshift(np.percentile(patterns, 10, axis=0)))
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

    probe = torch.as_tensor(probe_init,dtype=get_default_complex_dtype(),)
    print(f"synthesized {probe_init_type} probe, diameter {probe_diameter_m * 1e6:.2f} um, "
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
        padded, secondary_mode_energy=0, rotation_deg=probe_mode_rotation_deg
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
# obj = torch.ones((1, *object_shape), dtype=get_default_complex_dtype())  # (n_slices, h, w)
# obj = torch.full((1, *object_shape), 1j, dtype=get_default_complex_dtype())

# Seed the object from the patterns instead of starting flat. The data give
# |FFT(P.O)| but not its phase; under the weak-object approximation the exit wave's
# far field is dominated by the probe's, so seed the missing phase with the probe's
# own and invert:
#
#     psi_j = IFFT( sqrt(I_j) . exp(i . angle(FFT(P))) )
#     O_j   = psi_j . conj(P) / (|P|^2 + eps)
#
# The divide has to be Wiener-regularised because P is zero over most of the array,
# and the per-position patches are stitched with the same |P|^2 weight the
# reconstructor uses as its preconditioner.
#
# The seed is only as good as the probe it is given. Scored against the converged
# reference object it reaches corr 0.75 when fed that converged probe, but 0.02 when
# fed the synthesized speckle probe, whose far-field phase is unrelated to the real
# beam -- so this earns its keep when init_recon_file supplies a real probe, and
# does nothing for a synthesized one. Uncomment a line above to go back to flat.
def seed_object(patterns, probe_2d, positions_px, object_shape, valid_pixel_mask):
    """Weak-object estimate of the object from the patterns and a known probe."""
    P = np.asarray(probe_2d, dtype=np.complex64)
    n = P.shape[-1]
    p_hat = np.fft.fft2(P)
    seed_phase = np.exp(1j * np.angle(p_hat))
    # At dead detector pixels there is no measurement; the model's own amplitude is
    # the best stand-in and keeps the module gaps out of the seeded object.
    amp_model = np.abs(p_hat)
    keep = np.fft.ifftshift(valid_pixel_mask)

    p_conj = np.conj(P)
    p_sq = np.abs(P) ** 2
    eps = 1e-3 * p_sq.max()
    cy, cx = object_shape[0] // 2, object_shape[1] // 2

    num = np.zeros(object_shape, dtype=np.complex64)
    den = np.zeros(object_shape, dtype=np.float32)
    for (pos_y, pos_x), pattern in zip(positions_px, patterns):
        amp = np.where(keep, np.sqrt(np.fft.ifftshift(pattern)), amp_model)
        psi = np.fft.ifft2(amp * seed_phase)
        r0 = int(round(cy + pos_y - n / 2))
        c0 = int(round(cx + pos_x - n / 2))
        num[r0 : r0 + n, c0 : c0 + n] += psi * p_conj
        den[r0 : r0 + n, c0 : c0 + n] += p_sq

    est = num / (den + eps)
    lit = den > 0.01 * den.max()
    est = est / np.median(np.abs(est[lit]))   # mean transmission ~ 1, as for ones
    return np.where(lit, est, 1.0).astype(np.complex64)   # unlit buffer stays clear


obj = torch.as_tensor(
    seed_object(patterns, probe[0, 0].numpy(), positions_px, object_shape,
                valid_pixel_mask)[None],
    dtype=get_default_complex_dtype(),
)  # (n_slices, h, w)

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
options.probe_options.fourier_support_constraint.radius_px = 200

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
