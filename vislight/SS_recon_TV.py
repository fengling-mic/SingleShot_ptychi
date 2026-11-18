#%%
# RPI (Randomized Probe Imaging) single-shot reconstruction of one diffraction pattern
# from a laser-lab (780 nm) raster scan, with TOTAL-VARIATION regularization. Same data,
# geometry and forward model as SS_recon_blrelax.py (Levitan et al., Opt. Express 28,
# 37103 (2020), Eq. 1), minus its band-limit relaxation schedule.
#
# Why TV: on these frames the plain RPI fit is under-constrained. Its loss ends at
# 0.2-0.3x the loss of the step-scan object (the closest thing to ground truth) in the
# same forward model -- 0x with no band limit at all -- so the extra freedom goes into
# fitting noise and model error, not the object. TV = mean |grad O| charges a lot for
# pixel-to-pixel wiggle and little for a few sharp edges, which suits a USAF target
# (flat regions, sharp bars). Objective: data loss + rpi_tv_weight * TV(O).
#
# Measured on 20261114_..._1000mmlens_structure_2ms, frames 33 / 18, probe fixed, 6000
# epochs, scored against the step-scan object at the same position (the "compare" cell):
#
#   object grid                    TV     FRC period      complex corr   loss / step-scan loss
#   N126 (R 0.8 at kp ~ 79 px)     0      350 / 825 um    0.58 / 0.47    0.32 / 0.27
#   N126                           0.1    114 / 117-175   0.72 / 0.61    0.63 / 0.55
#   N126                           0.3    113 / 115       0.77 / 0.64    0.86 / 0.74
#   N512 (no band limit)           0     1186 / 822       0.33 / 0.31    0.00 / 0.00
#   N512                           1      270 / 121       0.69 / 0.58    0.67 / 0.57
#   N512                           2      277 / 138       0.72 / 0.59    0.93 / 0.79
#   N512                           3-5    410-510         lower          > 1 (over-smoothed)
#
# N158 and N200 were no better than N126, and coarse-to-fine growth of the grid did not
# help. The remaining limit is the data: the signal falls below 1 ADU/px at r ~ 50 px and
# fewer than 10% of pixels are lit beyond r ~ 70-85 px (8-bit camera, 2 ms), about the
# probe's own kp.

from utils import use_inline_backend
use_inline_backend()             # Plot Viewer-compatible backend, in its own cell

#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # This makes GPU N appear as GPU 0 to CuPy

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from tqdm.auto import tqdm

from ptychi.utils import (
    add_additional_opr_probe_modes_to_probe,
    get_default_complex_dtype,
    orthogonalize_initial_probe,
)

from utils import (
    center_crop_or_pad,
    centered_extent,
    collect_rpi_params,
    crop_at_position,
    fourier_resample,
    fourier_ring_correlation,
    fourier_upsample_object,
    frc_crossing_px,
    frc_half_bit_threshold,
    make_disk_probe,
    make_rpi_recon_dir_name,
    probe_fourier_radius,
    rescale_probe_to_counts,
    rpi_diffraction_loss,
    save_initial_conditions,
    save_rpi_reconstruction,
    simulate_rpi_diffraction,
    total_variation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


#%% ---------------------------------------------------------------- paths

scan = "20261114_FoV3mm_500umstp_1000mmlens_structure_2ms"
scan_initialGuess = "20261114_FoV3mm_500umstp_1000mmlens_structure_2ms"  # scan used to generate the initial guess (positions + probe)
data_root = Path(r"\\micdata\data3\fengling\date_Laserlab")
# data_root = Path("/mnt/micdata3/fengling/date_Laserlab")

dp_file = data_root / "preproc" / scan / "data_roi0_Ndp1024_dp.h5"
para_file = data_root / "preproc" / scan / "data_roi0_Ndp1024_para.h5"
# Prior multi-position Pty-Chi reconstruction: source of the probe, of the positions, and
# of the reference object the "compare" cell scores against. Set to None to start from
# the para file positions and a synthesized probe instead (no comparison then).
init_recon_file = (
    data_root / "ptychi_recons" / scan_initialGuess
    / "Ndp512_LSQML_c50_m0.25_gaussian_p1_opr1_pc0_probe10" / "recon_Niter1000.h5"
)

out_dir = data_root / "ptychi_recons" / "singleframe_recons"
recon_dir_suffix = "_TV"         # appended to the folder name, e.g. "_v2" or "_pos"

frame_index = 33   # index into the loaded scan's patterns/positions to reconstruct

#%% ---------------------------------------------------------------- read the parameters from the file

# Same reads as ptychi_recon.py: Preprocess.py stores the camera geometry alongside the
# data ("det_pixel_size", "lambda"), since the 12-ID readers hardcode the Pilatus pixel.
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
print(f"lambda {wavelength_m * 1e9:.1f} nm, distance {detector_distance_m * 1e3:.1f} mm, "
      f"camera pixel {det_pixel_m * 1e6:.2f} um, {len(ppx)} positions")


#%% ---------------------------------------------------------------- geometry & knobs

# Detector crop. Must match the Pty-Chi reconstruction the probe comes from (Ndp512_...)
# for the probe to need no resampling; the central 512 px also hold >99.9% of the counts
# of the 1024 px preprocessed frames on 20261114_..._1000mmlens.
n_dp = 512
det_dist_m = detector_distance_m    # Fourier-lens focal length
far_field = True

pixel_size_m = wavelength_m * det_dist_m / (n_dp * det_pixel_m)

n_probe_modes = None             # None = keep every incoherent mode in the prior probe
n_opr_modes = 1                  # a single frame carries no probe-variation information
probe_diameter_m = 3e-3          # only used when no probe comes from init_recon_file

# --- band limit -------------------------------------------------------------
# R = ko/kp, measured against the probe's own Fourier support (NOT the detector); the
# object grid is 2 R kp px. None = no band limit: the object lives on the full n_dp grid
# and only TV regularizes it -- then use rpi_tv_weight ~ 1-2 (see the table above).
rpi_resolution_ratio = 0.8
rpi_kp_quantile = 0.99           # fraction of probe far-field power used to define kp

# --- TV regularization --------------------------------------------------------
# Weight of total_variation(obj_lowres) against the normalized data loss; 0 = off. With
# the band limit (N126) 0.1-0.3 worked best; without it, 1-2. Too large flattens real
# detail: watch the "loss / step-scan loss" ratio in the compare cell climb past 1.
rpi_tv_weight = 0.1

# --- optimizer --------------------------------------------------------------
rpi_optimizer_cls = torch.optim.Adam
num_epochs = 6000                 # max optimizer iterations per restart
rpi_lr = 0.005                    # Adam step on an object of magnitude ~1
rpi_lr_decay_factor = 0.5
rpi_lr_decay_patience = 100      # if the objective doesn't improve for this many epochs, decay lr*0.5
rpi_min_lr = 1e-10                # stop when lr decays below this
rpi_loss_floor = 1e-20            # stop when loss falls below this
rpi_num_restarts = 1
rpi_object_init = "unit"         # "unit" (1 + noise, a transmission object) or "random" (paper)
rpi_init_sigma = 0.1             # init noise std ("random" uses 1.0 per the paper)
rpi_background_counts = 0.0      # known detector background (Eq. 2's B_ij), if any

# --- saving -------------------------------------------------------------
save_freq_iterations = 2000       # write a recon_Niter*.h5 snapshot every N epochs (per restart)

# --- staged probe update ----------------------------------------------------
# rpi_probe_start = N freezes the probe for N iterations and then refines it, anchored
# to the calibration probe; None keeps it fixed throughout (the paper). Releasing it at
# 5000 on frame 18 drove the loss to 0.04x the step-scan object's -- pure overfitting.
rpi_probe_start = None
rpi_probe_lr_rel = 0.001         # probe lr = this * mean|P| (probe entries are ~1e-3)
rpi_probe_anchor = 3.0           # weight of ||P - P0||^2 / ||P0||^2 keeping P near calibration

rpi_illumination_threshold = 0.10   # fraction of peak illumination defining the usable FOV

# fracPy flip switches -- keep identical to ptychi_recon.py, whose probe is used here.
flip_dp_x = False
# fracPy's preprocess_780nm.py flips every frame (np.flipud); vislight/Preprocess.py does
# not, so flip here to match (see ptychi_recon.py).
flip_dp_y = True
flip_positions_x = False
flip_positions_y = False
swap_position_axes = False       # True if stored positions are (x, y) not (y, x)

random_seed = 123

print(f"pixel size = {pixel_size_m * 1e6:.3f} um, FOV = {n_dp * pixel_size_m * 1e3:.2f} mm")

#%% ---------------------------------------------------------------- load diffraction patterns

with h5py.File(dp_file, "r") as f:
    print(f"keys in {dp_file.name}: {list(f.keys())}")
    patterns = f["dp"][()]
    det_mask = np.asarray(f["det_pixel_mask"][()], dtype=bool)
print(f"raw ptychogram: {patterns.shape}")

# The mask is cropped and flipped together with the patterns, or it no longer lines up.
patterns = center_crop_or_pad(patterns, n_dp)
det_mask = center_crop_or_pad(det_mask, n_dp)
if flip_dp_x:
    patterns = patterns[..., :, ::-1]
    det_mask = det_mask[:, ::-1]
if flip_dp_y:
    patterns = patterns[..., ::-1, :]
    det_mask = det_mask[::-1, :]
patterns = np.ascontiguousarray(patterns, dtype=np.float32)
det_mask = np.ascontiguousarray(det_mask)
np.clip(patterns, 0, None, out=patterns)

print(f"ptychogram: {patterns.shape}, total counts {patterns.sum():.3e}")

# Same detector mask as ptychi_recon.py, so the single frame is fit on the pixels the
# probe was: the file mask, minus "dead" pixels (zero in every frame but surrounded by
# signal -- hot in the dark frames, wiped out by the dark subtraction) and "stuck" ones
# (the identical count in all frames). Both need the whole scan, so this runs before the
# single frame is picked out.
frame_sum = patterns.sum(axis=0)
dead_pixels = (frame_sum == 0) & (median_filter(frame_sum, size=5) > 0)
stuck_pixels = (patterns.std(axis=0) == 0) & (frame_sum > 0)
n_file_masked = int((~det_mask).sum())
det_mask = det_mask & ~(dead_pixels | stuck_pixels)
print(
    f"detector mask: {n_file_masked} from file + {dead_pixels.sum()} dead + "
    f"{stuck_pixels.sum()} stuck -> {100 * det_mask.mean():.2f}% valid, "
    f"{100 * frame_sum[det_mask].sum() / frame_sum.sum():.3f}% of counts kept"
)

assert 0 <= frame_index < len(patterns), (
    f"frame_index={frame_index} out of range for {len(patterns)} patterns"
)
patterns = patterns[frame_index : frame_index + 1]
print(f"reconstructing frame {frame_index}: pattern {patterns.shape}, total counts {patterns.sum():.3e}")


#%% ---------------------------------------------------------------- positions + prior probe

prior = {}
if init_recon_file is not None:
    with h5py.File(init_recon_file, "r") as f:
        print(f"keys in {init_recon_file.name}: {list(f.keys())}")
        prior["probe"] = np.asarray(f["probe"][()]).view(np.complex64)
        prior["positions_px"] = np.asarray(f["positions_px"][()], dtype=np.float64)
        if "object" in f:
            prior["object"] = np.asarray(f["object"][()]).view(np.complex64).squeeze(0)
            prior["obj_pixel_size_m"] = float(np.asarray(f["obj_pixel_size_m"][()]))

if "positions_px" in prior:
    # In pixels of the prior reconstruction's grid; the FOV is fixed, so the pixel size
    # scales as 1 / n_dp and the positions as n_dp.
    positions_px_all = prior["positions_px"] * (n_dp / prior["probe"].shape[-1])
else:
    # ppX/ppY (meters) from the para file; we want pixels.
    positions_px_all = np.stack((ppy, ppx), axis=-1) / pixel_size_m

if swap_position_axes:
    positions_px_all = positions_px_all[:, ::-1].copy()
if flip_positions_y:
    positions_px_all[:, 0] = -positions_px_all[:, 0]
if flip_positions_x:
    positions_px_all[:, 1] = -positions_px_all[:, 1]

assert frame_index < len(positions_px_all), (
    f"frame_index={frame_index} out of range for {len(positions_px_all)} positions"
)
position_px = positions_px_all[frame_index]   # the single scan position reconstructed here
print(f"position: y={position_px[0]:.1f}, x={position_px[1]:.1f} px")

# Plot the whole scan (centered on its own mean) and mark the reconstructed frame.
plot_positions = positions_px_all - positions_px_all.mean(axis=0, keepdims=True)
plt.figure(figsize=(4, 4))
plt.plot(plot_positions[:, 1], plot_positions[:, 0], ".-", lw=0.3, ms=2,
         color="0.6", label=f"all positions ({len(plot_positions)})")
sel = plot_positions[frame_index]
plt.plot(sel[1], sel[0], "o", ms=7, mfc="none", mec="red", mew=1.5,
         label=f"reconstructed frame {frame_index}")
plt.legend(fontsize=7, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("x [px]")
plt.ylabel("y [px]")
plt.title("scan positions [px]")
plt.show()


#%% ---------------------------------------------------------------- initial probe

if "probe" in prior:
    # The probe must be RESAMPLED onto the n_dp grid, not center-cropped: the array FOV
    # is fixed at wavelength*z/det_pixel regardless of n_dp (see fourier_resample).
    if prior["probe"].shape[-1] != n_dp:
        print(f"resampling probe {prior['probe'].shape[-1]} -> {n_dp} px over the fixed "
              f"{n_dp * pixel_size_m * 1e3:.2f} mm FOV")
    probe = fourier_resample(prior["probe"], n_dp)
    # (n_opr, n_modes, h, w); fold any extra leading axes (e.g. a wavelength axis)
    # into the OPR axis.
    probe = probe.reshape((-1,) + probe.shape[-3:])
    probe = torch.as_tensor(np.ascontiguousarray(probe), dtype=get_default_complex_dtype())
    print(f"probe from {init_recon_file.name}: {tuple(probe.shape)}")
else:
    probe = torch.as_tensor(
        make_disk_probe(n_dp, probe_diameter_m / pixel_size_m),
        dtype=get_default_complex_dtype(),
    )
    print(f"synthesized disk probe, diameter {probe_diameter_m * 1e3:.2f} mm")

# Incoherent modes: default to keeping all of them. Dropping modes throws away real
# incoherent power and the rescale below then inflates the survivors to compensate,
# which distorts the forward model.
if n_probe_modes is None:
    n_probe_modes = probe.shape[1]
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

mode_power = (probe[0].abs() ** 2).sum(dim=(-2, -1))
mode_power = (mode_power / mode_power.sum()).numpy()
print(f"probe: {tuple(probe.shape)} (n_opr, n_modes, h, w)")
print(f"  incoherent mode power fractions: {np.round(mode_power, 4)}")

fig, axes = plt.subplots(1, probe.shape[1], figsize=(2.2 * probe.shape[1], 2.4))
for i, ax in enumerate(np.atleast_1d(axes)):
    ax.imshow(np.abs(probe[0, i].numpy()), cmap="gray")
    ax.set_title(f"mode {i}\n{mode_power[i] * 100:.1f}%", fontsize=8)
    ax.set_xticks([]), ax.set_yticks([])
plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------- RPI setup

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

measured = torch.as_tensor(patterns[0], dtype=torch.float32, device=torch_device)
mask = torch.as_tensor(det_mask.astype(np.float32), device=torch_device)
probe_modes = rescale_probe_to_counts(probe[0].to(torch_device), measured, mask)

# Band limit, defined against the probe's numerical aperture (see SS_recon_blrelax.py).
assert n_dp % 2 == 0, "n_dp must be even so that n_dp - n_obj_lowres stays even"
kp = probe_fourier_radius(probe_modes, rpi_kp_quantile)
if rpi_resolution_ratio is None:
    n_obj_lowres = n_dp            # no band limit: the object lives on the detector grid
else:
    # Forced even, which keeps the zero-padding in fourier_upsample_object symmetric; an
    # odd total pad puts DC one bin off and corrupts the whole upsampled array.
    n_obj_lowres = 2 * int(round(rpi_resolution_ratio * kp))
    n_obj_lowres = int(np.clip(n_obj_lowres, 2, n_dp))
achieved_R = (n_obj_lowres / 2) / kp
object_pixel_size_m = pixel_size_m * n_dp / n_obj_lowres
# fourier_upsample_object is the identity at n_obj_lowres == n_dp; skip its two FFTs.
upsample = (lambda o, n: o) if n_obj_lowres == n_dp else fourier_upsample_object
print(
    f"RPI: kp = {kp} px (detector half-width {n_dp // 2}) -> object {n_obj_lowres}x{n_obj_lowres}"
    + (" (no band limit)" if rpi_resolution_ratio is None else f", R = {achieved_R:.2f}")
    + f", pixel size {object_pixel_size_m * 1e6:.1f} um, TV weight {rpi_tv_weight:g}, "
    f"device {torch_device}"
)
if achieved_R > 0.6 and not rpi_tv_weight:
    print(f"  WARNING: R = {achieved_R:.2f} exceeds the paper's reliable range (<= 0.6) "
          f"with no TV regularization")

# Output folder, in the same out_dir/scan/<tag>/ layout the Pty-Chi scripts use --
# see make_rpi_recon_dir_name for the tag (it adds tv<weight> when TV is on).
recon_dir = out_dir / scan / make_rpi_recon_dir_name(
    recon_dir_suffix, optimizer_name=rpi_optimizer_cls.__name__
)
save_initial_conditions(recon_dir, params=collect_rpi_params(), positions_px=positions_px_all,
                        params_filename="rpi_params.json")

# Illuminated field of view: object pixels outside it are not constrained by the data.
illumination = (probe_modes.abs() ** 2).sum(0)
illumination = (illumination / illumination.max()).cpu().numpy()
ill_mask = illumination > rpi_illumination_threshold
ill_rows = np.where(ill_mask.any(axis=1))[0]
ill_cols = np.where(ill_mask.any(axis=0))[0]
ill_slice = (slice(ill_rows[0], ill_rows[-1] + 1), slice(ill_cols[0], ill_cols[-1] + 1))
print(
    f"  illuminated FOV (> {rpi_illumination_threshold:g} of peak): "
    f"{ill_rows[-1] - ill_rows[0] + 1} x {ill_cols[-1] - ill_cols[0] + 1} px "
    f"= {(ill_rows[-1] - ill_rows[0] + 1) * pixel_size_m * 1e3:.2f} mm"
)

# Shot-noise floor of the loss (see SS_recon_blrelax.py). Indicative only: it takes
# 1 camera ADU = 1 photon, which the 8-bit camera does not guarantee.
noise_floor = float(
    (torch.clamp(2 * measured, max=0.25) * mask).sum() / (measured * mask).sum()
)
print(f"  shot-noise floor of the loss ~ {noise_floor:.5f} (assuming 1 ADU = 1 photon)")


#%% ---------------------------------------------------------------- band-limit geometry

# kp is a circular quantile of the probe's Fourier power; n_obj_lowres crops the object's
# spectrum to a SQUARE box (fourier_upsample_object), which reaches further out along
# the diagonals than a disk of radius ko = R * kp. With no band limit the box is the
# whole detector.
ko = n_obj_lowres / 2
probe_far = (torch.abs(torch.fft.fftshift(torch.fft.fft2(probe_modes), dim=(-2, -1))) ** 2) \
    .sum(0).cpu().numpy()
probe_log_floor = 1e-3
probe_far_n = np.maximum(probe_far / probe_far.max(), probe_log_floor)

n_half = n_dp / 2
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im_probe = ax.imshow(np.log10(probe_far_n), cmap="gray", origin="lower",
                     vmin=np.log10(probe_log_floor), vmax=0,
                     extent=[-n_half, n_half, -n_half, n_half])
fig.colorbar(im_probe, ax=ax, location="right", fraction=0.046, pad=0.04,
             label="probe log10(I / I_max)")
ax.add_patch(plt.Circle((0, 0), kp, fill=False, color="C0", lw=1.5,
                        label=f"kp = {kp} px  (circle, {rpi_kp_quantile:g} quantile)"))
ax.add_patch(plt.Rectangle((-kp, -kp), 2 * kp, 2 * kp, fill=False, color="C1", lw=1.5,
                           label=f"2kp x 2kp = {2 * kp} px  (circumscribing box)"))
ax.add_patch(plt.Rectangle((-ko, -ko), 2 * ko, 2 * ko, fill=False, color="C2", lw=2,
                           label=f"2ko x 2ko = {n_obj_lowres} px  (n_obj_lowres, R = {achieved_R:.2f})"))
lim = max(2 * kp, ko)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.axhline(0, color="0.85", lw=0.5, zorder=0)
ax.axvline(0, color="0.85", lw=0.5, zorder=0)
ax.set_xlabel("kx [detector px]")
ax.set_ylabel("ky [detector px]")
ax.set_title("RPI band-limit geometry in reciprocal space")
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
fig.savefig(recon_dir / "bandlimit_init.png", dpi=120, bbox_inches="tight")
plt.show()


#%% ---------------------------------------------------------------- run

probe_optimizable = rpi_probe_start is not None
probe_ref = probe_modes.clone()                    # calibration probe, the anchor
probe_ref_power = (probe_ref.abs() ** 2).sum()
probe_lr = rpi_probe_lr_rel * probe_ref.abs().mean().item()

best_loss, best_obj_lowres, best_probe, best_losses = np.inf, None, None, None
for restart in range(rpi_num_restarts):
    torch.manual_seed(random_seed + restart)
    noise = torch.randn(n_obj_lowres, n_obj_lowres, dtype=torch.float32, device=torch_device) \
        + 1j * torch.randn(n_obj_lowres, n_obj_lowres, dtype=torch.float32, device=torch_device)
    if rpi_object_init == "random":                # the paper's init (its objects were random)
        obj_lowres = rpi_init_sigma * noise
    else:                                          # a transmission object sits near 1
        obj_lowres = torch.ones_like(noise) + rpi_init_sigma * noise
    obj_lowres = obj_lowres.to(get_default_complex_dtype()).requires_grad_(True)

    probe_var = probe_ref.clone().requires_grad_(probe_optimizable)
    groups = [{"params": [obj_lowres], "lr": rpi_lr}]
    if probe_optimizable:
        groups.append({"params": [probe_var], "lr": probe_lr})
    optimizer = rpi_optimizer_cls(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=rpi_lr_decay_factor, patience=rpi_lr_decay_patience
    )

    losses = []
    bar = tqdm(range(num_epochs), desc=f"restart {restart + 1}/{rpi_num_restarts}", leave=True)
    for epoch in bar:
        probe_live = probe_optimizable and epoch >= rpi_probe_start
        probe_now = probe_var if probe_live else probe_ref

        optimizer.zero_grad()
        pred = simulate_rpi_diffraction(obj_lowres, probe_now, n_dp, rpi_background_counts,
                                        upsample=upsample)
        loss = rpi_diffraction_loss(pred, measured, mask)
        tv = total_variation(obj_lowres)
        total = loss + rpi_tv_weight * tv
        if probe_live and rpi_probe_anchor > 0:
            total = total + rpi_probe_anchor * ((probe_now - probe_ref).abs() ** 2).sum() / probe_ref_power
        if not torch.isfinite(total):
            # Checked before backward()/step(), so obj_lowres keeps its last finite value.
            bar.write(f"  epoch {epoch}: non-finite objective, stopping here")
            break
        total.backward()
        optimizer.step()
        if probe_live:
            # Remove the O <-> P scaling ambiguity opened by releasing the probe.
            with torch.no_grad():
                probe_var *= torch.sqrt(probe_ref_power / (probe_var.abs() ** 2).sum())

        # losses holds the data term alone, so runs with different TV weights stay
        # comparable; the scheduler follows the objective actually minimized.
        losses.append(loss.item())
        scheduler.step(total.item())
        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            bar.set_postfix(loss=f"{losses[-1]:.5f}", tv=f"{tv.item():.4f}",
                            floor=f"{noise_floor:.5f}", lr=f"{lr_now:.2e}",
                            probe="on" if probe_live else "off")

        if save_freq_iterations and (epoch + 1) % save_freq_iterations == 0:
            snap_dir = recon_dir / f"restart{restart}" if rpi_num_restarts > 1 else recon_dir
            with torch.no_grad():
                snap_fullres = upsample(obj_lowres, n_dp)
            save_rpi_reconstruction(obj_lowres, snap_fullres, probe_now, losses, snap_dir,
                                    epoch + 1, pixel_size_m=pixel_size_m,
                                    object_pixel_size_m=object_pixel_size_m)

        if losses[-1] < rpi_loss_floor or lr_now < rpi_min_lr:
            break
    bar.close()
    if not losses:
        raise RuntimeError("non-finite objective at the first epoch -- check the inputs")

    print(f"  restart {restart}: {len(losses)} iters, final loss {losses[-1]:.5f} "
          f"(noise floor {noise_floor:.5f})")
    if losses[-1] < best_loss:
        best_loss = losses[-1]
        best_obj_lowres = obj_lowres.detach().clone()
        best_probe = probe_var.detach().clone()
        best_losses = losses

obj_lowres = best_obj_lowres
recon_probe = best_probe
with torch.no_grad():
    obj_fullres = upsample(obj_lowres, n_dp)
obj_lowres_np = obj_lowres.cpu().numpy()
obj_fullres_np = obj_fullres.cpu().numpy()
recon_probe_np = recon_probe.cpu().numpy()

if probe_optimizable:
    dprobe = float((recon_probe - probe_ref).abs().sum() / probe_ref.abs().sum()) * 100
    print(f"  probe changed by {dprobe:.1f}% after release at iteration {rpi_probe_start}")


#%% ---------------------------------------------------------------- inspect

# Main result: object amplitude, object phase, probe -- full array (top) and cropped to
# the illuminated FOV (bottom), which is the only region the data constrains. Axes in mm,
# centred on the array's middle pixel (centered_extent); the crop keeps those coordinates.
ext_full = centered_extent(obj_fullres_np.shape, pixel_size_m, unit="mm")
_c = (np.arange(n_dp) - n_dp // 2) * pixel_size_m * 1e3
ext_ill = [_c[ill_slice[1]][0], _c[ill_slice[1]][-1], _c[ill_slice[0]][-1], _c[ill_slice[0]][0]]
fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))
panels = [
    (np.abs(obj_fullres_np), "object amplitude", "gray"),
    (np.angle(obj_fullres_np), "object phase", "gray"),
    (np.abs(recon_probe_np[0]), "probe amplitude (mode 0)", "gray"),
]
for col, (img, title, cmap) in enumerate(panels):
    axes[0, col].imshow(img, cmap=cmap, extent=ext_full)
    axes[0, col].set_title(f"{title}\nfull {n_dp}px array")
    axes[1, col].imshow(img[ill_slice], cmap=cmap, extent=ext_ill)
    axes[1, col].set_title(f"{title}\nilluminated FOV")
for ax in axes.ravel():
    ax.set_xlabel("x [mm]"), ax.set_ylabel("y [mm]")
plt.tight_layout()
plt.show()

# Diagnostics: the array actually optimized, and convergence of the data term.
ext_lowres = centered_extent(obj_lowres_np.shape, object_pixel_size_m, unit="mm")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(np.abs(obj_lowres_np), cmap="gray", extent=ext_lowres)
axes[0].set_title(f"optimized object amplitude\n{n_obj_lowres}px, {object_pixel_size_m * 1e6:.1f} um/px")
axes[1].imshow(np.angle(obj_lowres_np), cmap="gray", extent=ext_lowres)
axes[1].set_title("optimized object phase")
for ax in axes[:2]:
    ax.set_xlabel("x [mm]"), ax.set_ylabel("y [mm]")
axes[2].semilogy(best_losses, label="diffraction loss")
if probe_optimizable:
    axes[2].axvline(rpi_probe_start, color="0.5", ls=":", lw=1, label="probe released")
axes[2].set_xlabel("iteration"), axes[2].set_ylabel("loss")
axes[2].set_title(f"R = {achieved_R:.2f}, TV {rpi_tv_weight:g}, final {best_loss:.5f}")
axes[2].legend(fontsize=7)
plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------- compare with the step-scan reconstruction

# The step-scan object at this frame's position is the closest thing to ground truth, and
# scores the run objectively:
#   ref_loss      the step-scan object's own loss in this forward model. A final loss far
#                 BELOW it means the single frame is fitting noise and model error.
#   complex corr  |<O - mean, ref - mean>| / norms over the illuminated pixels: blind to
#                 global phase, scale and DC.
#   FRC period    full period where the amplitude FRC against the step-scan crop meets
#                 the half-bit threshold, over the illuminated bounding box.
compare_metrics = {}
if "object" in prior:
    assert np.isclose(prior["obj_pixel_size_m"], pixel_size_m, rtol=1e-3), (
        f"step-scan object pixel {prior['obj_pixel_size_m']:.4g} m != {pixel_size_m:.4g} m; "
        f"set n_dp to the step-scan's detector crop"
    )
    ref_obj, _ = crop_at_position(prior["object"], position_px, (n_dp, n_dp))

    with torch.no_grad():
        # Scale the step-scan exit wave as a whole to the measured counts, not the probe
        # alone: the step-scan object's mean transmission is well below 1.
        exit_ref = rescale_probe_to_counts(
            probe_ref * torch.as_tensor(ref_obj, device=torch_device), measured, mask
        )
        pred_ref = (torch.fft.fftshift(torch.fft.fft2(exit_ref), dim=(-2, -1)).abs() ** 2).sum(0)
        ref_loss = float(rpi_diffraction_loss(pred_ref + rpi_background_counts, measured, mask))

    a = obj_fullres_np[ill_mask] - obj_fullres_np[ill_mask].mean()
    b = ref_obj[ill_mask] - ref_obj[ill_mask].mean()
    complex_corr = float(abs(np.vdot(b, a)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real))

    crop_shape = obj_fullres_np[ill_slice].shape
    frc_freq, frc, frc_n = fourier_ring_correlation(
        np.abs(obj_fullres_np[ill_slice]), np.abs(ref_obj[ill_slice]), pixel_size_m
    )
    frc_thr = frc_half_bit_threshold(frc_n)
    r_px = frc_crossing_px(frc_freq, frc, frc_thr, crop_shape, pixel_size_m)
    frc_period_m = min(crop_shape) * pixel_size_m / r_px if r_px else np.nan

    compare_metrics = {
        "ref_loss": ref_loss,
        "loss_ratio": best_loss / ref_loss,
        "complex_corr": complex_corr,
        "frc_period_m": frc_period_m,
    }
    print(f"vs step-scan: loss {best_loss:.4f} / step-scan {ref_loss:.4f} = "
          f"{best_loss / ref_loss:.2f}, complex corr {complex_corr:.3f}, "
          f"FRC period {frc_period_m * 1e6:.0f} um")

    # Global phase aligned to the step-scan object, for display only.
    obj_disp = obj_fullres_np * np.exp(1j * np.angle(np.vdot(a, b)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (fn, name) in enumerate(((np.abs, "amplitude"), (np.angle, "phase"))):
        axes[row, 0].imshow(fn(obj_disp[ill_slice]), cmap="gray", extent=ext_ill)
        axes[row, 0].set_title(f"single frame {frame_index}, {name}")
        axes[row, 1].imshow(fn(ref_obj[ill_slice]), cmap="gray", extent=ext_ill)
        axes[row, 1].set_title(f"step-scan, same position, {name}")
        for ax in axes[row, :2]:
            ax.set_xlabel("x [mm]"), ax.set_ylabel("y [mm]")
    axes[0, 2].plot(frc_freq * 1e-3, frc, label="amplitude FRC")
    axes[0, 2].plot(frc_freq * 1e-3, frc_thr, color="0.4", ls="--", lw=1, label="half-bit")
    if r_px:
        axes[0, 2].axvline(1e-3 / frc_period_m, color="C0", ls=":", lw=1.5)
    axes[0, 2].set_xlabel("spatial frequency [1/mm]"), axes[0, 2].set_ylabel("FRC")
    axes[0, 2].set_ylim(-0.05, 1.05)
    axes[0, 2].set_title(f"FRC period {frc_period_m * 1e6:.0f} um, corr {complex_corr:.3f}\n"
                         f"loss / step-scan loss {best_loss / ref_loss:.2f}")
    axes[0, 2].legend(fontsize=8)
    axes[1, 2].axis("off")
    plt.tight_layout()
    fig.savefig(recon_dir / "compare_stepscan.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("no step-scan object in init_recon_file -- skipping the comparison")


#%% ---------------------------------------------------------------- save

save_rpi_reconstruction(
    obj_lowres, obj_fullres, recon_probe, best_losses, recon_dir,
    pixel_size_m=pixel_size_m, object_pixel_size_m=object_pixel_size_m,
    position_px=position_px[None, :], illumination=illumination.astype(np.float32),
    extra_attrs={
        "wavelength_m": wavelength_m,
        "detector_distance_m": det_dist_m,
        "rpi_resolution_ratio": achieved_R,
        "probe_fourier_radius_px": kp,
        "rpi_tv_weight": rpi_tv_weight,
        "noise_floor": noise_floor,
        "final_loss": best_loss,
        "frame_index": frame_index,
        **compare_metrics,
    },
)
print(f"results in {recon_dir}")

# %%
