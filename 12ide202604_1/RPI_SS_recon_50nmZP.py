#%%
#%%
# RPI (Randomized Probe Imaging) single-shot reconstruction of one diffraction pattern
# from the 12-ID-C Siemens star scan, following Levitan et al., "Single-frame far-field
# diffractive imaging with randomized illumination," Opt. Express 28, 37103 (2020).
#
# Forward model (their Eq. 1):  E~ = F{ P . F^-1{ pad(F{O'}) } }
# The object O' lives on a grid COARSER than the probe/detector (band-limited), and is
# upsampled to full resolution by zero-padding its own Fourier transform. It is optimized
# directly with PyTorch autodiff + Adam against the single measured pattern; there is no
# Pty-Chi Task/task.run() here because Pty-Chi's forward model always keeps object and
# probe on the same pixel grid and cannot express the band limit. The probe comes from a
# prior multi-position Pty-Chi reconstruction (init_recon_file) and is held fixed, as in
# the paper -- see rpi_probe_start to release it.
#
# The size of the low-res object is set by the PROBE's numerical aperture, not by the
# detector: the paper's resolution ratio is R = ko/kp where kp is the probe's maximum
# spatial frequency (utils.probe_fourier_radius). Sizing it against the detector's
# Nyquist frequency instead silently runs at R ~ 1.5 here, far past the paper's
# reliability limit of ~0.6, and the reconstruction then just fits shot noise.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # This makes GPU N appear as GPU 0 to CuPy

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ptychi.utils import (
    add_additional_opr_probe_modes_to_probe,
    get_default_complex_dtype,
    get_suggested_object_size,
    orthogonalize_initial_probe,
)

from utils import (
    center_crop_or_pad,
    fourier_resample,
    fourier_upsample_object,
    make_disk_probe,
    probe_fourier_radius,
    rescale_probe_to_counts,
    rpi_diffraction_loss,
    siemens_star,
    simulate_rpi_diffraction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


#%% ---------------------------------------------------------------- paths

scan = "S0019"
scan_initialGuess = "S0019"  # scan used to generate the initial guess (positions + probe)
data_root = Path("/mnt/micdata2/12IDC/2026_Data/2026_3/01_piezo_test")

dp_file = data_root / "preproc" / scan / "data_roi0_Ndp1024_dp.hdf5"
para_file = data_root / "preproc" / scan / "data_roi0_Ndp1024_para.hdf5"

# Prior multi-position Pty-Chi reconstruction used as the source of the probe and of
# calibration positions. Set to None to start from the para file positions and a
# synthesized probe instead.
init_recon_file = (
    data_root / "ptychi_recons" / scan_initialGuess
    / "Ndp1024_LSQML_c20_m0.5_gaussian_p10_cp_mm_opr3_ic_pc1_f_ul2" / "recon_Niter400.h5"
)

out_dir = Path("/mnt/micdata3/fengling/2026_03_results") / "singleframe_recons"
# out_dir = Path(r"\\micdata\data3\fengling\2026_03_results") / "ptychi_recons"
recon_dir_suffix = ""            # appended to the folder name, e.g. "_v2" or "_pos"

frame_index = 62   # index into the loaded scan's patterns/positions to reconstruct

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

pixel_size_m = wavelength_m * det_dist_m / (n_dp * det_pixel_m)

n_probe_modes = 5             # None = keep every incoherent mode in the prior probe
n_opr_modes = 1                  # a single frame carries no probe-variation information
probe_diameter_m = 3.5e-6        # only used when no probe comes from init_recon_file

# --- band limit -------------------------------------------------------------
# R = ko/kp, measured against the probe's own Fourier support (NOT the detector).
# The paper: R < 0.4 "virtually guaranteed to succeed", R <= 0.6 workable, R ~ 0.94
# is the theoretical limit. Here kp ~ 87 px, so R = 0.5 gives an 88 px object at
# ~102 nm pitch -- the same regime as the paper's own X-ray demo (70 px, 83 nm).
# The 30 nm Siemens star features are below this limit and cannot be recovered.
rpi_resolution_ratio = 0.5
rpi_kp_quantile = 0.95           # fraction of probe far-field power used to define kp

# --- optimizer --------------------------------------------------------------
num_epochs = 3000                # max Adam iterations per restart
rpi_lr = 0.01                    # Adam step on an object of magnitude ~1
rpi_lr_decay_factor = 0.5
rpi_lr_decay_patience = 100      # iters without improvement before cutting lr
rpi_min_lr = 1e-5                # stop when lr decays below this
rpi_loss_floor = 1e-9            # stop when loss falls below this
rpi_num_restarts = 1             # measured: the solution is init-independent here
rpi_object_init = "unit"         # "unit" (1 + noise, a transmission object) or "random" (paper)
rpi_init_sigma = 0.1             # init noise std ("random" uses 1.0 per the paper)
rpi_background_counts = 0.0      # known detector background (Eq. 2's B_ij), if any

# --- staged probe update ----------------------------------------------------
# Set rpi_probe_start = 200 to freeze the probe for 200 iterations and then refine it.
# Measured on this dataset (frame 46), against the 81-position ptychography recon:
#     probe fixed          data loss 0.0097   eps 0.092   probe change  0.0%
#     release@200 anchor 0 data loss 0.0000   eps 0.092   probe change 18.5%
#     release@200 anchor 3 data loss 0.0046   eps 0.092   probe change  4.1%
# The Poisson noise floor is 0.0124, so every released run fits noise, and none improves
# the object. Default off. Useful only if the probe may have drifted between the
# calibration scan and the shot -- it has not here.
rpi_probe_start = 100
rpi_probe_lr_rel = 0.003         # probe lr = this * mean|P| (probe entries are ~1e-3)
rpi_probe_anchor = 3.0           # weight of ||P - P0||^2 / ||P0||^2 keeping P near calibration

rpi_illumination_threshold = 0.10   # fraction of peak illumination defining the usable FOV

# fracPy flip switches
flip_dp_x = False
flip_dp_y = False
flip_positions_x = False
flip_positions_y = False
swap_position_axes = False       # True if stored positions are (x, y) not (y, x)

random_seed = 123

print(f"pixel size = {pixel_size_m * 1e9:.3f} nm, FOV = {n_dp * pixel_size_m * 1e6:.2f} um")

#%% ---------------------------------------------------------------- load diffraction patterns

with h5py.File(dp_file, "r") as f:
    print(f"keys in {dp_file.name}: {list(f.keys())}")
    patterns = f["dp"][()]
    # Dead/hot detector pixels. Fitting them as if they were real counts biases the
    # solution badly (ground-truth loss 0.036 -> 0.020 once masked).
    det_mask = np.asarray(f["det_pixel_mask"][()], dtype=bool)
print(f"raw ptychogram: {patterns.shape}")

patterns = center_crop_or_pad(patterns, n_dp)
if flip_dp_x:
    patterns = patterns[..., :, ::-1]
if flip_dp_y:
    patterns = patterns[..., ::-1, :]
patterns = np.ascontiguousarray(patterns, dtype=np.float32)
np.clip(patterns, 0, None, out=patterns)

print(f"ptychogram: {patterns.shape}, total counts {patterns.sum():.3e}")

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

if "positions_px" in prior:
    positions_px_all = prior["positions_px"].copy()
else:
    # fracPy read ppX/ppY (meters) from the para file; we want pixels.
    with h5py.File(para_file, "r") as f:
        print(f"keys in {para_file.name}: {list(f.keys())}")
        ppx = np.asarray(f["ppX"][()], dtype=np.float64).squeeze()
        ppy = np.asarray(f["ppY"][()], dtype=np.float64).squeeze()
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
              f"{n_dp * pixel_size_m * 1e6:.2f} um FOV")
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
    print(f"synthesized disk probe, diameter {probe_diameter_m * 1e6:.2f} um")

# Incoherent modes: default to keeping all of them. Dropping modes throws away real
# incoherent power (the tail modes here carry ~27% of it) and the rescale below then
# inflates the survivors to compensate, which distorts the forward model.
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

# Band limit, defined against the probe's numerical aperture (see module docstring).
assert n_dp % 2 == 0, "n_dp must be even so that n_dp - n_obj_lowres stays even"
kp = probe_fourier_radius(probe_modes, rpi_kp_quantile)
# n_obj_lowres is forced even, which keeps the zero-padding in fourier_upsample_object
# symmetric; an odd total pad puts DC one bin off and corrupts the whole upsampled array.
n_obj_lowres = 2 * int(round(rpi_resolution_ratio * kp))
n_obj_lowres = int(np.clip(n_obj_lowres, 2, n_dp))
achieved_R = (n_obj_lowres / 2) / kp
object_pixel_size_m = pixel_size_m * n_dp / n_obj_lowres
print(
    f"RPI: kp = {kp} px (detector half-width {n_dp // 2}) -> object {n_obj_lowres}x{n_obj_lowres}, "
    f"R = {achieved_R:.2f}, pixel size {object_pixel_size_m * 1e9:.1f} nm, device {torch_device}"
)
if achieved_R > 0.6:
    print(f"  WARNING: R = {achieved_R:.2f} exceeds the paper's reliable range (<= 0.6)")

# Illuminated field of view: object pixels outside it are not constrained by the data.
illumination = (probe_modes.abs() ** 2).sum(0)
illumination = (illumination / illumination.max()).cpu().numpy()
ill_rows = np.where(illumination.max(axis=1) > rpi_illumination_threshold)[0]
ill_cols = np.where(illumination.max(axis=0) > rpi_illumination_threshold)[0]
ill_slice = (slice(ill_rows[0], ill_rows[-1] + 1), slice(ill_cols[0], ill_cols[-1] + 1))
print(
    f"  illuminated FOV (> {rpi_illumination_threshold:g} of peak): "
    f"{ill_rows[-1] - ill_rows[0] + 1} x {ill_cols[-1] - ill_cols[0] + 1} px "
    f"= {(ill_rows[-1] - ill_rows[0] + 1) * pixel_size_m * 1e6:.2f} um"
)

# Poisson shot-noise floor of the loss: Var(sqrt(I)) ~ 1/4 per valid pixel. A converged
# reconstruction should land NEAR this, not far below it -- below means it is fitting noise.
noise_floor = float((mask.sum() / 4) / (measured * mask).sum())
print(f"  Poisson noise floor of the loss ~ {noise_floor:.5f}")


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
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=rpi_lr_decay_factor, patience=rpi_lr_decay_patience
    )

    losses = []
    bar = tqdm(range(num_epochs), desc=f"restart {restart + 1}/{rpi_num_restarts}", leave=True)
    for epoch in bar:
        probe_live = probe_optimizable and epoch >= rpi_probe_start
        probe_now = probe_var if probe_live else probe_ref

        optimizer.zero_grad()
        pred = simulate_rpi_diffraction(obj_lowres, probe_now, n_dp, rpi_background_counts)
        loss = rpi_diffraction_loss(pred, measured, mask)
        total = loss
        if probe_live and rpi_probe_anchor > 0:
            total = total + rpi_probe_anchor * ((probe_now - probe_ref).abs() ** 2).sum() / probe_ref_power
        total.backward()
        optimizer.step()
        if probe_live:
            # Remove the O <-> P scaling ambiguity opened by releasing the probe.
            with torch.no_grad():
                probe_var *= torch.sqrt(probe_ref_power / (probe_var.abs() ** 2).sum())

        losses.append(loss.item())
        scheduler.step(losses[-1])
        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            bar.set_postfix(loss=f"{losses[-1]:.5f}", floor=f"{noise_floor:.5f}",
                            lr=f"{lr_now:.2e}", probe="on" if probe_live else "off")
        if losses[-1] < rpi_loss_floor or lr_now < rpi_min_lr:
            break
    bar.close()

    print(f"  restart {restart}: {len(losses)} iters, final loss {losses[-1]:.5f} "
          f"(noise floor {noise_floor:.5f})")
    if losses[-1] < best_loss:
        best_loss = losses[-1]
        best_obj_lowres = obj_lowres.detach().clone()
        best_probe = probe_var.detach().clone()
        best_losses = losses

obj_lowres = best_obj_lowres
recon_probe = best_probe
obj_fullres = fourier_upsample_object(obj_lowres, n_dp).detach()
obj_lowres_np = obj_lowres.cpu().numpy()
obj_fullres_np = obj_fullres.cpu().numpy()
recon_probe_np = recon_probe.cpu().numpy()

if best_loss < 0.7 * noise_floor:
    print(f"  NOTE: final loss {best_loss:.5f} is well below the Poisson floor "
          f"{noise_floor:.5f} -- the fit is absorbing shot noise. Lower rpi_resolution_ratio.")
if probe_optimizable:
    dprobe = float((recon_probe - probe_ref).abs().sum() / probe_ref.abs().sum()) * 100
    print(f"  probe changed by {dprobe:.1f}% after release at iteration {rpi_probe_start}")


#%% ---------------------------------------------------------------- inspect

# Main result: object amplitude, object phase, probe -- full array (top) and cropped to
# the illuminated FOV (bottom), which is the only region the data constrains.
fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))
panels = [
    (np.abs(obj_fullres_np), "object amplitude", "gray"),
    (np.angle(obj_fullres_np), "object phase", "gray"),
    (np.abs(recon_probe_np[0]), "probe amplitude (mode 0)", "gray"),
]
for col, (img, title, cmap) in enumerate(panels):
    axes[0, col].imshow(img, cmap=cmap)
    axes[0, col].set_title(f"{title}\nfull {n_dp}px array")
    axes[1, col].imshow(img[ill_slice], cmap=cmap)
    axes[1, col].set_title(f"{title}\nilluminated FOV")
for ax in axes.ravel():
    ax.set_xticks([]), ax.set_yticks([])
plt.tight_layout()
plt.show()

# Diagnostics: the array actually optimized, and convergence against the noise floor.
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(np.abs(obj_lowres_np), cmap="gray")
axes[0].set_title(f"low-res object amplitude\n{n_obj_lowres}px, {object_pixel_size_m * 1e9:.0f} nm/px")
axes[1].imshow(np.angle(obj_lowres_np), cmap="gray")
axes[1].set_title("low-res object phase")
for ax in axes[:2]:
    ax.set_xticks([]), ax.set_yticks([])
axes[2].semilogy(best_losses, label="diffraction loss")
axes[2].axhline(noise_floor, color="red", ls="--", lw=1, label=f"Poisson floor {noise_floor:.4f}")
if probe_optimizable:
    axes[2].axvline(rpi_probe_start, color="0.5", ls=":", lw=1, label="probe released")
axes[2].set_xlabel("iteration"), axes[2].set_ylabel("loss")
axes[2].set_title(f"R = {achieved_R:.2f}, final {best_loss:.5f}")
axes[2].legend(fontsize=7)
plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------- save

out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f"recon_frame{frame_index}_RPI_R{achieved_R:.2f}_{n_obj_lowres}px.h5"

with h5py.File(out_file, "w") as f:
    f.create_dataset("object_lowres", data=obj_lowres_np)
    # Full-resolution (Fourier-upsampled) object, kept under this name for
    # compatibility with init_recon_file's prior["object"] loader.
    f.create_dataset("object", data=obj_fullres_np)
    f.create_dataset("probe", data=recon_probe_np)
    f.create_dataset("illumination", data=illumination.astype(np.float32))
    f.create_dataset("positions_px", data=position_px[None, :])
    f.attrs["pixel_size_m"] = pixel_size_m
    f.attrs["object_pixel_size_m"] = object_pixel_size_m
    f.attrs["wavelength_m"] = wavelength_m
    f.attrs["detector_distance_m"] = det_dist_m
    f.attrs["rpi_resolution_ratio"] = achieved_R
    f.attrs["probe_fourier_radius_px"] = kp
    f.attrs["noise_floor"] = noise_floor
    f.attrs["final_loss"] = best_loss
    f.attrs["frame_index"] = frame_index

np.savetxt(out_dir / "rpi_loss.csv", best_losses, delimiter=",", header="loss", comments="")
print(f"saved {out_file}")
