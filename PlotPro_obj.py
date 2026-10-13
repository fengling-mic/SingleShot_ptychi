
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # This makes GPU N appear as GPU 0 to CuPy

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from tqdm.auto import tqdm

from ptychi.utils import (
    add_additional_opr_probe_modes_to_probe,
    get_default_complex_dtype,
    get_suggested_object_size,
    orthogonalize_initial_probe,
)

from utils import (
    center_crop_or_pad,
    collect_rpi_params,
    fourier_resample,
    fourier_upsample_object,
    make_disk_probe,
    make_rpi_recon_dir_name,
    probe_fourier_radius,
    rescale_probe_to_counts,
    rpi_diffraction_loss,
    save_initial_conditions,
    save_rpi_reconstruction,
    siemens_star,
    simulate_rpi_diffraction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


#%% ---------------------------------------------------------------- paths

scan = "S0019"
scan_initialGuess = "S0019"  # scan used to generate the initial guess (positions + probe)
data_root = Path("/mnt/micdata3/fengling/2026_03_results")

# Prior multi-position Pty-Chi reconstruction used as the source of the probe and of
# calibration positions. Set to None to start from the para file positions and a
# synthesized probe instead.
init_recon_file = (
    data_root / "singleframe_recons" / scan_initialGuess
    / "frame62_RPI_Adam_Nobj72_R0.4_relax100_0-1000_p10_opr1_prel100_blrelax" / "recon_Niter2000.h5"
)

n_dp = 1024

with h5py.File(init_recon_file, "r") as f:
    probe_recon = np.asarray(f["probe"][()]).view(np.complex64)
    pixel_size_m = float(f["obj_pixel_size_m"][()])
probe = center_crop_or_pad(probe_recon, n_dp)

#%%
# probe_det = np.fft.fftshift(np.fft.fft2(probe[0,:,:]))


def complex_to_rgb(field, vmax=None, gamma=0.6):
    """Domain coloring: hue = phase (-pi..pi), brightness = |field|.

    gamma < 1 compresses the amplitude range so faint structure stays visible
    next to a bright peak (use gamma=1 for linear amplitude, gamma=2 for
    brightness proportional to intensity |field|**2).
    """
    if vmax is None:
        vmax = np.abs(field).max()
    amp = np.clip(np.abs(field) / (vmax + 1e-30), 0, 1) ** gamma
    hue = (np.angle(field) + np.pi) / (2 * np.pi)
    return hsv_to_rgb(np.stack([hue, np.ones_like(amp), amp], axis=-1))


def add_scalebar(ax, n_px, pixel_size_m, bar_um=2.0, color="white", pad_frac=0.06):
    """Draw a horizontal scale bar in the bottom-right corner of an image axis
    shown in raw pixel coordinates (no extent, origin="upper")."""
    bar_px = bar_um * 1e-6 / pixel_size_m
    x1 = n_px * (1 - pad_frac)
    x0 = x1 - bar_px
    y = n_px * (1 - pad_frac)
    ax.plot([x0, x1], [y, y], color=color, lw=3, solid_capstyle="butt")
    # ax.text((x0 + x1) / 2, y - n_px * 0.015, f"{bar_um:g} um", color=color,
    #         ha="center", va="bottom", fontsize=9)


fig = plt.figure(figsize=(7, 5.5))
gs = fig.add_gridspec(1, 2, width_ratios=[20, 5])
ax_img = fig.add_subplot(gs[0, 0])
ax_wheel = fig.add_subplot(gs[0, 1], projection="polar")

ax_img.imshow(complex_to_rgb(probe[0,:,:]))
ax_img.set_title("probe (hue = phase, brightness = |E|)")
ax_img.set_axis_off()
add_scalebar(ax_img, probe.shape[-1], pixel_size_m, bar_um=2.0)

theta = np.linspace(-np.pi, np.pi, 256)
r = np.array([0.0, 1.0])
Theta, R = np.meshgrid(theta, r)
hue = (Theta + np.pi) / (2 * np.pi)
ax_wheel.pcolormesh(Theta, R, hue, cmap="hsv", vmin=0, vmax=1, shading="gouraud")
ax_wheel.set_yticks([])
ax_wheel.set_xticks([0, np.pi])
ax_wheel.set_xticklabels(["0", "pi"], fontsize=8)
ax_wheel.set_title("phase", fontsize=9)
plt.tight_layout()

#%%


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


# frame_index = 62   # index into the loaded scan's patterns/positions to reconstruct
# ko = n_obj_lowres / 2
# probe_far = (torch.abs(torch.fft.fftshift(torch.fft.fft2(probe_modes), dim=(-2, -1))) ** 2) \
#     .sum(0).cpu().numpy()
# obj_far = np.abs(np.fft.fftshift(np.fft.fft2(obj_lowres_init_np))) ** 2

# # Independent floors (in normalized-intensity units, i.e. decades below the peak): the
# # object's initial spectrum is nearly all DC + a flat noise floor (no real structure yet),
# # so it needs a much shallower floor than the probe to show that floor as texture instead
# # of crushing the whole square to one inferno color.
# probe_log_floor = 1e-3
# obj_log_floor = 1e-1
# probe_far_n = np.maximum(probe_far / probe_far.max(), probe_log_floor)
# obj_far_n = np.maximum(obj_far / obj_far.max(), obj_log_floor)

# n_half = n_dp / 2
# fig, ax = plt.subplots(figsize=(6.5, 5.5))
# im_probe = ax.imshow(np.log10(probe_far_n), cmap="gray", origin="lower",
#                      vmin=np.log10(probe_log_floor), vmax=0,
#                      extent=[-n_half, n_half, -n_half, n_half])
# # im_obj = ax.imshow(np.log10(obj_far_n), cmap="inferno", origin="lower", alpha=0.8,
# #                    vmin=np.log10(obj_log_floor), vmax=0, extent=[-ko, ko, -ko, ko])
# fig.colorbar(im_probe, ax=ax, location="right", fraction=0.046, pad=0.04,
#             label="probe log10(I / I_max)")
# # cbar_obj = fig.colorbar(im_obj, ax=ax, location="right", fraction=0.046, pad=0.15,
# #                         label="object log10(I / I_max)")
# # cbar_obj.ax.yaxis.set_ticks_position("left")
# # cbar_obj.ax.yaxis.set_label_position("left")
# ax.add_patch(plt.Circle((0, 0), kp, fill=False, color="C0", lw=1.5,
#                         label=f"kp = {kp} px  (circle, {rpi_kp_quantile:g} quantile)"))
# ax.add_patch(plt.Rectangle((-kp, -kp), 2 * kp, 2 * kp, fill=False, color="C1", lw=1.5,
#                            label=f"2kp x 2kp = {2 * kp} px  (circumscribing box)"))
# ax.add_patch(plt.Rectangle((-ko, -ko), 2 * ko, 2 * ko, fill=False, color="C2", lw=2,
#                            label=f"2ko x 2ko = {n_obj_lowres} px  (n_obj_lowres, R = {achieved_R:.2f})"))
# lim = 2 * kp
# ax.set_xlim(-lim, lim)
# ax.set_ylim(-lim, lim)
# ax.set_aspect("equal")
# ax.axhline(0, color="0.85", lw=0.5, zorder=0)
# ax.axvline(0, color="0.85", lw=0.5, zorder=0)
# ax.set_xlabel("kx [detector px]")
# ax.set_ylabel("ky [detector px]")
# ax.set_title("RPI band-limit geometry in reciprocal space")
# ax.legend(fontsize=8, loc="upper right")
# plt.tight_layout()
# fig.savefig(recon_dir / "bandlimit_init.png", dpi=120, bbox_inches="tight")
# plt.show()