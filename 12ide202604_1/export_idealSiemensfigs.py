from utils import center_crop_or_pad,show_ptychogram,make_randomized_zoneplate,siemens_star
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


n_dp = 1024
pixel_size_m = 8.8e-9
wavelength_m = 1.55e-10
probe_outer_zone_width_m = 5e-8
probe_focal_spot_diameter_m = 3.5e-6
central_stop = 0.15
seed = 0


out_dir = Path("/mnt/micdata3/fengling/figs")
out_dir.mkdir(parents=True, exist_ok=True)


#%% ------------------------------------------------------------- siemens star

n_spokes = 24
shape = (1024, 1024)

# False (repo default): siemens_star()'s spokes are a hard on/off square wave in angle.
# A square wave's Fourier series carries strong odd harmonics (3x, 5x, 7x, ... n_spokes),
# and each one adds its own finer ray-fan through the origin in the FFT -- their overlap is
# the dense crosshatch seen alongside the main 2*n_spokes rays.
# True: replace the hard threshold with a single-harmonic sinusoidal spoke weight, so the
# FFT shows just that one clean ray family with no crosshatch.
sinusoidal_spokes = False

if sinusoidal_spokes:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    theta = np.arctan2(yy - shape[0] / 2, xx - shape[1] / 2)
    r = np.hypot(yy - shape[0] / 2, xx - shape[1] / 2)
    spoke_weight = (0.5 + 0.5 * np.cos(n_spokes * theta)) * (r < 0.45 * min(shape))
    obj = ((1 - 0.15 * spoke_weight) * np.exp(1j * 0.8 * spoke_weight)).astype(np.complex64)
else:
    obj = siemens_star(shape, n_spokes=n_spokes)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
im0 = axes[0].imshow(np.angle(obj), cmap="gray")
axes[0].set_title("Siemens star — phase")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

im1 = axes[1].imshow(np.abs(obj), cmap="gray")
axes[1].set_title("Siemens star — amplitude")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

for ax in axes:
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal")

fig.tight_layout()
plt.show()

#%% ------------------------------------------------------------- siemens star fft

phase_img = np.angle(obj)
F = np.fft.fftshift(np.fft.fft2(phase_img))
fft_log_mag = np.log1p(np.abs(F))

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(-fft_log_mag, cmap="grey", origin="lower", 
               extent=[-n_dp / 2, n_dp / 2, -n_dp / 2, n_dp / 2], vmin=-8, vmax=-5)
ax.set_title("2D FFT (log magnitude) — Siemens star phase")
ax.set_xticks([]), ax.set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
fig.tight_layout()
plt.show()

#%%