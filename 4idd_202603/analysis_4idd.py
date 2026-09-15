#%%
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


recon_fly520_iter1000 = Path(
    "/mnt/micdata2/4IDD/2026_Sep/ptychi_recons/fly520"
    "/Ndp256_LSQML_c250_m0.25_gaussian_p10_mm_opr1_pc1_f_ul20/recon_Niter1000_deramped.h5"
)

recon_fly520_iter1500 = Path(
    "/mnt/micdata2/4IDD/2026_Sep/ptychi_recons/fly520"
    "/Ndp256_LSQML_c250_m0.25_gaussian_p10_mm_opr1_pc1_f_ul20/recon_Niter1500_deramped.h5"
)

recon_fly520_iter2000 = Path(
    "/mnt/micdata2/4IDD/2026_Sep/ptychi_recons/fly520"
    "/Ndp256_LSQML_c250_m0.25_gaussian_p10_mm_opr1_pc1_f_ul20/recon_Niter2000_deramped.h5"
)

recon_fly521_iter1000 = Path(
    "/mnt/micdata2/4IDD/2026_Sep/ptychi_recons/fly521"
    "/Ndp256_LSQML_c250_m0.25_gaussian_p10_mm_opr1_pc1_f_ul20/recon_Niter1000_deramped.h5"
)

recon_fly521_iter1500 = Path(
    "/mnt/micdata2/4IDD/2026_Sep/ptychi_recons/fly521"
    "/Ndp256_LSQML_c250_m0.25_gaussian_p10_mm_opr1_pc1_f_ul20/recon_Niter1500_deramped.h5"
)

recon_fly521_iter2000 = Path(
    "/mnt/micdata2/4IDD/2026_Sep/ptychi_recons/fly521"
    "/Ndp256_LSQML_c250_m0.25_gaussian_p10_mm_opr1_pc1_f_ul20/recon_Niter2000_deramped.h5"
)

#%% ------------------------------------------------- fly521 - fly520 difference

def load_object(path):
    with h5py.File(path, "r") as f:
        obj = f["object"][()]
        px = float(f["obj_pixel_size_m"][()])
    return (obj[0] if obj.ndim == 3 else obj), px


pairs = {
    1000: (recon_fly520_iter1000, recon_fly521_iter1000),
    1500: (recon_fly520_iter1500, recon_fly521_iter1500),
    2000: (recon_fly520_iter2000, recon_fly521_iter2000),
}

fig, axes = plt.subplots(3, 2, figsize=(10, 13.5), constrained_layout=True)

for ax_row, (niter, (file_a, file_b)) in zip(axes, pairs.items()):
    obj_a, px = load_object(file_a)
    obj_b, _ = load_object(file_b)
    h = min(obj_a.shape[0], obj_b.shape[0])      # 870 vs 868 rows
    w = min(obj_a.shape[1], obj_b.shape[1])
    obj_a, obj_b = obj_a[:h, :w], obj_b[:h, :w]

    d_mag = np.abs(obj_b) - np.abs(obj_a)
    d_ph = np.angle(obj_b * np.conj(obj_a))      # wrap-safe phase difference
    extent = [0, w * px * 1e6, h * px * 1e6, 0]

    for ax, img, name, unit in (
        (ax_row[0], d_mag, "|O| difference", ""),
        (ax_row[1], d_ph, "phase difference", "rad"),
    ):
        lim = np.percentile(np.abs(img), 99)
        im = ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=extent)
        ax.set_title(f"Niter {niter}  {name}  (fly521 - fly520)", fontsize=10)
        ax.set_xlabel("x (um)", fontsize=8)
        ax.set_ylabel("y (um)", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.85, label=unit)

    print(f"Niter {niter}:  d|O| mean {d_mag.mean():+.4f} rms "
          f"{np.sqrt((d_mag ** 2).mean()):.4f}   "
          f"dphase mean {d_ph.mean():+.4f} rms {np.sqrt((d_ph ** 2).mean()):.4f} rad")

plt.show()

# %%
