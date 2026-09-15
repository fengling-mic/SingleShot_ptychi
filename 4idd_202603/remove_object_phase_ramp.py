#%%
# Remove the linear phase ramp (and the constant phase offset) from a
# reconstructed object.
#
# A ptychographic solution is only defined up to  O(r) -> O(r) * exp(i(k.r + c)),
# P(r) -> P(r) * exp(-i(k.r))  : the extra ramp on the object is cancelled by the
# opposite ramp on the probe, and the per-position constant it leaves behind does
# not change |FFT(P*O)|^2.  A recon therefore often comes out with a phase wedge
# across the field of view, which swamps the sample contrast in object_ph.
#
# This script estimates k, divides it out of the object, and (unless told not to)
# multiplies it back into the probe so the saved file is still a valid solution.
#
# Estimators (--method):
#   fourier   argmax |FFT(mask * O)| gives k to one Fourier bin, then Nelder-Mead
#             refines it by maximising |sum(mask * O * exp(-i k.r))|, i.e. the
#             ramp that makes the magnitude-weighted phase as flat as possible.
#             Robust, subpixel, no unwrapping.  (default)
#   gradient  angle of the magnitude-weighted mean of O[i+1] * conj(O[i]) along
#             each axis.  One shot, no search; only valid for |k| < pi rad/px.
#   unwrap    skimage unwrap_phase + least-squares plane fit over the mask.
#
# The fit region defaults to the bounding box of positions_px (the interior of
# the scan, which is fully illuminated); the unscanned border is initialised to
# a constant by pty-chi and would otherwise bias the fit.
#
# Usage:
#   python remove_object_phase_ramp.py                       # the fly394 example
#   python remove_object_phase_ramp.py path/to/recon.h5
#   python remove_object_phase_ramp.py recon.h5 --method gradient --mask full
#
# Writes next to the input file:
#   <stem>_deramped.h5                       copy of the input, object/probe fixed
#   <stem>_deramp.png                        before / ramp / after preview
#   object_ph/object_ph_<stem>_deramped.tiff 16-bit phase, PEAR preview convention

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.optimize import minimize


#%% ---------------------------------------------------------------- defaults

# \\micdata\data2\4IDD\2026_Sep\ptychi_recons\fly394\... as mounted on the beamline
# Linux boxes.  Override on the command line.
default_recon_file = Path(
    "/mnt/micdata2/4IDD/2026_Sep/ptychi_recons/fly520"
    "/Ndp256_LSQML_c250_m0.25_gaussian_p10_mm_opr1_pc1_f_ul20/recon_Niter2000.h5"
)


#%% ---------------------------------------------------------------- helpers

def _gray_uint16(arr):
    """Min-max scaled 16-bit preview (PEAR's object_mag / object_ph tiffs)."""
    a = np.asarray(arr, dtype=np.float64)
    return ((a - a.min()) / max(np.ptp(a), 1e-12) * 65535).astype(np.uint16)


def centered_grid(shape):
    """Pixel coordinates measured from the array centre, as (y, x) 1-D vectors."""
    h, w = shape
    return (np.arange(h) - (h - 1) / 2.0, np.arange(w) - (w - 1) / 2.0)


def scan_bbox_mask(shape, positions_px, erode_px=0.0):
    """True inside the bounding box of the scan positions.

    pty-chi stores probe positions relative to the object centre, so the box in
    array indices is centre + (min, max).  Only the interior of that box is
    covered by every part of the probe, which is what we want to fit over.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    y0 = (h - 1) / 2.0 + positions_px[:, 0].min() + erode_px
    y1 = (h - 1) / 2.0 + positions_px[:, 0].max() - erode_px
    x0 = (w - 1) / 2.0 + positions_px[:, 1].min() + erode_px
    x1 = (w - 1) / 2.0 + positions_px[:, 1].max() - erode_px
    ys = slice(max(int(np.ceil(y0)), 0), min(int(np.floor(y1)) + 1, h))
    xs = slice(max(int(np.ceil(x0)), 0), min(int(np.floor(x1)) + 1, w))
    if ys.stop <= ys.start or xs.stop <= xs.start:
        raise ValueError("scan bounding box is empty after erosion")
    mask[ys, xs] = True
    return mask


def magnitude_mask(obj, quantile=0.2):
    """True where |O| is inside the bulk of its distribution.

    Fallback for files without positions_px: drops the very dark pixels (hot
    spots at the edge of the reconstructed area) but keeps the vacuum.
    """
    mag = np.abs(obj)
    lo, hi = np.quantile(mag, [quantile, 1.0 - quantile * 0.1])
    return (mag >= lo) & (mag <= hi)


#%% ---------------------------------------------------------------- estimators

def coherent_sum(obj, fy, fx, y, x):
    """sum(O * exp(-i 2pi (fy*y + fx*x))) -- the DFT of O at a continuous freq."""
    ex = np.exp(-2j * np.pi * fx * x)
    ey = np.exp(-2j * np.pi * fy * y)
    return ey @ (obj @ ex)


def estimate_ramp_fourier(obj, refine=True):
    """Ramp frequency (fy, fx) in cycles/px from the peak of |FFT(O)|.

    The DC content of a ramped object sits at the ramp frequency, so the FFT peak
    is the ramp to within one bin; Nelder-Mead on |coherent_sum| takes it the rest
    of the way (the coherent sum is largest when the phase is flattest).
    """
    spec = np.fft.fft2(obj)
    iy, ix = np.unravel_index(np.argmax(np.abs(spec)), spec.shape)
    f0 = np.array([np.fft.fftfreq(obj.shape[0])[iy], np.fft.fftfreq(obj.shape[1])[ix]])
    if not refine:
        return f0

    y, x = centered_grid(obj.shape)
    scale = 1.0 / max(np.abs(obj).sum(), 1e-30)
    bin_y, bin_x = 1.0 / obj.shape[0], 1.0 / obj.shape[1]
    res = minimize(
        lambda f: -np.abs(coherent_sum(obj, f[0], f[1], y, x)) * scale,
        f0,
        method="Nelder-Mead",
        options={"xatol": 1e-4 * min(bin_y, bin_x), "fatol": 1e-12, "maxiter": 2000},
    )
    return np.asarray(res.x)


def estimate_ramp_gradient(obj):
    """Ramp from the magnitude-weighted mean of the wrapped nearest-neighbour
    phase differences.  angle(sum(O[i+1] conj(O[i]))) is the per-pixel phase step
    along that axis, which is 2pi*f.  No unwrapping, but it aliases for
    |f| >= 0.5 cycles/px and it is pulled by strong sample gradients.
    """
    dy = np.angle(np.sum(obj[1:, :] * np.conj(obj[:-1, :])))
    dx = np.angle(np.sum(obj[:, 1:] * np.conj(obj[:, :-1])))
    return np.array([dy, dx]) / (2 * np.pi)


def estimate_ramp_unwrap(obj):
    """Least-squares plane fit to the unwrapped phase, weighted by |O|."""
    from skimage.restoration import unwrap_phase

    phase = unwrap_phase(np.angle(obj).astype(np.float64))
    y, x = centered_grid(obj.shape)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    w = np.abs(obj).ravel()
    a = np.stack([yy.ravel(), xx.ravel(), np.ones(phase.size)], axis=1)
    coef, *_ = np.linalg.lstsq(a * w[:, None], phase.ravel() * w, rcond=None)
    return coef[:2] / (2 * np.pi)


ESTIMATORS = {
    "fourier": estimate_ramp_fourier,
    "gradient": estimate_ramp_gradient,
    "unwrap": estimate_ramp_unwrap,
}


#%% ---------------------------------------------------------------- correction

def ramp_phase(shape, f):
    """2pi (fy*y + fx*x) on a grid centred on the array."""
    y, x = centered_grid(shape)
    return 2 * np.pi * (f[0] * y[:, None] + f[1] * x[None, :])


def remove_ramp(obj, f, mask=None, remove_offset=True):
    """Divide exp(i(k.r + c)) out of the object.  Returns (corrected, offset)."""
    out = obj * np.exp(-1j * ramp_phase(obj.shape, f))
    offset = 0.0
    if remove_offset:
        region = out if mask is None else out[mask]
        offset = float(np.angle(np.sum(region)))
        out = out * np.exp(-1j * offset)
    return out, offset


#%% ---------------------------------------------------------------- I/O

def load_recon(path):
    with h5py.File(path, "r") as f:
        obj = f["object"][()]
        probe = f["probe"][()] if "probe" in f else None
        positions = f["positions_px"][()] if "positions_px" in f else None
        pixel_size = float(f["obj_pixel_size_m"][()]) if "obj_pixel_size_m" in f else np.nan
    if obj.ndim == 2:                      # tolerate a 2-D object
        obj = obj[None]
    return obj, probe, positions, pixel_size


def write_recon(src_path, dst_path, obj, probe, deramp_info, overwrite=False):
    """Copy the input file, swap in the corrected object/probe, log the ramp."""
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"{dst_path} exists; pass --overwrite to replace it")
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key in src:
            if key not in ("object", "probe"):
                src.copy(key, dst)
        for key, val in src.attrs.items():
            dst.attrs[key] = val
        dst.create_dataset("object", data=obj.astype(np.complex64))
        if probe is not None:
            dst.create_dataset("probe", data=probe.astype(np.complex64))
        g = dst.create_group("deramp")
        for key, val in deramp_info.items():
            g.create_dataset(key, data=val)


def save_preview(path, before, after, f, mask):
    """Three-panel before / removed ramp / after, cropped to the fit region."""
    ys, xs = np.where(mask)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    ramp = np.angle(np.exp(1j * ramp_phase(before.shape, f)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, img, title in zip(
        axes,
        (np.angle(before)[box], ramp[box], np.angle(after)[box]),
        ("object phase", f"removed ramp  f=({f[0]:+.3e}, {f[1]:+.3e}) cyc/px", "deramped phase"),
    ):
        im = ax.imshow(img, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


#%% ---------------------------------------------------------------- driver

def deramp_file(
    recon_file,
    method="fourier",
    mask_mode="scan",
    erode_px=0.0,
    remove_offset=True,
    fix_probe=True,
    output=None,
    overwrite=False,
    make_preview=True,
):
    recon_file = Path(recon_file)
    obj, probe, positions, pixel_size = load_recon(recon_file)
    n_slices = obj.shape[0]

    if mask_mode == "scan" and positions is not None:
        mask = scan_bbox_mask(obj.shape[1:], positions, erode_px)
    elif mask_mode == "magnitude":
        mask = magnitude_mask(obj[0])
    else:
        mask = np.ones(obj.shape[1:], dtype=bool)
    ys, xs = np.where(mask)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))

    print(f"file        {recon_file}")
    print(f"object      {obj.shape}  pixel {pixel_size * 1e9:.4f} nm")
    print(f"fit region  {mask.sum()} px  rows {box[0].start}:{box[0].stop}  "
          f"cols {box[1].start}:{box[1].stop}  ({mask_mode})")

    corrected = np.empty_like(obj)
    ramps, offsets = [], []
    for i in range(n_slices):
        # Fit on the masked crop only; the unscanned border is a pty-chi constant.
        patch = obj[i][box].astype(np.complex128) * mask[box]
        f = np.asarray(ESTIMATORS[method](patch), dtype=np.float64)
        corrected[i], offset = remove_ramp(obj[i].astype(np.complex128), f, mask, remove_offset)
        ramps.append(f)
        offsets.append(offset)

        rad_px = 2 * np.pi * f
        fov = np.array(obj.shape[1:]) * pixel_size
        print(f"slice {i}: f = ({f[0]:+.6e}, {f[1]:+.6e}) cyc/px")
        print(f"          slope  ({rad_px[0]:+.4e}, {rad_px[1]:+.4e}) rad/px"
              f"   ({rad_px[0] / (pixel_size * 1e6):+.3f}, "
              f"{rad_px[1] / (pixel_size * 1e6):+.3f}) rad/um")
        print(f"          across the object: "
              f"({rad_px[0] * obj.shape[1] / (2 * np.pi):+.2f}, "
              f"{rad_px[1] * obj.shape[2] / (2 * np.pi):+.2f}) waves "
              f"over ({fov[0] * 1e6:.2f}, {fov[1] * 1e6:.2f}) um")
        if probe is not None:
            n_dp = probe.shape[-1]
            print(f"          equivalent diffraction shift: "
                  f"({f[0] * n_dp:+.3f}, {f[1] * n_dp:+.3f}) detector px")
        print(f"          constant offset removed: {offset:+.4f} rad")

    total_f = np.sum(ramps, axis=0)
    probe_out = probe
    if fix_probe and probe is not None:
        # exp(+i k.r) on the probe keeps P*O (and hence the fit to the data)
        # unchanged, up to a per-position constant phase the intensities cannot see.
        probe_out = probe * np.exp(1j * ramp_phase(probe.shape[-2:], total_f))
        print(f"probe       multiplied by the conjugate ramp "
              f"({total_f[0]:+.3e}, {total_f[1]:+.3e}) cyc/px")

    out_file = Path(output) if output else recon_file.with_name(recon_file.stem + "_deramped.h5")
    write_recon(
        recon_file,
        out_file,
        corrected,
        probe_out,
        {
            "ramp_cycles_per_px": np.asarray(ramps, dtype=np.float64),
            "phase_offset_rad": np.asarray(offsets, dtype=np.float64),
            "method": np.bytes_(method),
            "mask_mode": np.bytes_(mask_mode),
            "probe_ramp_applied": bool(fix_probe and probe is not None),
        },
        overwrite,
    )
    print(f"wrote       {out_file}")

    ph_dir = out_file.parent / "object_ph"
    ph_dir.mkdir(exist_ok=True)
    ph_file = ph_dir / f"object_ph_{out_file.stem}.tiff"
    tifffile.imwrite(ph_file, _gray_uint16(np.angle(corrected[0][box])))
    print(f"wrote       {ph_file}")

    if make_preview:
        png = recon_file.with_name(recon_file.stem + "_deramp.png")
        save_preview(png, obj[0], corrected[0], ramps[0], mask)
        print(f"wrote       {png}")

    return corrected, probe_out, np.asarray(ramps), np.asarray(offsets)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("recon_file", nargs="?", default=str(default_recon_file),
                   help="recon_Niter*.h5 to deramp")
    p.add_argument("--method", choices=list(ESTIMATORS), default="fourier")
    p.add_argument("--mask", dest="mask_mode", choices=["scan", "magnitude", "full"],
                   default="scan", help="region the ramp is fitted over")
    p.add_argument("--erode-px", type=float, default=0.0,
                   help="shrink the scan bounding box by this many pixels")
    p.add_argument("--keep-offset", action="store_true",
                   help="leave the constant phase offset alone")
    p.add_argument("--keep-probe", action="store_true",
                   help="do not put the conjugate ramp back on the probe")
    p.add_argument("--output", default=None, help="output .h5 (default <stem>_deramped.h5)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    deramp_file(
        args.recon_file,
        method=args.method,
        mask_mode=args.mask_mode,
        erode_px=args.erode_px,
        remove_offset=not args.keep_offset,
        fix_probe=not args.keep_probe,
        output=args.output,
        overwrite=args.overwrite,
        make_preview=not args.no_preview,
    )
