#%%
# Export diffraction patterns from a 12-ID-C preproc hdf5 to TIFF on the data3 share
# (Windows: \\micdata\data3\fengling\diffpatterns).
#
# Applies the same preprocessing the reconstruction scripts use (center crop to n_dp,
# optional flips, clip to >= 0) so what lands on disk is what the recon actually sees.
# Use --raw to skip it and dump the untouched 1024 px frames.
#
#   python export_patterns_tiff.py --scan S1567 --n-dp 512 --mask   # whole scan, one stack
#   python export_patterns_tiff.py --scan S1567 --frame 62          # single-shot frame only

import argparse
from pathlib import Path

import h5py
import numpy as np
import tifffile

data_root = Path("/mnt/micdata2/12IDC/2026_Data/2026_2/01_ptycho")
out_root = Path("/mnt/micdata3/fengling/diffpatterns")   # \\micdata\data3\fengling\diffpatterns


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


def parse_frames(spec):
    """'62' | '0-9' | '0,5,62' -> list of indices."""
    idx = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            idx.extend(range(int(a), int(b) + 1))
        else:
            idx.append(int(part))
    return idx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", default="S1567")
    p.add_argument("--dp-file", type=Path, default=None,
                   help="override the preproc dp hdf5 "
                        "(default: preproc/<scan>/data_roi0_Ndp1024_dp.hdf5)")
    p.add_argument("--out-dir", type=Path, default=None, help=f"default: {out_root}/<scan>")
    p.add_argument("--n-dp", type=int, default=512, help="detector crop size (0 = no crop)")
    p.add_argument("--raw", action="store_true", help="no crop, no flip, no clip")
    p.add_argument("--flip-x", action="store_true")
    p.add_argument("--flip-y", action="store_true")
    p.add_argument("--frame", type=int, default=None,
                   help="export ONE frame as a plain 2D TIFF, e.g. --frame 62 (the single-shot "
                        "frame the RPI scripts reconstruct)")
    p.add_argument("--frames", default=None, help="subset to export (default: all)")
    p.add_argument("--split", action="store_true",
                   help="one TIFF per frame instead of a single multi-page stack")
    p.add_argument("--mask", action="store_true",
                   help="also write det_pixel_mask as a uint8 TIFF")
    args = p.parse_args()
    if args.frame is not None:
        if args.frames:
            p.error("use --frame or --frames, not both")
        args.frames, args.split = str(args.frame), True

    dp_file = args.dp_file or data_root / "preproc" / args.scan / "data_roi0_Ndp1024_dp.hdf5"
    out_dir = args.out_dir or out_root / args.scan
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dp_file, "r") as f:
        patterns = f["dp"][()]
        det_mask = np.asarray(f["det_pixel_mask"][()], dtype=bool) if "det_pixel_mask" in f else None
    print(f"raw ptychogram: {patterns.shape} {patterns.dtype} from {dp_file}")

    n_dp = patterns.shape[-1] if (args.raw or not args.n_dp) else args.n_dp
    if not args.raw:
        patterns = center_crop_or_pad(patterns, n_dp)
        if det_mask is not None:
            det_mask = center_crop_or_pad(det_mask[None], n_dp)[0]
        if args.flip_x:
            patterns = patterns[..., :, ::-1]
            if det_mask is not None:
                det_mask = det_mask[:, ::-1]
        if args.flip_y:
            patterns = patterns[..., ::-1, :]
            if det_mask is not None:
                det_mask = det_mask[::-1, :]
        patterns = np.ascontiguousarray(patterns, dtype=np.float32)
        np.clip(patterns, 0, None, out=patterns)

    idx = parse_frames(args.frames) if args.frames else list(range(len(patterns)))
    patterns = patterns[idx]

    tag = f"{args.scan}_Ndp{n_dp}{'_raw' if args.raw else ''}"
    if args.split:
        for i, frame in zip(idx, patterns):
            tifffile.imwrite(out_dir / f"{tag}_f{i:04d}.tiff", frame.astype(np.float32))
        print(f"wrote {len(idx)} TIFFs to {out_dir}")
    else:
        # Name a subset stack after its frames so it never overwrites the full-scan stack.
        subset = f"_f{idx[0]:04d}-{idx[-1]:04d}" if args.frames else ""
        out_file = out_dir / f"{tag}{subset}_dp.tiff"
        tifffile.imwrite(out_file, patterns.astype(np.float32),
                         photometric="minisblack", metadata={"axes": "ZYX"})
        print(f"wrote {out_file} ({patterns.shape}, {out_file.stat().st_size / 1e6:.1f} MB)")

    if args.mask and det_mask is not None:
        mask_file = out_dir / f"{tag}_det_pixel_mask.tiff"
        tifffile.imwrite(mask_file, det_mask.astype(np.uint8) * 255)
        print(f"wrote {mask_file}")


if __name__ == "__main__":
    main()
