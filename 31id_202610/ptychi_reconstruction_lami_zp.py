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
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle
from scipy.ndimage import gaussian_filter, map_coordinates

from utils import center_crop_or_pad,show_ptychogram,make_regular_zoneplate_probe,hermite_secondary_modes

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

scan = "S0006"

data_root = Path("/home/beams/FENGLING.ZHANG/Downloads")

dp_file = data_root / "scan_00006.h5"        # summed Citius patterns (sum_dwells.py), read-only
position_data = data_root / "scan_00006.dat"  # measured scan positions [um], one row per point

out_dir = Path("/mnt/micdata3/fengling/2026_31id_results") / "ptychi_recons"
# out_dir = Path(r"\\micdata\data3\fengling\2026_03_results") / "ptychi_recons"
recon_dir_suffix = "_allpos_pc"            # appended to the folder name, e.g. "_v2" or "_pos"

#%% ---------------------------------------------------------------- read the summed Citius data and the measured scan positions
# scan_00006.h5 (sum_dwells.py): each scan point is the sum of the ~435 raw Citius
# frames recorded during that dwell. gzip-compressed, so plain h5py reads it.
# The file is only ever opened read-only.

with h5py.File(dp_file, "r") as f:
    entry = f["entry"]
    print(f"attributes of /entry in {dp_file.name}:")
    for key, value in entry.attrs.items():
        print(f"  {key} = {value}")
    g = entry["data"]
    print(f"datasets in /entry/data: {list(g.keys())}")
    patterns_raw = g["data"][()]               # (n_points, rows, cols) summed counts, float32
    frame_count = g["frame_count"][()]         # raw frames summed into each point
    file_boundary = g["file_boundary"][()]     # 1 if the point spans a raw file/train boundary
    edge_clipped = g["edge_clipped"][()]       # 1 if cut off by the start/end of acquisition
    point_index = g["point_index"][()]         # point number, 0..n_points-1
print(f"raw ptychogram: {patterns_raw.shape} {patterns_raw.dtype}, "
      f"frames per point {frame_count.min()}-{frame_count.max()}")

# The .h5 holds no positions; they come from the scan's .dat file. Line 1 is a
# "Scan 00006, lsamrot_encoder ..." header, line 2 the column names.
pos = np.genfromtxt(position_data, skip_header=1, names=True)
assert len(pos) == len(patterns_raw), f"{len(pos)} positions vs {len(patterns_raw)} patterns"
assert np.array_equal(pos["DataPoint"].astype(int), point_index), "DataPoint != point_index"

# Points whose dwell spans a raw-file boundary are flagged by sum_dwells.py as
# "may be clipped" (65 in S0006). With exclude_file_boundary = True they are
# dropped from BOTH the patterns and the positions; False keeps them.
exclude_file_boundary = False
keep = edge_clipped == 0
if exclude_file_boundary:
    keep &= file_boundary == 0
kept_index = point_index[keep]                 # original point number of each kept row
is_boundary = file_boundary[keep].astype(bool) # per kept row: spans a file boundary
print(f"keeping {int(keep.sum())} of {len(keep)} points "
      f"({int(is_boundary.sum())} of them span a file boundary); "
      f"excluding {int((~keep).sum())}: {np.flatnonzero(~keep).tolist()}")

# measured positions of the kept points, (y, x) in um
positions_um = np.stack((pos["Average_y_st_fzp"], pos["Average_x_st_fzp"]), axis=-1)[keep]


#%% ---------------------------------------------------------------- geometry & knobs

energy_kev = 8.57 # trasfer from keV to meters         
n_dp = 256                          # detector crop
wavelength_m = 1.24e-9 / energy_kev # trasfer from keV to meters         
det_pixel_m = 72.6e-6                # fracPy exampleData.dxd
det_dist_m = 1.5    # fracPy exampleData.zo
far_field = True                    # fracPy propagatorType 'Fraunhofer'

# fracPy fixed dxo = dxp = 1.76e-8; that is exactly the far-field sampling below.
pixel_size_m = wavelength_m * det_dist_m / (n_dp * det_pixel_m)

n_probe_modes = 5                # fracPy reconstruction.npsm, mix-state modes
n_opr_modes = 1                  # variable probe (OPR); 1 disables it
probe_focal_spot_diameter_m = 3.5e-6   # zone-plate aperture / OSA diameter (2*Rmax) -- the
                                        # actual focused spot is much smaller, set by the outer zone width
probe_diameter_m = probe_focal_spot_diameter_m  # feeds the real-space support constraint below
probe_outer_zone_width_m = 50e-9       # finest zone of the zone plate -- sets the focal-spot resolution
probe_zp_central_stop = 0.15           # fraction of the simulated zone plate's radius blocked
# The three zone-plate numbers above are used only by probe_init_method = "zoneplate_gs".
# Checked against S0006: the donut's outer edge (~29-35 detector px, 1.45 mrad) gives an
# outer zone of ~44-60 nm, so 50 nm is consistent; the hole/edge ratio gives a central
# stop of ~0.24; 3.5 um is not the real zone-plate diameter (it sizes the toy plate and
# the pinhole at its focus). That method makes an IN-FOCUS probe (~60 nm spot).

# Initial probe:
#   "data_defocus" -- measured beam far field (the pupil) + a defocus phase. Needs only
#                     probe_defocus_m. Recommended: no optics model involved.
#   "zoneplate_gs" -- the simulated zone plate above, far field fitted to the data by GS.
probe_init_method = "data_defocus"
# Sample distance from the focus. The speckle size inside the donut and the loss of
# similarity between neighbouring patterns both put the probe at ~0.7-0.9 um across,
# i.e. ~0.2-0.35 mm from the focus. The sign (before/after the focus) is unknown.
probe_defocus_m = 0.25e-3

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

num_epochs = 5500
save_freq_iterations = 1000       # write a recon_Niter*.h5 snapshot every N epochs
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

# Crop an n_dp x n_dp window centred on the donut. The zone plate's beam stop
# makes every pattern a bright ring around a dark hole, and the crop is centred
# on the middle of that hole. An intensity centroid is not used: the ring is
# brighter along one side and pulls a centroid toward it.
#
# Hole finder, on the log of the lightly smoothed mean of the kept patterns:
#   A. ring: cast rays from a start point, take the brightest radius on each ray,
#      fit a circle through those points, repeat -- a stable start for step B;
#   B. hole: on each ray, inside the ring, take the outermost radius that is still
#      darker than halfway between the hole floor and the ring peak; fit a circle
#      through those points, repeat.
# The centre is rounded to whole pixels, so the data are never resampled.
beam_centre_override = None      # (row, col) in the raw frame, to set the centre by hand

mean_pattern = np.mean(patterns_raw, axis=0, where=keep[:, None, None], dtype=np.float64)
n_rows, n_cols = mean_pattern.shape


def _fit_circle(y, x):
    """Least-squares circle through the points (Kasa fit) -> (row, col, radius)."""
    a = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    cx, cy, c = np.linalg.lstsq(a, x**2 + y**2, rcond=None)[0]
    return cy, cx, np.sqrt(c + cx**2 + cy**2)


def find_donut_centre(image, start_rc, max_radius_px=60, n_rays=360, n_iter=5):
    """Sub-pixel centre (row, col) and radius of the dark hole of a donut pattern."""
    log_img = np.log10(np.maximum(gaussian_filter(image, 1.0), 1.0))
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    sin_a, cos_a = np.sin(angles), np.cos(angles)
    radii = np.arange(0, max_radius_px, 0.25)

    def ray_profiles(cy, cx):
        return map_coordinates(log_img, [cy + np.outer(sin_a, radii),
                                         cx + np.outer(cos_a, radii)], order=1)

    cy, cx = start_rc
    for _ in range(n_iter):                                  # A. ring
        r_ring = radii[np.argmax(ray_profiles(cy, cx), axis=1)]
        cy, cx, _ = _fit_circle(cy + r_ring * sin_a, cx + r_ring * cos_a)
    for _ in range(n_iter):                                  # B. hole
        prof = ray_profiles(cy, cx)
        r_edge = np.full(n_rays, np.nan)
        for k, i_peak in enumerate(np.argmax(prof, axis=1)):
            inner = prof[k, : i_peak + 1]
            dark = np.flatnonzero(inner < 0.5 * (inner.min() + inner.max()))
            if len(inner) >= 3 and len(dark):
                r_edge[k] = radii[dark.max()]
        ok = np.isfinite(r_edge)
        cy, cx, radius = _fit_circle(cy + r_edge[ok] * sin_a[ok], cx + r_edge[ok] * cos_a[ok])
    return cy, cx, radius


# start point: intensity centroid in a window around the brightest pixel
_r, _c = np.unravel_index(np.argmax(mean_pattern), mean_pattern.shape)
for _ in range(2):
    _rs = slice(max(_r - 64, 0), min(_r + 65, n_rows))
    _cs = slice(max(_c - 64, 0), min(_c + 65, n_cols))
    _w = mean_pattern[_rs, _cs]
    _rr, _cc = np.mgrid[_rs, _cs]
    _r, _c = int(round((_w * _rr).sum() / _w.sum())), int(round((_w * _cc).sum() / _w.sum()))

donut_row, donut_col, donut_radius_px = find_donut_centre(mean_pattern, (_r, _c))
if beam_centre_override is None:
    beam_row, beam_col = int(round(donut_row)), int(round(donut_col))
else:
    beam_row, beam_col = (int(v) for v in beam_centre_override)


def _crop_centred(stack, rows_to_keep, centre_rc, n):
    """n x n window of the selected frames, centred on centre_rc. Parts of the
    window that fall outside the frame stay zero; zero in every frame, they are
    caught by the dead-pixel mask below and left out of the fit."""
    out = np.zeros((int(np.count_nonzero(rows_to_keep)), n, n), dtype=np.float32)
    r0, c0 = centre_rc[0] - n // 2, centre_rc[1] - n // 2
    sr0, sr1 = max(r0, 0), min(r0 + n, stack.shape[1])
    sc0, sc1 = max(c0, 0), min(c0 + n, stack.shape[2])
    out[:, sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = stack[rows_to_keep, sr0:sr1, sc0:sc1]
    return out


crop_r0, crop_c0 = beam_row - n_dp // 2, beam_col - n_dp // 2   # crop origin in the raw frame
patterns = _crop_centred(patterns_raw, keep, (beam_row, beam_col), n_dp)
pad_rows = max(-crop_r0, 0) + max(crop_r0 + n_dp - n_rows, 0)
pad_cols = max(-crop_c0, 0) + max(crop_c0 + n_dp - n_cols, 0)
_inside = mean_pattern[max(crop_r0, 0):crop_r0 + n_dp, max(crop_c0, 0):crop_c0 + n_dp].sum()
print(f"donut hole centre: row {donut_row:.2f}, col {donut_col:.2f}, radius "
      f"{donut_radius_px:.1f} px -> crop centre row {beam_row}, col {beam_col}"
      f"{' (override)' if beam_centre_override is not None else ''}")
print(f"crop rows {crop_r0}:{crop_r0 + n_dp}, cols {crop_c0}:{crop_c0 + n_dp} of the "
      f"{n_rows}x{n_cols} frame; zero-padded {pad_rows} rows, {pad_cols} cols; "
      f"{100 * _inside / mean_pattern.sum():.3f}% of counts inside")

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

# Check: crop box on the mean pattern, the donut in the middle of the crop, and
# which scan points were dropped.
fig, (ax_mean, ax_zoom, ax_pos) = plt.subplots(1, 3, figsize=(15, 5))
ax_mean.imshow(mean_pattern, cmap="inferno", interpolation="nearest",
               norm=LogNorm(vmin=1, vmax=float(mean_pattern.max())))
ax_mean.add_patch(Rectangle(
    (crop_c0 - 0.5, crop_r0 - 0.5), n_dp, n_dp, fill=False, edgecolor="cyan",
    linewidth=1.5, label=f"{n_dp}x{n_dp} crop",
))
ax_mean.plot(beam_col, beam_row, "+", color="cyan", ms=12, mew=1.5, label="donut centre")
ax_mean.set_title(f"mean of {int(keep.sum())} kept patterns (log)")
ax_mean.legend(loc="upper right", fontsize=8)

crop_mean = patterns.mean(axis=0)
_hole_r, _hole_c = donut_row - crop_r0, donut_col - crop_c0     # fitted hole, crop coords
if flip_dp_y:
    _hole_r = n_dp - 1 - _hole_r
if flip_dp_x:
    _hole_c = n_dp - 1 - _hole_c
ax_zoom.imshow(crop_mean, cmap="inferno", interpolation="nearest",
               norm=LogNorm(vmin=max(float(crop_mean.min()), 1.0), vmax=float(crop_mean.max())))
ax_zoom.axhline(n_dp // 2, color="cyan", lw=0.8)
ax_zoom.axvline(n_dp // 2, color="cyan", lw=0.8)
ax_zoom.add_patch(Circle((_hole_c, _hole_r), donut_radius_px, fill=False,
                         edgecolor="lime", linewidth=1.2, label="fitted hole edge"))
ax_zoom.set_xlim(n_dp // 2 - 40, n_dp // 2 + 40)
ax_zoom.set_ylim(n_dp // 2 + 40, n_dp // 2 - 40)
ax_zoom.set_title(f"cropped mean, zoom; crosshair = crop centre ({n_dp // 2}, {n_dp // 2})",
                  fontsize=9)
ax_zoom.legend(loc="upper right", fontsize=8)

ax_pos.plot(pos["Average_x_st_fzp"][keep], pos["Average_y_st_fzp"][keep], "o", ms=3,
            label=f"kept ({int(keep.sum())})")
ax_pos.plot(pos["Average_x_st_fzp"][~keep], pos["Average_y_st_fzp"][~keep], "x",
            color="tab:orange", ms=6, mew=1.5, label=f"excluded ({int((~keep).sum())})")
ax_pos.set_aspect("equal")
ax_pos.set_xlabel("x [um]"), ax_pos.set_ylabel("y [um]")
ax_pos.set_title("measured scan positions")
ax_pos.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()

#%% ---------------------------------------------------------------- positions

# No initial-guess reconstruction: positions are the measured ones from the .dat
# file, and the probe below is synthesized (`prior` stays empty).
prior = {}
positions_px = positions_um * 1e-6 / pixel_size_m    # um -> m -> object pixels

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
    print(f"probe from the initial-guess recon: {tuple(probe.shape)}")
else:
    # Beam far field, shared by both methods: a low percentile across positions
    # keeps what every frame has in common (the beam) and suppresses the
    # position-dependent object scatter. Gap-filled across dead/stuck pixels
    # (normalized convolution: blur data*mask and mask separately, divide).
    _percentile_10 = np.percentile(patterns, 10, axis=0)
    _mask = valid_pixel_mask.astype(np.float32)
    _num = gaussian_filter(_percentile_10 * _mask, sigma=3.0)
    _den = gaussian_filter(_mask, sigma=3.0)
    _gap_filled = np.where(valid_pixel_mask, _percentile_10, _num / np.maximum(_den, 1e-6))
    _ff_amp = np.sqrt(np.fft.ifftshift(_gap_filled))       # FFT layout, DC = crop centre

if "probe" not in prior and probe_init_method == "data_defocus":
    # Probe straight from the data. The measured beam far field IS the zone
    # plate's pupil -- cone edge, central-stop shadow and all -- so its amplitude
    # needs no optics model. What the intensity cannot tell is how far the sample
    # is from the focus; that enters as the Fresnel defocus phase on the pupil,
    #     phi(u) = -pi * dz * u^2 / (lambda * L^2)
    # for a detector offset u at distance L. Flipping the sign of dz (sample
    # before vs after the focus) leaves the intensity unchanged, so either sign
    # is a fair start; the reconstruction refines the phase from there.
    _u = np.fft.fftfreq(n_dp, d=1.0 / n_dp) * det_pixel_m   # detector offset [m], FFT layout
    _phi = -np.pi * probe_defocus_m * (_u[:, None] ** 2 + _u[None, :] ** 2) / (
        wavelength_m * det_dist_m**2
    )
    _mode = np.fft.fftshift(np.fft.ifft2(_ff_amp * np.exp(1j * _phi)))
    probe_init = _mode[None, None].astype(np.complex64)

    # footprint: diameter of the disk (about the power centroid) holding 90 % of the power
    _I = np.abs(_mode) ** 2
    _yy, _xx = np.indices(_I.shape)
    _rr = np.hypot(_yy - (_I * _yy).sum() / _I.sum(), _xx - (_I * _xx).sum() / _I.sum()).ravel()
    _order = np.argsort(_rr)
    _cum = np.cumsum(_I.ravel()[_order]) / _I.sum()
    probe_d90_m = 2 * _rr[_order][np.searchsorted(_cum, 0.9)] * pixel_size_m
    probe = torch.as_tensor(probe_init, dtype=get_default_complex_dtype())
    print(f"probe from the measured beam far field, defocus {probe_defocus_m * 1e3:.3f} mm: "
          f"90% of the power within {probe_d90_m * 1e6:.2f} um "
          f"(window {n_dp * pixel_size_m * 1e6:.2f} um)")
    if probe_d90_m > 0.5 * n_dp * pixel_size_m:
        print("  WARNING: probe wider than half the window -- the far field is undersampled")

elif "probe" not in prior and probe_init_method == "zoneplate_gs":
    probe_init = make_regular_zoneplate_probe(
        n_dp, pixel_size_m, wavelength_m,
        probe_outer_zone_width_m, probe_focal_spot_diameter_m,
        central_stop=probe_zp_central_stop,
    )
    # The regular-zone-plate simulation already gives the probe a physically
    # realistic far field -- a ring carved out by the central stop, peaking
    # off DC -- the qualitative signature the real beam (recon_Niter400.h5,
    # same method) also shows, rather than the randomized probe's speckle-
    # at-DC mismatch this block was originally written to repair. What the
    # idealized simulation parameters (central_stop fraction, outer zone
    # width, aperture radius) still can't capture is this specific optic's
    # actual fabrication and alignment, so the ring's exact radius/width
    # need not match the data yet.
    #
    # Gerchberg-Saxton against the measured illumination refines that: a low
    # percentile across positions keeps what every frame has in common (the
    # beam) and suppresses the position-dependent object scatter. Alternate
    # imposing that amplitude in the far field with the simulated probe's own
    # real-space envelope, so the probe keeps the footprint and resolution
    # already validated -- only its far-field amplitude is corrected, from
    # the data.
    #
    # Note this fixes the far-field AMPLITUDE only. The far-field PHASE stays
    # arbitrary -- GS converges to some field consistent with both amplitudes,
    # not to the real beam's actual phase, which only a reconstruction
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
    # (_gap_filled / _ff_amp: the shared beam far field computed above.)
    gs_iterations = 300
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
    print(f"synthesized regular-zone-plate probe: focal spot "
          f"{probe_focal_spot_diameter_m * 1e6:.2f} um, outer zone "
          f"{probe_outer_zone_width_m * 1e9:.0f} nm, "
          f"far field matched to the data in {gs_iterations} GS iterations")

elif "probe" not in prior:
    raise ValueError(f"unknown probe_init_method {probe_init_method!r}")

if "probe" not in prior:
    # Check: mode 0 in real space, and its far field next to the measured beam.
    _m0 = np.asarray(probe_init).reshape(-1, n_dp, n_dp)[0]
    _amp0 = np.abs(_m0)
    _far0 = np.fft.fftshift(np.abs(np.fft.fft2(_m0)) ** 2)
    _far0 *= _gap_filled.sum() / _far0.sum()
    _ext = np.array([-0.5, 0.5, 0.5, -0.5]) * n_dp * pixel_size_m * 1e6
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(_amp0, cmap="inferno", extent=_ext)
    axes[0].set_title(f"initial probe |mode 0| ({probe_init_method})", fontsize=9)
    axes[1].imshow(np.where(_amp0 > 0.05 * _amp0.max(), np.angle(_m0), np.nan),
                   cmap="twilight", extent=_ext)
    axes[1].set_title("phase where |p| > 5% of max", fontsize=9)
    for ax in axes[:2]:
        ax.set_xlabel("x [um]"), ax.set_ylabel("y [um]")
    _norm = LogNorm(vmin=max(float(_gap_filled[_gap_filled > 0].min()), 1.0),
                    vmax=float(_gap_filled.max()))
    axes[2].imshow(_gap_filled, cmap="inferno", norm=_norm)
    axes[2].set_title("measured beam far field (10th percentile)", fontsize=9)
    axes[3].imshow(_far0, cmap="inferno", norm=_norm)
    axes[3].set_title("|FFT(mode 0)|^2", fontsize=9)
    for ax in axes[2:]:
        ax.set_xlim(n_dp // 2 - 60, n_dp // 2 + 60)
        ax.set_ylim(n_dp // 2 + 60, n_dp // 2 - 60)
    plt.tight_layout()
    plt.show()

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

#%%
probe = torch.as_tensor(rescale_probe(probe, patterns), dtype=get_default_complex_dtype())
opr_weights = generate_initial_opr_mode_weights(len(positions_px), probe.shape[0], probe=probe)
print(f"probe: {tuple(probe.shape)} (n_opr, n_modes, h, w)")

#%% ---------------------------------------------------------------- initial object

# object_shape = get_suggested_object_size(positions_px, probe.shape[-2:], extra=object_padding_px)
# obj = torch.ones((1, *object_shape), dtype=get_default_complex_dtype())  # (n_slices, h, w)
# # obj = torch.full((1, *object_shape), 1j, dtype=get_default_complex_dtype())

# print(f"object buffer: {tuple(obj.shape)}")

# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].imshow(np.abs(obj[0].numpy()), cmap="gray")
# axes[0].set_title("initial object magnitude")
# axes[1].imshow(np.angle(obj[0].numpy()), cmap="gray")
# axes[1].set_title("initial object phase")
# for ax in axes:
#     ax.set_aspect("equal")
# plt.tight_layout()
# plt.show()

#%%
#%% ---------------------------------------------------------------- initial object

# PRL-style per-position ESW estimate: Williams et al., "Fresnel Coherent Diffractive
# Imaging," Phys. Rev. Lett. 97, 025506 (2006), Eq. (3). At each position, iterate
#
#     rho_{k+1} = F^-1[ M( F[rho_k] + Psi_inc ) - Psi_inc ]
#
# where rho = T.psi_inc is the object-induced perturbation of the exit wave (rho_0 = 0,
# i.e. "no object yet"), Psi_inc = F[psi_inc] is the known illumination's own far field
# (precomputed once), and M enforces the measured modulus at valid detector pixels,
# leaving the current estimate alone at dead/stuck pixels (same `valid_pixel_mask`
# convention used for the probe's own GS fit above). Unlike a single Wiener division,
# this iterates the actual modulus constraint against the measured data at every
# position, so it isn't limited to a weak-object approximation.
#
# CAVEAT: this omits the paper's real-space support constraint (pi_s in their Eq. 1/3).
# Their support -- the isolated gold sample surrounded by vacuum -- is what gave plain ER
# its resistance to the twin image and its clean, ~30-iteration convergence. The Siemens
# star fills the field of view with no equivalent vacuum region, so there's nothing
# physically correct to threshold to zero, and this was deliberately run without one on
# request. What's left is the known-illumination modulus trick alone -- still legitimate,
# since the paper credits the curved/structured illumination itself (not the support)
# with making the solution unique -- but expect noisier, slower, less certain convergence
# per position than the paper reports, and no protection against stagnation or the twin
# image.
#
# Positions are processed independently (no overlap redundancy shared during the ER
# iterations themselves), then combined afterward with the same |P|^2-weighted stitch
# `seed_object` used in the non-ESW script. Runs batched in Torch on GPU since
# n_iterations x n_positions x 2 FFTs of size n_dp^2 is too slow as a per-position NumPy
# loop.
def esw_object_prl(
    patterns, probe_2d, positions_px, object_shape, valid_pixel_mask,
    n_iterations=500, batch_size=50, verbose=True,
):
    """Per-position ESW estimate via known-illumination ER (no support constraint)."""
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    P = torch.as_tensor(np.asarray(probe_2d, dtype=np.complex64), device=torch_device)
    n = P.shape[-1]
    Psi_inc = torch.fft.fft2(P)  # illumination's own far field, precomputed once
    keep = torch.as_tensor(np.fft.ifftshift(valid_pixel_mask), device=torch_device)

    p_conj = torch.conj(P)
    p_sq = P.abs() ** 2
    eps = 1e-3 * p_sq.max()
    cy, cx = object_shape[0] // 2, object_shape[1] // 2

    num = torch.zeros(object_shape, dtype=torch.complex64, device=torch_device)
    den = torch.zeros(object_shape, dtype=torch.float32, device=torch_device)

    n_pos = len(positions_px)
    err_num, err_den = 0.0, 0.0
    for start in range(0, n_pos, batch_size):
        sl = slice(start, start + batch_size)
        batch_patterns = np.fft.ifftshift(patterns[sl], axes=(-2, -1)).astype(np.float32)
        target_amp = torch.sqrt(
            torch.clamp(torch.as_tensor(batch_patterns, device=torch_device), min=0)
        )
        b = target_amp.shape[0]
        rho = torch.zeros((b, n, n), dtype=torch.complex64, device=torch_device)

        for _ in range(n_iterations):
            far = torch.fft.fft2(rho) + Psi_inc  # (i) propagate, (ii) add illumination
            amp = far.abs()
            corrected = torch.where(
                amp > 1e-12, target_amp * far / amp, target_amp.to(far.dtype)
            )
            far = torch.where(keep, corrected, far)  # (iii) modulus at valid pixels only
            rho = torch.fft.ifft2(far - Psi_inc)      # (iv) subtract illum., (v) backpropagate

        # Track a chi^2-like relative modulus error (paper's Eq. 2) for convergence sanity.
        far_final = (torch.fft.fft2(rho) + Psi_inc).abs()
        err_num += (((far_final - target_amp) ** 2) * keep).sum().item()
        err_den += ((target_amp ** 2) * keep).sum().item()

        pos_batch = positions_px[sl]
        for i in range(b):
            pos_y, pos_x = pos_batch[i]
            r0 = int(round(cy + pos_y - n / 2))
            c0 = int(round(cx + pos_x - n / 2))
            num[r0 : r0 + n, c0 : c0 + n] += rho[i] * p_conj
            den[r0 : r0 + n, c0 : c0 + n] += p_sq

    if verbose:
        print(f"ESW/PRL seed: relative modulus chi^2 after {n_iterations} iters "
              f"= {err_num / err_den:.4e}")

    est = num / (den + eps)
    lit = den > 0.01 * den.max()
    est = est / torch.median(est[lit].abs())
    est = torch.where(lit, est, torch.ones_like(est))
    return est.cpu().numpy().astype(np.complex64)


object_shape = get_suggested_object_size(positions_px, probe.shape[-2:], extra=object_padding_px)
# obj = torch.ones((1, *object_shape), dtype=get_default_complex_dtype())  # (n_slices, h, w)
# obj = torch.full((1, *object_shape), 1j, dtype=get_default_complex_dtype())

obj = torch.as_tensor(
    esw_object_prl(patterns, probe[0, 0].numpy(), positions_px, object_shape,
                   valid_pixel_mask)[None],
    dtype=get_default_complex_dtype(),
)  # (n_slices, h, w)

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
options.object_options.step_size = 0.2
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
options.probe_options.step_size = 0.2
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
# In far field the probe's FFT bins are detector pixels (independent of n_dp). The
# measured beam cone reaches ~35-40 px from the centre (99.45% of the counts within
# 35 px), so 45 px keeps the whole cone and cuts what lies outside it. (200 px was
# no constraint at all for n_dp = 256: the farthest bin is ~181 px from DC.)
options.probe_options.fourier_support_constraint.radius_px = 45

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
