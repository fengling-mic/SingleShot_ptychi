#%% Fourier Ring Correlation: step-scan vs. single-frame ptychography, at frame 62's area
#
# Three single-frame curves (one per RPI variant in `singleframe_recons`: band-limited, no
# band limit, band-limit relaxation) plus the step-scan reference, all over the same area.
# The step-scan side is a two-group split:
#   group 1 = dir1 + dir4 (both gaussian, pc0, good init probe; dir4 adds esw). They share
#             dir3's probe and positions and agree to Nyquist, so they go in the SAME group;
#             splitting them would inflate the step-scan curve.
#   group 2 = dir2 (gaussian, position-corrected, 15500 iterations) + dir3 (the original
#             piezo_test LSQML recon the others and the RPI run were bootstrapped from)
# Each group is registered onto its first member and complex-averaged. dir1-dir4 all
# reconstruct the IDENTICAL raw S0019 diffraction data (confirmed via their pear_params.json):
#
#   1-3. group 1 vs each single-frame reconstruction
#   4.   group 1 vs group 2      <- step-scan's own resolution ceiling, for the single-frame
#        curves to be read against.
#
# The area is frame 62's window, cut down to where the single-frame probe is above
# `illumination_threshold` of its peak: outside that the RPI object is unconstrained init
# noise and would only dilute the FRC. Each step-scan recon is cropped at its own
# positions_px[62] (dir3's is frame 62's own scan position; the pc0 recons inherit it).
# Each object's phase ramp/offset (gauge freedom: O -> O*exp(i(k.r+c)) is invisible to the
# data) is removed over that lit area, and every pair is registered on PHASE before the FRC.

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import h5py
import torch
from pathlib import Path

from scipy.ndimage import rotate
from skimage.measure import profile_line
from skimage.registration import phase_cross_correlation

from utils import (
    crop_at_position,
    fourier_ring_correlation,
    frc_crossing_px,
    frc_half_bit_threshold,
    plot_frc,
    plot_frc_rings,
    probe_fourier_radius,
    remove_phase_ramp,
)

#%% FRC from multiple step scans
step_recon_dir1 = Path(r"\\micdata\data3\fengling\2026_03_results") / "ptychi_recons" / "S0019" / "Ndp1024_LSQML_c100_m0_gaussian_p5_opr1_pc0_goodinitProbe" / "recon_Niter500.h5"
step_recon_dir2 = Path(r"\\micdata\data3\fengling\2026_03_results") / "ptychi_recons" / "S0019" / "Ndp1024_LSQML_c100_m0.25_gaussian_p5_opr1_pc1_f_ul20_initprobe_fromzpOSA_GS_eswonj_stepsize" / "recon_Niter15500.h5"
step_recon_dir3 = Path(r"\\micdata\data2\12IDC\2026_Data\2026_3\01_piezo_test") / "ptychi_recons" / "S0019" / "Ndp1024_LSQML_c20_m0.5_gaussian_p10_cp_mm_opr3_ic_pc1_f_ul2" / "recon_Niter400.h5"
step_recon_dir4 = Path(r"\\micdata\data3\fengling\2026_03_results") / "ptychi_recons" / "S0019" / "Ndp1024_LSQML_c120_m0_gaussian_p5_opr1_pc0_goodinitProbe_esw" / "recon_Niter1000.h5"


#%% FRC from single-frame ptychography
# singleframe_recon_dir_bl = Path("/mnt/micdata3/fengling/2026_03_results") / "singleframe_recons" / "S0019" / "frame62_RPI_Adam_Nobj84_R0.5_p10_opr1_prel10_esw_1" / "recon_Niter2000.h5"
singleframe_recon_dir_nobl = Path(r"\\micdata\data3\fengling\2026_03_results") / "singleframe_recons" / "S0019" / "frame62_RPI_Adam_Nobj1024_R100_p10_opr1_prel10_esw_1" / "recon_Niter10000.h5"
singleframe_recon_dir_blrelax = Path(r"\\micdata\data3\fengling\2026_03_results") / "singleframe_recons" / "S0019" / "frame62_RPI_Adam_Nobj72_R0.4_relax20_0-2000_p10_opr1_prel100_blrelax" / "recon_Niter10000.h5"
# singleframe_recon_dir = Path(r"\\micdata\data3\fengling\2026_03_results") / "singleframe_recons" / "S0019" / "frame63_RPI_Adam_Nobj1024_R100_p10_opr1_prel10_esw_1" / "recon_Niter10000.h5"
singleframe_recon_dir_bl = Path(r"\\micdata\data3\fengling\2026_03_results") / "singleframe_recons" / "S0019" / "frame62_RPI_Adam_Nobj92_R0.5_relax10000_0-10000_p10_opr1_prel100_blrelax" / "recon_Niter10000.h5"


singleframe_recons = {   # legend label -> recon file, in order of decreasing band limit
    "Band-limited": singleframe_recon_dir_bl,
    "Band-limit relax": singleframe_recon_dir_blrelax,
    "No band limit": singleframe_recon_dir_nobl,
}
# Light trio (very light blue / aqua / blue, in dict order) + a contrasting orange; validated
# all-pairs for normal and colorblind vision. Relax and no band limit get the most separated pair.
singleframe_colors = ["#86b6ef", "#1baf7a", "#3987e5"]
step_scan_color = "#eb6834"

frame_index = 62   # must match rpi_params.json's "frame_index" next to each single-frame recon
illumination_threshold = 0.20   # FRC area = square bbox of single-frame probe intensity above this fraction of peak


#%% ---------------------------------------------------------------- helpers

def load_step_recon(path):
    """(object [2D, slice axis dropped], positions_px, obj_pixel_size_m)."""
    with h5py.File(path, "r") as f:
        obj = f["object"][()]
        positions_px = f["positions_px"][()].astype(np.float64)
        pixel_size_m = float(f["obj_pixel_size_m"][()])
    if obj.ndim == 3:
        obj = obj[0]
    return obj, positions_px, pixel_size_m

def lit_square(probe, threshold):
    """(row_slice, col_slice): square bbox of where sum_modes |probe|^2 exceeds `threshold`
    of its peak. Square because `fourier_ring_correlation` bins by integer FFT radius,
    which is only isotropic in frequency for a square image."""
    ill = (np.abs(probe.reshape(-1, *probe.shape[-2:])) ** 2).sum(0)
    rows, cols = np.where(ill > threshold * ill.max())
    side = max(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
    side += side % 2
    slices = []
    for lo, hi, n in ((rows.min(), rows.max(), ill.shape[0]), (cols.min(), cols.max(), ill.shape[1])):
        start = int(np.clip((lo + hi + 1) // 2 - side // 2, 0, n - side))
        slices.append(slice(start, start + side))
    return tuple(slices)


def crop_deramped(path, frame_index, out_shape, lit):
    """Load a step-scan recon, crop `out_shape` centered on ITS OWN
    `positions_px[frame_index]`, cut that down to `lit`, then remove the phase ramp/offset
    (fitted over the lit area only). Returns (corrected_crop, pixel_size_m)."""
    obj, positions_px, pixel_size_m = load_step_recon(path)
    position_rc = positions_px[frame_index]
    crop, center_rc = crop_at_position(obj, position_rc, out_shape)
    corrected, ramp, offset = remove_phase_ramp(crop[lit])
    print(f"  {path.parent.name}:\n"
          f"    frame {frame_index} position (row, col) = ({position_rc[0]:.2f}, "
          f"{position_rc[1]:.2f}) px -> crop center {center_rc} in {obj.shape} object\n"
          f"    pixel {pixel_size_m * 1e9:.4f} nm, ramp removed ({ramp[0]:+.2e}, "
          f"{ramp[1]:+.2e}) cyc/px, offset {offset:+.3f} rad")
    return corrected, pixel_size_m


#%% ---------------------------------------------------------------- load everything

print("=== loading the 3 single-frame reconstructions and the 4 step-scan crops at their area ===")

sf_full, sf_pixel_size = {}, {}
for label, path in singleframe_recons.items():
    with h5py.File(path, "r") as f:
        sf_full[label] = f["object"][()]
        sf_pixel_size[label] = float(f["obj_pixel_size_m"][()])

# All three probes give the same lit box today; take it from the no-band-limit recon.
with h5py.File(singleframe_recon_dir_nobl, "r") as f:
    sf_probe = f["probe"][()]
    window_shape = f["object"].shape
    sf_pixel_size_m = float(f["obj_pixel_size_m"][()])
lit = lit_square(sf_probe, illumination_threshold)
lit_px = lit[0].stop - lit[0].start
print(f"  FRC area: probe intensity > {illumination_threshold:g} of peak -> {lit_px} x {lit_px} px "
      f"= {lit_px * sf_pixel_size_m * 1e6:.2f} um, rows {lit[0].start}-{lit[0].stop - 1}, "
      f"cols {lit[1].start}-{lit[1].stop - 1} of the {window_shape} window")

sf_objects = {}
for label, obj in sf_full.items():
    sf_objects[label], sf_ramp, sf_offset = remove_phase_ramp(obj[lit])
    print(f"  {label}: {singleframe_recons[label].parent.name}\n"
          f"    pixel {sf_pixel_size[label] * 1e9:.4f} nm, ramp removed ({sf_ramp[0]:+.2e}, "
          f"{sf_ramp[1]:+.2e}) cyc/px, offset {sf_offset:+.3f} rad")

dir1_crop, dir1_pixel_size_m = crop_deramped(step_recon_dir1, frame_index, window_shape, lit)
dir2_crop, dir2_pixel_size_m = crop_deramped(step_recon_dir2, frame_index, window_shape, lit)
dir3_crop, dir3_pixel_size_m = crop_deramped(step_recon_dir3, frame_index, window_shape, lit)
dir4_crop, dir4_pixel_size_m = crop_deramped(step_recon_dir4, frame_index, window_shape, lit)
step_pixel_size_m = dir1_pixel_size_m
print(f"  step-scan pixel sizes (dir1/dir2/dir3/dir4): "
      f"{dir1_pixel_size_m * 1e9:.4f} / {dir2_pixel_size_m * 1e9:.4f} / "
      f"{dir3_pixel_size_m * 1e9:.4f} / {dir4_pixel_size_m * 1e9:.4f} nm")


#%% ---------------------------------------------------------------- group 1 = dir1 + dir4, group 2 = dir2 + dir3, align manually

print("group 1: registering dir4 onto dir1:")
# dir4_aligned = rotate(dir4_crop, 0, reshape=False)
dir4_aligned = np.roll(dir4_crop, -1,axis=1)
group1 = 0.5 * (dir1_crop + dir4_aligned)

print("group 2: registering dir3 onto dir2:")
dir3_aligned = np.roll(dir3_crop, -1,axis=1)
group2 = 0.5 * (dir2_crop + dir3_aligned)

#%% sanity check: do all groups show the same feature, before trusting the FRC?
# panels = [(group1, "dir1+dir4 (group 1)"), (group2, "dir2+dir3 (group 2)"), (dir1_crop, "dir1"), (dir2_crop, "dir2"), (dir3_crop, "dir3"), (dir4_crop, "dir4")]
# panels += [(obj, f"single-frame: {label}") for label, obj in sf_objects.items()]
# fig, axes = plt.subplots(2, len(panels), figsize=(3.8 * len(panels), 8))
# for col, (img, title) in enumerate(panels):
#     axes[0, col].imshow(np.abs(img), cmap="gray")
#     axes[0, col].set_title(f"{title}\n|O|", fontsize=9)
#     axes[1, col].imshow(np.angle(img), cmap="gray")
#     axes[1, col].set_title("phase", fontsize=9)
# for ax in axes.ravel():
#     ax.set_xticks([]), ax.set_yticks([])
# fig.suptitle(f"frame {frame_index} lit area ({lit_px} px): the reconstructions being compared")
# plt.tight_layout()
# plt.show()
#%% ---------------------------------------------------------------- the FRC curves for phase

print("computing the FRC curves:")

curves, labels, pixel_sizes = [], [], []
sf_aligned = np.zeros_like(dir3_crop,shape=(3,408,408))   # same crop size for all single-frame reconstructions
item_iterator = iter(enumerate(sf_objects.items(), start=1))

i, (label, sf_object) = next(item_iterator)
print(f"  {i}) dir3 vs {label}:")
sf_aligned[0,:,:] = np.roll(sf_object,(2,2),axis=(0,1))
plt.figure()
plt.imshow(np.abs(dir3_crop - sf_aligned[0,:,:]))

freq, frc, n_per_ring = fourier_ring_correlation(np.angle(dir3_crop), np.angle(sf_aligned[0,:,:]),
                                                    pixel_size_m=sf_pixel_size[label])
curves.append(frc), labels.append(label), pixel_sizes.append(sf_pixel_size[label])

i, (label, sf_object) = next(item_iterator)
print(f"  {i}) dir3 vs {label}:")
sf_aligned[1,:,:] = np.roll(sf_object,(1,3),axis=(0,1))
freq, frc, n_per_ring = fourier_ring_correlation(np.angle(dir3_crop), np.angle(sf_aligned[1,:,:]),
                                                    pixel_size_m=sf_pixel_size[label])
curves.append(frc), labels.append(label), pixel_sizes.append(sf_pixel_size[label])

i, (label, sf_object) = next(item_iterator)
print(f"  {i}) dir3 vs {label}:")
sf_aligned[2,:,:] = np.roll(sf_object,(1,1),axis=(0,1))
freq, frc, n_per_ring = fourier_ring_correlation(np.angle(dir3_crop), np.angle(sf_aligned[2,:,:]),
                                                    pixel_size_m=sf_pixel_size[label])
curves.append(frc), labels.append(label), pixel_sizes.append(sf_pixel_size[label])


# plt.figure()
# plt.imshow(np.abs(dir3_crop - sf_aligned[0,:,:]))


print(f"  {len(curves) + 1}) dir1+dir4 vs dir2+dir3 (step-scan resolution ceiling):")
group2_aligned= np.roll(group2, (-1,-1),axis=(0,1))
# plt.figure()
# plt.imshow(np.abs(group1 - group2_aligned))
_,frc_step,_ = fourier_ring_correlation(np.angle(group1), np.angle(group2_aligned),
                                          pixel_size_m=step_pixel_size_m)
curves.append(frc_step), labels.append("Step-scan"), pixel_sizes.append(step_pixel_size_m)
colors = singleframe_colors + [step_scan_color]

half_bit = frc_half_bit_threshold(n_per_ring)   # same crop size for all -> same n_per_ring

# Paper figure: single-column width, type sized for print at that width.
paper_rc = {"font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
            "legend.fontsize": 8.5, "lines.linewidth": 1.8, "figure.dpi": 300}
def paper_frc_figure(keep, legend_kwargs, figsize=(3.5, 3.0)): #figsize=(3.4, 2.7)
    """One paper FRC figure with only the curves whose label is in `keep`; each run keeps
    its color across figures."""
    idx = [labels.index(k) for k in keep]
    with plt.rc_context(paper_rc):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_frc(
            freq, [curves[i] for i in idx], threshold=half_bit, labels=[labels[i] for i in idx],
            ax=ax, colors=[colors[i] for i in idx], resolution_in_label=False,
            top_tick_spacing=0.11, legend_kwargs=legend_kwargs,
        )
        plt.xlim(0,20)
        plt.show()


legend_inside = {"frameon": False, "fontsize": 7.5, "loc": "upper right"}

print("FRC half-bit crossings, figure 1 (the three single-frame runs):")
paper_frc_figure(list(singleframe_recons), legend_inside)

print("FRC half-bit crossings, figure 2 (step-scan vs. band-limit relax):")
# Legend below the axes: inside, it runs into the step-scan curve.
legend_below = {"frameon": False, "fontsize": 7.5, "loc": "upper center", "bbox_to_anchor": (0.5, -0.2),
                "ncol": 2, "columnspacing": 1.0, "handlelength": 1.6}
# paper_frc_figure(["Band-limit relax", "Step-scan"], legend_below, figsize=(3.4, 3.0))
paper_frc_figure(["Band-limit relax", "Step-scan"], legend_inside, figsize=(3.4, 3.0))

#%% FRC rings: where each curve's resolution cutoff sits in Fourier space, on group 1's
# own power spectrum (group 1 is the common image in every curve above).
radii = [frc_crossing_px(freq, frc, half_bit, group1.shape, px)
         for frc, px in zip(curves, pixel_sizes)]
plot_frc_rings(
    np.angle(group1), radii, labels=labels, colors=colors,
    log_floor=1e-10, title=f"FRC rings at frame {frame_index}'s lit area (dir1+dir4 power spectrum)",
)
# plt.show()

#%% plot the reonstruction strategy with loss functions
# extract the loss function from the RPI single-frame recon's h5 file.
# plot the double y-axis: loss function vs. iteration, and the band-limit ratio vs. iteration.
#
# The h5 doesn't store the band-limit schedule, so it's rebuilt from rpi_params.json with
# the same rule as `_schedule_size` in RPI_SS_recon_50nmZP_blrelax.py. kp comes from the
# probe the run started from; the cap from the saved object_lowres size (the script's
# rpi_relax_max_ratio isn't in the json).

def relax_schedule(n_init, n_cap, freq, start, end, n_epochs):
    """Per-iteration n_obj_lowres: grow from n_init to n_cap in (end - start) // freq even
    steps, reach n_cap by iteration end - 1, hold n_cap outside [start, end)."""
    n_events = max((end - start) // freq, 1)
    n_obj = np.full(n_epochs, n_cap, dtype=int)
    for epoch in range(max(start, 0), min(end - 1, n_epochs)):
        frac = min(((epoch - start) // freq) / n_events, 1.0)
        n_obj[epoch] = int(2 * round((n_init + (n_cap - n_init) * frac) / 2))
    return n_obj


strategy = {}      # label -> {"loss": per-iteration loss, "ratio": per-iteration R or None}
init_probes = {}   # all three runs start from the same probe; load it once
for label, path in singleframe_recons.items():
    params = json.load(open(path.parent / "rpi_params.json"))
    with h5py.File(path, "r") as f:
        loss = f["loss"][()]
        n_obj_final = f["object_lowres"].shape[-1]
    probe_key = (params["path_to_init_recon"], params["n_probe_modes"])
    if probe_key not in init_probes:
        # rpi_params.json was written on Linux: map /mnt/micdataN/... onto the \\micdata\dataN share
        init_recon = Path(probe_key[0].replace("/mnt/micdata", r"\\micdata\data", 1))
        with h5py.File(init_recon, "r") as f:
            init_probes[probe_key] = torch.as_tensor(f["probe"][0, :probe_key[1]])
    kp = probe_fourier_radius(init_probes[probe_key], params["rpi_kp_quantile"])
    if params["rpi_relaxation_freq"]:
        n_obj = relax_schedule(2 * round(params["rpi_resolution_ratio"] * kp), n_obj_final,
                               params["rpi_relaxation_freq"], params["rpi_relax_start"],
                               params["rpi_relax_end"], len(loss))
    else:
        n_obj = np.full(len(loss), n_obj_final)
    ratio = None if n_obj_final >= params["n_dp"] else n_obj / 2 / kp
    strategy[label] = {"loss": loss, "ratio": ratio}
    r_text = "no band limit" if ratio is None else f"R {ratio[0]:.2f} -> {ratio[-1]:.2f}"
    print(f"{label}: {len(loss)} iterations, kp = {kp} px, grid {n_obj[0]} -> {n_obj[-1]} px "
          f"({r_text}), loss {loss[0]:.4f} -> {loss[-1]:.4f}")

ratio_label = r"$R = k_o/k_p$"
iteration_xmax = 10000   # every run has done ~99% of its loss decrease by here


def set_iteration_axis(ax, scale):
    """Square-root ("sqrt") or natural-log ("ln") iteration axis, cropped at iteration_xmax.
    A log base only moves the ticks; sqrt is what spreads the early iterations."""
    if scale == "sqrt":
        ax.set_xscale("function", functions=(lambda x: np.sqrt(np.clip(x, 0, None)), np.square))
        ax.set_xlim(0, iteration_xmax)
        ax.set_xticks([0, 100, 500, 1000, 2000, 5000, 10000])
        # ax.set_xticks([0, 100, 500, 1000, 2000, 5000])
    else:
        ax.set_xscale("log", base=np.e)
        ax.set_xlim(1, iteration_xmax)
        ax.xaxis.set_major_locator(FixedLocator(np.exp(np.arange(0, 9, 2))))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"$e^{{{np.log(v):.0f}}}$"))
        ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Iteration")


def plot_strategy_stacked(scale):
    """Two stacked panels sharing the iteration axis: loss on top, R below."""
    with plt.rc_context(paper_rc):
        fig, (ax_loss, ax_ratio) = plt.subplots(
            2, 1, figsize=(4.4, 3.0), sharex=True, constrained_layout=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        for (label, run), color in zip(strategy.items(), singleframe_colors):
            iterations = np.arange(1, len(run["loss"]) + 1)
            ax_loss.semilogy(iterations, run["loss"], color=color, label=label)
            if run["ratio"] is not None:
                ax_ratio.plot(iterations, run["ratio"], color=color, drawstyle="steps-post")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(frameon=False, loc="upper right")
        ax_ratio.set_ylabel(ratio_label)
        ax_ratio.set_ylim(0.3, 1.0)
        set_iteration_axis(ax_ratio, scale)   # shared x: the loss panel follows
        plt.show()


def plot_strategy_dual(scale):
    """Double y-axis: loss (solid) on the left, band-limit ratio (dashed) on the right."""
    with plt.rc_context(paper_rc):
        fig, ax_loss = plt.subplots(figsize=(3.4, 3.2), constrained_layout=True)
        ax_ratio = ax_loss.twinx()
        loss_lines, ratio_lines = [], []
        for (label, run), color in zip(strategy.items(), singleframe_colors):
            iterations = np.arange(1, len(run["loss"]) + 1)
            loss_lines += ax_loss.semilogy(iterations, run["loss"], color=color, label=label)
            if run["ratio"] is not None:
                ratio_lines += ax_ratio.plot(iterations, run["ratio"], color=color, ls="--",
                                             drawstyle="steps-post", label=f"{label}, R")
        ax_loss.set_ylabel("Loss")
        set_iteration_axis(ax_loss, scale)   # the twinx R axis shares it
        ax_ratio.set_ylabel(ratio_label)
        ax_ratio.set_ylim(0.3, 1.0)
        ax_loss.legend(handles=loss_lines + ratio_lines, frameon=False, loc="upper center",
                       bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0, handlelength=1.6)
        plt.show()


for scale in ("sqrt", "ln"):
    print(f"strategy figure: stacked panels, {scale} iteration axis")
    plot_strategy_stacked(scale)
    print(f"strategy figure: double y-axis, {scale} iteration axis")
    plot_strategy_dual(scale)

#%% only plot the loss curve of the band-limit relax run, with the RPI relaxation schedule overlaid
# def plot_run_strategy(label, scale="sqrt"):
#     """One run alone: loss (solid, left axis) with its band-limit ratio R (dashed, right
#     axis) overlaid. A run with no band limit has no R, so it gets the loss only.
#     `scale` is "sqrt" or "ln" for the iteration axis."""
#     run = strategy[label]
#     color = singleframe_colors[list(singleframe_recons).index(label)]
#     iterations = np.arange(1, len(run["loss"]) + 1)
#     print(f"loss figure: {label}, {scale} iteration axis")
#     with plt.rc_context(paper_rc):
#         fig, ax_loss = plt.subplots(figsize=(4.4, 2.6), constrained_layout=True)
#         line_loss, = ax_loss.semilogy(iterations, run["loss"], color=color, label="Loss")
#         ax_loss.set_ylabel("Loss")
#         set_iteration_axis(ax_loss, scale)
#         if run["ratio"] is not None:
#             ax_ratio = ax_loss.twinx()
#             line_ratio, = ax_ratio.plot(iterations, run["ratio"], color="#52514e", ls="--",
#                                         drawstyle="steps-post", label=ratio_label)
#             ax_ratio.set_ylabel(ratio_label)
#             ax_ratio.set_ylim(0.3, 1.0)
#             ax_loss.legend(handles=[line_loss, line_ratio], frameon=False, loc="center right")
#         plt.show()


# plot_run_strategy("Band-limit relax")

#%% phase line cuts: phase (rad) vs distance (nm) along one line through the lit area
# On dir3_crop's grid: every sf_aligned[k] was rolled onto it in the FRC section, so the
# same pixel coordinates pick out the same feature in every image.
cut_start_xy, cut_end_xy = (257, 164), (268, 192)   # (x, y) = (col, row), as imshow's cursor reads
cut_src_rc, cut_dst_rc = cut_start_xy[::-1], cut_end_xy[::-1]   # profile_line takes (row, col)
cut_length_px = np.hypot(cut_dst_rc[0] - cut_src_rc[0], cut_dst_rc[1] - cut_src_rc[1])


def phase_line_cut(img, pixel_size_m):
    """(distance_nm, phase_rad) from cut_start_xy to cut_end_xy, bilinear-interpolated;
    distance is measured from the start point."""
    phase = profile_line(np.angle(img), cut_src_rc, cut_dst_rc, order=1)
    distance_nm = np.linspace(0, cut_length_px * pixel_size_m * 1e9, len(phase))
    return distance_nm, phase


line_cuts = {"Step-scan": (dir3_crop, dir3_pixel_size_m, step_scan_color)}   # label -> (image, pixel size, color)
for k, label in enumerate(singleframe_recons):
    line_cuts[label] = (sf_aligned[k], sf_pixel_size[label], singleframe_colors[k])
print(f"line cut {cut_start_xy} -> {cut_end_xy} (x, y): {cut_length_px:.1f} px = "
      f"{cut_length_px * dir3_pixel_size_m * 1e9:.1f} nm at dir3's pixel; pixel sizes "
      + ", ".join(f"{label} {px * 1e9:.4f} nm" for label, (_, px, _) in line_cuts.items()))

# where the cut runs, on dir3's phase; the dot marks x = 0
with plt.rc_context(paper_rc):
    fig, ax = plt.subplots(figsize=(3.4, 3.4), constrained_layout=True)
    ax.imshow(np.angle(dir3_crop), cmap="gray")
    ax.plot([cut_start_xy[0], cut_end_xy[0]], [cut_start_xy[1], cut_end_xy[1]], color=step_scan_color)
    # ax.plot(*cut_start_xy, "o", color=step_scan_color, markersize=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("Line cut on step-scan (dir3) phase")
    plt.show()


def paper_line_cut_figure(keep, legend_kwargs, figsize=(3.4, 2.7)):
    """One paper line-cut figure with only the runs whose label is in `keep`; each run
    keeps its FRC-figure color."""
    with plt.rc_context(paper_rc):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        for label in keep:
            if label == 'Step-scan':
                img, pixel_size_m, color = line_cuts[label]
                ax.plot(*phase_line_cut(img, pixel_size_m), ".-",color=color, label=label)
            else:
                img, pixel_size_m, color = line_cuts[label]
                ax.plot(*phase_line_cut(img, pixel_size_m), ".--",color=color, label=label)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("Phase (rad)")
        ax.legend(**legend_kwargs)
        plt.show()

paper_line_cut_figure(['Band-limited', 'Band-limit relax', 'No band limit', 'Step-scan'], legend_below)

paper_line_cut_figure(["Band-limit relax", "Step-scan"], legend_inside)
paper_line_cut_figure(list(singleframe_recons), legend_inside)

# %%
# plot_run_strategy("No band limit")   # no band limit -> loss only
plot_run_strategy("Band-limited")
# %%

sf_aligned[0,:,:] = np.roll(sf_object,(1,2),axis=(0,1))
plt.figure()
plt.imshow(np.abs(dir3_crop - sf_aligned[0,:,:]))