"""Propagate a reconstructed ptychography probe along z and explore the caustic.

Uses pty-chi's ``AngularSpectrumPropagator`` (ptychi/propagate.py) to move the
reconstructed probe to a stack of planes around the sample, then opens an
interactive window:

    +-----------------+-----------------+
    |  I(x, z)        |  widths / peak  |
    +-----------------+-----------------+
    |  I(x, y)        |  I(z, y)        |
    +-----------------+-----------------+
                 [ z slider ]

Every panel is the intensity summed incoherently over the probe modes selected
in ``MODES``. Set ``DISPLAY`` to "complex", "amplitude" or "phase" instead to
inspect the coherent field of a single mode.

Drag the slider to step through z. Keys: left/right = one plane, up/down = five
planes, home/end = the ends of the scan, ``f`` = jump to the brightest plane.

Real-space pixel size and wavelength come from the ``*_para.hdf5`` file
referenced by ``pear_params.json`` -- ``det_pixel_size_m`` in the recon params
is not trustworthy -- and are cross-checked against ``obj_pixel_size_m`` in the
recon file.

Runs top to bottom as a script, or cell by cell (``#%%``) in VS Code. The
backend is chosen automatically: ipympl in a notebook kernel (works over plain
SSH, no X server), TkAgg when a desktop session is present, otherwise Agg with
the figure written to a PNG.
"""

# %% -- setup ---------------------------------------------------------------
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import json
import re
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch


def select_backend():
    """Pick an interactive matplotlib backend that works wherever this is run.

    Hard-coding TkAgg fails over plain SSH with "Cannot load backend 'TkAgg'
    ... as 'headless' is currently running", because Tk needs an X display and
    there is none. The three cases that matter here:

    * VS Code interactive window / Jupyter kernel -> ipympl, which draws into
      the notebook and needs no X server. This is the SSH-friendly one.
    * A real desktop session (DISPLAY or WAYLAND_DISPLAY set) -> TkAgg.
    * Neither -> Agg, and the figure gets written to a PNG instead of shown.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except ImportError:
        shell = None

    if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
        try:
            import ipympl  # noqa: F401
        except ImportError:
            print("Running in a kernel but ipympl is missing -- the slider will be dead.\n"
                  "  Fix with:  pip install ipympl   (then restart the kernel)")
        else:
            shell.run_line_magic("matplotlib", "widget")
            return matplotlib.get_backend(), True

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("TkAgg")
        return "TkAgg", True

    matplotlib.use("Agg")
    print("No display and no notebook kernel -- falling back to a static PNG.\n"
          "  For the slider, run this in the VS Code interactive window (#%% cells),\n"
          "  or from a terminal inside a remote-desktop session with DISPLAY set.")
    return "Agg", False


BACKEND, INTERACTIVE = select_backend()

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import Slider

from ptychi.propagate import AngularSpectrumPropagator, WavefieldPropagatorParameters

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
RECON_FILE = (
    Path("/mnt/micdata2/4IDD/2026_Sep/ptychi_recons/batch_JD/S0197")
    / "Ndp128_LSQML_c500_m0.5_gaussian_p10_cp_pc500_f_ul2"
    / "recon_Niter2000.h5"
)

Z_RANGE_UM = (-200.0, 700.0)  # propagation distances to scan (focus sits near +475 um)
N_Z = 91                      # number of planes
PAD = 2                       # zero-pad factor before propagating (avoids wrap-around)
MODES = "all"                 # probe modes to add incoherently; "all", or a list like [0, 1, 2]
DISPLAY = "intensity"         # "intensity" sums all MODES; "complex"/"amplitude"/"phase" show MODES[0]
NORMALIZE = "per_plane"       # transverse panel scaling: "per_plane" or "global"
CUT = "sum"                   # x-z / y-z panels: "sum" over the third axis, or "center" line cut
FOV_UM = None                 # zoom the transverse panel, e.g. 4.0; None = full padded FOV
DEVICE = "cpu"                # torch device for the FFTs (these grids are tiny)


# %% -- loading -------------------------------------------------------------
def read_params(recon_file):
    params_file = Path(recon_file).parent / "pear_params.json"
    if not params_file.is_file():
        return {}
    with open(params_file) as fh:
        return json.load(fh)


def sampling_from_para_file(recon_file, n_dp):
    """Get (dx, lambda) from the processed ``*_para.hdf5`` written by the preprocessor.

    The para file stores ``dx`` for the diffraction-pattern size it was built
    with (``Ndp<N>`` in its name). A recon that crops to a smaller pattern has a
    proportionally coarser real-space pixel, so rescale by the size ratio.
    """
    para_path = read_params(recon_file).get("path_to_processed_hdf5_pos", "")
    if not para_path or not Path(para_path).is_file():
        print(f"  note: para file not readable ({para_path or 'not listed'})")
        return None, None

    with h5py.File(para_path, "r") as f:
        dx = float(np.ravel(f["dx"][()])[0])
        lam = float(np.ravel(f["lambda"][()])[0]) if "lambda" in f else None

    match = re.search(r"Ndp(\d+)", Path(para_path).name)
    if match and int(match.group(1)) != n_dp:
        n_para = int(match.group(1))
        print(f"  para file built at Ndp{n_para}, recon at Ndp{n_dp}: scaling dx by {n_para / n_dp:g}")
        dx *= n_para / n_dp
    return dx, lam


def load_probe(recon_file):
    """Read the probe and its sampling from a pty-chi recon file.

    Returns
    -------
    probe : (n_modes, h, w) complex64 ndarray
    dx_m : float
        Real-space pixel size of the probe/object grid.
    wavelength_m : float
    """
    with h5py.File(recon_file, "r") as f:
        probe = f["probe"][()]
        dx_h5 = float(f["obj_pixel_size_m"][()]) if "obj_pixel_size_m" in f else None
    # (n_opr_modes, n_probe_modes, h, w) -> keep the main OPR mode only
    while probe.ndim > 3:
        probe = probe[0]

    dx_para, lam_para = sampling_from_para_file(recon_file, n_dp=probe.shape[-1])

    dx_m = dx_para if dx_para is not None else dx_h5
    if dx_m is None:
        raise RuntimeError(f"No pixel size found for {recon_file}")
    if dx_para is not None and dx_h5 is not None:
        rel = abs(dx_para - dx_h5) / dx_h5
        print(f"  dx (para file) = {dx_para * 1e9:.4f} nm")
        print(f"  dx (recon h5)  = {dx_h5 * 1e9:.4f} nm  ({rel:+.2%})")
        if rel > 0.01:
            print("  WARNING: the two pixel sizes disagree; using the para-file value.")

    params = read_params(recon_file)
    wavelength_m = lam_para
    if wavelength_m is None and "wavelength_m" in params:
        wavelength_m = float(params["wavelength_m"])
    if wavelength_m is None and "beam_energy_kev" in params:
        wavelength_m = 1.23984193e-9 / float(params["beam_energy_kev"])
    if wavelength_m is None:
        raise RuntimeError(f"No wavelength found for {recon_file}")
    return probe.astype(np.complex64), float(dx_m), float(wavelength_m)


# %% -- propagation ---------------------------------------------------------
def make_params(wavelength_m, h, w, dx_m, z_m):
    return WavefieldPropagatorParameters.create_simple(
        wavelength_m=wavelength_m,
        width_px=w,
        height_px=h,
        pixel_width_m=dx_m,
        pixel_height_m=dx_m,
        propagation_distance_m=z_m,
    )


def propagate_stack(probe, dx_m, wavelength_m, z_m, pad=2, device="cpu"):
    """Propagate every probe mode to every plane in ``z_m``.

    Returns
    -------
    fields : (nz, n_modes, H, W) complex64 ndarray
        H, W include the zero padding.
    """
    field = torch.from_numpy(probe).to(device)
    if pad > 1:
        h, w = field.shape[-2:]
        ph, pw = (pad - 1) * h // 2, (pad - 1) * w // 2
        field = torch.nn.functional.pad(field, (pw, pw, ph, ph))
    h, w = field.shape[-2:]

    z_crit = min(h, w) * dx_m**2 / wavelength_m
    print(f"  grid {h}x{w} px, angular-spectrum critical distance |z| < {z_crit * 1e6:.1f} um")
    if np.abs(z_m).max() > z_crit:
        print("  WARNING: requested |z| exceeds the critical distance -- the transfer")
        print("           function is undersampled there and the result will alias.")

    # One propagator, retuned per plane: the transfer function is the only
    # z-dependent piece and `update()` rebuilds just that.
    propagator = AngularSpectrumPropagator(make_params(wavelength_m, h, w, dx_m, float(z_m[0])))

    fields = np.empty((len(z_m), *field.shape), dtype=np.complex64)
    for i, z in enumerate(z_m):
        propagator.update(make_params(wavelength_m, h, w, dx_m, float(z)))
        # The transfer function is held as two plain (unregistered) tensors so
        # that DataParallel does not choke on a complex buffer -- which also
        # means `.to()` on the module would not move it. Move it by hand.
        propagator._transfer_function_real = propagator._transfer_function_real.to(device)
        propagator._transfer_function_imag = propagator._transfer_function_imag.to(device)
        fields[i] = propagator.propagate_forward(field).cpu().numpy()
    return fields


# %% -- analysis / display helpers ------------------------------------------
def fwhm(coord, profile):
    """Full width at half maximum of a 1D profile, by linear interpolation."""
    p = np.asarray(profile, dtype=float)
    p = p - p.min()
    if p.max() <= 0:
        return np.nan
    p /= p.max()
    ipk = int(np.argmax(p))
    left = np.flatnonzero(p[: ipk + 1] <= 0.5)
    right = np.flatnonzero(p[ipk:] <= 0.5)
    if left.size == 0 or right.size == 0:
        return np.nan
    il, ir = left[-1], right[0] + ipk
    xl = np.interp(0.5, [p[il], p[il + 1]], [coord[il], coord[il + 1]])
    xr = np.interp(0.5, [p[ir], p[ir - 1]], [coord[ir], coord[ir - 1]])
    return abs(xr - xl)


def complex_to_rgb(field, vmax, gamma=0.6):
    """Domain colouring: hue = phase, brightness = amplitude."""
    amp = np.clip(np.abs(field) / (vmax + 1e-30), 0, 1) ** gamma
    hue = (np.angle(field) + np.pi) / (2 * np.pi)
    return hsv_to_rgb(np.stack([hue, np.ones_like(amp), amp], axis=-1))


# %% -- load and propagate --------------------------------------------------
print(f"Loading {RECON_FILE}")
probe_all, dx_m, wavelength_m = load_probe(RECON_FILE)
mode_power = (np.abs(probe_all) ** 2).sum(axis=(-1, -2))
print(f"  probe {probe_all.shape}, mode powers {np.round(mode_power / mode_power.sum(), 4)}")
print(f"  dx = {dx_m * 1e9:.4f} nm, lambda = {wavelength_m * 1e12:.4f} pm "
      f"({1.23984193e-9 / wavelength_m:.4f} keV)")

modes = list(range(probe_all.shape[0])) if MODES == "all" else list(MODES)
probe = probe_all[modes]
print(f"  using modes {modes} ({mode_power[modes].sum() / mode_power.sum():.1%} of the power)")
if len(modes) > 1 and DISPLAY != "intensity":
    print(f'  note: the caustic panels add all {len(modes)} modes incoherently, but '
          f'DISPLAY="{DISPLAY}" can only\n        show one coherent field -- it draws mode '
          f'{modes[0]}. Use DISPLAY="intensity" for the sum.')

z_um = np.linspace(Z_RANGE_UM[0], Z_RANGE_UM[1], N_Z)
print(f"Propagating {N_Z} planes over z = [{Z_RANGE_UM[0]:g}, {Z_RANGE_UM[1]:g}] um")
fields = propagate_stack(probe, dx_m, wavelength_m, z_um * 1e-6, pad=PAD, device=DEVICE)

# transverse coordinates of the padded grid, centred on the probe centre
ny, nx = fields.shape[-2:]
x_um = (np.arange(nx) - nx // 2) * dx_m * 1e6
y_um = (np.arange(ny) - ny // 2) * dx_m * 1e6

intensity = (np.abs(fields) ** 2).sum(axis=1)  # incoherent sum over modes -> (nz, ny, nx)
main_mode = fields[:, 0]  # first selected mode, the only one with a meaningful phase map

i_xz = intensity.sum(axis=1) if CUT == "sum" else intensity[:, ny // 2, :]
i_yz = intensity.sum(axis=2) if CUT == "sum" else intensity[:, :, nx // 2]

peak = intensity.max(axis=(1, 2))
width_x = np.array([fwhm(x_um, row) for row in i_xz])
width_y = np.array([fwhm(y_um, row) for row in i_yz])
i_focus = int(np.nanargmax(peak))
print(f"  brightest plane: z = {z_um[i_focus]:+.2f} um, "
      f"FWHM = {width_x[i_focus]:.3f} x {width_y[i_focus]:.3f} um")

int_max = float(intensity.max())
# Brightness scale for the transverse panel. The peak intensity swings by orders
# of magnitude through focus, so a single global scale leaves most planes black;
# "per_plane" renormalises each one instead.
if NORMALIZE == "per_plane":
    amp_scale = np.abs(main_mode).max(axis=(1, 2))
    int_scale = intensity.max(axis=(1, 2))
else:
    amp_scale = np.full(N_Z, np.abs(main_mode).max())
    int_scale = np.full(N_Z, int_max)


# %% -- interactive figure --------------------------------------------------
fig = plt.figure(figsize=(11.5, 7.8))
gs = fig.add_gridspec(
    2, 3, width_ratios=[1, 1, 0.035], height_ratios=[1, 1.5],
    left=0.07, right=0.92, top=0.90, bottom=0.14, wspace=0.28, hspace=0.22,
)
ax_xz = fig.add_subplot(gs[0, 0])
ax_w = fig.add_subplot(gs[0, 1])
ax_img = fig.add_subplot(gs[1, 0], sharex=ax_xz)
ax_yz = fig.add_subplot(gs[1, 1], sharey=ax_img)
cax = fig.add_subplot(gs[1, 2])

ext_x = [x_um[0], x_um[-1]]
ext_y = [y_um[0], y_um[-1]]
ext_z = [z_um[0], z_um[-1]]

# --- x vs z ----------------------------------------------------------------
ax_xz.imshow(i_xz, extent=[*ext_x, ext_z[1], ext_z[0]], origin="upper",
             aspect="auto", cmap="inferno")
ax_xz.set_ylabel("z (um)")
ax_xz.set_title(f"I(x, z)   [{CUT} over y]", fontsize=10)
ax_xz.axhline(z_um[i_focus], color="w", lw=0.6, ls=":")
hline = ax_xz.axhline(z_um[0], color="cyan", lw=1.0)
ax_xz.tick_params(labelbottom=False)

# --- y vs z ----------------------------------------------------------------
ax_yz.imshow(i_yz.T, extent=[*ext_z, ext_y[1], ext_y[0]], origin="upper",
             aspect="auto", cmap="inferno")
ax_yz.set_xlabel("z (um)")
ax_yz.set_title(f"I(z, y)   [{CUT} over x]", fontsize=10)
ax_yz.axvline(z_um[i_focus], color="w", lw=0.6, ls=":")
vline = ax_yz.axvline(z_um[0], color="cyan", lw=1.0)
ax_yz.tick_params(labelleft=False)

# --- width / peak vs z ------------------------------------------------------
ax_w.plot(z_um, width_x, color="tab:blue", label="FWHM x")
ax_w.plot(z_um, width_y, color="tab:orange", label="FWHM y")
ax_w.set_xlabel("z (um)")
ax_w.set_ylabel("FWHM (um)")
ax_w.set_title("beam width and peak intensity", fontsize=10)
ax_w.legend(fontsize=8, loc="upper left")
ax_w.grid(alpha=0.3)
ax_peak = ax_w.twinx()
ax_peak.plot(z_um, peak / int_max, color="tab:green", lw=1, ls="--")
ax_peak.set_ylabel("peak I (norm.)", color="tab:green")
ax_peak.tick_params(axis="y", labelcolor="tab:green")
wline = ax_w.axvline(z_um[0], color="k", lw=1.0)

# --- transverse plane -------------------------------------------------------
if DISPLAY == "complex":
    frames, clim = None, None
    im = ax_img.imshow(complex_to_rgb(main_mode[0], amp_scale[0]),
                       extent=[*ext_x, ext_y[1], ext_y[0]], origin="upper",
                       interpolation="nearest")
    hue_strip = hsv_to_rgb(np.stack([np.linspace(0, 1, 256)[:, None].repeat(8, 1),
                                     np.ones((256, 8)), np.ones((256, 8))], axis=-1))
    cax.imshow(hue_strip, extent=[0, 1, -np.pi, np.pi], aspect="auto", origin="lower")
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_yticks([-np.pi, 0, np.pi], ["-pi", "0", "pi"])
    cax.set_ylabel(f"phase of mode {modes[0]}  (brightness = |psi|)", fontsize=8)
else:
    frames, cmap, clim, cbar_label = {
        "intensity": (intensity, "inferno", int_scale,
                      f"intensity (sum of {len(modes)} mode{'s' if len(modes) > 1 else ''})"),
        "amplitude": (np.abs(main_mode), "inferno", amp_scale, f"|psi| (mode {modes[0]})"),
        "phase": (np.angle(main_mode), "twilight", None, f"phase (mode {modes[0]})"),
    }[DISPLAY]
    im = ax_img.imshow(frames[0], extent=[*ext_x, ext_y[1], ext_y[0]], origin="upper",
                       cmap=cmap, interpolation="nearest",
                       vmin=-np.pi if clim is None else 0,
                       vmax=np.pi if clim is None else clim[0])
    fig.colorbar(im, cax=cax, label=cbar_label)

ax_img.set_xlabel("x (um)")
ax_img.set_ylabel("y (um)")
ax_img.set_aspect("equal")
if FOV_UM:
    ax_img.set_xlim(-FOV_UM / 2, FOV_UM / 2)
    ax_img.set_ylim(FOV_UM / 2, -FOV_UM / 2)
img_title = ax_img.set_title("", fontsize=10)

fig.suptitle(
    f"{RECON_FILE.parent.name} / {RECON_FILE.name}\n"
    f"angular-spectrum propagation   dx = {dx_m * 1e9:.2f} nm, "
    f"lambda = {wavelength_m * 1e12:.2f} pm, modes {modes}",
    fontsize=10,
)


def show_plane(iz):
    """Redraw everything that depends on the selected z index."""
    iz = int(np.clip(iz, 0, N_Z - 1))
    if frames is None:
        im.set_data(complex_to_rgb(main_mode[iz], amp_scale[iz]))
    else:
        im.set_data(frames[iz])
        if clim is not None:
            im.set_clim(0, clim[iz])
    hline.set_ydata([z_um[iz]] * 2)
    vline.set_xdata([z_um[iz]] * 2)
    wline.set_xdata([z_um[iz]] * 2)
    img_title.set_text(
        f"z = {z_um[iz]:+8.2f} um     FWHM = {width_x[iz]:.3f} x {width_y[iz]:.3f} um"
        f"     peak I = {peak[iz] / int_max:.3f}"
    )
    fig.canvas.draw_idle()


slider = Slider(
    fig.add_axes([0.10, 0.04, 0.72, 0.03]),
    "z (um)", z_um[0], z_um[-1], valinit=z_um[0], valstep=z_um,
)
slider.on_changed(lambda v: show_plane(np.argmin(np.abs(z_um - v))))


def on_key(event):
    """Left/right arrows step one plane; home/end jump to the ends."""
    iz = int(np.argmin(np.abs(z_um - slider.val)))
    step = {"left": -1, "right": 1, "down": -5, "up": 5}.get(event.key)
    if step is not None:
        slider.set_val(z_um[int(np.clip(iz + step, 0, N_Z - 1))])
    elif event.key == "home":
        slider.set_val(z_um[0])
    elif event.key == "end":
        slider.set_val(z_um[-1])
    elif event.key == "f":  # jump to the brightest plane
        slider.set_val(z_um[i_focus])


fig.canvas.mpl_connect("key_press_event", on_key)
show_plane(0)

if hasattr(fig.canvas, "header_visible"):
    # ipympl only: trim the notebook canvas chrome so the figure gets the full cell.
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

if INTERACTIVE:
    plt.show()
else:
    png = RECON_FILE.parent / "beam_propagation.png"
    show_plane(i_focus)
    fig.savefig(png, dpi=110)
    print(f"Wrote {png}")


#%%
print("Done.")