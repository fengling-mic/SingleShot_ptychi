import json
import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import tifffile
except ImportError:                     # only the preview writers need it
    tifffile = None


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


# ---------------------------------------------------------------------------
# probe synthesis
#
# Two ways to build a probe, picked with `plane`:
#
#   plane="sample"  the aperture is drawn directly on the sample grid (dx =
#                   pixel_size_m). Use it for unfocused beams: a pinhole, a
#                   Gaussian of a known size, a slit.
#
#   plane="pupil"   the aperture is drawn on the *pupil* grid -- the optic
#                   (zone plate, KB aperture) as seen at distance
#                   focal_length_m -- and the probe is its Fourier transform.
#                   This is the way to get a focused beam: the Airy rings, the
#                   donut from a central stop and the spot size all come out
#                   right, and the sampling is exact however long the focal
#                   length is. The pupil grid has du = lambda f / (N dx), so
#                   the optic must be smaller than lambda f / dx across.
#
# Defocus is one angular-spectrum step in both cases, applied to the same
# frequency grid, so `defocus_m` means "sample is this far downstream of the
# focus" either way.
# ---------------------------------------------------------------------------


def gaussian2D(X, Y, FWHM):
    """2D Gaussian *amplitude* with the given full-width-half-max.

    :param X: 2D x meshgrid
    :param Y: 2D y meshgrid
    :param FWHM: full-width-half-max, in the units of X/Y
    :return: 2D Gaussian profile, 0.5 at radius FWHM/2
    """
    return np.exp(-(4 * np.log(2)) * (X**2 + Y**2) / FWHM**2)


def circ(X, Y, D):
    """Binary disk of diameter D on a 2D grid.

    :param X: 2D x meshgrid
    :param Y: 2D y meshgrid
    :param D: diameter, in the units of X/Y
    :return: binary 2D array
    """
    return (X**2 + Y**2) < (D / 2) ** 2


def rect(arr, threshold=0.5):
    """Binary rect: True where |arr| < threshold. Feed it X/width.

    :param arr: coordinate array, normally X or Y divided by the wanted width
    :param threshold: half-width of the pass band, default 0.5
    :return: binary array
    """
    return np.abs(arr) < threshold


def _soft_step(coord, half_width, edge):
    """Falling edge at `half_width`: a hard step when edge <= 0, else tanh."""
    if edge <= 0:
        return (np.abs(coord) < half_width).astype(np.float64)
    return 0.5 * (1 - np.tanh((np.abs(coord) - half_width) / edge))


def _ft(a):
    """Centred forward FFT (origin at pixel n//2 on both sides)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a), norm="ortho"))


def _ift(a):
    """Centred inverse FFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a), norm="ortho"))


def _grid(n, step):
    """Centred (X, Y) meshgrid of n points spaced `step`, origin at index n//2."""
    c = (np.arange(n) - n // 2) * step
    return np.meshgrid(c, c, indexing="xy")


def _asm_transfer(FX, FY, wavelength_m, z_m):
    """Band-limited angular-spectrum transfer function for a hop of z_m."""
    arg = 1.0 / wavelength_m**2 - FX**2 - FY**2
    H = np.zeros(arg.shape, dtype=np.complex128)
    ok = arg > 0                                  # drop the evanescent corner
    H[ok] = np.exp(2j * np.pi * z_m * np.sqrt(arg[ok]))
    return H


def _smooth_random_phase(size, rms_rad, corr_px, rng):
    """Low-pass filtered random phase screen, `rms_rad` rad rms."""
    noise = rng.standard_normal((size, size))
    FX, FY = _grid(size, 1.0 / size)
    screen = np.real(_ift(_ft(noise) * np.exp(-2 * (np.pi * corr_px) ** 2 * (FX**2 + FY**2))))
    screen -= screen.mean()
    if screen.std() > 0:
        screen *= rms_rad / screen.std()
    return screen


def _hermite_gauss_orders(n):
    """(nx, ny) pairs for the first n Hermite-Gauss modes, lowest order first."""
    pairs = [(i, j) for i in range(8) for j in range(8)]
    return sorted(pairs, key=lambda p: (p[0] + p[1], p[0]))[:n]


def _orthonormalize(modes):
    """Gram-Schmidt the flattened modes in place order."""
    basis = []
    for m in modes:
        v = m.ravel().astype(np.complex128).copy()
        for u in basis:
            v -= np.vdot(u, v) * u
        norm = np.linalg.norm(v)
        basis.append(v / norm if norm > 1e-12 else np.zeros_like(v))
    return np.stack(basis).reshape(len(modes), *modes[0].shape)


def _fwhm(coord, profile):
    """Full width at half maximum of a 1D profile, by linear interpolation."""
    p = np.asarray(profile, dtype=float)
    p = p - p.min()
    if p.max() <= 0:
        return np.nan
    p = p / p.max()
    ipk = int(np.argmax(p))
    left = np.flatnonzero(p[: ipk + 1] <= 0.5)
    right = np.flatnonzero(p[ipk:] <= 0.5)
    if left.size == 0 or right.size == 0:
        return np.nan
    il, ir = left[-1], right[0] + ipk
    xl = np.interp(0.5, [p[il], p[il + 1]], [coord[il], coord[il + 1]])
    xr = np.interp(0.5, [p[ir], p[ir - 1]], [coord[ir], coord[ir - 1]])
    return abs(xr - xl)


def aperture(X, Y, kind="disk", diameter=1.0, central_stop=0.0, aspect=1.0,
             edge_softness=0.0, min_edge=0.0):
    """Real-valued aperture transmission on the (X, Y) grid.

    kind
        "disk" / "circ" / "pinhole"   circular aperture of `diameter`
        "gauss"                       Gaussian, `diameter` = amplitude FWHM
        "rect" / "square"             square aperture of side `diameter`
        "annulus" / "zp" / "zoneplate"  disk with a central stop (0.3 by default)
        "slit"                        one-dimensional, open along x
    central_stop
        Blocked central fraction of `diameter` (zone-plate beam stop).
    aspect
        y/x ratio of the aperture: 1.0 round, != 1.0 elliptical / rectangular.
    edge_softness
        tanh roll-off width as a fraction of the aperture radius. 0 = hard edge.
    min_edge
        Floor on that roll-off width in grid units, so a soft edge stays
        resolved (pass the pixel size).
    """
    kind = kind.lower()
    Ys = Y / aspect
    r = np.hypot(X, Ys)
    edge = max(edge_softness * diameter / 2, min_edge) if edge_softness > 0 else 0.0

    if kind in ("disk", "circ", "pinhole"):
        amp = _soft_step(r, diameter / 2, edge)
    elif kind in ("gauss", "gaussian"):
        amp = gaussian2D(X, Ys, diameter)
    elif kind in ("rect", "square", "kb"):
        amp = _soft_step(X, diameter / 2, edge) * _soft_step(Ys, diameter / 2, edge)
    elif kind in ("slit",):
        amp = _soft_step(X, diameter / 2, edge)
    elif kind in ("annulus", "ring", "zp", "zoneplate"):
        amp = _soft_step(r, diameter / 2, edge)
        central_stop = central_stop or 0.3
    else:
        raise ValueError(f"unknown aperture kind {kind!r}")

    if central_stop > 0:
        amp = amp * (1 - _soft_step(r, central_stop * diameter / 2, edge))
    return amp


def make_probe(
    size,
    kind="disk",
    diameter=None,
    pixel_size_m=None,
    wavelength_m=None,
    plane="sample",
    focal_length_m=None,
    outermost_zone_m=None,
    central_stop=0.0,
    aspect=1.0,
    edge_softness=0.1,
    defocus_m=0.0,
    astigmatism_m=0.0,
    vortex_charge=0,
    shift_px=(0.0, 0.0),
    random_phase_rad=0.0,
    random_phase_corr_px=8.0,
    seed=0,
    n_modes=1,
    secondary_mode_power=0.02,
    power=1.0,
    verbose=True,
    dtype=np.complex64,
):
    """Synthesize an initial probe guess, shaped (1, n_modes, size, size).

    Everything is in metres once `pixel_size_m` is given; with `pixel_size_m`
    left at None the grid is in pixels, `diameter` is in pixels, and only
    plane="sample" with no propagation is available.

    Parameters
    ----------
    kind, diameter, central_stop, aspect, edge_softness
        The aperture, see `aperture()`. `diameter` defaults to 30% of the grid
        in the sample plane, or 50% of the pupil in the pupil plane.
    plane
        "sample" draws the aperture on the sample grid; "pupil" draws it on the
        optic and Fourier transforms to the focus (needs wavelength_m and a
        focal length). See the block comment above.
    focal_length_m, outermost_zone_m
        Focal length of the optic, given directly or as a zone plate's
        outermost zone width: f = diameter * dr_N / lambda.
    defocus_m
        Sample distance downstream of the focus (angular spectrum, exact).
    astigmatism_m
        Difference in focus between x and y; splits `defocus_m` by +-a/2.
    vortex_charge
        Topological charge l: multiplies the aperture by exp(i l theta). That
        is the phase plate, so the dark core only opens up after focusing
        (plane="pupil") or propagating.
    shift_px
        (y, x) translation of the probe on the grid, as a linear phase.
    random_phase_rad, random_phase_corr_px
        Smooth random phase screen on the aperture -- speckle / partial
        coherence, and it breaks the symmetry that stalls a recon.
    n_modes, secondary_mode_power
        Incoherent modes: Hermite-Gauss modulations of the base field,
        orthonormalized, sharing `secondary_mode_power` of the total power.
    power
        Total sum|psi|^2 of the returned probe; None leaves the raw aperture
        scaling alone. Pty-Chi's `rescale_probe` overrides this anyway.
    """
    dx = 1.0 if pixel_size_m is None else float(pixel_size_m)
    in_pixels = pixel_size_m is None
    plane = plane.lower()
    rng = np.random.default_rng(seed)

    if plane not in ("sample", "pupil"):
        raise ValueError(f"plane must be 'sample' or 'pupil', not {plane!r}")
    needs_lambda = plane == "pupil" or defocus_m or astigmatism_m
    if needs_lambda and (in_pixels or wavelength_m is None):
        raise ValueError("pupil-plane construction and defocus need pixel_size_m and wavelength_m")

    FX, FY = _grid(size, 1.0 / (size * dx))          # frequency grid, cycles/m

    # ---- the aperture, on whichever plane it lives ------------------------
    if plane == "pupil":
        if focal_length_m is None:
            if outermost_zone_m is None or diameter is None:
                raise ValueError(
                    "pupil plane needs focal_length_m, or both diameter and outermost_zone_m"
                )
            focal_length_m = diameter * outermost_zone_m / wavelength_m
        pupil_span = wavelength_m * focal_length_m / dx      # width of the pupil grid
        if diameter is None:
            diameter = 0.5 * pupil_span
        X, Y = wavelength_m * focal_length_m * FX, wavelength_m * focal_length_m * FY
        min_edge = pupil_span / size
    else:
        if diameter is None:
            diameter = 0.3 * size * dx
        X, Y = _grid(size, dx)
        min_edge = dx

    amp = aperture(X, Y, kind, diameter, central_stop, aspect, edge_softness, min_edge)
    field = amp.astype(np.complex128)

    if vortex_charge:
        field = field * np.exp(1j * vortex_charge * np.arctan2(Y, X))
    if random_phase_rad:
        field = field * np.exp(1j * _smooth_random_phase(size, random_phase_rad,
                                                         random_phase_corr_px, rng))

    # ---- angular spectrum: defocus, astigmatism, translation --------------
    spectrum = field if plane == "pupil" else _ft(field)
    if defocus_m or astigmatism_m:
        zx = defocus_m + astigmatism_m / 2
        zy = defocus_m - astigmatism_m / 2
        if astigmatism_m:
            # Fresnel form, so the two axes can carry different focal terms.
            spectrum = spectrum * np.exp(
                -1j * np.pi * wavelength_m * (zx * FX**2 + zy * FY**2)
            )
        else:
            spectrum = spectrum * _asm_transfer(FX, FY, wavelength_m, defocus_m)
    if shift_px[0] or shift_px[1]:
        spectrum = spectrum * np.exp(
            -2j * np.pi * (FY * shift_px[0] * dx + FX * shift_px[1] * dx)
        )
    psi = _ift(spectrum)

    # ---- incoherent modes --------------------------------------------------
    if n_modes > 1:
        w = np.sqrt((np.abs(psi) ** 2 * (X**2 + Y**2)).sum() / (np.abs(psi) ** 2).sum()) \
            if plane == "sample" else None
        Xs, Ys = _grid(size, dx)
        if w is None or w <= 0:
            w = np.sqrt((np.abs(psi) ** 2 * (Xs**2 + Ys**2)).sum() / (np.abs(psi) ** 2).sum())
        stack = []
        for nx, ny in _hermite_gauss_orders(n_modes):
            hx = np.polynomial.hermite.hermval(Xs / w, np.eye(nx + 1)[nx])
            hy = np.polynomial.hermite.hermval(Ys / w, np.eye(ny + 1)[ny])
            stack.append(psi * hx * hy)
        modes = _orthonormalize(np.stack(stack))
        weights = np.full(n_modes, secondary_mode_power / (n_modes - 1))
        weights[0] = 1.0 - secondary_mode_power
        modes = modes * np.sqrt(weights)[:, None, None]
    else:
        modes = psi[None]

    if power is not None:
        modes = modes * np.sqrt(power / max((np.abs(modes) ** 2).sum(), 1e-300))
    modes = modes.astype(dtype)[None]                 # (n_opr, n_modes, h, w)

    if verbose:
        unit, to_um = ("px", 1.0) if in_pixels else ("um", 1e6)
        msg = [f"probe {tuple(modes.shape)} {kind} on the {plane} plane, "
               f"aperture {diameter * to_um if not in_pixels else diameter:.3f} {unit}"]
        if plane == "pupil":
            msg.append(f"  f = {focal_length_m * 1e3:.2f} mm, pupil grid spans "
                       f"{pupil_span * 1e6:.1f} um "
                       f"({'fits' if diameter <= pupil_span else 'TOO SMALL -- aperture clipped'})")
        if defocus_m or astigmatism_m:
            z_crit = size * dx**2 / wavelength_m if not in_pixels else np.inf
            msg.append(f"  defocus {defocus_m * 1e6:+.1f} um, astigmatism "
                       f"{astigmatism_m * 1e6:+.1f} um (angular-spectrum limit "
                       f"|z| < {z_crit * 1e6:.0f} um)")
        prof = (np.abs(modes[0, 0]) ** 2).sum(axis=0)
        coord = (np.arange(size) - size // 2) * dx * to_um
        msg.append(f"  mode 0 FWHM ~ {_fwhm(coord, prof):.4g} {unit} (x, summed over y)")
        print("\n".join(msg))
    return modes


def make_disk_probe(size, diameter_px, **kwargs):
    """Soft-edged disk, used when there is no probe to inherit.

    Thin wrapper on `make_probe`; pass any of its keywords through. The disk is
    centred on pixel size//2 (the FFT origin), a half pixel off the old
    (size-1)/2 centring, so that propagating it adds no phase ramp.
    """
    kwargs.setdefault("verbose", False)
    kwargs.setdefault("power", None)
    return make_probe(size, kind="disk", diameter=diameter_px, **kwargs)


def show_probe(probe, pixel_size_m=None, mode=0, opr=0, cmap="inferno",
               log_floor=1e-5, fov=None, figsize=(13.0, 3.3)):
    """Amplitude, phase, log intensity and far field of one probe mode.

    `fov` zooms the real-space panels (um if pixel_size_m is given, else px).
    """
    psi = probe.detach().cpu().numpy() if torch.is_tensor(probe) else np.asarray(probe)
    if psi.ndim == 4:
        psi = psi[opr, mode]
    elif psi.ndim == 3:
        psi = psi[mode]

    n = psi.shape[-1]
    unit, step = ("px", 1.0) if pixel_size_m is None else ("um", pixel_size_m * 1e6)
    c = (np.arange(n) - n // 2) * step
    ext = [c[0], c[-1], c[-1], c[0]]

    amp = np.abs(psi)
    inten = amp**2
    inten_n = inten / max(inten.max(), 1e-300)
    far = np.abs(_ft(psi)) ** 2
    far_n = far / max(far.max(), 1e-300)
    fc = (np.arange(n) - n // 2)                      # far field in detector px
    ext_f = [fc[0], fc[-1], fc[-1], fc[0]]

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    panels = [
        (amp, ext, cmap, None, f"|psi|   ({unit})"),
        (np.ma.masked_where(amp < 0.01 * amp.max(), np.angle(psi)), ext, "twilight",
         (-np.pi, np.pi), "phase"),
        (np.log10(np.maximum(inten_n, log_floor)), ext, cmap,
         (np.log10(log_floor), 0), "log10 I"),
        (np.log10(np.maximum(far_n, log_floor)), ext_f, cmap,
         (np.log10(log_floor), 0), "log10 far field (det px)"),
    ]
    for ax, (img, extent, cm, clim, title) in zip(axes, panels):
        im = ax.imshow(img, extent=extent, cmap=cm,
                       **({} if clim is None else dict(vmin=clim[0], vmax=clim[1])))
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if fov:
        for ax in axes[:3]:
            ax.set_xlim(-fov / 2, fov / 2)
            ax.set_ylim(fov / 2, -fov / 2)

    fx = _fwhm(c, inten.sum(axis=0))
    fy = _fwhm(c, inten.sum(axis=1))
    fig.suptitle(f"probe mode {mode}   FWHM {fx:.4g} x {fy:.4g} {unit}   "
                 f"power {inten.sum():.4g}", fontsize=10)
    plt.tight_layout()
    plt.show()


def siemens_star(shape, n_spokes=24):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    theta = np.arctan2(yy - shape[0] / 2, xx - shape[1] / 2)
    r = np.hypot(yy - shape[0] / 2, xx - shape[1] / 2)
    spokes = (np.cos(n_spokes * theta) > 0) & (r < 0.45 * min(shape))
    return ((1 - 0.15 * spokes) * np.exp(1j * 0.8 * spokes)).astype(np.complex64)


def _dp_frame(patterns, i, log, transpose):
    frame = np.asarray(patterns[i], dtype=np.float32)
    if transpose:                       # fracPy swaps axes 1,2 before showing
        frame = frame.T
    return np.log10(np.maximum(frame, 0) + 1) if log else frame


def _dp_clim(patterns, log, transpose, n_sample=20):
    """Common color scale, taken from up to `n_sample` frames spread over the stack."""
    idx = np.unique(np.linspace(0, len(patterns) - 1, min(len(patterns), n_sample)).astype(int))
    return 0.0, float(max(_dp_frame(patterns, i, log, transpose).max() for i in idx))


def use_interactive_backend(verbose=True):
    """Switch matplotlib to a backend whose widgets actually respond.

    The VS Code interactive window and Jupyter default to the inline backend,
    which renders each figure to a PNG -- a Slider drawn on it is a picture of
    a slider, and dragging does nothing. The three cases that matter here:

    * a notebook kernel (VS Code interactive window included) -> ipympl, which
      draws into the cell and needs no X server, so it works over plain SSH;
    * a real desktop session (DISPLAY or WAYLAND_DISPLAY) -> TkAgg;
    * neither -> leave the backend alone.

    Returns True if the backend in force can handle widgets. Call it *before*
    creating the figure -- switching afterwards leaves the old canvas behind.
    """
    backend = matplotlib.get_backend().lower()
    if any(k in backend for k in ("ipympl", "nbagg", "widget", "qt", "tk", "gtk", "macosx")):
        return True

    try:
        from IPython import get_ipython
        shell = get_ipython()
    except ImportError:
        shell = None

    if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
        try:
            import ipympl  # noqa: F401
        except ImportError:
            if verbose:
                print("Running in a kernel but ipympl is missing -- the slider will be dead.\n"
                      "  Fix with:  pip install ipympl   (then restart the kernel)")
            return False
        shell.run_line_magic("matplotlib", "widget")
        if verbose:
            print(f"matplotlib backend -> {matplotlib.get_backend()} (was inline; "
                  "widgets need it)")
        return True

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("TkAgg")
        if verbose:
            print("matplotlib backend -> TkAgg")
        return True

    return False


def show_ptychogram(patterns, index=0, log=True, cmap="inferno", transpose=False,
                    clim=None, figsize=(6.0, 6.6), interactive=True):
    """fracPy's exampleData.showPtychogram(): scroll through the diffraction stack.

    Drag the slider or use the left/right arrow keys. The matplotlib backend is
    switched to ipympl first (see `use_interactive_backend`), because on the
    inline backend the slider is only a picture of a slider; if no interactive
    backend can be had, this falls back to `show_ptychogram_grid`. Pass
    interactive=False to skip the switch and draw just frame `index`.

    In the VS Code interactive window the first call prints the backend change,
    and the figure then appears in an ipympl canvas -- if that canvas comes up
    blank, re-run the cell once (a known ipympl quirk on the very first switch).
    """
    from matplotlib.widgets import Slider

    if interactive and not use_interactive_backend():
        print("No interactive backend available -- showing a montage instead.\n"
              "  For the slider, run this in the VS Code interactive window (#%% cells).")
        return show_ptychogram_grid(patterns, log=log, cmap=cmap, transpose=transpose,
                                    clim=clim)

    n = len(patterns)
    if clim is None:
        clim = _dp_clim(patterns, log, transpose)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.12)
    im = ax.imshow(_dp_frame(patterns, index, log, transpose),
                   cmap=cmap, vmin=clim[0], vmax=clim[1])
    ax.set_xticks([]), ax.set_yticks([])
    unit = "log10(I + 1)" if log else "I"
    title = ax.set_title(f"frame {index} / {n - 1}   [{unit}]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    slider = Slider(fig.add_axes([0.15, 0.03, 0.7, 0.03]), "frame", 0, n - 1,
                    valinit=index, valstep=1)

    def update(_):
        i = int(slider.val)
        im.set_data(_dp_frame(patterns, i, log, transpose))
        title.set_text(f"frame {i} / {n - 1}   [{unit}]")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("left", "right"):
            slider.set_val(int(np.clip(slider.val + (1 if event.key == "right" else -1), 0, n - 1)))

    slider.on_changed(update)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig._ptychogram_slider = slider     # keep the widget alive after this returns
    plt.show()
    return slider


def show_ptychogram_grid(patterns, n_show=16, log=True, cmap="inferno", transpose=False,
                         clim=None, share_clim=True, tile=2.0):
    """Static montage of `n_show` frames spread over the stack (inline backend)."""
    n = len(patterns)
    idx = np.unique(np.linspace(0, n - 1, min(n, n_show)).astype(int))
    ncol = int(np.ceil(np.sqrt(len(idx))))
    nrow = int(np.ceil(len(idx) / ncol))
    if share_clim and clim is None:
        clim = _dp_clim(patterns, log, transpose)

    fig, axes = plt.subplots(nrow, ncol, figsize=(tile * ncol, tile * nrow), squeeze=False)
    axes = axes.ravel()
    kw = dict(vmin=clim[0], vmax=clim[1]) if clim is not None else {}
    for ax, i in zip(axes, idx):
        ax.imshow(_dp_frame(patterns, i, log, transpose), cmap=cmap, **kw)
        ax.set_title(f"{i}", fontsize=8)
    for ax in axes[len(idx):]:
        ax.set_visible(False)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("ptychogram, log10(I + 1)" if log else "ptychogram")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# output in the PEAR / beamline layout
#
# Results go to <recon_parent_dir>/<scan>/<make_recon_dir_name(...)>/, the same
# layout PEAR (pear.ptycho_recon, used by the beamline batch script) produces:
#
#   ptychi_recons/S0019/Ndp1024_LSQML_c20_m0.5_gaussian_p10_cp_mm_opr3_ic_pc1_f_ul2/
#       pear_params.json  dp_sum.tiff  init_positions.png  init_probe_mag.tiff
#       recon_Niter200.h5  recon_Niter400.h5  ...
#       loss/  object_mag/  object_ph/  positions/  probe_mag/
#
# In the reconstruction scripts these knobs are module-level variables, so both
# make_recon_dir_name() and collect_params() take a `cfg` namespace and are
# normally called as `make_recon_dir_name(globals())`. A dict, a SimpleNamespace
# or any object with the attributes works too, and keyword arguments override
# whatever comes out of it.
# ---------------------------------------------------------------------------

_MISSING = object()


def _lookup(cfg, name, default=_MISSING):
    """Read `name` from a mapping or an object; raise if it is not there."""
    if cfg is not None:
        if hasattr(cfg, "keys") and name in cfg:
            return cfg[name]
        if hasattr(cfg, name):
            return getattr(cfg, name)
    if default is _MISSING:
        raise KeyError(f"{name!r} is not in the config namespace and has no default")
    return default


def _enum_value(x):
    """api.BatchingModes.COMPACT -> 'compact'; a plain string passes through."""
    return getattr(x, "value", x)


def rgb_uint8(arr, cmap="gray", log=False):
    """Colormapped preview, the format PEAR writes its *.tiff previews in."""
    a = np.asarray(arr, dtype=np.float64)
    if log:
        a = np.log10(a - a.min() + 1.0)
    a = (a - a.min()) / max(np.ptp(a), 1e-12)
    return (plt.get_cmap(cmap)(a)[..., :3] * 255).astype(np.uint8)


def gray_uint16(arr):
    """Min-max scaled 16-bit preview (PEAR's object_mag / object_ph tiffs)."""
    a = np.asarray(arr, dtype=np.float64)
    return ((a - a.min()) / max(np.ptp(a), 1e-12) * 65535).astype(np.uint16)


def make_recon_dir_name(cfg=None, suffix="", **knobs):
    """Folder name in the beamline convention, e.g.

        Ndp1024_LSQML_c20_m0.5_gaussian_p10_cp_mm_opr3_ic_pc1_f_ul2

    Ndp<detector crop>, algorithm, batch selection scheme + batch size,
    m<momentum>, noise model, p<probe modes>, cp = probe centering, mm = object
    updated with the higher probe modes, opr<n>, ic = intensity correction,
    pc0/pc1 = position correction, gradient method initial, ul<update limit>.

    Knobs come from `cfg` (pass `globals()`) unless given as keywords.
    """
    def k(name, default=_MISSING):
        return knobs[name] if name in knobs else _lookup(cfg, name, default)

    batch_tag = {"compact": "c", "random": "r", "uniform": "u"}[_enum_value(k("batching_mode"))]
    position_start = k("position_start", None)
    parts = [
        f"Ndp{k('n_dp')}",
        k("algorithm", "LSQML"),
        f"{batch_tag}{k('batch_size')}",
        f"m{k('momentum_gain')}",
        _enum_value(k("noise_model")),
        f"p{k('n_probe_modes')}",
    ]
    if k("center_probe", False):
        parts.append("cp")
    if k("update_object_with_higher_probe_modes", False):
        parts.append("mm")
    parts.append(f"opr{k('n_opr_modes', 1)}")
    if k("optimize_intensity_variation", False):
        parts.append("ic")
    if position_start is None:
        parts.append("pc0")
    else:
        parts += ["pc1", "f", f"ul{k('position_update_limit_px', 0):g}"]
    return "_".join(parts) + suffix


def collect_params(cfg=None, **overrides):
    """The knob dump written to pear_params.json, keyed like PEAR's params.

    Script variables are read from `cfg` (pass `globals()`); `overrides` are
    keyed by PEAR name and win over it. Anything absent from both falls back to
    the default listed below, so a script that does not define, say,
    `probe_propagation_m` still gets a valid file.
    """
    def k(name, default=_MISSING):
        return _lookup(cfg, name, default)

    init_recon_file = k("init_recon_file", None)
    init_path = str(init_recon_file) if init_recon_file else ""
    position_start = k("position_start", None)

    params = {
        "data_directory": str(k("data_root", "")),
        "path_to_init_probe": init_path,
        "init_probe_propagation_m": k("probe_propagation_m", 0.0),  # extra: not a PEAR key
        "path_to_init_object": "",
        "path_to_init_positions": init_path,
        "scan_num": k("scan", ""),
        "instrument": k("instrument", ""),
        "det_sample_dist_m": k("det_dist_m"),
        "diff_pattern_size_pix": k("n_dp"),
        "load_processed_hdf5": not k("use_simulated_data", False),
        "path_to_processed_hdf5_dp": str(k("dp_file", "")),
        "path_to_processed_hdf5_pos": str(k("para_file", "")),
        "position_correction": position_start is not None,
        "position_correction_update_limit": k("position_update_limit_px", 0.0),
        "position_correction_gradient_method": "fourier",
        "position_correction_affine_constraint": False,
        "intensity_correction": k("optimize_intensity_variation", False),
        "center_probe": k("center_probe", False),
        "number_probe_modes": k("n_probe_modes"),
        "update_object_w_higher_probe_modes": k("update_object_with_higher_probe_modes", False),
        "number_opr_modes": k("n_opr_modes", 1),
        "update_batch_size": k("batch_size"),
        "batch_selection_scheme": _enum_value(k("batching_mode")),
        "momentum_acceleration": k("momentum_gain", 0.0) > 0,
        "number_of_slices": k("n_slices", 1),
        "number_of_iterations": k("num_epochs"),
        "save_freq_iterations": k("save_freq_iterations", None),
        "noise_model": _enum_value(k("noise_model")),
        "det_pixel_size_m": k("det_pixel_m"),
        "wavelength_m": k("wavelength_m"),
        "obj_pixel_size_m": k("pixel_size_m"),
        "probe_start_epoch": k("probe_start", None),
        "opr_start_epoch": k("opr_start", None),
        "position_start_epoch": position_start,
    }
    unknown = set(overrides) - set(params)
    if unknown:
        raise KeyError(f"unknown pear_params key(s): {sorted(unknown)}")
    params.update(overrides)
    return params


def save_initial_conditions(recon_dir, params, patterns, probe, positions_px):
    """Write pear_params.json and the dp_sum / init_probe / init_positions previews.

    `probe` is the initial guess in any of the shapes the scripts hold it in --
    (h, w), (n_modes, h, w) or (n_opr, n_modes, h, w) -- torch or numpy.
    """
    if tifffile is None:
        raise ImportError("tifffile is needed for the *.tiff previews")
    recon_dir = Path(recon_dir)
    recon_dir.mkdir(parents=True, exist_ok=True)

    with open(recon_dir / "pear_params.json", "w") as f:
        json.dump(params, f, indent=4)

    tifffile.imwrite(recon_dir / "dp_sum.tiff", rgb_uint8(np.asarray(patterns).sum(0), log=True))

    p = probe.detach().cpu().numpy() if torch.is_tensor(probe) else np.asarray(probe)
    while p.ndim > 3:                                   # (n_opr, n_modes, h, w) -> modes
        p = p[0]
    if p.ndim == 2:
        p = p[None]
    probe_montage = np.concatenate([np.abs(p[i]) for i in range(p.shape[0])], axis=1)
    tifffile.imwrite(recon_dir / "init_probe_mag.tiff", rgb_uint8(probe_montage))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(positions_px[:, 1], positions_px[:, 0], ".-", lw=0.3, ms=2)
    ax.set_aspect("equal")
    ax.set_title("initial positions [px]")
    fig.savefig(recon_dir / "init_positions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Reconstruction results will be saved in: {recon_dir}")


# Affine components of the position correction, keyed by iteration; the beamline
# files carry one entry per saved iteration, so this accumulates over the run.
# One history per output folder, so two runs in the same kernel do not mix.
_affine_history = {}


def save_reconstruction(task, recon_dir, pixel_size_m, n_iter=None, affine_history=None):
    """Write recon_Niter{n_iter}.h5 plus previews, matching the beamline files.

    `n_iter` defaults to the number of epochs the task has actually run, so this
    can be called straight after a hand-issued task.run(n) without tracking the
    count yourself. `affine_history` is the dict the position-correction
    components accumulate into; the default keeps one per `recon_dir`.

    Datasets (identical names, shapes and dtypes to PEAR's recon_Niter*.h5, so
    these results can be fed straight back in through `init_recon_file`):

        init_positions_px  (n, 2)                float64
        loss               (n_epochs,)           float64
        obj_pixel_size_m   scalar                float64
        object             (n_slices, h, w)      complex64
        opr_mode_weights   (n, n_opr)            float32
        positions_px       (n, 2)                float32
        pos_corr/affine_matrix                   (2, 3) float32
        pos_corr/{scale,asymmetry,rotation,shear}  (n_saves,) float32
        pos_corr/iterations                      (n_saves,) int64
    """
    if tifffile is None:
        raise ImportError("tifffile is needed for the *.tiff previews")
    recon_dir = Path(recon_dir)
    recon_dir.mkdir(parents=True, exist_ok=True)
    if affine_history is None:
        affine_history = _affine_history.setdefault(str(recon_dir), {})

    recon_obj = task.get_data_to_cpu("object", as_numpy=True)
    recon_probe = task.get_data_to_cpu("probe", as_numpy=True)
    recon_pos = task.get_data_to_cpu("probe_positions", as_numpy=True)
    weights = task.get_data_to_cpu("opr_mode_weights", as_numpy=True)
    init_pos = task.probe_positions.initial_positions.detach().cpu().numpy()
    loss = task.reconstructor.loss_tracker.table["loss"].to_numpy()
    if n_iter is None:
        n_iter = len(loss)

    comps = task.probe_positions.affine_transform_components
    affine_history[n_iter] = {key: float(np.asarray(comps[key])) for key in comps}
    iters = sorted(affine_history)
    affine_matrix = task.probe_positions.affine_transform_matrix.detach().cpu().numpy()

    with h5py.File(recon_dir / f"recon_Niter{n_iter}.h5", "w") as f:
        f.create_dataset("object", data=recon_obj.astype(np.complex64))
        f.create_dataset("probe", data=recon_probe.astype(np.complex64))
        f.create_dataset("positions_px", data=recon_pos.astype(np.float32))
        f.create_dataset("init_positions_px", data=init_pos.astype(np.float64))
        f.create_dataset("opr_mode_weights", data=weights.astype(np.float32))
        f.create_dataset("loss", data=loss.astype(np.float64))
        f.create_dataset("obj_pixel_size_m", data=np.float64(pixel_size_m))
        g = f.create_group("pos_corr")
        g.create_dataset("affine_matrix", data=affine_matrix.astype(np.float32))
        g.create_dataset("iterations", data=np.asarray(iters, dtype=np.int64))
        for key in ("scale", "asymmetry", "rotation", "shear"):
            g.create_dataset(
                key, data=np.asarray([affine_history[i][key] for i in iters], dtype=np.float32)
            )

    # Previews, one file per saved iteration in its own subfolder.
    roi = task.object.roi_bbox.get_slicer()
    obj_roi = recon_obj[0][roi]
    for sub, img in (
        ("object_mag", gray_uint16(np.abs(obj_roi))),
        ("object_ph", gray_uint16(np.angle(obj_roi))),
        (
            "probe_mag",
            rgb_uint8(
                np.concatenate(
                    [np.abs(recon_probe[0, i]) for i in range(recon_probe.shape[1])], axis=1
                )
            ),
        ),
    ):
        (recon_dir / sub).mkdir(exist_ok=True)
        tifffile.imwrite(recon_dir / sub / f"{sub}_Niter{n_iter}.tiff", img)

    (recon_dir / "loss").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(np.arange(1, len(loss) + 1), loss)
    ax.set_xlabel("epoch"), ax.set_ylabel("loss")
    fig.savefig(recon_dir / "loss" / f"loss_Niter{n_iter}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    (recon_dir / "positions").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(init_pos[:, 1], init_pos[:, 0], ".", ms=3, label="initial")
    ax.plot(recon_pos[:, 1], recon_pos[:, 0], ".", ms=3, label="corrected")
    ax.set_aspect("equal"), ax.legend()
    fig.savefig(recon_dir / "positions" / f"positions_Niter{n_iter}.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)

    print(f"saved {recon_dir / f'recon_Niter{n_iter}.h5'}")
    