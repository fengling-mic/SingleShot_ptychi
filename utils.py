import inspect
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


def _smooth(a, sigma_px):
    """Gaussian blur of `sigma_px`, done on the FFT grid like everything else."""
    FX, FY = _grid(a.shape[-1], 1.0 / a.shape[-1])
    return np.real(_ift(_ft(a) * np.exp(-2 * (np.pi * sigma_px) ** 2 * (FX**2 + FY**2))))


def _lowpass_noise(size, corr_px, rng, complex_noise=False):
    """White noise low-passed to a Gaussian blob of `corr_px` pixels rms."""
    noise = rng.standard_normal((size, size))
    if complex_noise:
        noise = (noise + 1j * rng.standard_normal((size, size))) / np.sqrt(2)
    FX, FY = _grid(size, 1.0 / size)
    return _ift(_ft(noise) * np.exp(-2 * (np.pi * corr_px) ** 2 * (FX**2 + FY**2)))


def _smooth_random_phase(size, rms_rad, corr_px, rng):
    """Low-pass filtered random phase screen, `rms_rad` rad rms."""
    screen = np.real(_lowpass_noise(size, corr_px, rng))
    screen -= screen.mean()
    if screen.std() > 0:
        screen *= rms_rad / screen.std()
    return screen


def _speckle_modulation(size, contrast, grain_px, rng, phase_only=False):
    """Complex speckle modulation with unit mean intensity.

    A diffuser dropped onto the plane the aperture is drawn on: white complex
    Gaussian noise low-passed to a grain of `grain_px` pixels, blended against
    a flat field with weight `contrast`. At weight 1 that is fully developed
    speckle -- Rayleigh |s|, uniform arg(s).

    The weight is not itself the intensity contrast: for a blend a + b*s with
    a = 1 - w, b = w, sigma_I / <I> works out to sqrt(2a^2b^2 + b^4)/(a^2 + b^2),
    which rises faster than w and is already saturated at 1 by w ~ 0.7:

        w     0.1   0.2   0.3   0.4   0.5   0.7   1.0
        K     0.18  0.34  0.53  0.72  0.89  0.99  1.00

    With `phase_only` the grains are pure phase (a random phase plate, |s| == 1
    everywhere) and `contrast` is the phase excursion in radians instead.
    """
    if phase_only:
        return np.exp(1j * _smooth_random_phase(size, contrast, grain_px, rng))
    s = _lowpass_noise(size, grain_px, rng, complex_noise=True)
    rms = np.sqrt((np.abs(s) ** 2).mean())
    if rms > 0:
        s = s / rms
    m = (1.0 - contrast) + contrast * s
    return m / np.sqrt(max((np.abs(m) ** 2).mean(), 1e-300))


# A Gaussian-smoothed screen of sigma has autocorrelation FWHM 2.3548*sigma*sqrt(2).
_SCREEN_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.sqrt(2.0)   # 3.3302


def _speckle_sigma_for_grain(grain_px, rms_rad, phase_only):
    """Sigma of the smoothing kernel that delivers a `grain_px` wide wavefront.

    The grain is the FWHM of the field autocorrelation -- the speckle you see
    when you look at the probe -- which is not the correlation length of the
    screen behind it once the phase wraps.

    For a Gaussian phase screen of rms `s` and normalized screen correlation
    exp(-r^2 / 4 sigma^2), the field correlation is

        C(r) = exp(-s^2 (1 - exp(-r^2 / 4 sigma^2)))

    and setting C = 1/2 inverts to the expression below. Strong phase only:
    at small `s` a coherent unscattered component survives at large r, C never
    reaches 1/2 on the diffuse part alone, and the formula runs away (at 1 rad
    it overpredicts by 2x). Below s^2 = log(2) there is no solution at all.

    Without `phase_only` the modulation is low-passed complex noise that does
    not wrap, so the field grain is just the kernel's own autocorrelation.
    """
    if not phase_only:
        return grain_px / _SCREEN_FWHM_PER_SIGMA
    if rms_rad ** 2 <= np.log(2.0):
        raise ValueError(
            f"speckle_grain_m needs a phase rms above {np.sqrt(np.log(2.0)):.2f} rad to be "
            f"well defined (got speckle={rms_rad:g}): below that the wavefront keeps a "
            f"coherent component and its correlation never falls to half. Raise `speckle`, "
            f"or set the kernel width directly with speckle_grain_px."
        )
    return grain_px / (4.0 * np.sqrt(-np.log(1.0 - np.log(2.0) / rms_rad ** 2)))


def _autocorr_fwhm_px(a):
    """FWHM of the autocorrelation of `a`, in pixels; nan if it never halves."""
    a = np.asarray(a)
    a = a - a.mean()
    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(a)) ** 2)))
    peak = ac.max()
    if peak <= 0:
        return np.nan
    line = ac[ac.shape[0] // 2, ac.shape[1] // 2:] / peak
    below = np.flatnonzero(line < 0.5)
    if below.size == 0:
        return np.nan
    j = below[0]
    if j == 0:
        return 0.0
    return 2.0 * (j - 1 + (line[j - 1] - 0.5) / (line[j - 1] - line[j]))


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
    speckle=0.0,
    speckle_grain_px=None,
    speckle_grain_m=None,
    speckle_phase_only=False,
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
    speckle, speckle_phase_only
        Granular structure inside the aperture, as if a diffuser sat on it.
        Without `speckle_phase_only` the diffuser modulates amplitude and
        `speckle` is how much of it, 0 to 1 -- not the contrast itself, which
        saturates at fully developed speckle (Rayleigh amplitude, uniform
        phase) by about 0.7:

            speckle  0.1   0.2   0.3   0.4   0.5   0.7   1.0
            K        0.18  0.34  0.53  0.72  0.89  0.99  1.00

        so 0.2-0.3 mottles the probe while keeping it beam-shaped. With
        `speckle_phase_only` the amplitude is left alone -- |field| is exactly
        the aperture, so a flat-top aperture gives a constant-amplitude
        wavefront -- and `speckle` is the phase excursion in radians instead.
        Verbose prints what was actually delivered either way.
    speckle_grain_px, speckle_grain_m
        Size of the grains, one or the other, never both. `speckle_grain_px`
        is the smoothing kernel's sigma in pixels of the plane the aperture is
        drawn on (with plane="pupil" that is the pupil grid, where finer
        grains spread the focus further); it defaults to 4.0. `speckle_grain_m`
        instead asks for a delivered wavefront grain -- the FWHM of the field
        autocorrelation, the speckle you actually see -- in metres, and solves
        for the sigma that produces it.

        The two differ once the phase wraps: at speckle = 2*pi a 30 nm
        wavefront grain comes from a 188 nm screen, so passing 30 nm to
        `speckle_grain_m` and to `speckle_grain_px / pixel` are very different
        requests. `speckle_grain_m` needs a phase rms above ~0.83 rad to be
        well posed, and needs `pixel_size_m`.
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

    # ---- grain: a kernel sigma in pixels, or a wanted grain in metres ------
    if speckle_grain_px is not None and speckle_grain_m is not None:
        raise ValueError("give speckle_grain_px or speckle_grain_m, not both")
    if speckle_grain_m is not None:
        if in_pixels:
            raise ValueError("speckle_grain_m needs pixel_size_m; use speckle_grain_px instead")
        speckle_grain_px = _speckle_sigma_for_grain(
            speckle_grain_m / dx, speckle, speckle_phase_only
        )
    elif speckle_grain_px is None:
        speckle_grain_px = 4.0

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
    if speckle:
        field = field * _speckle_modulation(size, speckle, speckle_grain_px, rng,
                                            speckle_phase_only)

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
        if speckle:
            where = "pupil" if plane == "pupil" else "sample"
            msg.append(f"  speckle {'phase ' if speckle_phase_only else ''}{speckle:.2f}, "
                       f"kernel sigma {speckle_grain_px:.2f} px on the {where} plane "
                       f"(screen {_SCREEN_FWHM_PER_SIGMA * speckle_grain_px:.1f} px"
                       + ("" if in_pixels else
                          f" = {_SCREEN_FWHM_PER_SIGMA * speckle_grain_px * dx * 1e9:.0f} nm")
                       + ")")
            # Both diagnostics only make sense where the probe is still on the
            # plane the grains were drawn on. Focus or propagate and the grain
            # is reset by the aperture size, not by the kernel.
            if plane == "sample" and not (defocus_m or astigmatism_m):
                # The delivered grain: FWHM of the field autocorrelation. This
                # is the speckle you see, and once the phase wraps it is much
                # finer than the screen printed above.
                grain = _autocorr_fwhm_px(np.real(modes[0, 0]))
                if np.isfinite(grain) and grain > 0:
                    got = f"  grain {grain:.2f} px"
                    if not in_pixels:
                        got += f" = {grain * dx * 1e9:.1f} nm"
                    if grain < 2.0:
                        got += (f"  -- UNDER-SAMPLED, a grain needs >= 2 px "
                                f"({2 * dx * 1e9:.1f} nm at this pixel); it will alias")
                    msg.append(got)
                # Amplitude contrast is only a thing in blend mode; phase_only
                # leaves |field| equal to the aperture by construction, so the
                # only spread left would be the rim.
                #
                # The support has to come from the blurred envelope: thresholding
                # the speckled intensity itself keeps only the brightest grains
                # and reports a contrast near zero. Six grains of blur averages
                # the grains away and leaves the aperture shape; the cap stops
                # that blurring past the aperture itself.
                if not speckle_phase_only:
                    inten = np.abs(modes[0, 0]) ** 2
                    env = _smooth(inten, min(max(6 * speckle_grain_px, 3.0), size / 16))
                    lit = inten[env > 0.5 * env.max()]
                    if lit.size >= 100:
                        msg.append(f"  intensity contrast {lit.std() / lit.mean():.2f} "
                                   f"over the lit area")
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


def _fresnel_propagate(field_np, distance_m, dx_m, wavelength_m, device="cpu"):
    """Single-FFT Fresnel transform of a 2D field (Pty-Chi's FresnelTransformPropagator).

    Unlike angular-spectrum propagation, the destination plane comes back on a
    DIFFERENT pixel pitch, `wavelength_m * |distance_m| / (n * dx_m)` -- exactly
    the rescaling a zone plate's near field needs -- so this is used instead of
    hand-rolling the FFT/scaling algebra. Returns (field_out, dx_out_m).
    """
    from ptychi.propagate import FresnelTransformPropagator, WavefieldPropagatorParameters

    n = field_np.shape[-1]
    params = WavefieldPropagatorParameters.create_simple(
        wavelength_m=wavelength_m, width_px=n, height_px=n,
        pixel_width_m=dx_m, pixel_height_m=dx_m,
        propagation_distance_m=distance_m,
    )
    field = torch.as_tensor(np.ascontiguousarray(field_np), dtype=torch.complex64, device=device)
    propagator = FresnelTransformPropagator(params).to(device)
    out = propagator.propagate_forward(field)
    dx_out_m = wavelength_m * abs(distance_m) / (n * dx_m)
    return out.detach().cpu().numpy(), dx_out_m


def make_randomized_zoneplate_probe(
    size,
    pixel_size_m,
    wavelength_m,
    outer_zone_width_m,
    focal_spot_diameter_m,
    zp_radius_m=None,
    central_stop=0.15,
    zp_transmission=-1.0 + 0j,
    oversample=8,
    osa_edge_softness=0.1,
    seed=0,
    power=1.0,
    verbose=True,
    device="cpu",
    dtype=np.complex64,
):
    """Simulate the focal-spot probe of a randomized (speckle-coded) Fresnel zone plate.

    A randomized zone plate jitters its zone boundaries by a random amount so
    the focused beam is a broad, high-spatial-frequency speckle pattern rather
    than a clean diffraction-limited spot -- the point being illumination
    diversity for single-shot/few-frame ptychography, not a tight focus. This
    builds that field from the optic's two physical specs directly, rather than
    fitting speckle statistics borrowed from some other reconstruction's probe:

        1. Random phase over a disk of radius Rmax = focal_spot_diameter_m / 2,
           sampled at pixel_size_m -- the field as it should look once it
           reaches the sample.
        2. Back-propagate (Fresnel) by -foc to the zone-plate plane. foc solves
           the zone-plate equation foc = 2 * zp_radius_m * outer_zone_width_m /
           wavelength_m; the destination pixel pitch comes back rescaled from
           `_fresnel_propagate`, not forced to match pixel_size_m.
        3. Build the zone-plate transmission on THAT plane's own grid by
           binarizing the SIGN of the backpropagated field's own phase --
           `angle(field_zp) > 0` gets `zp_transmission`, the rest passes
           untouched -- with no geometric zone-boundary formula involved.
           `field_zp` is fully-developed speckle (a linear propagation of a
           random-phase source), with grain size wavelength_m * foc /
           (2 * Rmax) -- equal to outer_zone_width_m when zp_radius_m is left
           at its default -- so thresholding its phase directly produces a
           binary random mask at exactly that feature size; this IS the
           "randomization". (An earlier version instead thresholded
           floor((chirp + angle(field_zp)) / pi) against the deterministic
           thin-lens chirp pi * R^2 / (wavelength_m * foc). Don't do that: since
           `field_zp` is also the illumination here, the trig cross-term
           between the two leaves a coherent, undegraded focusing wave riding
           on top of the speckle -- an unphysical bright central spike,
           confirmed by testing, that a real randomized zone plate does not
           show.) A central stop (`central_stop` * zp_radius_m) and the outer
           edge both get a soft tanh roll-off (`_soft_step`) rather than a
           hard cutoff.
        4. Forward-propagate by +foc back to the focus/object plane. Going out
           and back by the same distance is self-inverse regardless of the
           grid size used, so this plane's pixel pitch is exactly
           `pixel_size_m` again -- no approximation needed to make that hold.
        5. Order-sorting aperture: multiply by one more soft-edged circular
           aperture of radius Rmax (`_soft_step`, edge width
           `osa_edge_softness * Rmax`). A real (randomized or not) zone plate
           always sends light into more than the one order that focuses, and
           every real zone-plate microscope/ptychography setup blocks
           everything else with a physical pinhole at the focus for exactly
           this reason -- without it, step 4's output is a genuinely
           unconfined diffraction pattern, most of its power scattered well
           outside the intended footprint (confirmed by testing: only
           50/70/90% of the power inside 3.05/4.31/6.73 um diameters at
           osa_edge_softness=0 i.e. no OSA, against a 3.5 um target). At the
           default 0.1 this confines 99.1% of the power inside
           `focal_spot_diameter_m`.
        6. Crop/pad to `size` (`center_crop_or_pad`), rescale total power.

    Resolution depends only on `outer_zone_width_m`, not on `zp_radius_m`
    (NA = wavelength_m / (2 * outer_zone_width_m) regardless of radius; the
    radius only sets the self-consistent focal length above) -- so leaving
    `zp_radius_m` at its default (`focal_spot_diameter_m / 2`, i.e. no larger
    unlit zone-plate area beyond the illuminated footprint) does not cost any
    resolution. Validated against on-site parameters (12-ID-C S0019, 8 keV,
    8.8 nm sample-plane pixels): with the defaults here, a 50 nm outer zone
    and a 3.5 um focal-spot spec deliver a probe with 99%+ of its power
    inside that 3.5 um diameter and ~46-49 nm speckle grain -- both within a
    few percent of the inputs, with no further tuning.

    Parameters
    ----------
    size
        Output array size (crop/pad target), e.g. the detector crop n_dp.
    pixel_size_m, wavelength_m
        Sample-plane pixel size and the illumination wavelength.
    outer_zone_width_m
        The zone plate's finest (outermost) zone width -- sets resolution and
        speckle grain.
    focal_spot_diameter_m
        Desired illuminated footprint at the sample (2 * Rmax).
    zp_radius_m
        The simulated zone plate's own radius. None (default) sets it equal
        to Rmax = focal_spot_diameter_m / 2 -- see the resolution note above
        for why that loses nothing. Pass the real optic's clear-aperture
        radius instead if it is known and much larger than Rmax; the
        simulation grid (`oversample` px per outermost zone) scales with it.
    central_stop
        Fraction of zp_radius_m blocked by a central beam stop, 0 to disable.
    zp_transmission
        Complex transmission of an "odd" zone; default -1+0j is an idealized
        lossless pi-phase zone plate. Override with the real optic's
        delta/beta-derived transmission at its design energy for better
        fidelity.
    oversample
        Minimum pixels per delivered speckle grain (== outer_zone_width_m at
        the default zp_radius_m) at the zone-plate-plane grid; the grid size
        is derived from this, not given directly.
    osa_edge_softness
        Edge width of the order-sorting aperture (step 5), as a fraction of
        Rmax = focal_spot_diameter_m / 2; 0 disables it (not recommended --
        see step 5). Matches the name and "fraction of aperture radius"
        convention of `make_probe`'s own `edge_softness`.
    seed
        Realization of the random phase / zone jitter.
    power
        Total sum|psi|^2 of the returned probe; None leaves the raw scaling.
    device
        Where the two Fresnel transforms run ("cpu" by default -- the grid is
        small enough that CPU is fast, and it avoids competing for GPU memory
        with the reconstruction itself; pass "cuda" for a much larger
        zp_radius_m).

    Returns
    -------
    numpy.ndarray
        Complex probe, shaped (1, 1, size, size) like `make_disk_probe`.
    """
    rng = np.random.default_rng(seed)
    Rmax = focal_spot_diameter_m / 2.0
    r_zp = Rmax if zp_radius_m is None else float(zp_radius_m)
    bs_zp = central_stop * r_zp
    foc = 2.0 * r_zp * outer_zone_width_m / wavelength_m

    n_zp = int(np.ceil(2 * oversample * r_zp / pixel_size_m))
    n_zp += n_zp % 2

    # 1) random phase over a disk of radius Rmax, at the focus/object plane
    Xf, Yf = _grid(n_zp, pixel_size_m)
    support = np.hypot(Xf, Yf) <= Rmax
    field = np.zeros((n_zp, n_zp), dtype=np.complex128)
    field[support] = np.exp(2j * np.pi * rng.random(int(support.sum())))

    # 2) back-propagate to the zone-plate plane
    field_zp, res_zp = _fresnel_propagate(field, -foc, pixel_size_m, wavelength_m, device)

    # 3) zone-plate transmission, on the zone-plate-plane's own grid: binarize
    # the SIGN of the field's own phase -- no geometric zone-boundary formula.
    # field_zp is fully-developed speckle already (a linear propagation of a
    # random-phase source), with grain size wavelength_m * foc / (2 * Rmax)
    # (== outer_zone_width_m at the default zp_radius_m), so this directly
    # produces a binary random mask at that feature size.
    #
    # Do NOT thread the deterministic thin-lens chirp (pi * R^2 /
    # (wavelength_m * foc)) into this threshold instead -- tried first, and
    # since field_zp is also the illumination here, the trig cross-term
    # between the fixed chirp and the field's own phase leaves a coherent,
    # undegraded focusing wave riding on top of the speckle: an unphysical
    # bright central spike, confirmed by testing, that a real randomized zone
    # plate does not show.
    Xz, Yz = _grid(n_zp, res_zp)
    Rz = np.hypot(Xz, Yz)
    transmission = np.where(np.angle(field_zp) > 0, zp_transmission, 1.0 + 0j)

    edge = 4.0 * res_zp
    envelope = _soft_step(Rz, r_zp, edge)
    if central_stop > 0:
        envelope = envelope * (1.0 - _soft_step(Rz, bs_zp, edge))
    field_after_zp = field_zp * transmission * envelope

    # 4) forward-propagate back to the focus/object plane
    probe_full, res_out = _fresnel_propagate(field_after_zp, foc, res_zp, wavelength_m, device)

    # 5) order-sorting aperture: a real zone plate always sends light into more
    # than the one order that focuses, and every real ZP setup blocks the rest
    # with a physical pinhole at the focus -- without this, the field above is
    # a genuinely unconfined diffraction pattern, most of its power scattered
    # well outside Rmax, not a focused beam. See the docstring for validated
    # confinement numbers.
    if osa_edge_softness > 0:
        Xo, Yo = _grid(n_zp, res_out)
        Ro = np.hypot(Xo, Yo)
        probe_full = probe_full * _soft_step(Ro, Rmax, osa_edge_softness * Rmax)

    # 6) crop/pad, rescale power
    probe = center_crop_or_pad(probe_full, size)
    if power is not None:
        probe = probe * np.sqrt(power / max((np.abs(probe) ** 2).sum(), 1e-300))
    probe = probe.astype(dtype)

    if verbose:
        # A speckled field is not a single smooth lobe, so `_fwhm` on a raw summed
        # profile locks onto the width of whichever single bright grain lands
        # nearest the peak (tens of nm) rather than the illuminated footprint --
        # tried that first and it printed a wildly misleading "FWHM". The radius
        # that contains a given fraction of the total power is the robust
        # equivalent for this kind of pattern, so that is what is reported here.
        inten = (np.abs(probe) ** 2).astype(np.float64)
        yy, xx = np.mgrid[:size, :size] - size // 2
        r_m = np.hypot(yy, xx) * pixel_size_m
        order = np.argsort(r_m.ravel())
        cum = np.cumsum(inten.ravel()[order])
        cum /= max(cum[-1], 1e-300)
        r_sorted = r_m.ravel()[order]
        d50, d70 = (2 * r_sorted[np.searchsorted(cum, f)] for f in (0.5, 0.7))

        # Measured from a small patch near the beam center, not the whole array:
        # the OSA gives the field a real, well-defined amplitude taper over the
        # full ~focal_spot_diameter_m disk, and that taper's own broad
        # autocorrelation swamps _autocorr_fwhm_px's half-max crossing when run
        # on the whole field (it reports the ENVELOPE size, off by >10x, not the
        # fine speckle grain). A patch much smaller than the disk sees that taper
        # as effectively flat, isolating the fine structure the same way a local
        # crop would if measured off a real recorded probe image.
        patch = int(np.clip(0.3 * focal_spot_diameter_m / pixel_size_m, 32, size))
        c = size // 2
        center_patch = probe[c - patch // 2 : c + patch // 2, c - patch // 2 : c + patch // 2]
        grain_px = _autocorr_fwhm_px(np.real(center_patch))
        grain_msg = (f"{grain_px * pixel_size_m * 1e9:.1f} nm"
                     if np.isfinite(grain_px) and grain_px > 0 else "n/a")
        print(
            f"randomized zone plate: outer zone {outer_zone_width_m * 1e9:.1f} nm, "
            f"zp radius {r_zp * 1e6:.3f} um, central stop {central_stop:.2f}, "
            f"focal length {foc * 1e3:.3f} mm\n"
            f"  zone-plate-plane grid {n_zp}x{n_zp} px at {res_zp * 1e9:.2f} nm/px "
            f"(round-trip object-plane pixel {res_out * 1e9:.4f} nm vs input "
            f"{pixel_size_m * 1e9:.4f} nm)\n"
            f"  probe footprint (50%/70% power) ~ {d50 * 1e6:.2f}/{d70 * 1e6:.2f} um "
            f"(target {focal_spot_diameter_m * 1e6:.2f} um), "
            f"speckle grain ~ {grain_msg} (target {outer_zone_width_m * 1e9:.0f} nm)"
        )

    return probe[None, None]


def _square_mode_orders(n_modes):
    """(a, b) exponent pairs in square order: (0,0), (1,0), (0,1), (1,1), ...

    Sorted by (max(a, b), a + b, a), so the two dipoles come before the
    quadrupole. Pty-Chi instead walks a rectangular grid sized
    m = ceil(sqrt(n)) - 1, n = ceil(n / (m + 1)) - 1, which at 5 modes gives
    (0,0), u, u^2, v, uv -- the second dipole is pushed to index 3 and u^2
    lands at index 2.
    """
    k = int(np.ceil(np.sqrt(n_modes))) + 1
    pairs = [(a, b) for a in range(k) for b in range(k)]
    return sorted(pairs, key=lambda p: (max(p), p[0] + p[1], p[0]))[:n_modes]


def hermite_secondary_modes(probe, secondary_mode_energy=0.02, rotation_deg=0.0):
    """Fill incoherent modes 1.. with Hermite-Gauss modulations of mode 0.

    Drop-in for Pty-Chi's `orthogonalize_initial_probe`: takes an
    (n_opr, n_modes, h, w) tensor, uses probe[0, 0] as the source, overwrites
    probe[0, 1:], and normalizes each mode to its share of mode 0's energy.
    Two differences, both needed to start from the mode basis a measured probe
    actually has:

      * modes are walked in square order (see `_square_mode_orders`), so the
        ladder is dipole, dipole, quadrupole rather than dipole, u^2, dipole;
      * `rotation_deg` turns the basis, so the lobes can split along the
        diagonals instead of the axes.

    `secondary_mode_energy` is the energy of EACH secondary mode, not the
    total, matching Pty-Chi: mode 0 keeps 1 - (n_modes - 1) * that.

    Note this only sets the *initial* modes. With
    `probe_options.orthogonalize_incoherent_modes` enabled the reconstructor
    re-orthogonalizes by SVD every few epochs, and the data reshapes them.
    """
    is_tensor = torch.is_tensor(probe)
    p = probe.detach().cpu().numpy().copy() if is_tensor else np.array(probe, copy=True)
    n_modes = p.shape[1]
    psi = p[0, 0]
    if n_modes < 2:
        return probe

    h, w = psi.shape
    x = np.arange(w) - w / 2 + 1                  # Pty-Chi's centring
    y = np.arange(h) - h / 2 + 1
    xx, yy = np.meshgrid(x, y, indexing="xy")
    inten = np.abs(psi) ** 2
    total = inten.sum()
    X = xx - (xx * inten).sum() / total
    Y = yy - (yy * inten).sum() / total

    t = np.deg2rad(rotation_deg)
    U = X * np.cos(t) + Y * np.sin(t)
    V = -X * np.sin(t) + Y * np.cos(t)
    var_u = (U**2 * inten).sum() / total
    var_v = (V**2 * inten).sum() / total
    damp = np.exp(-(U**2 / (2 * var_u)) - (V**2 / (2 * var_v)))

    basis = []
    for i, (a, b) in enumerate(_square_mode_orders(n_modes)):
        f = (U**a) * (V**b) * psi
        if i > 0:
            f = f * damp
        f = f / np.sqrt(max((np.abs(f) ** 2).sum(), 1e-300))
        for g in basis:                           # Gram-Schmidt, as Pty-Chi does
            f = f - g * (g * f.conj()).sum()
        basis.append(f / np.sqrt(max((np.abs(f) ** 2).sum(), 1e-300)))

    energies = np.full(n_modes, secondary_mode_energy, dtype=np.float64)
    energies[0] = 1.0 - secondary_mode_energy * (n_modes - 1)
    energies = energies * total
    for i, f in enumerate(basis):
        p[0, i] = f * np.sqrt(energies[i] / max((np.abs(f) ** 2).sum(), 1e-300))

    if is_tensor:
        return torch.as_tensor(p, dtype=probe.dtype, device=probe.device)
    return p


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

    Nothing calls this on your behalf: an ipympl canvas takes over every later
    figure in the session, so opting in is yours to do. `use_inline_backend()`
    undoes it.
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


def _show_one_frame(patterns, index, log, cmap, transpose, clim, figsize):
    """One diffraction pattern, full size, on whatever backend is in force."""
    n = len(patterns)
    if clim is None:
        clim = _dp_clim(patterns, log, transpose)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(_dp_frame(patterns, index, log, transpose),
                   cmap=cmap, vmin=clim[0], vmax=clim[1])
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"frame {index} / {n - 1}   [{'log10(I + 1)' if log else 'I'}]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    plt.tight_layout()
    plt.show()


def use_inline_backend(verbose=True):
    """Go back to plain inline PNG figures, undoing `use_interactive_backend()`.

    Only meaningful inside a kernel; elsewhere it leaves the backend alone.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except ImportError:
        shell = None
    if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
        return False
    shell.run_line_magic("matplotlib", "inline")
    if verbose:
        print(f"matplotlib backend -> {matplotlib.get_backend()}")
    return True


def show_ptychogram(patterns, index=None, log=True, cmap="inferno", transpose=False,
                    clim=None, figsize=(6.0, 6.6), interactive=False, force_inline=True,
                    **grid_kwargs):
    """fracPy's exampleData.showPtychogram(): look through the diffraction stack.

    Plain inline figures by default -- no ipympl canvas, nothing to click:

        index=None (default)  a montage of frames spread over the stack,
                              i.e. `show_ptychogram_grid`; extra keywords
                              (n_show, tile, ...) are passed on to it
        index=<int>           just that one frame, full size

    If the session is left on a widget backend (by an earlier call, or by
    beamPropagation.py), this switches it back to inline first, so the figure
    is a plain PNG that VS Code's plot viewer can expand. Pass
    force_inline=False to leave whatever backend is in force alone.

    interactive=True instead switches the backend to ipympl and draws the
    slider version (drag it, or use the left/right arrow keys). That canvas
    then takes over every later figure in the session, so it is opt-in;
    `use_inline_backend()` puts things back.
    """
    from matplotlib.widgets import Slider

    if not interactive:
        if force_inline and any(k in matplotlib.get_backend().lower()
                                for k in ("ipympl", "nbagg", "widget")):
            use_inline_backend()
        if index is None:
            return show_ptychogram_grid(patterns, log=log, cmap=cmap, transpose=transpose,
                                        clim=clim, **grid_kwargs)
        return _show_one_frame(patterns, index, log, cmap, transpose, clim, figsize)

    if not use_interactive_backend():
        print("No interactive backend available -- showing a montage instead.\n"
              "  For the slider, run this in the VS Code interactive window (#%% cells).")
        return show_ptychogram_grid(patterns, log=log, cmap=cmap, transpose=transpose,
                                    clim=clim, **grid_kwargs)

    index = 0 if index is None else index
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
# In the reconstruction scripts these knobs are module-level variables, so the
# four functions below read them from a `cfg` namespace that defaults to the
# globals of whatever called them. Run cell-by-cell in an interactive window
# that is the notebook namespace, so `make_recon_dir_name(recon_dir_suffix)`
# and `save_initial_conditions(recon_dir)` just work, exactly as they do in
# 4idd_202603/ptychi_reconstruction_4idd.py where these were local functions.
#
# Pass the namespace explicitly -- `make_recon_dir_name(globals(), "_v2")` -- to
# call them from inside another function, where the caller's globals are the
# defining module's and not the knobs. A dict, a SimpleNamespace or any object
# with the attributes works, and keyword arguments override whatever it holds.
# ---------------------------------------------------------------------------

_MISSING = object()


def _caller_globals(depth=2):
    """Module globals of the frame `depth` levels up (2 = our caller's caller)."""
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            if frame.f_back is None:
                break
            frame = frame.f_back
        return frame.f_globals
    finally:
        del frame


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


def _jsonable(x):
    """Plain JSON types out of the objects the scripts actually hold knobs in.

    Values read straight out of the para file are 0-d numpy arrays rather than
    Python floats -- `energy` and `detector_distance` come back from h5py as
    `np.asarray(...).squeeze()` -- and paths are `Path`. json.dump chokes on
    both, so everything going into pear_params.json passes through here.
    """
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.item()) if x.ndim == 0 else [_jsonable(v) for v in x.tolist()]
    if torch.is_tensor(x):
        return _jsonable(x.detach().cpu().numpy())
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return _enum_value(x)


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

    Knobs come from `cfg`, which defaults to the caller's globals, unless given
    as keywords. A bare string first argument is taken as the suffix, so both
    `make_recon_dir_name("_v2")` and `make_recon_dir_name(globals(), "_v2")`
    do the same thing.
    """
    if isinstance(cfg, str):
        if suffix:
            raise TypeError("suffix given both positionally and as a keyword")
        cfg, suffix = None, cfg
    if cfg is None:
        cfg = _caller_globals()

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

    Script variables are read from `cfg`, which defaults to the caller's
    globals; `overrides` are keyed by PEAR name and win over it. Anything absent
    from both falls back to the default listed below, so a script that does not
    define, say, `probe_propagation_m` still gets a valid file.
    """
    if cfg is None:
        cfg = _caller_globals()

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
    return {key: _jsonable(value) for key, value in params.items()}


def save_initial_conditions(recon_dir, params=None, patterns=None, probe=None,
                            positions_px=None, cfg=None):
    """Write pear_params.json and the dp_sum / init_probe / init_positions previews.

    `probe` is the initial guess in any of the shapes the scripts hold it in --
    (h, w), (n_modes, h, w) or (n_opr, n_modes, h, w) -- torch or numpy.

    Everything but `recon_dir` defaults to the same-named variable in `cfg`,
    which itself defaults to the caller's globals, so from a reconstruction
    script this is just `save_initial_conditions(recon_dir)`. `params` defaults
    to `collect_params()` over that same namespace.
    """
    if tifffile is None:
        raise ImportError("tifffile is needed for the *.tiff previews")
    if cfg is None:
        cfg = _caller_globals()
    if params is None:
        params = collect_params(cfg)
    if patterns is None:
        patterns = _lookup(cfg, "patterns")
    if probe is None:
        probe = _lookup(cfg, "probe")
    if positions_px is None:
        positions_px = _lookup(cfg, "positions_px")
    recon_dir = Path(recon_dir)
    recon_dir.mkdir(parents=True, exist_ok=True)

    with open(recon_dir / "pear_params.json", "w") as f:
        json.dump(params, f, indent=4, default=_jsonable)

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


def save_reconstruction(task, recon_dir, n_iter=None, *, pixel_size_m=None,
                        affine_history=None, cfg=None):
    """Write recon_Niter{n_iter}.h5 plus previews, matching the beamline files.

    `n_iter` defaults to the number of epochs the task has actually run, so this
    can be called straight after a hand-issued task.run(n) without tracking the
    count yourself. `pixel_size_m` is keyword-only -- it can never swallow a
    positional epoch count -- and defaults to the variable of that name in
    `cfg`, itself defaulting to the caller's globals. `affine_history` is the
    dict the position-correction components accumulate into; the default keeps
    one per `recon_dir`.

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
    if pixel_size_m is None:
        pixel_size_m = _lookup(cfg if cfg is not None else _caller_globals(), "pixel_size_m")
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
    