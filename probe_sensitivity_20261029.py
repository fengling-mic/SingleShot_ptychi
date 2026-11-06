"""First-order object-frequency sensitivity of a focused vs. a randomized zone plate.

Why this exists
---------------
The usual argument for RPI is a *counting* one: a randomized probe spreads the beam over
more detector pixels, so there are more measurements per object unknown. For THIS pair of
optics that argument does not work, because the two zone plates have the same NA by
construction. `make_randomized_zoneplate` builds its optic by back-propagating a
random-phase disk of radius Rmax = focal_spot_diameter_m/2 whose speckle grain is
lambda*foc/(2*Rmax) = outer_zone_width_m, so the back-propagated field fills exactly the
same 1.75 um aperture as the regular zone plate:

    conventional ZP : r = 1.75 um, dr_N = 50 nm, f = 1.129 mm -> NA = 1.55 mrad
    randomized  ZP : r = 1.75 um, grain = 50 nm, f = 1.129 mm -> NA = 1.55 mrad

Same NA, same number of lit detector pixels. Counting cannot tell them apart, and with the
sample at focus the *conventional* probe actually has the smaller real-space footprint, so
the naive "detector pixels in beam / object pixels in probe" ratio comes out LARGER for the
conventional ZP.

What separates them is the quantity counting ignores: the relative complex phase of the
probe's Fourier components. For a weak phase object O = exp(i*a*cos(2*pi*k.r)) and a probe
with pupil A(q) = F{P}(q), the first-order response of the measured intensity is

    dI(q)/da = -Im[ A*(q) ( A(q-k) + A(q+k) ) ]                                    (*)

Two things follow:

  1. SUPPORT. The term needs A(q) != 0 *and* A(q-+k) != 0, i.e. q in S_A and q-+k in S_A.
     So only detector pixels in the overlap S_A ^ (S_A +- k) respond at all, and first-order
     sensitivity dies for |k| > 2*theta_d/lambda = 2*kp. Beyond that there is no heterodyne
     beat against the direct beam and the signal is dark-field, O(a^2). This is the honest
     form of the oversampling argument -- a statement about the *support* of the response,
     not a raw pixel count.

  2. PHASE. If A(q) has constant phase across its support -- an aberration-free pupil at
     exact focus -- then A*(q)(A(q-k)+A(q+k)) is real and (*) vanishes for EVERY k. A
     focused zone plate is therefore first-order blind to phase, no matter how many pixels
     are lit. Randomize arg A(q) and the same term becomes O(|A|^2) with no systematic
     zeros.

Analytic case this script must reproduce: for a defocused ideal pupil
A(q) = A0*exp(i*pi*lambda*dz*q^2),

    dI(q)/da = -2*A0^2 * sin(pi*lambda*dz*k^2) * cos(2*pi*lambda*dz*q.k)

-- the textbook contrast transfer function, with zeros at lambda*dz*k^2 = integer and
identically zero at dz = 0.

What it measures
----------------
For each test frequency k, through the project's own forward model
(`utils.simulate_rpi_diffraction`, so mixed probe modes are handled automatically):

    T(k) = sum_q (dI/da)^2                  total first-order response
    F(k) = sum_q (dI/da)^2 / (I0 + bkg)     shot-noise Fisher information for a

T(k) says how much signal a given object frequency produces; F(k) says how well it can be
estimated at fixed dose. F(k) is the one that answers the dose objection in Jacobsen,
Deng & Nashed, J. Synchrotron Rad. 24, 1078 (2017) -- in the heterodyne regime the local
oscillator |A(q)|^2 cancels out of (dI/da)^2/I, so spreading the direct beam over more
pixels costs no information.

dI/da is taken as a CENTRAL difference in a. That matters: the forward difference leaves an
O(a) residue from the second-order term, which would swamp the focused-ZP null. The central
difference is even-order-cancelling, so the null is measurable down to the O(a^2) floor.

Running it
----------
SYNTHETIC_ONLY = True (default) needs no beamline data and runs anywhere the ptychi env is
installed -- use it to verify the physics. Set it False on the Linux host to add the two
experimental probes from the prior LSQML reconstructions.

Outputs (never overwrites; every file carries the date stamp and the grid):
    <out_dir>/probe_sensitivity_<STAMP>_n<n_dp>.png
    <out_dir>/probe_metrics_<STAMP>_n<n_dp>.txt
"""

#%% ------------------------------------------------------------------ imports + config
import math
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import (
    center_crop_or_pad,
    crop_at_position,
    fourier_resample,
    make_randomized_zoneplate_probe,
    make_regular_zoneplate_probe,
    probe_fourier_center,
    probe_fourier_power,
    probe_fourier_radius,
    simulate_rpi_diffraction,
)

STAMP = "20261101"   # bump per revision; outputs also carry the grid, so re-running at a
                     # different n_dp never overwrites a previous result

# The beamline shares are mounted at /mnt/... on the Linux host and reachable as UNC paths
# from Windows. Branch rather than hardcode: on Windows, Path("/mnt/...").mkdir(parents=True)
# does not fail, it silently creates C:\mnt\... .
WINDOWS = os.name == "nt"
DATA2 = Path("//micdata/data2") if WINDOWS else Path("/mnt/micdata2")
out_dir = (Path.home() / "Pictures" / "RPI_paper" if WINDOWS
           else Path("/mnt/micdata3/fengling/figs"))   # cf. 12ide202604_1/export_figs.py:16

SYNTHETIC_ONLY = False         # True -> skip the experimental probes entirely. When False,
                               # each one is still skipped individually (with a message) if
                               # its reconstruction file is not reachable.
max_modes = None               # cap incoherent modes on the experimental probes; None keeps
                               # all 10. Dropping modes discards real incoherent power, so
                               # this is a GPU-headroom escape hatch, not a default.

# Optic + geometry. Canonical set from 12ide202604_1/export_figs.py:7-13.
# n_dp = 512 matches the reconstruction geometry of BOTH single-frame scripts now that
# RPI_SS_recon_50nmZP_blrelax.py has moved to a 512 crop. Everything downstream is derived
# from it, and fourier_resample holds the probe FOV fixed, so 1024 can be run unchanged as a
# grid-independence cross-check: T(k) and F(k) are expressed against |k| in *detector*
# pixels, which does not depend on this choice.
n_dp                 = int(os.environ.get("PROBE_NDP", 512))
det_pixel_m          = 172e-6
det_dist_m           = 10.0
wavelength_m         = 1.55e-10          # 8 keV
outer_zone_width_m   = 50e-9
focal_spot_diameter_m = 3.5e-6           # = OSA diameter = 2 * zp_radius_m
central_stop         = 0.15
seed                 = 0

pixel_size_m = wavelength_m * det_dist_m / (n_dp * det_pixel_m)   # 8.8 nm
zp_radius_m  = focal_spot_diameter_m / 2
foc_m        = 2 * zp_radius_m * outer_zone_width_m / wavelength_m
NA           = zp_radius_m / foc_m
kp_nominal   = NA * det_dist_m / det_pixel_m          # pupil radius in detector pixels

# Measurement knobs.
amp          = 1e-3        # object phase amplitude a [rad]; central difference in +-a
n_radii      = 56          # radial samples of |k|
n_azim       = 6           # azimuths per radius, over [0, pi) (cos is even in k)
k_max_bins   = int(min(n_dp // 2 - 2, round(2.6 * kp_nominal)))   # past the 2*kp cutoff
total_counts = 1e9         # every probe rescaled to this, so comparisons are dose-matched
bkg_counts   = 1e-6        # keeps F(k) finite where |A|^2 -> 0 (both numerator and
                           # denominator vanish there, so the ratio is physically finite)
kp_quantile  = 0.95        # forced EQUAL for both probes -- the scripts disagree (0.95/0.98)
panel_b_k_over_kp = 1.0    # test-object frequency for panel (b), in units of kp_nominal.
                           # At >= 1.0 the triple overlap S_A ^ (S_A+k) ^ (S_A-k) is empty
                           # (the +-k shifted pupils are tangent at exactly |k| = kp), so
                           # every responding pixel sees ONE sideband and the CTF-zero
                           # cancellation of (3) cannot occur. Below 1.0 both sidebands
                           # coexist over part of the pupil. The response itself survives
                           # either way out to 2 kp, via the pairwise overlap.
defocus_m    = 250e-6      # for the CTF validation panel; puts ~3 zeros inside |k| < kp.
                           # It must be kp, not 2*kp: the sin(pi*lambda*dz*k^2) cancellation
                           # needs BOTH sidebands at the same q, i.e. q in the triple overlap
                           # S_A ^ (S_A+k) ^ (S_A-k), which is empty once |k| > kp. Beyond kp
                           # each sideband is detected alone, Im[A*(q)A(q-k)] does not vanish
                           # at lambda*dz*k^2 = n, and the zeros simply do not exist.

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
# complex128, not the complex64 the reconstructions use. The central difference subtracts
# two intensities that agree to ~1 part in 1/a, so in float32 the focused-ZP null would be
# measuring roundoff rather than physics. At n_dp=1024 the extra cost is irrelevant.
CDTYPE = torch.complex128

# Experimental probes (only used when SYNTHETIC_ONLY is False).
EXPERIMENTAL = {
    "ZP (exp, S0040)": dict(
        recon=DATA2 / "12IDC/2026_Data/2026_1/03_ptycho/ptychi_recons/S0040"
        / "Ndp512_LSQML_c50_m0.5_gaussian_p10_cp_mm_ic_pc1_f_ul2" / "recon_Niter1000.h5",
        dp=DATA2 / "12IDC/2026_Data/2026_1/03_ptycho/results/S0040"
        / "data_roi0_Ndp512_dp.hdf5",
    ),
    # The Ndp512 S0019 reconstruction, not the Ndp1024 one. Paired with the ZP's Ndp512
    # S0040 run it gives the same detector crop, the same 4.4 mrad recorded angular range
    # and the same iteration count -- removing the largest confound in the earlier
    # comparison, where the RPI frame recorded twice the angular range of the ZP frame.
    "RPI (exp, S0019)": dict(
        recon=DATA2 / "12IDC/2026_Data/2026_3/01_piezo_test/ptychi_recons/S0019"
        / "Ndp512_LSQML_c150_m0.5_gaussian_p10_cp_mm_opr3_ic_pc1_f_ul2_pd-0.15"
        / "recon_Niter1000.h5",
        # S0019 only ships Ndp1024 data; the mask is center-cropped to n_dp below, exactly
        # as the reconstruction scripts crop their patterns.
        dp=DATA2 / "12IDC/2026_Data/2026_3/01_piezo_test/preproc/S0019"
        / "data_roi0_Ndp1024_dp.hdf5",
    ),
}

# Step-scan reconstruction of the same sample (S0019), used ONLY to measure which spatial
# frequencies the real object actually contains. It never enters T(k), F(k) or any bound --
# it is context for reading them, and the justification for the panel-(b) test frequency.
STEPSCAN = dict(
    recon=Path("//micdata/data3" if WINDOWS else "/mnt/micdata3")
    / "fengling/2026_03_results/ptychi_recons/S0019"
    / "Ndp1024_LSQML_c100_m0.25_gaussian_p5_opr1_pc1_f_ul20_initprobe_fromzpOSA_GS_eswonj_stepsize"
    / "recon_Niter15500.h5",
    frame_index=62,        # the single scan position the single-frame reconstructions use
)

print(f"pixel {pixel_size_m*1e9:.2f} nm | f {foc_m*1e3:.3f} mm | NA {NA*1e3:.3f} mrad | "
      f"kp_nominal {kp_nominal:.1f} det px | k_max {k_max_bins} bins | {torch_device}")


#%% ------------------------------------------------------------------ probe construction
def rescale_to_counts(probe_modes, counts=total_counts):
    """Scale so the total far-field power equals `counts`.

    Dose matching is not cosmetic here: T(k) and F(k) are both extensive in the photon
    budget, so without this the comparison would just be reporting which probe happened to
    be normalized brighter. Parseval for the unnormalized FFT used by the forward model
    gives sum_q |A|^2 = n^2 * sum_r |P|^2.
    """
    power = (probe_modes.abs() ** 2).sum() * probe_modes.shape[-1] ** 2
    return probe_modes * torch.sqrt(counts / power)


def as_modes(arr):
    """(..., n, n) numpy -> (n_modes, n, n) torch on the compute device."""
    a = np.asarray(arr)
    a = a.reshape((-1,) + a.shape[-2:])
    return torch.as_tensor(np.ascontiguousarray(a), dtype=CDTYPE, device=torch_device)


def ideal_pupil_probe(defocus=0.0, stop=0.0):
    """Probe defined directly by its pupil: a flat REAL disk of radius NA/lambda times the
    Fresnel defocus phase exp(i*pi*lambda*dz*q^2).

    This is the analytic model behind the CTF result, so it is the reference case:
    defocus=0 must give an exact first-order null, defocus!=0 must give sin(pi*lambda*dz*k^2)
    zeros. It is deliberately NOT `make_regular_zoneplate_probe` -- that one carries residual
    unfocused orders through the OSA, which break the perfectly-real pupil and so can only
    be *strongly suppressed*, not exactly null.
    """
    dq = 1.0 / (n_dp * pixel_size_m)                       # frequency per detector pixel
    qy, qx = np.mgrid[:n_dp, :n_dp] - n_dp // 2
    q = np.hypot(qy, qx) * dq
    q_na = NA / wavelength_m
    A = (q <= q_na).astype(np.complex128)
    if stop > 0:
        A[q < stop * q_na] = 0
    A *= np.exp(1j * np.pi * wavelength_m * defocus * q ** 2)
    # fftshift(fft2(P)) == A  requires  P = ifft2(ifftshift(A))
    return np.fft.ifft2(np.fft.ifftshift(A))[None]


def load_experimental(recon_file):
    """Prior LSQML probe -> (n_modes, n_dp, n_dp), resampled onto the common grid.

    Resampled with `fourier_resample`, not center-cropped: the array FOV is
    lambda*z/det_pixel independent of n_dp, so cropping would silently shrink the probe's
    field of view. This is what puts the 512-px conventional probe and the 1024-px RPI
    probe on the same footing.
    """
    with h5py.File(recon_file, "r") as f:
        probe = np.asarray(f["probe"][()]).view(np.complex64)
    probe = fourier_resample(probe, n_dp)
    # (n_opr, n_modes, h, w). Take OPR mode 0 only, exactly as the reconstruction scripts
    # do (`probe_modes = rescale_probe_to_counts(probe[0], ...)`). OPR modes are variation
    # directions across scan positions, NOT incoherent modes -- folding them into the mode
    # axis would add power the detector never saw incoherently.
    probe = probe.reshape((-1,) + probe.shape[-3:])
    probe = probe[0]
    return probe if max_modes is None else probe[:max_modes]


probes = {}
probes["ideal pupil, in focus"] = as_modes(ideal_pupil_probe(0.0))
probes[f"ideal pupil, dz={defocus_m*1e6:.0f} um"] = as_modes(ideal_pupil_probe(defocus_m))
probes["ZP (synth)"] = as_modes(make_regular_zoneplate_probe(
    n_dp, pixel_size_m, wavelength_m, outer_zone_width_m, focal_spot_diameter_m,
    zp_radius_m=zp_radius_m, central_stop=central_stop, verbose=True))
probes["RPI (synth)"] = as_modes(make_randomized_zoneplate_probe(
    n_dp, pixel_size_m, wavelength_m, outer_zone_width_m, focal_spot_diameter_m,
    zp_radius_m=zp_radius_m, central_stop=central_stop, seed=seed, verbose=True))

def load_sensor_mask(dp_file):
    """The detector's own good-pixel mask, brought to the n_dp grid.

    `det_pixel_mask` in the *_dp.hdf5 marks the sensor's inherent structure -- the gaps
    between modules, in both axes. ~9% of pixels. Those pixels carry no signal, so counting
    them when measuring the pupil centre or kp biases both. center_crop_or_pad matches what
    the reconstruction scripts do to their patterns, and its zero-pad correctly marks
    never-measured area as invalid when the stored mask is smaller than n_dp.
    """
    with h5py.File(dp_file, "r") as f:
        m = np.asarray(f["det_pixel_mask"][()], dtype=bool)
    return center_crop_or_pad(m.astype(np.float32), n_dp) > 0.5


sensor_mask = {}          # label -> bool array, only for probes that have one
if not SYNTHETIC_ONLY:
    for label, cfg in EXPERIMENTAL.items():
        if cfg["recon"].exists():
            probes[label] = as_modes(load_experimental(cfg["recon"]))
            if cfg["dp"].exists():
                sensor_mask[label] = load_sensor_mask(cfg["dp"])
                frac = sensor_mask[label].mean() * 100
            else:
                frac = float("nan")
                print(f"  no dp file for {label}; proceeding without a sensor mask")
            print(f"loaded {label}: {tuple(probes[label].shape)}, mask {frac:.1f}% valid")
        else:
            print(f"SKIP {label}: {cfg['recon']} not reachable")

def phase_randomize(probe_modes, rng_seed=0):
    """Same |A(q)|, randomized arg A(q).

    This is the control that isolates the claim. It holds the pupil's amplitude, support
    and therefore kp exactly fixed, and changes only the relative phase of the Fourier
    components -- the single quantity the counting argument cannot see. Any difference in
    T(k) between a probe and its phase-randomized twin is attributable to pupil phase alone.
    """
    gen = torch.Generator(device="cpu").manual_seed(rng_seed)
    a_q = torch.fft.fftshift(torch.fft.fft2(probe_modes), dim=(-2, -1))
    ph = (torch.rand(a_q.shape, generator=gen) * 2 * math.pi).to(a_q.device)
    a_q = a_q.abs() * torch.exp(1j * ph)
    return torch.fft.ifft2(torch.fft.ifftshift(a_q, dim=(-2, -1)))


probes["ZP (synth), rand phase"] = phase_randomize(probes["ZP (synth)"])
if "ZP (exp, S0040)" in probes:
    probes["ZP (exp), rand phase"] = phase_randomize(probes["ZP (exp, S0040)"])

probes = {k: rescale_to_counts(v) for k, v in probes.items()}


#%% ----------------------------------------------- what the real sample actually contains
def sample_power_spectrum(obj_patch):
    """Radially averaged power of a complex object patch's phase, and its cumulative sum.

    Returned against |k| in DETECTOR pixels. That mapping is exact and needs no conversion
    factor as long as the patch spans the detector field of view lambda*z/Delta: the object
    grid's conjugate bins are then detector pixels one-for-one. Hann-windowed, because a
    hard crop's edge discontinuity puts a 1/k tail across the whole spectrum and would fake
    high-frequency content.
    """
    ph = np.angle(obj_patch)
    ph = ph - ph.mean()
    n = ph.shape[-1]
    ph = ph * np.outer(np.hanning(n), np.hanning(n))
    f2 = np.abs(np.fft.fftshift(np.fft.fft2(ph))) ** 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - n / 2, xx - n / 2).astype(int)
    power = np.bincount(r.ravel(), f2.ravel())
    return np.arange(len(power)), power, np.cumsum(power) / max(power.sum(), 1e-300)


def sample_percentiles(cum, fracs=(0.25, 0.50, 0.75, 0.90, 0.95, 0.99)):
    return {f: int(np.searchsorted(cum, f)) for f in fracs}


sample = {}        # region label -> (k, power, cumulative)
if STEPSCAN["recon"].exists():
    with h5py.File(STEPSCAN["recon"], "r") as _f:
        _obj = np.asarray(_f["object"][()]).squeeze()
        _dx = float(_f["obj_pixel_size_m"][()])
        _pos = np.asarray(_f["positions_px"][()], dtype=np.float64)
    # Patch side that spans exactly the detector FOV, so bins == detector px at any n_dp.
    _n_fov = int(round(wavelength_m * det_dist_m / det_pixel_m / _dx))
    _patch, _ctr = crop_at_position(_obj, _pos[STEPSCAN["frame_index"]], (_n_fov, _n_fov))
    print(f"step-scan object {_obj.shape} at {_dx*1e9:.3f} nm/px; FOV patch {_n_fov} px")
    print(f"  frame {STEPSCAN['frame_index']} position "
          f"({_pos[STEPSCAN['frame_index']][0]:.1f}, {_pos[STEPSCAN['frame_index']][1]:.1f}) "
          f"-> patch centre {_ctr}  (object centre {_obj.shape[0]//2}, {_obj.shape[1]//2})")
    sample[f"frame {STEPSCAN['frame_index']}"] = sample_power_spectrum(_patch)
    _c0, _c1 = _obj.shape[0] // 2, _obj.shape[1] // 2
    sample["central crop"] = sample_power_spectrum(
        _obj[_c0 - _n_fov // 2:_c0 + _n_fov // 2, _c1 - _n_fov // 2:_c1 + _n_fov // 2])
else:
    print(f"no step-scan recon at {STEPSCAN['recon']}; skipping sample spectrum")


#%% ------------------------------------------------------------------ the measurement
def obj_sinusoid(ky, kx, a):
    """O = exp(i*a*cos(2*pi*k.r)) with k an INTEGER number of detector pixels.

    Integer k means the +-k sidebands in (*) land exactly on detector pixels, so no
    interpolation enters anywhere and the analytic cross-check below is exact.
    """
    # float64 explicitly: torch.arange gives int64, and int64/int would demote the whole
    # expression to float32, silently undoing the complex128 chosen for the differencing.
    y = torch.arange(n_dp, device=torch_device, dtype=torch.float64).view(-1, 1)
    x = torch.arange(n_dp, device=torch_device, dtype=torch.float64).view(1, -1)
    return torch.exp(1j * a * torch.cos(2 * math.pi * (ky * y + kx * x) / n_dp)).to(CDTYPE)


def dI_da(probe_modes, ky, kx, a=amp):
    """Central difference of the project's own forward model. Returns (n_dp, n_dp) real."""
    ip = simulate_rpi_diffraction(obj_sinusoid(ky, kx, +a), probe_modes, n_dp, 0.0)
    im = simulate_rpi_diffraction(obj_sinusoid(ky, kx, -a), probe_modes, n_dp, 0.0)
    return (ip - im) / (2 * a)


def dI_da_analytic(probe_modes, ky, kx):
    """-sum_modes Im[A*(q) (A(q-k) + A(q+k))], evaluated by rolling the shifted pupil.

    np.roll(A, +k)[q] == A[q-k], so the two rolls are the -+k sidebands. Summed over
    incoherent modes, since the detector adds their intensities.
    """
    A = torch.fft.fftshift(torch.fft.fft2(probe_modes), dim=(-2, -1))
    a_minus = torch.roll(A, shifts=(ky, kx), dims=(-2, -1))
    a_plus = torch.roll(A, shifts=(-ky, -kx), dims=(-2, -1))
    return -(torch.conj(A) * (a_minus + a_plus)).imag.sum(0)


def radial_k_list():
    """[(radius_bins, [(ky, kx), ...]), ...] over azimuths in [0, pi)."""
    radii = np.linspace(2, k_max_bins, n_radii)
    # Add the predicted CTF zero radii explicitly. The generic grid has ~4 px spacing while
    # a zero of sin^2(pi*lambda*dz*k^2) is far narrower than that, so without this the
    # validation measures wherever the grid happened to land, not the dip.
    _z = np.sqrt(np.arange(1, 6) / (wavelength_m * defocus_m)) * n_dp * pixel_size_m
    radii = np.concatenate([radii, _z[(_z >= 2) & (_z <= k_max_bins)]])
    radii = np.unique(np.round(radii).astype(int))
    out = []
    for m in radii:
        ks = {(int(round(m * math.sin(t))), int(round(m * math.cos(t))))
              for t in np.linspace(0, math.pi, n_azim, endpoint=False)}
        ks.discard((0, 0))
        if ks:
            out.append((int(m), sorted(ks)))
    return out


def sensitivity_curves(probe_modes):
    """T(k), F(k) radially averaged over azimuth, plus a sub-photon diagnostic.

    The fourth return is the fraction of each F(k) contributed by pixels holding less than
    one count. F = (dI/da)^2 / I divides out the local intensity, so a faint halo pixel is
    weighted like a bright core pixel; below one count the Poisson expression is still
    finite and large but the information is not physically realizable. That is why F(k)
    stays flat past the 2kp knee where T(k) has collapsed. F itself is UNCHANGED -- this
    only measures how much of it sits on pixels that record no photons, so the caveat can
    be quoted rather than asserted.
    """
    i0 = simulate_rpi_diffraction(
        torch.ones((n_dp, n_dp), dtype=CDTYPE, device=torch_device), probe_modes, n_dp, 0.0)
    denom = i0 + bkg_counts
    lit = i0 >= 1.0
    radii, t_of_k, f_of_k, sub_of_k = [], [], [], []
    for m, ks in radial_k_list():
        t_acc, f_acc, f_lit = 0.0, 0.0, 0.0
        for ky, kx in ks:
            d = dI_da(probe_modes, ky, kx)
            d2 = d ** 2
            t_acc += d2.sum().item()
            f = d2 / denom
            f_acc += f.sum().item()
            f_lit += (f * lit).sum().item()
        radii.append(m)
        t_of_k.append(t_acc / len(ks))
        f_of_k.append(f_acc / len(ks))
        sub_of_k.append(1.0 - f_lit / max(f_acc, 1e-300))
    return (np.array(radii), np.array(t_of_k), np.array(f_of_k), np.array(sub_of_k))


curves = {}
with torch.no_grad():          # nothing here needs a graph; keeps peak memory flat
    for label, pm in probes.items():
        curves[label] = sensitivity_curves(pm)
        print(f"  T(k) done: {label}")


#%% ------------------------------------------------------------------ self-tests
report = []


def check(name, ok, detail):
    report.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(report[-1])


# 1. Numerics vs. the analytic formula (*), on the probe with the richest pupil phase.
#    Fixed at 0.5 kp deliberately, NOT tied to panel_b_k_over_kp: the validation wants a
#    frequency where both sidebands coexist over a good fraction of the pupil, so changing
#    what the figure displays must not quietly change what the test exercises.
_ref = probes["RPI (synth)"]
_ky, _kx = 0, max(2, int(round(0.5 * kp_nominal)))
_num, _ana = dI_da(_ref, _ky, _kx), dI_da_analytic(_ref, _ky, _kx)
_rel = (torch.linalg.norm(_num - _ana) / torch.linalg.norm(_ana)).item()
check("analytic agreement", _rel < 1e-2, f"relative RMS {_rel:.2e} at k=({_ky},{_kx})")

# 2. Linearity: the central difference must be independent of a while a is small.
_d1 = dI_da(_ref, _ky, _kx, a=1e-4)
_d2 = dI_da(_ref, _ky, _kx, a=1e-2)
_lin = (torch.linalg.norm(_d2 - _d1) / torch.linalg.norm(_d1)).item()
check("linearity in a", _lin < 1e-2, f"a=1e-4 vs 1e-2 differ by {_lin:.2e}")

# 3. In-focus null: a flat real pupil must be first-order blind.
_null = curves["ideal pupil, in focus"][1].max() / curves["RPI (synth)"][1].max()
check("in-focus null", _null < 1e-3, f"ideal in-focus T(k) is {_null:.2e} x RPI")

# 4. CTF zeros of the defocused ideal pupil at lambda*dz*k^2 = integer.
#    Checking that a predicted zero lands near a sampled radius proves nothing -- with any
#    dense sampling it always does. The real test is that T(k) actually DIPS there, so
#    compare T at each predicted zero against the local median.
_r, _t, _, _ = curves[f"ideal pupil, dz={defocus_m*1e6:.0f} um"]
_k_phys = _r / (n_dp * pixel_size_m)
_pred = np.sqrt(np.arange(1, 6) / (wavelength_m * defocus_m))
# Only inside |k| < kp, where both sidebands coexist at the same q -- see `defocus_m`.
_kp_phys = kp_nominal / (n_dp * pixel_size_m)
_pred = _pred[(_pred >= _k_phys.min()) & (_pred <= min(_k_phys.max(), _kp_phys))]
#    Summed over the WHOLE detector the zeros are only partially filled in, even below kp:
#    the pupil splits into a triple-overlap region, where both sidebands interfere and the
#    sin(pi*lambda*dz*k^2) cancellation happens, and a single-sideband annulus where
#    Im[A*(q)A(q-k)] = A0^2 sin(pi*lambda*dz*(k^2 - 2 q.k)) has no zero at all and fills the
#    dip back in. So test the cancellation where the analytic result actually applies, by
#    restricting the q-sum to S_A ^ (S_A+k) ^ (S_A-k). A dense scan across the first zero
#    then has to show a deep, narrow dip.
def t_triple_overlap(label, k_bins):
    """T(k) summed only over q where BOTH sidebands are present."""
    pm = probes[label]
    a_q = torch.fft.fftshift(torch.fft.fft2(pm), dim=(-2, -1))
    supp = (a_q.abs() > 1e-8 * a_q.abs().max()).any(0)
    vals = []
    for _m in k_bins:
        _k = (0, int(_m))
        tri = (supp
               & torch.roll(supp, shifts=_k, dims=(-2, -1))
               & torch.roll(supp, shifts=(-_k[0], -_k[1]), dims=(-2, -1)))
        vals.append(((dI_da(pm, *_k) ** 2) * tri).sum().item())
    return np.array(vals)


_lbl = f"ideal pupil, dz={defocus_m*1e6:.0f} um"
if len(_pred):
    _z1 = _pred[0] * n_dp * pixel_size_m                       # first zero, in detector px
    _scan = np.unique(np.round(np.linspace(0.7 * _z1, 1.3 * _z1, 25)).astype(int))
    _tt = t_triple_overlap(_lbl, _scan)
    _dip = _tt.min() / max(float(np.median(_tt)), 1e-300)
    check("CTF zero (triple-overlap)", _dip < 0.1,
          f"first zero predicted at {_z1:.1f} px, measured min at "
          f"{_scan[int(np.argmin(_tt))]} px, depth {_dip:.2e} x median over the scan")
else:
    check("CTF zero (triple-overlap)", False, "no predicted zero inside |k| < kp")

# 5. Band limit: response must collapse past |k| = 2*kp.
_r, _t, _, _ = curves["RPI (synth)"]
_in = _t[_r < 1.8 * kp_nominal].max()
_out = _t[_r > 2.2 * kp_nominal]
check("2*kp band limit", _out.size > 0 and _out.max() / _in < 1e-2,
      f"T beyond 2kp is {(_out.max()/_in if _out.size else float('nan')):.2e} x in-band")

# 6. kp at a matched quantile. The claim being tested is that the two probes have the same
#    angular support, so counting cannot explain any resolution difference. That claim is
#    about the EXPERIMENTAL pair -- assert on those when present. The synthetic pair is
#    reported too but not asserted: the randomized builder's binary phase mask has sharp
#    edges at the 6.25 nm zone-plate-plane grid, and the OSA cuts harder into its broad
#    speckle field than into a 50 nm focused core, both of which broaden its far field
#    beyond the geometric aperture. That is a property of the idealized construction.
# kp about the measured pupil centre, with the sensor's dead pixels excluded. Keep the
# array-centre and unmasked values too: this is a correction to how every previous run was
# measured, so it should be auditable rather than silently applied.
pupil_c, kp, kp_arr, kp_nomask = {}, {}, {}, {}
for _lbl, _pm in probes.items():
    _v = sensor_mask.get(_lbl)
    _ff = probe_fourier_power(_pm.cpu())
    pupil_c[_lbl] = probe_fourier_center(ff=_ff, valid=_v)
    kp[_lbl] = probe_fourier_radius(_pm.cpu(), kp_quantile, valid=_v)
    kp_nomask[_lbl] = probe_fourier_radius(_pm.cpu(), kp_quantile)
    kp_arr[_lbl] = probe_fourier_radius(_pm.cpu(), kp_quantile,
                                        center=(n_dp / 2, n_dp / 2), valid=_v)
print("  kp at q=%.2f: " % kp_quantile
      + ", ".join(f"{k}={v}" for k, v in kp.items()))
_pair = ("ZP (exp, S0040)", "RPI (exp, S0019)")
if all(p in kp for p in _pair):
    _d = abs(kp[_pair[0]] - kp[_pair[1]])
    check("kp matched (experimental)", _d <= 0.1 * kp_nominal,
          f"ZP {kp[_pair[0]]} px vs RPI {kp[_pair[1]]} px, differ by {_d} px "
          f"(<= {0.1*kp_nominal:.0f} required); synthetic pair "
          f"{kp['ZP (synth)']}/{kp['RPI (synth)']} px")
else:
    check("kp matched (experimental)", False,
          "experimental probes not loaded; synthetic pair "
          f"{kp['ZP (synth)']}/{kp['RPI (synth)']} px")


# 7. The RPI band-limit heuristic R <= 0.94 should fall out of the classical sigma >= 2
#    criterion rather than being an arbitrary safety margin. Object on a square n_obj grid
#    (Fourier half-side ko), probe pupil a disk of radius kp: the exit-wave spectrum is
#    their Minkowski sum, so the detector pixels carrying signal number
#    4ko^2 + 8 ko kp + pi kp^2, against 2*n_obj^2 = 8 ko^2 REAL unknowns for a complex
#    object. Requiring measurements >= sigma * unknowns and dividing through by kp^2 gives
#        (8 sigma - 4) R^2 - 8 R - pi <= 0,   R = ko/kp
#    whose positive root is the largest admissible R. This test asserts the arithmetic, so
#    it fails loudly if the geometry assumptions above are ever edited.
#    The count above treats every pixel of the object grid as an independent unknown, i.e.
#    assumes the probe fills the whole reconstruction FOV. It never does. Only the
#    illuminated part is constrained by the data, so the unknown count carries the fill
#    fraction f = (pi/4)(d_footprint/n_dp)^2, giving (8 sigma f - 4) R^2 - 8 R - pi <= 0.
#    f = 1 recovers the familiar bound; f < 1/(2 sigma) makes the leading coefficient
#    non-positive and removes the bound entirely.
def r_max(sigma, f=1.0):
    a, b, c = 8 * sigma * f - 4, -8.0, -math.pi
    if a <= 0:                      # opens downward (or linear): no positive root
        return float("inf")
    return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)


check("sigma>=2 reproduces R<=0.94", abs(r_max(2) - 0.944) < 0.01,
      f"sigma>=1 -> R <= {r_max(1):.3f};  sigma>=2 -> R <= {r_max(2):.3f} "
      f"(hardcoded heuristic 0.94, RPI_SS_recon_50nmZP.py:335)")

check("bound relaxes as fill fraction falls",
      r_max(2, 0.688) > r_max(2, 1.0) and math.isinf(r_max(2, 0.25)),
      f"f=1.00 -> R <= {r_max(2, 1.0):.3f};  f=0.69 -> R <= {r_max(2, 0.688):.3f};  "
      f"f=0.25 -> unbounded")


#%% ------------------------------------------------------------------ figure
out_dir.mkdir(parents=True, exist_ok=True)


def _fresh_path(name):
    """Never overwrite: if `name` exists, append _v2, _v3, ... and return that instead.

    Two reasons. Results from a previous run are kept rather than silently replaced, and on
    Windows a PNG held open by an image viewer or a sync client makes savefig fail with
    OSError 22 -- writing a new name sidesteps the lock instead of losing the figure.
    """
    path = out_dir / name
    if not path.exists():
        return path
    stem, ext, n = path.stem, path.suffix, 2
    while (out_dir / f"{stem}_v{n}{ext}").exists():
        n += 1
    path = out_dir / f"{stem}_v{n}{ext}"
    print(f"  (target existed; writing {path.name})")
    return path


# Which sample region drives the figure annotations: the patch the single-frame
# reconstructions actually cover. The central crop stays in the metrics file as the
# representativeness check.
_sample_key = next((k for k in sample if k.startswith("frame")), None)

main = [k for k in probes if "ideal" not in k]
# One colour per probe, shared by every panel AND by every figure. Letting each panel run
# the default cycle over a different subset gives the same probe different colours in
# adjacent panels; keying off the full probe dict also keeps a probe the same colour
# between the all-probe figure and the experiment-only one.
COLOR = {lbl: c for lbl, c in zip(probes, plt.rcParams["axes.prop_cycle"].by_key()["color"]
                                  + ["k", "0.5", "tab:olive", "tab:cyan"])}


def make_figure(image_labels, curve_labels_T, curve_labels_F, norm_label, suffix,
                show_ctf=True):
    """Render the four-panel figure for a chosen subset of probes.

    image_labels  probes tiled in panels (a) and (b)
    curve_labels_T / _F   what panels (c) and (d) draw
    norm_label    probe whose peak sets the shared T/F normalization. It must stay a single
                  scale shared by every curve: the probes were deliberately dose-matched in
                  rescale_to_counts, so per-curve normalization would throw away exactly the
                  comparison the figure exists to make.
    show_ctf      draw the predicted CTF zero lines, which belong to the defocused ideal
                  pupil -- meaningless, and misleading, in a figure that does not show it.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (a) pupil intensity, log, with the matched-quantile kp circle
    ax = axes[0, 0]
    panel_a = list(image_labels)
    stack = []
    for label in panel_a:
        a_q = torch.fft.fftshift(torch.fft.fft2(probes[label]), dim=(-2, -1)).abs() ** 2
        stack.append(a_q.sum(0).cpu().numpy())
    half = int(2.6 * kp_nominal)
    sl = slice(n_dp // 2 - half, n_dp // 2 + half)
    comp = np.concatenate([s[sl, sl] for s in stack], axis=1)
    im = ax.imshow(np.log10(comp + comp.max() * 1e-8), cmap="inferno", origin="lower")
    for j, label in enumerate(panel_a):
        # Centre the circle on the MEASURED pupil centre, not the tile centre. The direct
        # beam sits a few px off the array centre, which is plainly visible against the
        # annulus when the circle is drawn at n_dp/2.
        cy, cx = pupil_c[label]
        ax.add_patch(plt.Circle((cx - sl.start + 2 * half * j, cy - sl.start), kp[label],
                                fill=False, color="w", lw=0.8, ls="--"))
        ax.text(2 * half * j + 4, 2 * half - 20,
                f"{label}\n$k_p$={kp[label]}  offset "
                f"({cy - n_dp/2:+.1f}, {cx - n_dp/2:+.1f})", color="w", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(r"$\log_{10}|A(q)|^2$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # (b) dI/da at one mid-band k -- the null vs the full-field response
    ax = axes[0, 1]
    mid_k = (0, max(2, int(round(panel_b_k_over_kp * kp_nominal))))
    maps = [dI_da(probes[l], *mid_k)[sl, sl].cpu().numpy() for l in panel_a]
    # 99.5th percentile, not the max: a handful of bright pixels near the optical axis set a
    # vmax two orders above everything else and render the whole panel flat grey. One shared
    # scale across both tiles, so they are directly comparable.
    vmax = max(np.percentile(np.abs(m), 99.5) for m in maps)
    im = ax.imshow(np.concatenate(maps, axis=1), cmap="coolwarm", origin="lower",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    half_period_nm = n_dp * pixel_size_m * 1e9 / (2 * mid_k[1])
    _pct_txt = ""
    if _sample_key:
        _c = sample[_sample_key][2]
        _pct_txt = f", {100 * _c[min(mid_k[1], len(_c) - 1)]:.0f}th pct"
    ax.set_ylabel(rf"$\partial I/\partial a$,  $|k|$={mid_k[1]} px"
                  f" ({half_period_nm:.0f} nm{_pct_txt})")
    # Both tiles span the same area of k-space -- that is forced by the matched pupil
    # support, and is the point rather than a defect. What differs is the MAGNITUDE inside
    # that area, which the colour scale alone does not convey, so state T = sum (dI/da)^2
    # per tile.
    # The sideband regime is per PROBE, not per figure: the triple overlap
    # S_A ^ (S_A+k) ^ (S_A-k) is empty once |k| >= that probe's own kp, and the measured kp
    # differ (85, 84 experimental; 90, 132 synthetic). Judging it against kp_nominal would
    # mislabel whichever probe's pupil is smaller. Visually, one sideband shows as two
    # separated lobes, two sidebands as a connected region.
    for j, label in enumerate(panel_a):
        _sb = "1 sideband" if mid_k[1] >= kp[label] else "2 sidebands"
        ax.text(2 * half * j + 4, 2 * half - 20,
                f"{label}\n$T$={np.sum(maps[j] ** 2):.2e}, {_sb}", color="k", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"counts / rad  (shared scale, clipped at 99.5th pct)", fontsize=8)

    # (c) T(k) and (d) F(k), both normalized by ONE common scale -- see `norm_label`.
    norm_T = max(curves[norm_label][1].max(), 1e-300)
    norm_F = max(curves[norm_label][2].max(), 1e-300)
    for ax, idx, sym, scale, labels in (
        (axes[1, 0], 1, r"$T(k)$", norm_T, curve_labels_T),
        (axes[1, 1], 2, r"$F(k)$", norm_F, curve_labels_F),
    ):
        for label in labels:
            r = curves[label][0]
            y = curves[label][idx] / scale
            drawn = y.max() > 1e-11          # the in-focus null sits ~1e-26; say so, don't
            ax.semilogy(r, y, lw=1.2, color=COLOR[label],                   # silently omit it
                        ls="--" if ("ideal" in label or "rand phase" in label) else "-",
                        label=label + ("" if drawn else f"  (off scale, {y.max():.0e})"))
        for x, ls in ((kp_nominal, ":"), (2 * kp_nominal, "--")):
            ax.axvline(x, color="k", lw=0.8, ls=ls)
        # x in data coords, y in axes fraction -- a hardcoded data y lands outside the axes
        # whenever ylim changes, and drags the saved figure's bounding box with it.
        ax.text(2 * kp_nominal, 0.02, r"  $2k_p$", fontsize=8,
                transform=ax.get_xaxis_transform())
        if idx == 1 and show_ctf:
            # predicted CTF zeros, lambda*dz*k^2 = 1, 2, 3, ... converted to detector px
            for z in _pred:
                ax.axvline(z * n_dp * pixel_size_m, color="0.6", lw=0.6, ls="-.")
        ax.set_xlabel(r"$|k|$  [detector px]")
        ax.set_ylabel(sym + "  (norm.)")
        ax.set_ylim(1e-5, 3)
        # Pin the x range: the sample spectrum drawn below extends to the patch's corner
        # radius (~724 bins), and on a shared-x twin axis that autoscales the whole panel
        # out to it, squashing everything of interest into the left eighth.
        ax.set_xlim(0, k_max_bins)
        ax.legend(fontsize=7, loc="lower left")
        sec = ax.secondary_xaxis("top", functions=(
            lambda v: np.divide(n_dp * pixel_size_m * 1e9, 2 * np.maximum(v, 1e-9)),
            lambda v: np.divide(n_dp * pixel_size_m * 1e9, 2 * np.maximum(v, 1e-9))))
        # Explicit ticks: the k -> half-period map diverges at k=0, so the automatic locator
        # emits nonsense near the origin.
        sec.set_xticks([hp for hp in (500, 200, 100, 50, 30, 25, 20)
                        if 2 <= n_dp * pixel_size_m * 1e9 / (2 * hp) <= k_max_bins])
        sec.set_xlabel("half-period  [nm]", fontsize=8)
        # Which frequencies the real sample actually occupies. Without this the transfer
        # functions float free of any statement about what matters: F(k) stays high well
        # past kp, but the sample has almost no power out there.
        if _sample_key:
            kk, _, cum = sample[_sample_key]
            ax2 = ax.twinx()
            ax2.plot(kk, cum, color="0.45", lw=0.9, ls=":")
            ax2.set_ylim(0, 1.02)
            ax2.set_ylabel("cumulative sample power", fontsize=8, color="0.45")
            ax2.tick_params(axis="y", labelsize=7, colors="0.45")

    fig.tight_layout()
    fig_path = _fresh_path(f"probe_sensitivity_{STAMP}_n{n_dp}{suffix}.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


# The full figure: every probe, with the ideal-pupil reference cases that make the
# in-focus null and the CTF zeros legible as correctness checks.
make_figure(main[:2], list(curves), main, "RPI (synth)", "")

# The experiment-only figure: strictly the two reconstructed probes. No ideal pupils, no
# synthetic optics, and not the phase-randomized control -- that one is derived from a
# reconstructed probe rather than reconstructed itself, and the mechanism argument it
# supports stays in the full figure above. CTF guide lines are off: they are a property of
# the defocused ideal pupil, so drawing them here would imply these curves should dip there.
EXP = ["ZP (exp, S0040)", "RPI (exp, S0019)"]
if all(e in probes for e in EXP):
    make_figure(EXP, EXP, EXP, "RPI (exp, S0019)", "_exp", show_ctf=False)
else:
    print(f"skipping experiment-only figure: {[e for e in EXP if e not in probes]} not loaded")


#%% ------------------------------------------------------------------ metrics table
def footprint_px(probe_modes, frac=0.95):
    """Diameter, in object pixels, of the disk holding `frac` of the probe's power."""
    inten = (probe_modes.abs() ** 2).sum(0).cpu().numpy().astype(np.float64)
    yy, xx = np.mgrid[:n_dp, :n_dp] - n_dp // 2
    r = np.hypot(yy, xx)
    order = np.argsort(r.ravel())
    cum = np.cumsum(inten.ravel()[order])
    return 2 * r.ravel()[order][np.searchsorted(cum, frac * cum[-1])]


lines = [
    f"probe sensitivity metrics  [{STAMP}]",
    f"n_dp={n_dp}  pixel={pixel_size_m*1e9:.2f} nm  lambda={wavelength_m*1e12:.1f} pm  "
    f"z={det_dist_m} m  NA={NA*1e3:.3f} mrad  kp_nominal={kp_nominal:.1f} px",
    f"all probes rescaled to {total_counts:.1e} counts; kp quantile forced to {kp_quantile}",
    "",
    f"{'probe':<28}{'kp[px]':>8}{'fp50':>8}{'fp95':>8}"
    f"{'rho':>9}{'s_lin50':>9}{'s_lin95':>9}{'s_ar50':>9}{'s_ar95':>9}"
    f"{'T_hi/T_lo':>12}{'F_band':>12}",
]


def t_at(label, k_bins, idx=1):
    """T(k) (idx=1) or F(k) (idx=2) interpolated at a radius in detector pixels."""
    r = curves[label][0]
    return float(np.interp(k_bins, r, curves[label][idx]))


def f_band(label):
    """Mean shot-noise Fisher information over 0.2..1.5 kp.

    Directly comparable between probes because every probe was rescaled to the same total
    counts, so this is information per unit dose -- the quantity the resolution claim
    ultimately rests on.
    """
    r, _, f, _ = curves[label]
    sel = (r >= 0.2 * kp_nominal) & (r <= 1.5 * kp_nominal)
    return float(np.mean(f[sel])) if sel.any() else float("nan")


for label, pm in probes.items():
    k_p = kp[label]
    fp50, fp = footprint_px(pm, 0.50), footprint_px(pm, 0.95)
    # rho is what the first version of this table reported: detector pixels inside the
    # pupil over object pixels inside the footprint. It is NOT the classical oversampling
    # ratio -- its numerator counts only the direct-beam disk, so it is smaller by
    # (2 kp / n_dp)^2 and cannot be compared with the literature's ">= 2". Reported here
    # only so the two runs can be reconciled; sigma_lin / sigma_area are the real ones.
    n_det = math.pi * k_p ** 2
    n_obj = math.pi / 4 * fp ** 2
    rho = n_det / max(n_obj, 1e-9)
    s_lin50, s_lin95 = n_dp / max(fp50, 1e-9), n_dp / max(fp, 1e-9)
    s_ar50, s_ar95 = 4 / math.pi * s_lin50 ** 2, 4 / math.pi * s_lin95 ** 2
    # Peak T says how much signal the probe makes in total, which is dominated by low k and
    # so does not answer the question that matters. T_hi/T_lo is the shape: how much of the
    # response survives out at high spatial frequency. It is dose-independent by
    # construction, so it compares probes directly.
    t_ratio = t_at(label, 1.5 * kp_nominal) / max(t_at(label, 0.2 * kp_nominal), 1e-300)
    lines.append(
        f"{label:<28}{k_p:>8d}{fp50:>8.0f}{fp:>8.0f}"
        f"{rho:>9.3f}{s_lin50:>9.2f}{s_lin95:>9.2f}{s_ar50:>9.2f}{s_ar95:>9.2f}"
        f"{t_ratio:>12.3e}{f_band(label):>12.3e}")

lines += [
    "",
    "fp50/fp95 = footprint diameter [object px] holding 50% / 95% of the probe power.",
    "rho       = (det px inside kp)/(object px inside the 95% footprint). NOT the classical",
    "            oversampling ratio -- the numerator counts only the direct-beam disk, so rho",
    "            is smaller by (2 kp/n_dp)^2 and must NOT be compared with the '>= 2'",
    "            threshold. Reported only to reconcile with the first version of this table:",
    "                sigma_lin = (n_dp / 2 kp) * sqrt(rho)      [exact]",
    "s_lin     = n_dp / footprint: the linear ratio of RPI_SS_recon_50nmZP.py:453. The",
    "            familiar support-doubling rule wants this >= 2.",
    "s_ar      = (4/pi) * s_lin^2: Miao's degrees-of-freedom ratio, whose own threshold is",
    "            also 2 (so s_lin >= sqrt(pi/2) ~ 1.25). These are DIFFERENT criteria that",
    "            share the number 2; the 4/pi is the disk-vs-square support convention.",
    "            Quoting both at 50% and 95% exposes how much the answer depends on where",
    "            the support is cut -- for a probe with a halo, a great deal.",
    "T_hi/T_lo = T(1.5 kp)/T(0.2 kp): fraction of the first-order response surviving to high",
    "            spatial frequency. Dose-independent.",
    "F_band    = mean shot-noise Fisher information over 0.2..1.5 kp at matched total",
    "            counts: information per unit dose.",
    "",
    "The 'rand phase' rows hold |A(q)| -- and therefore kp and every counting quantity that",
    "depends on the pupil -- exactly fixed, and randomize only arg A(q).",
    "",
    "kp audit. The direct beam is not centred on the detector crop, and the sensor's",
    "inter-module gaps carry no signal; measuring the radial profile about n_dp/2 over all",
    "pixels -- as every run before this one did -- inflates kp on both counts.",
    "",
    f"{'probe':<28}{'offset (dy,dx)':>18}{'kp@array':>10}{'kp@ctr':>9}{'kp@ctr+mask':>13}",
]
for label in probes:
    cy, cx = pupil_c[label]
    lines.append(f"{label:<28}"
                 f"{f'({cy-n_dp/2:+.1f}, {cx-n_dp/2:+.1f})':>18}"
                 f"{kp_arr[label]:>10d}{kp_nomask[label]:>9d}{kp[label]:>13d}")
lines += [
    "",
    "The kp column in the table above is the rightmost one (centred, masked).",
    "",
]

def _fmt_r(v):
    return "unbounded" if math.isinf(v) else f"{v:.3f}"


lines += [
    "RPI band-limit ratio R = ko/kp, largest value admitted by each criterion,",
    "assuming the probe fills the whole reconstruction FOV (fill fraction f = 1):",
    f"    support / heterodyne ceiling  (S_A ^ (S_A +- k) non-empty)   R <= 2",
    f"    bare determinacy    sigma >= 1                               R <= {_fmt_r(r_max(1))}",
    f"    classical CDI       sigma >= 2                               R <= {_fmt_r(r_max(2))}",
    "",
    "The last line reproduces the 0.94 hardcoded at RPI_SS_recon_50nmZP.py:335, so that",
    "heuristic is the sigma >= 2 criterion for this geometry rather than an arbitrary",
    "safety margin -- but only at f = 1.",
    "",
    "f = 1 never holds. The object is parameterised over the full FOV while the data",
    "constrain only the illuminated part, so the unknown count carries",
    "f = (pi/4)(footprint/n_dp)^2 and the bound becomes (8 sigma f - 4) R^2 - 8R - pi <= 0.",
    "Per probe, at sigma >= 2:",
    "",
    f"{'probe':<28}{'f(50%)':>9}{'R<=':>12}{'f(95%)':>9}{'R<=':>12}",
]
def fill_fraction(pm, frac):
    """Illuminated fraction of the field of view, clamped to 1.

    footprint_px returns an enclosing DIAMETER, and the enclosing radius can reach the
    array corners (n_dp/sqrt(2)), so the raw (pi/4)(d/n_dp)^2 can exceed 1 for a probe whose
    power is not confined to the FOV -- the ideal pupil's jinc tails do exactly that. Such a
    probe fills the FOV by any reasonable reading, so 1 is the right answer and the
    footprint measure has simply saturated.
    """
    return min(1.0, math.pi / 4 * (footprint_px(pm, frac) / n_dp) ** 2)


for label, pm in probes.items():
    f50, f95 = fill_fraction(pm, 0.50), fill_fraction(pm, 0.95)
    lines.append(f"{label:<28}{f50:>9.3g}{_fmt_r(r_max(2, f50)):>12}"
                 f"{f95:>9.3g}{_fmt_r(r_max(2, f95)):>12}")
lines += [
    "",
    "So the 0.94 heuristic is the f = 1 special case and these experiments are nowhere",
    "near it. Relaxing the band limit past 0.94 is not operating without redundancy -- the",
    "bound was simply computed for the wrong unknown count.",
    "",
]

if sample:
    fov_nm = wavelength_m * det_dist_m / det_pixel_m * 1e9
    lines += [
        "Where the REAL sample's phase power lies, from the step-scan reconstruction.",
        "k is in detector pixels (the patch spans the detector FOV, so object FFT bins map",
        "one-for-one). This is diagnostic only -- it enters nothing above.",
        "",
        f"{'cum. power':>11}" + "".join(f"{r:>22}" for r in sample)
        + f"{'half-period [nm]':>19}",
    ]
    for _frac in (0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        _ks = [sample_percentiles(c)[_frac] for _, _, c in sample.values()]
        _hp = fov_nm / (2 * max(_ks[0], 1))
        lines.append(f"{_frac:>11.2f}" + "".join(f"{k:>22d}" for k in _ks)
                     + f"{_hp:>19.1f}")
    _ks50 = [sample_percentiles(c)[0.50] for _, _, c in sample.values()]
    _ks95 = [sample_percentiles(c)[0.95] for _, _, c in sample.values()]
    _agree = abs(_ks50[0] - _ks50[-1]) <= 5 and abs(_ks95[0] - _ks95[-1]) <= 10
    lines += [
        "",
        ("The reconstructed frame is representative of the scan: the two regions agree at "
         "the\n50% and 95% levels." if _agree else
         "WARNING: the reconstructed frame's spectrum differs from the whole-scan average,\n"
         "so the single-frame comparison is NOT being judged on representative content."),
        "",
    ]
    if _sample_key:
        _cum = sample[_sample_key][2]
        _mid = max(2, int(round(panel_b_k_over_kp * kp_nominal)))
        lines += [
            f"Panel (b) uses k = {_mid} px ({fov_nm/(2*_mid):.0f} nm half-period) = the "
            f"{100*_cum[min(_mid, len(_cum)-1)]:.0f}th percentile",
            f"of sample power. Above kp ~ {kp_nominal:.0f} px only "
            f"{100*(1-_cum[min(int(kp_nominal), len(_cum)-1)]):.1f}% of the sample's phase",
            "power remains -- so the high-k band where F(k) is still strong carries a small",
            "fraction of the total signal. That is where a Siemens star's fine inner spokes",
            "live and is what resolution is judged on, but any claim resting on that band is",
            "resting on a few percent of the power.",
            "",
        ]
lines += report
# Curves to CSV. Without this they exist only in memory, so any statement about their shape
# in the writeup would be read off a PNG by eye instead of traced to a number.
_cols = ["k_detpx", "half_period_nm"]
_data = [curves[next(iter(curves))][0].astype(float)]
_data.append(wavelength_m * det_dist_m / det_pixel_m * 1e9 / (2 * np.maximum(_data[0], 1e-9)))
for _lbl in curves:
    _slug = _lbl.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    _cols += [f"T__{_slug}", f"F__{_slug}", f"Fsubphoton__{_slug}"]
    _data += [curves[_lbl][1], curves[_lbl][2], curves[_lbl][3]]
if _sample_key:
    _cum = sample[_sample_key][2]
    _cols.append("sample_cum_power")
    _data.append(np.array([_cum[min(int(k), len(_cum) - 1)] for k in _data[0]]))
csv_path = _fresh_path(f"probe_curves_{STAMP}_n{n_dp}.csv")
np.savetxt(csv_path, np.column_stack(_data), delimiter=",", header=",".join(_cols),
           comments="", fmt="%.6e")
print(f"wrote {csv_path}")

txt_path = _fresh_path(f"probe_metrics_{STAMP}_n{n_dp}.txt")
txt_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"wrote {txt_path}")
