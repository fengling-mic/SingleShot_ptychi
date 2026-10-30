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

Outputs (never overwrites; everything carries the dated suffix):
    <out_dir>/probe_sensitivity_20261029.png
    <out_dir>/probe_metrics_20261029.txt
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
    fourier_resample,
    make_randomized_zoneplate_probe,
    make_regular_zoneplate_probe,
    probe_fourier_radius,
    simulate_rpi_diffraction,
)

STAMP = "20261029"

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
n_dp                 = 1024
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
    ),
    "RPI (exp, S0019)": dict(
        recon=DATA2 / "12IDC/2026_Data/2026_3/01_piezo_test/ptychi_recons/S0019"
        / "Ndp1024_LSQML_c20_m0.5_gaussian_p10_cp_mm_opr3_ic_pc1_f_ul2" / "recon_Niter400.h5",
    ),
}

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

if not SYNTHETIC_ONLY:
    for label, cfg in EXPERIMENTAL.items():
        if cfg["recon"].exists():
            probes[label] = as_modes(load_experimental(cfg["recon"]))
            print(f"loaded {label}: {tuple(probes[label].shape)}")
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
    """T(k), F(k) radially averaged over azimuth."""
    i0 = simulate_rpi_diffraction(
        torch.ones((n_dp, n_dp), dtype=CDTYPE, device=torch_device), probe_modes, n_dp, 0.0)
    denom = i0 + bkg_counts
    radii, t_of_k, f_of_k = [], [], []
    for m, ks in radial_k_list():
        t_acc, f_acc = 0.0, 0.0
        for ky, kx in ks:
            d = dI_da(probe_modes, ky, kx)
            d2 = d ** 2
            t_acc += d2.sum().item()
            f_acc += (d2 / denom).sum().item()
        radii.append(m)
        t_of_k.append(t_acc / len(ks))
        f_of_k.append(f_acc / len(ks))
    return np.array(radii), np.array(t_of_k), np.array(f_of_k)


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
_r, _t, _ = curves[f"ideal pupil, dz={defocus_m*1e6:.0f} um"]
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
_r, _t, _ = curves["RPI (synth)"]
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
kp = {k: probe_fourier_radius(v.cpu(), kp_quantile) for k, v in probes.items()}
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


#%% ------------------------------------------------------------------ figure
out_dir.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
main = [k for k in probes if "ideal" not in k]
# One colour per probe, shared by every panel. Letting each panel run the default cycle
# over a different subset gives the same probe different colours in adjacent panels.
COLOR = {lbl: c for lbl, c in zip(probes, plt.rcParams["axes.prop_cycle"].by_key()["color"]
                                  + ["k", "0.5", "tab:olive", "tab:cyan"])}

# (a) pupil intensity, log, with the matched-quantile kp circle
ax = axes[0, 0]
panel_a = main[:2]
stack = []
for label in panel_a:
    a_q = torch.fft.fftshift(torch.fft.fft2(probes[label]), dim=(-2, -1)).abs() ** 2
    stack.append(a_q.sum(0).cpu().numpy())
half = int(2.6 * kp_nominal)
sl = slice(n_dp // 2 - half, n_dp // 2 + half)
comp = np.concatenate([s[sl, sl] for s in stack], axis=1)
im = ax.imshow(np.log10(comp + comp.max() * 1e-8), cmap="inferno", origin="lower")
for j, label in enumerate(panel_a):
    ax.add_patch(plt.Circle((half + 2 * half * j, half), kp[label], fill=False,
                            color="w", lw=0.8, ls="--"))
    ax.text(2 * half * j + 4, 2 * half - 14, f"{label}\n$k_p$={kp[label]}", color="w", fontsize=8)
ax.set_xticks([]); ax.set_yticks([])
ax.set_ylabel(r"$\log_{10}|A(q)|^2$")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

# (b) dI/da at one mid-band k -- the null vs the full-field response
ax = axes[0, 1]
mid_k = (0, max(2, int(round(0.5 * kp_nominal))))
maps = [dI_da(probes[l], *mid_k)[sl, sl].cpu().numpy() for l in panel_a]
# 99.5th percentile, not the max: a handful of bright pixels near the optical axis set a
# vmax two orders above everything else and render the whole panel flat grey.
vmax = max(np.percentile(np.abs(m), 99.5) for m in maps)
im = ax.imshow(np.concatenate(maps, axis=1), cmap="coolwarm", origin="lower",
               vmin=-vmax, vmax=vmax)
ax.set_xticks([]); ax.set_yticks([])
ax.set_ylabel(rf"$\partial I/\partial a$,  $|k|$={mid_k[1]} px")
for j, label in enumerate(panel_a):
    ax.text(2 * half * j + 4, 2 * half - 14, label, color="k", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

# (c) T(k) over EVERY probe, including the two ideal-pupil reference cases -- the
#     in-focus null and the defocused CTF zeros are the correctness check, and they only
#     mean anything next to the real probes.
# (d) F(k) over the real probes only.
#
# Both panels are normalized by ONE common scale (the RPI peak), never per-curve: the
# probes were deliberately dose-matched in rescale_to_counts, so rescaling each curve to
# its own maximum would throw away exactly the comparison this figure exists to make.
norm_T = max(curves["RPI (synth)"][1].max(), 1e-300)
norm_F = max(curves["RPI (synth)"][2].max(), 1e-300)
for ax, idx, sym, scale, labels in (
    (axes[1, 0], 1, r"$T(k)$", norm_T, list(curves)),
    (axes[1, 1], 2, r"$F(k)$", norm_F, main),
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
    if idx == 1:
        # predicted CTF zeros, lambda*dz*k^2 = 1, 2, 3, ... converted to detector px
        for z in _pred:
            ax.axvline(z * n_dp * pixel_size_m, color="0.6", lw=0.6, ls="-.")
    ax.set_xlabel(r"$|k|$  [detector px]")
    ax.set_ylabel(sym + "  (norm.)")
    ax.set_ylim(1e-5, 3)
    ax.legend(fontsize=7, loc="lower left")
    sec = ax.secondary_xaxis("top", functions=(
        lambda v: np.divide(n_dp * pixel_size_m * 1e9, 2 * np.maximum(v, 1e-9)),
        lambda v: np.divide(n_dp * pixel_size_m * 1e9, 2 * np.maximum(v, 1e-9))))
    # Explicit ticks: the k -> half-period map diverges at k=0, so the automatic locator
    # emits nonsense near the origin.
    sec.set_xticks([hp for hp in (500, 200, 100, 50, 30, 25, 20)
                    if 2 <= n_dp * pixel_size_m * 1e9 / (2 * hp) <= k_max_bins])
    sec.set_xlabel("half-period  [nm]", fontsize=8)

fig.tight_layout()
fig_path = out_dir / f"probe_sensitivity_{STAMP}.png"
fig.savefig(fig_path, dpi=120, bbox_inches="tight")
print(f"wrote {fig_path}")


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
    f"{'probe':<28}{'kp[px]':>8}{'theta_d[mrad]':>15}{'95% fp[px]':>12}"
    f"{'det px<kp':>11}{'ratio':>10}{'T_hi/T_lo':>12}{'F_band':>12}",
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
    r, _, f = curves[label]
    sel = (r >= 0.2 * kp_nominal) & (r <= 1.5 * kp_nominal)
    return float(np.mean(f[sel])) if sel.any() else float("nan")


for label, pm in probes.items():
    k_p = kp[label]
    fp = footprint_px(pm)
    n_det = math.pi * k_p ** 2
    n_obj = math.pi / 4 * fp ** 2
    # Peak T says how much signal the probe makes in total, which is dominated by low k and
    # so does not answer the question that matters. T_hi/T_lo is the shape: how much of the
    # response survives out at high spatial frequency. It is dose-independent by
    # construction, so it compares probes directly.
    t_ratio = t_at(label, 1.5 * kp_nominal) / max(t_at(label, 0.2 * kp_nominal), 1e-300)
    lines.append(
        f"{label:<28}{k_p:>8d}{k_p*det_pixel_m/det_dist_m*1e3:>15.3f}{fp:>12.1f}"
        f"{n_det:>11.0f}{n_det/max(n_obj,1e-9):>10.2f}"
        f"{t_ratio:>12.3e}{f_band(label):>12.3e}")
lines += ["", "ratio     = (detector px inside kp) / (object px inside 95% footprint),",
          "            i.e. the oversampling ratio computed identically for every probe.",
          "T_hi/T_lo = T(1.5 kp) / T(0.2 kp): fraction of the first-order response that",
          "            survives to high spatial frequency. Dose-independent.",
          "F_band    = mean shot-noise Fisher information over 0.2..1.5 kp, at matched",
          "            total counts: information per unit dose.",
          "",
          "The 'rand phase' rows hold |A(q)| -- and therefore kp and the oversampling",
          "ratio -- exactly fixed, and randomize only arg A(q). They isolate pupil phase",
          "from every counting quantity.", ""]
lines += report
txt_path = out_dir / f"probe_metrics_{STAMP}.txt"
txt_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"wrote {txt_path}")
