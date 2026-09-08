#%%
# RPI (Randomized Probe Imaging) single-shot reconstruction of one diffraction pattern
# from the 12-ID-C Siemens star scan, following Levitan et al., "Single-frame far-field
# diffractive imaging with randomized illumination," Opt. Express 28, 37103 (2020).
#
# Forward model (their Eq. 1):  E~ = F{ P . F^-1{ pad(F{O'}) } }
# The object O' lives on a grid COARSER than the probe/detector (band-limited), and is
# upsampled to full resolution by zero-padding its own Fourier transform. It is optimized
# directly with PyTorch autodiff + Adam against the single measured pattern; there is no
# Pty-Chi Task/task.run() here because Pty-Chi's forward model always keeps object and
# probe on the same pixel grid and cannot express the band limit. The probe comes from a
# prior multi-position Pty-Chi reconstruction (init_recon_file) and is held fixed, as in
# the paper -- see rpi_probe_start to release it.
#
# The size of the low-res object is set by the PROBE's numerical aperture, not by the
# detector: the paper's resolution ratio is R = ko/kp where kp is the probe's maximum
# spatial frequency (probe_fourier_radius below). Sizing it against the detector's
# Nyquist frequency instead silently runs at R ~ 1.5 here, far past the paper's
# reliability limit of ~0.6, and the reconstruction then just fits shot noise.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # This makes GPU N appear as GPU 0 to CuPy

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import ptychi.api  # noqa: F401  -- must precede the submodule import below, or pty-chi
                   # hits a circular import (forward_models has no attribute ForwardModel)
from ptychi.reconstructors.nn.models.autoencoder import Autoencoder
from ptychi.utils import (
    add_additional_opr_probe_modes_to_probe,
    get_default_complex_dtype,
    get_suggested_object_size,
    orthogonalize_initial_probe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


#%% ---------------------------------------------------------------- paths

scan = "S1567"
scan_initialGuess = "S1567"  # scan used to generate the initial guess (positions + probe)
data_root = Path("/mnt/micdata2/12IDC/2026_Data/2026_2/02_levitan")

dp_file = data_root / "preproc" / scan / "data_roi0_Ndp1024_dp.hdf5"
para_file = data_root / "preproc" / scan / "data_roi0_Ndp1024_para.hdf5"

# Prior multi-position Pty-Chi reconstruction used as the source of the probe and of
# calibration positions. Set to None to start from the para file positions and a
# synthesized probe instead.
init_recon_file = (
    data_root / "ptychi_recons" / scan_initialGuess
        / "Ndp600_LSQML_c30_m0.5_gaussian_p10_cp_mm_opr3_ic_pc1_f_ul2" / "recon_Niter1000.h5"
)

out_dir = Path(__file__).parent / "recon_out" / scan

# Set True to skip the beamline files and run the whole pipeline on a small
# synthetic multi-position dataset (useful to debug the structure without the data share).
use_simulated_data = False

frame_index = 62   # index into the loaded scan's patterns/positions to reconstruct


#%% ---------------------------------------------------------------- geometry & knobs

n_dp = 512                       # detector crop
wavelength_m = 0.155e-9
det_pixel_m = 172e-6             # detector pixel size
det_dist_m = 10.0                # sample-detector distance

pixel_size_m = wavelength_m * det_dist_m / (n_dp * det_pixel_m)

n_probe_modes = 5                # None = keep every incoherent mode in the prior probe
n_opr_modes = 1                  # a single frame carries no probe-variation information
probe_diameter_m = 3.0e-6        # only used when no probe comes from init_recon_file

# --- band limit -------------------------------------------------------------
# R = ko/kp, measured against the probe's own Fourier support (NOT the detector).
# The paper: R < 0.4 "virtually guaranteed to succeed", R <= 0.6 workable, R ~ 0.94
# is the theoretical limit. Here kp ~ 87 px, so R = 0.5 gives an 88 px object at
# ~102 nm pitch -- the same regime as the paper's own X-ray demo (70 px, 83 nm).
# The 30 nm Siemens star features are below this limit and cannot be recovered.
rpi_resolution_ratio = 0.4
rpi_kp_quantile = 0.95           # fraction of probe far-field power used to define kp

# --- optimizer --------------------------------------------------------------
num_epochs = 2500                # max Adam iterations per restart. Plain pixels converge
                                 # by ~250; DIP keeps improving and bottoms out near 4500.
rpi_lr = 0.01                    # Adam step on an object of magnitude ~1
rpi_lr_decay_factor = 0.5
rpi_lr_decay_patience = 100      # iters without improvement before cutting lr
rpi_min_lr = 1e-5                # stop when lr decays below this
rpi_loss_floor = 1e-9            # stop when loss falls below this
rpi_num_restarts = 1             # measured: the solution is init-independent here
rpi_object_init = "unit"         # "unit" (1 + noise, a transmission object) or "random" (paper)
rpi_init_sigma = 0.1             # init noise std ("random" uses 1.0 per the paper)
rpi_background_counts = 0.0      # known detector background (Eq. 2's B_ij), if any

# --- staged probe update ----------------------------------------------------
# Set rpi_probe_start = 200 to freeze the probe for 200 iterations and then refine it.
# Measured on this dataset (frame 46), against the 81-position ptychography recon:
#     probe fixed          data loss 0.0097   eps 0.092   probe change  0.0%
#     release@200 anchor 0 data loss 0.0000   eps 0.092   probe change 18.5%
#     release@200 anchor 3 data loss 0.0046   eps 0.092   probe change  4.1%
# The Poisson noise floor is 0.0124, so every released run fits noise, and none improves
# the object. Default off. Useful only if the probe may have drifted between the
# calibration scan and the shot -- it has not here.
rpi_probe_start = 100
rpi_probe_lr_rel = 0.003         # probe lr = this * mean|P| (probe entries are ~1e-3)
rpi_probe_anchor = 2.0           # weight of ||P - P0||^2 / ||P0||^2 keeping P near calibration
# rpi_probe_anchor = 0

rpi_illumination_threshold = 0.10   # fraction of peak illumination defining the usable FOV

# --- Deep Image Prior (object) ----------------------------------------------
# The object is generated by a CNN from a fixed random input and the network weights are
# optimized instead of the pixels (see DIPObjectGenerator). This supplies the object prior
# the plain-pixel fit lacks. Modeled on test_DIP_seimenStar.py, but wired in by hand: that
# script flips a Pty-Chi option, which is unavailable here because this reconstruction
# bypasses PtychographyTask to express the band limit.
use_dip = True                   # False -> plain directly-optimized object pixels
# Two-stage schedule: warm up with ordinary plain-pixel RPI (the probe releasing at
# rpi_probe_start as usual), then hand the object over to DIP. With a residual
# parameterization DIP starts from exactly the stage-1 object and learns a correction to
# it; with "direct" the network ignores the warm start and only the refined probe carries
# over. Set dip_stage1_iters = 0 to run DIP from the first iteration (no warm start).
dip_stage1_iters = 0           # plain-pixel RPI iterations before DIP takes over
dip_stage2_probe = "continue"      # "frozen": hold the stage-1 probe while DIP runs, so DIP
                                 #   acts on the object ONLY
                                 # "continue": keep refining it (measured to overfit)
dip_grid = "fullres"              # "lowres": CNN makes the band-limited object, then the
                                 #   Fourier upsample runs as usual (band limit AND DIP).
                                 # "fullres": CNN makes the n_dp object directly and the
                                 #   band limit is dropped, so DIP is the only regularizer.
# Defaults below are measured, not guessed. Scored on this frame with the probe frozen,
# as eps vs the 121-position ptychography recon (complex gamma, illumination > 10%):
#     plain pixels (use_dip=False)             eps 0.1224   loss 1.18x noise floor
#     DIP direct levels=2 sigmoid=False        eps 0.1087   loss 1.42x   <- default
#     DIP direct levels=2 sigmoid=True         eps 0.1135   loss 1.37x
#     DIP direct levels=3 sigmoid=True         eps 0.1134   loss 1.43x  (needs ~6k iters)
#     DIP direct levels=3 sigmoid=False        eps 0.1183   (unstable: spikes to 0.185)
#     DIP direct, fullres grid                 eps 0.1370
#     DIP residual_zeroconv                    eps 0.1622
#     DIP residual                             eps 0.1876
# DIP beats plain pixels by ~11%, and note HOW: it sits FURTHER from the noise floor while
# landing closer to truth -- the signature of a real regularizer. Plain pixels converge by
# iteration ~250 and then never improve; DIP keeps improving for thousands of iterations.
# Two results contradicted expectation, hence the measurements: "residual_zeroconv" is the
# WORST despite starting at the exact vacuum value (its zero-init convs gate the encoder
# off early in training), and dropping the sigmoid helps at 2 levels but destabilizes 3.
dip_parameterization = "direct"  # | "residual" | "residual_zeroconv"  (see the class docstring)
dip_num_levels = 3               # autoencoder depth; 2/3/4 -> 0.37M/1.26M/4.80M weights
dip_base_channels = 32
dip_input_channels = 32          # channels of the fixed random network input
dip_use_batchnorm = True
dip_sigmoid_on_magnitude = False   # the sigmoid caps |O| in (0,1), but a transmission
                                   # object sits at |O| ~ 1, i.e. on the saturated edge
dip_scaled_tanh_on_phase = True    # phase = pi * tanh(x), so it is hard-capped at +/-pi.
                                   # Measured here: 84% of pixels sit beyond 2.8 rad, i.e.
                                   # deep in tanh saturation where the phase gradient
                                   # nearly vanishes. It still reconstructs correctly (the
                                   # saturation is absorbed by the unobservable GLOBAL
                                   # phase -- see the gauge fix after the run), but set
                                   # False for a thick sample whose phase must exceed pi.
dip_lr = 1e-4                    # Adam lr on network WEIGHTS, not on pixels. The reference
                                 # uses 1e-6, but that suits Pty-Chi's unnormalized loss;
                                 # this loss is normalized to ~1e-2.
dip_lr_decay_patience = 300      # DIP needs a longer high-lr phase than plain pixels do:
                                 # measured eps 0.109 at patience 200 vs 0.118 at 100
dip_seed = 0                     # seeds the fixed network input
dip_input_speckle_px = 1         # speckle grain of the fixed input z, in pixels. z is drawn
                                 # on an (n/grain, n/grain) grid and bilinearly upsampled, so
                                 # neighbouring pixels stay correlated over ~grain px. 1 (or
                                 # anything below it) gives the DIP reference input, white
                                 # U[0, 0.1) noise -- one pixel is the finest grain a pixel
                                 # grid can carry, so there is nothing below 1 to reach for.
                                 #
                                 # MEASURED: this changes z as asked but does NOT change the
                                 # untrained object. Its phase grain sits at 3-4 px for every
                                 # value from 1 to 64, because the random decoder's upsampling
                                 # and ReLU harmonics regenerate pixel-scale structure. Object
                                 # grain is set by dip_num_levels (2 levels -> ~2 px, 3 -> ~5),
                                 # not here.

# fracPy flip switches
flip_dp_x = False
flip_dp_y = False
flip_positions_x = False
flip_positions_y = False
swap_position_axes = False       # True if stored positions are (x, y) not (y, x)

random_seed = 123

print(f"pixel size = {pixel_size_m * 1e9:.3f} nm, FOV = {n_dp * pixel_size_m * 1e6:.2f} um")


#%% ---------------------------------------------------------------- helpers


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


def make_disk_probe(size, diameter_px):
    """Soft-edged disk, used when there is no probe to inherit."""
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2
    r = np.hypot(yy, xx)
    edge = max(0.1 * diameter_px / 2, 1.0)
    disk = 0.5 * (1 - np.tanh((r - diameter_px / 2) / edge))
    return disk.astype(np.complex64)[None, None]  # (n_opr, n_modes, h, w)


def siemens_star(shape, n_spokes=24):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    theta = np.arctan2(yy - shape[0] / 2, xx - shape[1] / 2)
    r = np.hypot(yy - shape[0] / 2, xx - shape[1] / 2)
    spokes = (np.cos(n_spokes * theta) > 0) & (r < 0.45 * min(shape))
    return ((1 - 0.15 * spokes) * np.exp(1j * 0.8 * spokes)).astype(np.complex64)


def probe_fourier_radius(probe_modes, quantile=0.95):
    """kp in detector pixels: the radius of the probe's own far-field containing
    `quantile` of its power. RPI's resolution ratio R = ko/kp is defined against this,
    not against the detector's Nyquist frequency (= n_dp / 2)."""
    ff = (torch.abs(torch.fft.fftshift(torch.fft.fft2(probe_modes), dim=(-2, -1))) ** 2)
    ff = ff.sum(0).detach().cpu().numpy()
    n = ff.shape[-1]
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - n / 2, xx - n / 2).astype(int)
    cum = np.cumsum(np.bincount(r.ravel(), ff.ravel()))
    return int(np.searchsorted(cum, quantile * cum[-1]))


def rescale_probe_to_counts(probe_modes, measured, mask):
    """Scale the probe so its far-field power matches the measured counts over the VALID
    detector pixels -- the same pixels the loss is computed on. (Pty-Chi's rescale_probe
    ignores the mask, which biases the scale by the dead-pixel fraction.)"""
    ff = (torch.abs(torch.fft.fftshift(torch.fft.fft2(probe_modes), dim=(-2, -1))) ** 2).sum(0)
    return probe_modes * torch.sqrt((measured * mask).sum() / (ff * mask).sum())


def fourier_resample(arr, n_out):
    """Resample the last two axes to (n_out, n_out) over the SAME field of view, by
    cropping or zero-padding the centred spectrum.

    This is the real-space counterpart of cropping the detector, and it is what a probe
    stored on a different grid needs. The array's field of view is
        n_dp * pixel_size_m = wavelength_m * det_dist_m / det_pixel_m
    which does NOT depend on n_dp -- changing the detector crop changes the *sampling* of
    a fixed FOV. Center-cropping the probe instead would shrink its FOV (e.g. 512 -> 256
    would give a 4.5 um probe against a 9.01 um object) and silently break the geometry.
    Cropping the spectrum is also exactly what cropping the detector does to the measured
    field, so this keeps probe and patterns consistent."""
    n_in = arr.shape[-1]
    if n_in == n_out:
        return arr
    assert (n_in - n_out) % 2 == 0, (
        f"|{n_in} - {n_out}| must be even to keep the spectrum centred"
    )
    axes = (-2, -1)
    spec = np.fft.fftshift(np.fft.fft2(arr, norm="ortho", axes=axes), axes=axes)
    if n_out < n_in:
        lo = (n_in - n_out) // 2
        spec = spec[..., lo : lo + n_out, lo : lo + n_out]
    else:
        lo = (n_out - n_in) // 2
        pad = [(0, 0)] * arr.ndim
        pad[-2] = pad[-1] = (lo, lo)
        spec = np.pad(spec, pad)
    out = np.fft.ifft2(np.fft.ifftshift(spec, axes=axes), norm="ortho", axes=axes)
    return out * (n_out / n_in)


def fourier_upsample_object(obj_lowres, n_full):
    """Zero-pad the object's own FFT out to n_full x n_full (RPI Eq. 1's `pad(F{O'})`
    step): band-limited upsampling that reproduces obj_lowres exactly at the
    corresponding full-res grid points. n_full - obj_lowres.shape[-1] must be even.
    (Torch/autograd twin of fourier_resample's padding branch, kept separate because it
    runs in the optimization hot loop.)"""
    n_low = obj_lowres.shape[-1]
    spec = torch.fft.fftshift(torch.fft.fft2(obj_lowres, norm="ortho"))
    padded = torch.zeros((n_full, n_full), dtype=spec.dtype, device=spec.device)
    lo = (n_full - n_low) // 2
    padded[lo : lo + n_low, lo : lo + n_low] = spec
    return torch.fft.ifft2(torch.fft.ifftshift(padded), norm="ortho") * (n_full / n_low)


def rpi_far_field(obj_fullres, probe_modes, background):
    """Multiply a full-resolution object by the probe and propagate to the detector."""
    exit_waves = obj_fullres * probe_modes
    # Unnormalized FFT, matching the convention the probe was rescaled against.
    far_field = torch.fft.fftshift(torch.fft.fft2(exit_waves), dim=(-2, -1))
    return (far_field.abs() ** 2).sum(0) + background


def simulate_rpi_diffraction(obj_lowres, probe_modes, n_full, background):
    """RPI forward model (Eq. 1): upsample -> multiply by probe -> propagate -> incoherent sum."""
    return rpi_far_field(fourier_upsample_object(obj_lowres, n_full), probe_modes, background)


def rpi_diffraction_loss(predicted_intensity, measured_intensity, mask):
    """Normalized amplitude MSE (Eq. 2), over valid detector pixels only."""
    resid = (predicted_intensity.sqrt() - measured_intensity.sqrt()) ** 2
    return (resid * mask).sum() / (measured_intensity * mask).sum()


def simulated_noise_floor(model_intensity, measured, mask, n_draws=10, seed=0):
    """Expected value of rpi_diffraction_loss at the true solution, by drawing Poisson
    noise around `model_intensity`.

    The analytic form (n_valid/4)/sum(I) assumes Var[sqrt(I)] = 1/4, which only holds for
    bright pixels. This data is sparse (median 0 counts), where that overestimates the
    floor by ~1.9x -- enough to make a converged fit look like it is overfitting."""
    g = torch.Generator(device=model_intensity.device).manual_seed(seed)
    lam = model_intensity.clamp_min(0).detach()
    denom = (measured * mask).sum()
    vals = [
        float((((torch.poisson(lam, generator=g).sqrt() - lam.sqrt()) ** 2) * mask).sum() / denom)
        for _ in range(n_draws)
    ]
    return float(np.mean(vals))


class DIPObjectGenerator:
    """Deep image prior: the object is the OUTPUT of a CNN driven by a FIXED random
    input, and the network WEIGHTS are optimized instead of the object pixels. The CNN's
    inductive bias acts as an implicit regularizer -- the prior this reconstruction
    otherwise lacks, since the band limit is its only constraint.

    Note this is NOT a dimensionality reduction: the network has far MORE free parameters
    than the object has pixels (1.26M vs 15.5k at 88px/3 levels). The regularization comes
    from the architecture and from early stopping, so watch the loss against the noise
    floor rather than assuming DIP cannot overfit.

    parameterization:
      "residual_zeroconv" -- O = base + net(z), final convs zero-initialized so the object
                             starts at exactly `base` and the network learns only the
                             deviation. With a warm-started base this is a true continuation
                             of the plain-pixel solve; the cost is that the encoder is
                             gradient-gated for the first few steps.
      "residual"          -- O = base + net(z) with ordinary init; trains everything at once
                             but perturbs the base immediately.
      "direct"            -- O = mag * exp(i*phase) straight from the net, ignoring `base`,
                             as in test_DIP_seimenStar.py (residual_generation = False).

    base: the object the residual is measured against. Defaults to 1.0 (vacuum, i.e. a
    thin non-absorbing sample). Pass the stage-1 object to warm-start DIP from it -- the
    same role `initial_data` plays in Pty-Chi's DIPPlanarObject.generate().
    """

    def __init__(self, n_out, parameterization="residual_zeroconv", num_levels=3,
                 base_channels=32, in_channels=32, use_batchnorm=True,
                 sigmoid_on_magnitude=True, scaled_tanh_on_phase=True,
                 seed=0, device="cpu", base=None, input_speckle_px=1):
        if parameterization not in ("residual_zeroconv", "residual", "direct"):
            raise ValueError(f"unknown dip_parameterization {parameterization!r}")
        self.parameterization = parameterization
        self.residual = parameterization.startswith("residual")
        if base is None:
            self.base = torch.ones((n_out, n_out), dtype=get_default_complex_dtype(), device=device)
        else:
            if tuple(base.shape[-2:]) != (n_out, n_out):
                raise ValueError(f"base is {tuple(base.shape[-2:])}, expected {(n_out, n_out)}")
            self.base = base.detach().clone().to(device)
        self.net = Autoencoder(
            num_in_channels=in_channels,
            num_levels=num_levels,
            base_channels=base_channels,
            use_batchnorm=use_batchnorm,
            zero_conv=(parameterization == "residual_zeroconv"),
            sigmoid_on_magnitude=sigmoid_on_magnitude,
            scaled_tanh_on_phase=scaled_tanh_on_phase,
        ).to(device)
        # The DIP input is drawn ONCE and held fixed; only the weights are optimized.
        # Drawing z per pixel decorrelates at 1 px, so to get a coarser grain draw it on an
        # (n_low, n_low) grid instead and upsample back, keeping neighbours correlated over
        # ~input_speckle_px pixels. n_low is clamped to n_out, so any grain <= 1 is the
        # reference white U[0, 0.1) input -- 1 px is the finest a pixel grid can carry.
        self.input_speckle_px = input_speckle_px
        n_low = min(n_out, max(2, round(n_out / input_speckle_px)))
        g = torch.Generator().manual_seed(seed)
        z = torch.rand((1, in_channels, n_low, n_low), generator=g)
        if n_low != n_out:
            z = torch.nn.functional.interpolate(z, size=(n_out, n_out), mode="bilinear",
                                                align_corners=False)
        self.z = (z * 0.1).to(device)

    def __call__(self):
        amp, phase = self.net(self.z)
        obj = amp[:, 0] * torch.exp(1j * phase[:, 0])
        if self.residual:
            obj = obj + self.base   # learn only the deviation from `base`
        return obj[0]

    def parameters(self):
        return list(self.net.parameters())

    def n_parameters(self):
        return sum(p.numel() for p in self.net.parameters())


#%% ---------------------------------------------------------------- load diffraction patterns

if use_simulated_data:
    # Small synthetic far-field dataset: 8x8 jittered grid over a Siemens star.
    n_dp = 64
    pixel_size_m = wavelength_m * det_dist_m / (n_dp * det_pixel_m)
    n_probe_modes, n_opr_modes = 2, 2
    sim_object_padding_px = 8

    rng = np.random.default_rng(0)
    grid = (np.arange(8) - 3.5) * 12.0
    gy, gx = np.meshgrid(grid, grid, indexing="ij")
    positions_px_all = np.stack([gy.ravel(), gx.ravel()], -1) + rng.normal(0, 0.5, (64, 2))

    sim_shape = get_suggested_object_size(positions_px_all, (n_dp, n_dp), extra=sim_object_padding_px)
    sim_obj = siemens_star(sim_shape)
    sim_probe = make_disk_probe(n_dp, 40)[0, 0]

    patterns = np.empty((len(positions_px_all), n_dp, n_dp), dtype=np.float32)
    for i, (py, px) in enumerate(positions_px_all):
        y0 = int(round(sim_shape[0] / 2 + py - n_dp / 2))
        x0 = int(round(sim_shape[1] / 2 + px - n_dp / 2))
        psi = sim_obj[y0 : y0 + n_dp, x0 : x0 + n_dp] * sim_probe
        patterns[i] = np.abs(np.fft.fftshift(np.fft.fft2(psi, norm="ortho"))) ** 2
    patterns = rng.poisson(patterns / patterns.max() * 1e4).astype(np.float32)
    det_mask = np.ones((n_dp, n_dp), dtype=bool)
    prior = {}
else:
    with h5py.File(dp_file, "r") as f:
        print(f"keys in {dp_file.name}: {list(f.keys())}")
        patterns = f["dp"][()]
        # Dead/hot detector pixels. Fitting them as if they were real counts biases the
        # solution badly (ground-truth loss 0.036 -> 0.020 once masked).
        det_mask = np.asarray(f["det_pixel_mask"][()], dtype=bool)
    print(f"raw ptychogram: {patterns.shape}")

    patterns = center_crop_or_pad(patterns, n_dp)
    det_mask = center_crop_or_pad(det_mask[None], n_dp)[0]
    if flip_dp_x:
        patterns = patterns[..., :, ::-1]
        det_mask = det_mask[:, ::-1]
    if flip_dp_y:
        patterns = patterns[..., ::-1, :]
        det_mask = det_mask[::-1, :]
    patterns = np.ascontiguousarray(patterns, dtype=np.float32)
    det_mask = np.ascontiguousarray(det_mask)
    np.clip(patterns, 0, None, out=patterns)
    print(f"detector mask: {det_mask.mean() * 100:.1f}% of pixels valid")

assert 0 <= frame_index < len(patterns), (
    f"frame_index={frame_index} out of range for {len(patterns)} patterns"
)
patterns = patterns[frame_index : frame_index + 1]
print(f"reconstructing frame {frame_index}: pattern {patterns.shape}, total counts {patterns.sum():.3e}")


#%% ---------------------------------------------------------------- positions + prior probe

if not use_simulated_data:
    prior = {}
    if init_recon_file is not None:
        with h5py.File(init_recon_file, "r") as f:
            print(f"keys in {init_recon_file.name}: {list(f.keys())}")
            prior["probe"] = np.asarray(f["probe"][()]).view(np.complex64)
            prior["positions_px"] = np.asarray(f["positions_px"][()], dtype=np.float64)
            if "obj_pixel_size_m" in f:
                prior["pixel_size_m"] = float(f["obj_pixel_size_m"][()])

    if "positions_px" in prior:
        positions_px_all = prior["positions_px"].copy()
        # Those positions are in the PRIOR reconstruction's pixels, which are only the
        # same as ours when it used the same n_dp. Convert, or every position is silently
        # off by the grid ratio (here 15.02 nm vs 17.60 nm, a 17% error).
        prior_px = prior.get("pixel_size_m")
        # atol=0 is essential: these are ~1e-8 metres, so np.isclose's default atol=1e-8
        # would call two grids 17% apart "equal".
        if prior_px is not None and not np.isclose(prior_px, pixel_size_m, rtol=1e-6, atol=0.0):
            print(f"converting positions from the prior's {prior_px * 1e9:.3f} nm grid "
                  f"to this run's {pixel_size_m * 1e9:.3f} nm grid")
            positions_px_all = positions_px_all * (prior_px / pixel_size_m)
    else:
        # fracPy read ppX/ppY (meters) from the para file; we want pixels.
        with h5py.File(para_file, "r") as f:
            print(f"keys in {para_file.name}: {list(f.keys())}")
            ppx = np.asarray(f["ppX"][()], dtype=np.float64).squeeze()
            ppy = np.asarray(f["ppY"][()], dtype=np.float64).squeeze()
        positions_px_all = np.stack((ppy, ppx), axis=-1) / pixel_size_m

    if swap_position_axes:
        positions_px_all = positions_px_all[:, ::-1].copy()
    if flip_positions_y:
        positions_px_all[:, 0] = -positions_px_all[:, 0]
    if flip_positions_x:
        positions_px_all[:, 1] = -positions_px_all[:, 1]

assert frame_index < len(positions_px_all), (
    f"frame_index={frame_index} out of range for {len(positions_px_all)} positions"
)
position_px = positions_px_all[frame_index]   # the single scan position reconstructed here
print(f"position: y={position_px[0]:.1f}, x={position_px[1]:.1f} px")

# Plot the whole scan (centered on its own mean) and mark the reconstructed frame.
plot_positions = positions_px_all - positions_px_all.mean(axis=0, keepdims=True)
plt.figure(figsize=(4, 4))
plt.plot(plot_positions[:, 1], plot_positions[:, 0], ".-", lw=0.3, ms=2,
         color="0.6", label=f"all positions ({len(plot_positions)})")
sel = plot_positions[frame_index]
plt.plot(sel[1], sel[0], "o", ms=7, mfc="none", mec="red", mew=1.5,
         label=f"reconstructed frame {frame_index}")
plt.legend(fontsize=7, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("x [px]")
plt.ylabel("y [px]")
plt.title("scan positions [px]")
plt.show()


#%% ---------------------------------------------------------------- initial probe

if "probe" in prior:
    # The probe must be RESAMPLED onto the n_dp grid, not center-cropped: the array FOV
    # is fixed at wavelength*z/det_pixel regardless of n_dp (see fourier_resample).
    if prior["probe"].shape[-1] != n_dp:
        print(f"resampling probe {prior['probe'].shape[-1]} -> {n_dp} px over the fixed "
              f"{n_dp * pixel_size_m * 1e6:.2f} um FOV")
    probe = fourier_resample(prior["probe"], n_dp)
    # (n_opr, n_modes, h, w); fold any extra leading axes (e.g. a wavelength axis)
    # into the OPR axis.
    probe = probe.reshape((-1,) + probe.shape[-3:])
    probe = torch.as_tensor(np.ascontiguousarray(probe), dtype=get_default_complex_dtype())
    print(f"probe from {init_recon_file.name}: {tuple(probe.shape)}")
else:
    probe = torch.as_tensor(
        make_disk_probe(n_dp, probe_diameter_m / pixel_size_m),
        dtype=get_default_complex_dtype(),
    )
    print(f"synthesized disk probe, diameter {probe_diameter_m * 1e6:.2f} um")

# Incoherent modes: default to keeping all of them. Dropping modes throws away real
# incoherent power (the tail modes here carry ~27% of it) and the rescale below then
# inflates the survivors to compensate, which distorts the forward model.
if n_probe_modes is None:
    n_probe_modes = probe.shape[1]
if probe.shape[1] > n_probe_modes:
    probe = probe[:, :n_probe_modes]
elif probe.shape[1] < n_probe_modes:
    padded = torch.zeros(
        (probe.shape[0], n_probe_modes, *probe.shape[-2:]), dtype=get_default_complex_dtype()
    )
    padded[:, : probe.shape[1]] = probe
    probe = orthogonalize_initial_probe(padded, secondary_mode_energy=0.02)

# OPR modes.
if probe.shape[0] > n_opr_modes:
    probe = probe[:n_opr_modes]
elif probe.shape[0] < n_opr_modes:
    probe = add_additional_opr_probe_modes_to_probe(probe, n_opr_modes - probe.shape[0])

mode_power = (probe[0].abs() ** 2).sum(dim=(-2, -1))
mode_power = (mode_power / mode_power.sum()).numpy()
print(f"probe: {tuple(probe.shape)} (n_opr, n_modes, h, w)")
print(f"  incoherent mode power fractions: {np.round(mode_power, 4)}")

fig, axes = plt.subplots(1, probe.shape[1], figsize=(2.2 * probe.shape[1], 2.4))
for i, ax in enumerate(np.atleast_1d(axes)):
    ax.imshow(np.abs(probe[0, i].numpy()), cmap="gray")
    ax.set_title(f"mode {i}\n{mode_power[i] * 100:.1f}%", fontsize=8)
    ax.set_xticks([]), ax.set_yticks([])
plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------- RPI setup

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

measured = torch.as_tensor(patterns[0], dtype=torch.float32, device=torch_device)
mask = torch.as_tensor(det_mask.astype(np.float32), device=torch_device)
probe_modes = rescale_probe_to_counts(probe[0].to(torch_device), measured, mask)

# Band limit, defined against the probe's numerical aperture (see module docstring).
assert n_dp % 2 == 0, "n_dp must be even so that n_dp - n_obj_lowres stays even"
kp = probe_fourier_radius(probe_modes, rpi_kp_quantile)
# n_obj_lowres is forced even, which keeps the zero-padding in fourier_upsample_object
# symmetric; an odd total pad puts DC one bin off and corrupts the whole upsampled array.
n_obj_lowres = 2 * int(round(rpi_resolution_ratio * kp))
n_obj_lowres = int(np.clip(n_obj_lowres, 2, n_dp))
achieved_R = (n_obj_lowres / 2) / kp
object_pixel_size_m = pixel_size_m * n_dp / n_obj_lowres
print(
    f"RPI: kp = {kp} px (detector half-width {n_dp // 2}) -> object {n_obj_lowres}x{n_obj_lowres}, "
    f"R = {achieved_R:.2f}, pixel size {object_pixel_size_m * 1e9:.1f} nm, device {torch_device}"
)
if achieved_R > 0.6:
    print(f"  WARNING: R = {achieved_R:.2f} exceeds the paper's reliable range (<= 0.6)")

# Illuminated field of view: object pixels outside it are not constrained by the data.
illumination = (probe_modes.abs() ** 2).sum(0)
illumination = (illumination / illumination.max()).cpu().numpy()
ill_rows = np.where(illumination.max(axis=1) > rpi_illumination_threshold)[0]
ill_cols = np.where(illumination.max(axis=0) > rpi_illumination_threshold)[0]
ill_slice = (slice(ill_rows[0], ill_rows[-1] + 1), slice(ill_cols[0], ill_cols[-1] + 1))
print(
    f"  illuminated FOV (> {rpi_illumination_threshold:g} of peak): "
    f"{ill_rows[-1] - ill_rows[0] + 1} x {ill_cols[-1] - ill_cols[0] + 1} px "
    f"= {(ill_rows[-1] - ill_rows[0] + 1) * pixel_size_m * 1e6:.2f} um"
)

# Poisson shot-noise floor of the loss. A converged reconstruction should land NEAR this,
# not far below it -- below means it is fitting noise. Estimated by drawing Poisson noise
# around the measured pattern itself rather than from (n_valid/4)/sum(I): that analytic
# form assumes Var[sqrt(I)] = 1/4, which fails on sparse data (median 0 counts here) and
# overestimates the floor by ~1.9x, enough to make a converged fit look overfitted.
noise_floor = simulated_noise_floor(measured, measured, mask, seed=random_seed)
analytic_floor = float((mask.sum() / 4) / (measured * mask).sum())
print(f"  Poisson noise floor of the loss ~ {noise_floor:.5f} "
      f"(analytic 1/4 form would say {analytic_floor:.5f})")


#%% ---------------------------------------------------------------- run

probe_optimizable = rpi_probe_start is not None
probe_ref = probe_modes.clone()                    # calibration probe, the anchor
probe_ref_power = (probe_ref.abs() ** 2).sum()
probe_lr = rpi_probe_lr_rel * probe_ref.abs().mean().item()

# DIP generates the object on its own grid: the band-limited array when dip_grid is
# "lowres" (the Fourier upsample still runs), or the full detector grid when "fullres"
# (the band limit is dropped and DIP is the only regularizer).
dip_n_out = n_obj_lowres if dip_grid == "lowres" else n_dp
if use_dip:
    n_obj_pixels = 2 * dip_n_out ** 2
    print(f"DIP: {dip_parameterization} autoencoder on the {dip_grid} grid "
          f"({dip_n_out}x{dip_n_out}), lr {dip_lr:g}")

def run_phase(label, n_iters, generator, obj_leaf, param_groups, probe_var,
              probe_start_iter, patience, epoch0=0):
    """Run one optimization phase and return (losses, obj_lowres, obj_fullres).

    `generator` is None for the plain-pixel phase, in which case `obj_leaf` is stepped
    directly; otherwise the object is regenerated from the network each iteration.
    `probe_start_iter` counts GLOBAL iterations (epoch0 + local) so a probe release
    schedule spans both stages. The probe is only ever updated when probe_var requires
    grad, which is how stage 2 pins it for a DIP-on-the-object-only run."""
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=rpi_lr_decay_factor, patience=patience
    )
    losses, obj_lowres_out, obj_fullres_out = [], None, None
    bar = tqdm(range(n_iters), desc=label, leave=True)
    for local in bar:
        probe_live = (
            probe_var.requires_grad
            and probe_start_iter is not None
            and (epoch0 + local) >= probe_start_iter
        )
        probe_now = probe_var if probe_live else probe_var.detach()

        optimizer.zero_grad()
        if generator is not None:
            # Regenerate the object from the network every iteration; the weights, not
            # the pixels, are what Adam is stepping.
            obj_gen = generator()
            if dip_grid == "lowres":
                obj_lowres_out, obj_fullres_out = obj_gen, None
                pred = simulate_rpi_diffraction(obj_gen, probe_now, n_dp, rpi_background_counts)
            else:
                obj_lowres_out, obj_fullres_out = None, obj_gen
                pred = rpi_far_field(obj_gen, probe_now, rpi_background_counts)
        else:
            obj_lowres_out, obj_fullres_out = obj_leaf, None
            pred = simulate_rpi_diffraction(obj_leaf, probe_now, n_dp, rpi_background_counts)

        loss = rpi_diffraction_loss(pred, measured, mask)
        total = loss
        if probe_live and rpi_probe_anchor > 0:
            total = total + rpi_probe_anchor * ((probe_now - probe_ref).abs() ** 2).sum() / probe_ref_power
        total.backward()
        optimizer.step()
        if probe_live:
            # Remove the O <-> P scaling ambiguity opened by releasing the probe.
            with torch.no_grad():
                probe_var *= torch.sqrt(probe_ref_power / (probe_var.abs() ** 2).sum())

        losses.append(loss.item())
        scheduler.step(losses[-1])
        lr_now = optimizer.param_groups[0]["lr"]
        if local % 20 == 0 or local == n_iters - 1:
            bar.set_postfix(loss=f"{losses[-1]:.5f}", floor=f"{noise_floor:.5f}",
                            lr=f"{lr_now:.2e}", probe="on" if probe_live else "off")
        if losses[-1] < rpi_loss_floor or lr_now < rpi_min_lr:
            break
    bar.close()
    return losses, obj_lowres_out, obj_fullres_out


if use_dip and dip_stage1_iters > 0 and dip_parameterization != "direct":
    # Measured on this frame. A residual stage 2 launched from a CONVERGED stage 1 fails
    # in one of two ways, because the warm start is already a stationary point of the
    # data term:
    #   lowres  -- the object is stationary within the band limit, so the residual net
    #              starts at ~zero gradient and never moves: eps identical to 4 decimals
    #              whether the probe was released (0.1268) or frozen (0.1224).
    #   fullres -- the band limit is gone, so the net DOES move, but only into the extra
    #              unconstrained frequencies: loss fell to 0.36x the noise floor while
    #              eps got slightly WORSE (0.1268 -> 0.1282). That is noise fitting.
    # Only "direct" improved on stage 1 (0.1113 released / 0.1183 frozen), by discarding
    # the warm-started object -- so the warm start's only real contribution is the probe.
    _how = ("freeze there and change nothing" if dip_grid == "lowres"
            else "move only into the unconstrained extra frequencies, i.e. fit noise")
    print(f"  WARNING: dip_parameterization={dip_parameterization!r} continues from the "
          f"converged stage-1 object, which is a stationary point -- DIP is likely to "
          f"{_how}. Use 'direct', or set dip_stage1_iters = 0.")

best_loss, best_obj_lowres, best_probe, best_losses = np.inf, None, None, None
best_obj_fullres, best_net, stage_boundary = None, None, 0


def _gauge_fixed(obj_np):
    """O -> O * exp(-i arg<O>) over the illuminated region, for display only.

    The same free gauge choice applied to the final result below. Needed on the
    diagnostic snapshots too: without it a global phase near +/-pi puts the real
    structure across the np.angle branch cut and the phase map reads as binary.
    """
    m = illumination > rpi_illumination_threshold
    g = np.mean(obj_np[m])
    return obj_np if abs(g) == 0 else obj_np * np.conj(g / abs(g))


def _loss_of(obj_fullres_t, probe_t):
    """Data loss of an object snapshot, so the plots can be labelled with a number."""
    with torch.no_grad():
        return rpi_diffraction_loss(
            rpi_far_field(obj_fullres_t, probe_t, rpi_background_counts), measured, mask,
        ).item()


# Snapshots for the "before DIP" cell below (last restart wins).
init_obj_np, init_obj_loss = None, None          # iteration 0, the plain-pixel init
stage1_obj_np, stage1_loss, stage1_iters_done = None, None, 0   # what stage 2 inherits
dip_init_obj_np, dip_init_loss = None, None      # the untrained net's first emission

for restart in range(rpi_num_restarts):
    torch.manual_seed(random_seed + restart)
    probe_var = probe_ref.clone().requires_grad_(probe_optimizable)
    generator, obj_leaf, losses = None, None, []
    obj_lowres, obj_fullres_live = None, None

    # ---- stage 1: ordinary plain-pixel RPI (the warm start) --------------------
    n_stage1 = dip_stage1_iters if use_dip else num_epochs
    if n_stage1 > 0:
        noise = torch.randn(n_obj_lowres, n_obj_lowres, dtype=torch.float32, device=torch_device) \
            + 1j * torch.randn(n_obj_lowres, n_obj_lowres, dtype=torch.float32, device=torch_device)
        if rpi_object_init == "random":            # the paper's init (its objects were random)
            obj_leaf = rpi_init_sigma * noise
        else:                                      # a transmission object sits near 1
            obj_leaf = torch.ones_like(noise) + rpi_init_sigma * noise
        obj_leaf = obj_leaf.to(get_default_complex_dtype()).requires_grad_(True)

        # Where the whole run starts: the object before a single gradient step. Kept so the
        # cell below can show it next to what stage 1 hands DIP -- the pair is what says how
        # much of the stage-2 input is data and how much is still the rpi_object_init guess.
        with torch.no_grad():
            _init0 = fourier_upsample_object(obj_leaf.detach(), n_dp)
        init_obj_loss = _loss_of(_init0, probe_var.detach())
        init_obj_np = _gauge_fixed(_init0.cpu().numpy())
        print(f"  initial object ({rpi_object_init}, sigma {rpi_init_sigma:g}): |O| mean "
              f"{np.abs(init_obj_np).mean():.3f} (a vacuum object is 1.000), loss "
              f"{init_obj_loss:.5f} ({init_obj_loss / noise_floor:.1f}x the noise floor)")

        groups = [{"params": [obj_leaf], "lr": rpi_lr}]
        if probe_optimizable:
            groups.append({"params": [probe_var], "lr": probe_lr})
        l1, obj_lowres, obj_fullres_live = run_phase(
            f"stage1 RPI {restart + 1}/{rpi_num_restarts}", n_stage1, None, obj_leaf,
            groups, probe_var, rpi_probe_start, rpi_lr_decay_patience, epoch0=0,
        )
        losses += l1
        stage_boundary = len(l1)

        if use_dip:
            # What stage 1 produced, captured BEFORE DIP touches it -- this is the object
            # stage 2 starts from (up to the best-vs-last iterate difference).
            with torch.no_grad():
                _s1 = fourier_upsample_object(obj_lowres.detach(), n_dp)
            stage1_obj_np = _gauge_fixed(_s1.cpu().numpy())
            stage1_loss, stage1_iters_done = l1[-1], len(l1)
            print(f"  stage 1 done: loss {l1[-1]:.5f} ({l1[-1] / noise_floor:.2f}x floor)")

    # ---- stage 2: DIP takes over the object ------------------------------------
    if use_dip:
        # With no stage 1 to inherit, start DIP from the SAME random field plain-pixel RPI
        # starts from, instead of the flat vacuum object. Drawn i.i.d. per pixel, so its
        # grain is 1 px in both amplitude and phase -- finer than anything the untrained
        # network emits, whose decoder correlates neighbours over ~3 px.
        noise = torch.randn(dip_n_out, dip_n_out, dtype=torch.float32, device=torch_device) \
            + 1j * torch.randn(dip_n_out, dip_n_out, dtype=torch.float32, device=torch_device)
        if rpi_object_init == "random":            # the paper's init (its objects were random)
            warm = rpi_init_sigma * noise
        else:                                      # a transmission object sits near 1
            warm = torch.ones_like(noise) + rpi_init_sigma * noise
        warm = warm.to(get_default_complex_dtype())
        if obj_leaf is not None:                   # a real stage-1 result beats the random field
            warm = obj_leaf.detach()
            if dip_grid == "fullres":
                warm = fourier_upsample_object(warm, n_dp).detach()
        generator = DIPObjectGenerator(
            dip_n_out, parameterization=dip_parameterization, num_levels=dip_num_levels,
            base_channels=dip_base_channels, in_channels=dip_input_channels,
            use_batchnorm=dip_use_batchnorm, sigmoid_on_magnitude=dip_sigmoid_on_magnitude,
            scaled_tanh_on_phase=dip_scaled_tanh_on_phase,
            seed=dip_seed + restart, device=torch_device, base=warm,
            input_speckle_px=dip_input_speckle_px,
        )
        if restart == 0:
            print(f"  network has {generator.n_parameters() / 1e6:.2f}M weights for "
                  f"{n_obj_pixels / 1e3:.1f}k object values -- DIP is not a dimensionality "
                  f"reduction, its prior is the architecture plus early stopping")
            if dip_parameterization == "direct":
                print("  WARNING: dip_parameterization='direct' IGNORES the base, so the "
                      "1 px RPI random init above is discarded and DIP starts from the "
                      "untrained network's own ~3 px output. Use 'residual_zeroconv' to "
                      "actually start from it.")

        # What DIP actually STARTS from, before a single weight update. Worth seeing: for
        # "direct" and "refine" the untrained network emits randomness regardless of its
        # input, so the object here is NOT the stage-1 result and the loss jumps by ~100x
        # before recovering. For the residual modes it starts at (or very near) the base.
        with torch.no_grad():
            _dip0 = generator()
            _dip0 = _dip0 if dip_grid == "fullres" else fourier_upsample_object(_dip0, n_dp)
        dip_init_loss = _loss_of(_dip0, probe_var.detach())
        dip_init_obj_np = _gauge_fixed(_dip0.cpu().numpy())
        print(f"  DIP initial object (input grain {dip_input_speckle_px:g} px): |O| mean "
              f"{np.abs(dip_init_obj_np).mean():.3f} (a vacuum object is 1.000), phase std "
              f"{np.angle(dip_init_obj_np).std():.3f} rad (uniform speckle would be 1.814), "
              f"loss {dip_init_loss:.5f} ({dip_init_loss / noise_floor:.1f}x the noise floor)")

        # "DIP on the object only" means the probe stops moving here unless asked.
        probe_var.requires_grad_(probe_optimizable and dip_stage2_probe == "continue")
        groups = [{"params": generator.parameters(), "lr": dip_lr}]
        if probe_var.requires_grad:
            groups.append({"params": [probe_var], "lr": probe_lr})
        l2, obj_lowres, obj_fullres_live = run_phase(
            f"stage2 DIP {restart + 1}/{rpi_num_restarts}", num_epochs, generator, None,
            groups, probe_var, rpi_probe_start, dip_lr_decay_patience, epoch0=len(losses),
        )
        if losses and l2:
            gained = (losses[-1] - l2[-1]) / max(losses[-1], 1e-30)
            if abs(gained) < 1e-3:
                print(f"  WARNING: stage 2 left the loss unchanged ({losses[-1]:.5f} -> "
                      f"{l2[-1]:.5f}) -- DIP froze on the warm start and contributed "
                      "nothing. See the note above dip_stage1_iters.")
            elif gained < -0.01:
                # Expected for "direct", which restarts the object from scratch: stage 2
                # begins far from the stage-1 optimum and needs thousands of iterations.
                print(f"  NOTE: stage 2 ended at a HIGHER loss than stage 1 "
                      f"({losses[-1]:.5f} -> {l2[-1]:.5f}, {l2[-1] / noise_floor:.2f}x the "
                      f"noise floor). With dip_parameterization={dip_parameterization!r} "
                      "stage 2 does not inherit the stage-1 object, so this usually just "
                      "means it has not converged yet -- raise num_epochs.")
        losses += l2

    print(f"  restart {restart}: {len(losses)} iters, final loss {losses[-1]:.5f} "
          f"(noise floor {noise_floor:.5f})")
    if losses[-1] < best_loss:
        best_loss = losses[-1]
        best_obj_lowres = None if obj_lowres is None else obj_lowres.detach().clone()
        with torch.no_grad():
            best_obj_fullres = (
                obj_fullres_live if obj_fullres_live is not None
                else fourier_upsample_object(best_obj_lowres, n_dp)
            ).detach().clone()
        best_probe = probe_var.detach().clone()
        best_net = generator.net if use_dip else None
        best_losses = losses

obj_lowres = best_obj_lowres
obj_fullres = best_obj_fullres
recon_probe = best_probe
obj_lowres_np = None if obj_lowres is None else obj_lowres.cpu().numpy()
obj_fullres_np = obj_fullres.cpu().numpy()
recon_probe_np = recon_probe.cpu().numpy()

# Remove the unobservable GLOBAL phase, O -> O * exp(-i arg<O>), averaged over the
# illuminated region. Only |O| and phase DIFFERENCES are measurable -- a global phase
# cancels in |FFT(O.P)|^2 -- so this is a free gauge choice, not a change to the result.
#
# It matters for reading the phase map. Under DIP the global phase tends to settle near
# +/-pi (the network's phase output is pi*tanh(x), whose saturated regime is exactly
# there). The genuine +/-0.25 rad structure then straddles the +/-pi branch cut and
# np.angle wraps it into two piles at opposite ends of the colour scale, which reads as a
# "binary" phase image. Measured here before the fix: 84% of pixels beyond 2.8 rad;
# after it, phase std 0.254 rad against the ptychography reference's 0.261 rad.
_ill_mask = illumination > rpi_illumination_threshold
_gauge = np.mean(obj_fullres_np[_ill_mask])
if abs(_gauge) > 0:
    _gauge = _gauge / abs(_gauge)
    obj_fullres_np = obj_fullres_np * np.conj(_gauge)
    if obj_lowres_np is not None:
        obj_lowres_np = obj_lowres_np * np.conj(_gauge)
    print(f"  removed global phase {np.angle(_gauge):+.3f} rad; object phase now spans "
          f"[{np.angle(obj_fullres_np[_ill_mask]).min():+.2f}, "
          f"{np.angle(obj_fullres_np[_ill_mask]).max():+.2f}] rad "
          f"(std {np.angle(obj_fullres_np[_ill_mask]).std():.3f})")

if best_loss < 0.7 * noise_floor:
    print(f"  NOTE: final loss {best_loss:.5f} is well below the Poisson floor "
          f"{noise_floor:.5f} -- the fit is absorbing shot noise."
          + ("  Stop DIP earlier (lower num_epochs) or lower rpi_resolution_ratio."
             if use_dip else "  Lower rpi_resolution_ratio."))
if probe_optimizable:
    dprobe = float((recon_probe - probe_ref).abs().sum() / probe_ref.abs().sum()) * 100
    print(f"  probe changed by {dprobe:.1f}% after release at iteration {rpi_probe_start}")


#%% ------------------------------------------------- object before DIP takes over

# Plotted in its own cell rather than inside the run loop, where the tqdm output buries it.
# Re-runnable without redoing the reconstruction. Chronological: (1) the initial guess
# stage 1 starts from, (2) what stage 1 hands to stage 2, (3) what the untrained network
# actually emits on its first forward pass. With dip_stage1_iters = 0 there is no stage 1,
# so only (3) appears -- that emission IS the initial object in that case.

if init_obj_np is not None:
    # Iteration 0. Featureless by construction -- any structure here is the rpi_init_sigma
    # noise, not the sample, and is the baseline the next panel is read against.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5))
    axes[0].imshow(np.abs(init_obj_np), cmap="gray")
    axes[0].set_title(f"initial object amplitude\n|O| mean "
                      f"{np.abs(init_obj_np).mean():.3f} (vacuum = 1.000)")
    axes[1].imshow(np.angle(init_obj_np), cmap="gray")
    axes[1].set_title("initial object phase")
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"object at iteration 0 -- rpi_object_init={rpi_object_init!r}, "
                 f"sigma {rpi_init_sigma:g}, loss {init_obj_loss:.5f} "
                 f"({init_obj_loss / noise_floor:.1f}x floor)", fontsize=10)
    plt.tight_layout()
    plt.show()

if stage1_obj_np is not None:
    # The object DIP inherits: amplitude and phase, full array (top) and cropped to the
    # illuminated FOV (bottom).
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5))
    for col, (img, name) in enumerate([(np.abs(stage1_obj_np), "amplitude"),
                                       (np.angle(stage1_obj_np), "phase")]):
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"stage 1 object {name}\nfull {n_dp}px array")
        axes[1, col].imshow(img[ill_slice], cmap="gray")
        axes[1, col].set_title(f"stage 1 object {name}\nilluminated FOV")
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"object entering stage 2 -- after {stage1_iters_done} plain-pixel RPI "
                 f"iterations, loss {stage1_loss:.5f} "
                 f"({stage1_loss / noise_floor:.2f}x floor)", fontsize=10)
    plt.tight_layout()
    plt.show()

if dip_init_obj_np is not None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5))
    axes[0].imshow(np.abs(dip_init_obj_np), cmap="gray")
    axes[0].set_title(f"DIP initial object amplitude\n|O| mean "
                      f"{np.abs(dip_init_obj_np).mean():.3f} (vacuum = 1.000)")
    axes[1].imshow(np.angle(dip_init_obj_np), cmap="gray")
    axes[1].set_title(f"DIP initial object phase\nstd "
                      f"{np.angle(dip_init_obj_np).std():.3f} rad "
                      f"(uniform speckle = 1.814)")
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"what DIP starts from ({dip_parameterization}, input grain "
                 f"{dip_input_speckle_px:g} px), before any weight update -- loss "
                 f"{dip_init_loss:.5f} "
                 f"({dip_init_loss / noise_floor:.1f}x floor)", fontsize=10)
    plt.tight_layout()
    plt.show()


#%% ---------------------------------------------------------------- inspect

# Main result: object amplitude, object phase, probe -- full array (top) and cropped to
# the illuminated FOV (bottom), which is the only region the data constrains.
fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))
panels = [
    (np.abs(obj_fullres_np), "object amplitude", "gray"),
    (np.angle(obj_fullres_np), "object phase", "gray"),
    (np.abs(recon_probe_np[0]), "probe amplitude (mode 0)", "gray"),
]
for col, (img, title, cmap) in enumerate(panels):
    axes[0, col].imshow(img, cmap=cmap)
    axes[0, col].set_title(f"{title}\nfull {n_dp}px array")
    axes[1, col].imshow(img[ill_slice], cmap=cmap)
    axes[1, col].set_title(f"{title}\nilluminated FOV")
for ax in axes.ravel():
    ax.set_xticks([]), ax.set_yticks([])
plt.tight_layout()
plt.show()

# Diagnostics: the band-limited array (when there is one) and convergence vs the floor.
if obj_lowres_np is not None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(np.abs(obj_lowres_np), cmap="gray")
    axes[0].set_title(f"low-res object amplitude\n{n_obj_lowres}px, {object_pixel_size_m * 1e9:.0f} nm/px")
    axes[1].imshow(np.angle(obj_lowres_np), cmap="gray")
    axes[1].set_title("low-res object phase")
    for ax in axes[:2]:
        ax.set_xticks([]), ax.set_yticks([])
else:
    # dip_grid = "fullres": the network writes the full array, there is no low-res one.
    fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    axes = np.array([None, None, axes])
axes[2].semilogy(best_losses, label="diffraction loss")
axes[2].axhline(noise_floor, color="red", ls="--", lw=1, label=f"Poisson floor {noise_floor:.4f}")
if probe_optimizable:
    axes[2].axvline(rpi_probe_start, color="0.5", ls=":", lw=1, label="probe released")
if use_dip and stage_boundary > 0:
    axes[2].axvline(stage_boundary, color="tab:blue", ls="-.", lw=1, label="DIP takes over")
axes[2].set_xlabel("iteration"), axes[2].set_ylabel("loss")
axes[2].set_title(f"R = {achieved_R:.2f}, final {best_loss:.5f}")
axes[2].legend(fontsize=7)
plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------- save

out_dir.mkdir(parents=True, exist_ok=True)
tag = f"_DIP_{dip_parameterization}_{dip_grid}" if use_dip else ""
out_file = out_dir / f"recon_frame{frame_index}_RPI_R{achieved_R:.2f}_{n_obj_lowres}px{tag}.h5"

with h5py.File(out_file, "w") as f:
    if obj_lowres_np is not None:
        f.create_dataset("object_lowres", data=obj_lowres_np)
    # Full-resolution object, kept under this name for compatibility with
    # init_recon_file's prior["object"] loader.
    f.create_dataset("object", data=obj_fullres_np)
    f.create_dataset("probe", data=recon_probe_np)
    f.create_dataset("illumination", data=illumination.astype(np.float32))
    f.create_dataset("positions_px", data=position_px[None, :])
    f.attrs["pixel_size_m"] = pixel_size_m
    f.attrs["object_pixel_size_m"] = object_pixel_size_m
    f.attrs["wavelength_m"] = wavelength_m
    f.attrs["detector_distance_m"] = det_dist_m
    f.attrs["rpi_resolution_ratio"] = achieved_R
    f.attrs["probe_fourier_radius_px"] = kp
    f.attrs["noise_floor"] = noise_floor
    f.attrs["final_loss"] = best_loss
    f.attrs["frame_index"] = frame_index
    f.attrs["use_dip"] = use_dip
    if use_dip:
        f.attrs["dip_grid"] = dip_grid
        f.attrs["dip_parameterization"] = dip_parameterization
        f.attrs["dip_num_levels"] = dip_num_levels
        f.attrs["dip_lr"] = dip_lr

if use_dip:
    # The object is only reproducible together with the weights that generated it.
    torch.save(best_net.state_dict(), out_file.with_suffix(".net.pt"))

np.savetxt(out_dir / "rpi_loss.csv", best_losses, delimiter=",", header="loss", comments="")
print(f"saved {out_file}")
