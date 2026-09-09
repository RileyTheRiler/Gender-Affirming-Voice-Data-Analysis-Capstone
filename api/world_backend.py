"""WORLD-vocoder resynthesis backend (experimental spike).

Praat's ``Change gender`` moves formants by resampling the whole signal and then
puts the pitch back with PSOLA. Everything is therefore coupled: resonance and
weight interact, the formant scale has to stay near 1.0 to hide the resampling,
and the excitation is whatever PSOLA leaves behind, so there is no breathiness
control at all.

WORLD decomposes speech into three streams that can be edited separately:

* ``f0``  - the fundamental frequency contour
* ``sp``  - the smooth spectral envelope (CheapTrick), i.e. the formants
* ``ap``  - band aperiodicity (D4C), i.e. how noise-like each band is

That buys three things the Praat path cannot express:

1. pitch and formants move independently, so neither drags the other;
2. aperiodicity is a first-class control, which is what "breathy" means
   acoustically;
3. the envelope can be warped per band, so F1 can move without F2/F3 -- the
   oral / pharyngeal distinction that voice therapy actually teaches.

``pyworld`` is an optional dependency. Import errors are reported through
:func:`world_available` so the API can fall back to Praat rather than 500.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised by whichever environment lacks the wheel
    import pyworld as _pw

    _IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    _pw = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# Analysis frame period in ms. WORLD's default is 5.0; keeping it there means the
# published quality figures for the vocoder still apply.
FRAME_PERIOD_MS = 5.0

# CheapTrick's FFT size is a function of the F0 floor, so pinning the floor keeps
# the envelope grid identical for every speaker and every parameter setting.
CHEAPTRICK_F0_FLOOR = 71.0

# Envelope warp. The crossover sits above a typical F1 and below a typical F2, so
# `resonance_low_scale` acts on F1 (mouth / tongue height) and
# `resonance_high_scale` on F2-F3 (front cavity, the part therapy calls
# "brightness of placement").
WARP_CROSSOVER_HZ = 1150.0
WARP_CROSSOVER_WIDTH_OCT = 0.55

# Above this the warp relaxes back to identity so the mapping stays monotonic and
# nothing is pushed past Nyquist.
WARP_TAPER_START_HZ = 5000.0

# Breathiness weighting. Breathy phonation raises the noise floor everywhere but
# most strongly in the upper bands, so the control is frequency dependent.
BREATH_LOW_WEIGHT = 0.55
BREATH_HIGH_WEIGHT = 1.25
BREATH_KNEE_HZ = 2200.0
BREATH_KNEE_WIDTH_OCT = 0.9

# How far a full-scale breathiness request moves each band toward pure noise
# (positive) or toward a fully periodic source (negative). Kept below 1.0 in both
# directions: an all-noise band is a whisper, and a zero-noise band buzzes.
BREATH_ADD_GAIN = 0.35
BREATH_REMOVE_GAIN = 0.70
BREATH_ADD_MAX = 0.90
BREATH_REMOVE_MAX = 0.95

# Aperiodicity is a ratio; keep it strictly inside (0, 1) or synthesis degenerates
# into either a buzz or pure noise.
AP_MIN = 1e-6
AP_MAX = 0.999999


def world_available() -> bool:
    """True when ``pyworld`` imported successfully."""
    return _pw is not None


def world_import_error() -> str | None:
    """The import failure message, or None when the backend is usable."""
    return _IMPORT_ERROR


def _sigmoid_oct(freqs: np.ndarray, knee_hz: float, width_oct: float) -> np.ndarray:
    """Smooth 0 -> 1 ramp centred on ``knee_hz``, measured in octaves."""
    safe = np.maximum(np.asarray(freqs, dtype=float), 1e-6)
    octaves = np.log2(safe / knee_hz) / max(width_oct, 1e-6)
    return 1.0 / (1.0 + np.exp(-4.0 * octaves))


def formant_scale_curve(freqs: np.ndarray, low_scale: float, high_scale: float) -> np.ndarray:
    """Per-frequency formant scale factor.

    A resonance sitting at ``f`` in the source ends up at ``curve[f] * f`` in the
    output, so the curve is directly readable as "how far this band moves".
    Blends ``low_scale`` into ``high_scale`` across the crossover, then relaxes to
    1.0 above :data:`WARP_TAPER_START_HZ` so the warp cannot run off Nyquist.
    """
    freqs = np.asarray(freqs, dtype=float)
    blend = _sigmoid_oct(freqs, WARP_CROSSOVER_HZ, WARP_CROSSOVER_WIDTH_OCT)
    scale = float(low_scale) + (float(high_scale) - float(low_scale)) * blend

    nyquist = float(freqs[-1]) if freqs.size else 1.0
    if nyquist > WARP_TAPER_START_HZ:
        span = np.clip((freqs - WARP_TAPER_START_HZ) / (nyquist - WARP_TAPER_START_HZ), 0.0, 1.0)
        taper = 0.5 * (1.0 - np.cos(np.pi * span))  # raised cosine, 0 -> 1
        scale = scale * (1.0 - taper) + taper
    return scale


def _warp_axis(freqs: np.ndarray, low_scale: float, high_scale: float) -> np.ndarray:
    """Source frequency to read for each output bin (the inverse warp).

    ``target[i] = curve[i] * freqs[i]`` is where source content lands. Inverting
    that mapping by interpolation gives, for every output bin, the source
    frequency whose envelope value belongs there.
    """
    curve = formant_scale_curve(freqs, low_scale, high_scale)
    target = curve * freqs
    # Enforce strict monotonicity before inverting; extreme scale combinations can
    # otherwise produce a flat or folded segment that np.interp would smear.
    target = np.maximum.accumulate(target)
    step = np.diff(target)
    if np.any(step <= 0.0):
        target = target + np.arange(target.size) * 1e-6
    return np.interp(freqs, target, freqs)


def warp_spectral_envelope(sp: np.ndarray, freqs: np.ndarray, low_scale: float, high_scale: float) -> np.ndarray:
    """Move the formants without touching the excitation.

    Interpolation happens in log power so the peaks keep their shape instead of
    being smeared by linear-domain averaging.
    """
    if abs(low_scale - 1.0) < 1e-9 and abs(high_scale - 1.0) < 1e-9:
        return sp.copy()

    source_freqs = _warp_axis(freqs, low_scale, high_scale)
    log_sp = np.log(np.maximum(sp, 1e-16))
    # np.interp is 1-D, but the sample points are the same for every frame, so the
    # per-frame loop is over a small number of rows with a vectorised body.
    warped = np.empty_like(log_sp)
    for i in range(log_sp.shape[0]):
        warped[i] = np.interp(source_freqs, freqs, log_sp[i])
    return np.exp(warped)


def apply_envelope_tilt(sp: np.ndarray, freqs: np.ndarray, tilt_db: np.ndarray) -> np.ndarray:
    """Apply a spectral tilt inside the envelope rather than as a post filter.

    The Praat path has to run a separate STFT pass to tilt the output. Here the
    tilt is part of the envelope the vocoder synthesises from, so it costs
    nothing and cannot add its own overlap-add smearing.
    """
    gain = np.power(10.0, np.asarray(tilt_db, dtype=float) / 10.0)  # power domain
    return sp * gain[None, :]


def breathiness_weight(freqs: np.ndarray) -> np.ndarray:
    """How strongly each band responds to the breathiness control."""
    blend = _sigmoid_oct(freqs, BREATH_KNEE_HZ, BREATH_KNEE_WIDTH_OCT)
    return BREATH_LOW_WEIGHT + (BREATH_HIGH_WEIGHT - BREATH_LOW_WEIGHT) * blend


def apply_breathiness(ap: np.ndarray, freqs: np.ndarray, amount: float) -> np.ndarray:
    """Raise or lower band aperiodicity.

    Modelled as a blend rather than a gain. A multiplicative or power-law control
    can only scale the noise that is already in a band, so it does nothing for a
    band that starts near zero -- which is exactly the case for a clear,
    non-breathy voice, the one a breathiness control is most needed for. Blending
    each band toward full noise (or toward none) adds or removes aspiration
    regardless of the starting point, and stays inside (0, 1) by construction.
    """
    if abs(amount) < 1e-9:
        return ap.copy()

    weight = breathiness_weight(freqs)[None, :]
    ap = np.clip(ap, AP_MIN, AP_MAX)
    if amount > 0.0:
        blend = np.minimum(BREATH_ADD_MAX, amount * BREATH_ADD_GAIN * weight)
        out = ap + (1.0 - ap) * blend
    else:
        blend = np.minimum(BREATH_REMOVE_MAX, -amount * BREATH_REMOVE_GAIN * weight)
        out = ap * (1.0 - blend)
    return np.clip(out, AP_MIN, AP_MAX)


def transform_f0(f0: np.ndarray, semitones: float, range_scale: float, floor: float, ceiling: float) -> tuple[np.ndarray, float | None]:
    """Shift the median and scale the excursion around it, in the log domain.

    Praat scales the pitch range linearly in hertz, which makes a given range
    factor stretch the top of the contour further than the bottom. Doing it in
    log hertz keeps the excursion symmetric in semitones, which is how intonation
    range is described in the therapy literature.
    """
    f0 = np.asarray(f0, dtype=float).copy()
    voiced = f0 > 0.0
    if not np.any(voiced):
        return f0, None

    log_f0 = np.log2(f0[voiced])
    median = float(np.median(log_f0))
    shifted = median + (semitones / 12.0) + (log_f0 - median) * float(range_scale)
    f0[voiced] = np.clip(np.power(2.0, shifted), floor, ceiling)
    return f0, float(np.power(2.0, median))


def analyze(audio: np.ndarray, sr: int, f0_floor: float, f0_ceil: float, use_harvest: bool = True) -> dict[str, Any]:
    """Decompose into F0 / spectral envelope / aperiodicity."""
    if _pw is None:  # pragma: no cover
        raise RuntimeError(f"pyworld is not available: {_IMPORT_ERROR}")

    x = np.ascontiguousarray(np.asarray(audio, dtype=np.float64))
    floor = float(max(40.0, min(f0_floor, 400.0)))
    ceil = float(max(floor * 2.0, min(f0_ceil, 1000.0)))

    if use_harvest:
        f0, t = _pw.harvest(x, sr, f0_floor=floor, f0_ceil=ceil, frame_period=FRAME_PERIOD_MS)
    else:
        f0, t = _pw.dio(x, sr, f0_floor=floor, f0_ceil=ceil, frame_period=FRAME_PERIOD_MS)
        f0 = _pw.stonemask(x, f0, t, sr)

    fft_size = _pw.get_cheaptrick_fft_size(sr, CHEAPTRICK_F0_FLOOR)
    sp = _pw.cheaptrick(x, f0, t, sr, fft_size=fft_size)
    ap = _pw.d4c(x, f0, t, sr, fft_size=fft_size)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    return {"f0": f0, "t": t, "sp": sp, "ap": ap, "fft_size": fft_size, "freqs": freqs, "sr": sr}


def synthesize(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray, sr: int, length: int | None = None) -> np.ndarray:
    if _pw is None:  # pragma: no cover
        raise RuntimeError(f"pyworld is not available: {_IMPORT_ERROR}")
    out = _pw.synthesize(
        np.ascontiguousarray(f0, dtype=np.float64),
        np.ascontiguousarray(sp, dtype=np.float64),
        np.ascontiguousarray(ap, dtype=np.float64),
        sr,
        FRAME_PERIOD_MS,
    )
    out = np.asarray(out, dtype=np.float64).reshape(-1)
    if length is not None:
        if out.size > length:
            out = out[:length]
        elif out.size < length:
            out = np.pad(out, (0, length - out.size))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
