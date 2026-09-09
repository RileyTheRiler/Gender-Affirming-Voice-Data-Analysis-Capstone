"""Measure the WORLD spike against the current Praat path.

The point of the spike is a set of specific claims. This script tries to falsify
each of them rather than just demonstrate that WORLD produces audio:

1. **Formant fidelity** - does a requested resonance scale of 1.15 actually move
   the formants by 15%? Praat's ceiling of 1.08 exists because resampling starts
   to be audible; if WORLD tracks the request accurately at 1.15-1.25 with no
   worse artifact score, the higher ceiling is justified.
2. **Independence** - Praat resamples and then restores pitch, so pitch and
   resonance are coupled. Measured as: move pitch only, see how far the formants
   drift; move resonance only, see how far the pitch drifts.
3. **Per-band control** - can F1 move while F2/F3 stay put? Praat cannot express
   this at all.
4. **Breathiness** - does the aperiodicity control produce a monotonic change in
   harmonics-to-noise ratio? Praat has no such control.
5. **Cost** - wall-clock time per second of audio, which decides whether this can
   run inside a serverless function.

Run with:  python -m bench.compare_backends [--fast] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from typing import Any

import numpy as np
import parselmouth
from parselmouth.praat import call

from api.index import transform_audio
from api import world_backend
from bench import signals

SR = 24000


# --------------------------------------------------------------------------
# Measurement primitives
# --------------------------------------------------------------------------
# Bands for the shift measurement. "low" brackets F1, "high" brackets F2/F3, and
# "full" is the overall resonance move.
BANDS: dict[str, tuple[float, float]] = {
    "low": (300.0, 1150.0),
    "high": (1400.0, 4200.0),
    "full": (220.0, 4200.0),
}
LOG_GRID_POINTS = 1024
LOG_GRID_LO_HZ = 120.0
LOG_GRID_HI_HZ = 6000.0
SMOOTH_OCTAVES = 0.12
# The broad glottal/radiation tilt does not move when the formants do. Left in, it
# anchors the cross-correlation and biases every measured shift toward 1.0, so it
# is subtracted out and only the formant structure is correlated.
DETREND_OCTAVES = 1.0
MAX_SHIFT_OCTAVES = 0.6


def _gaussian_smooth(values: np.ndarray, octaves: float, octaves_per_point: float) -> np.ndarray:
    sigma = max(1.0, octaves / octaves_per_point)
    half = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# Cepstral coefficients kept when the comparison is meant to be envelope-only.
# The pitch-adaptive cutoff used for shift estimation keeps enough quefrency to
# carry some harmonic fine structure, which is fine for locating formants but
# would score a harmless difference in harmonic placement as distortion.
ENVELOPE_ONLY_COEFFICIENTS = 40


def _cepstral_envelope(audio: np.ndarray, sr: int, f0_hz: float, coefficients: int | None = None) -> np.ndarray:
    """Harmonic-free log spectral envelope, averaged over voiced frames.

    Smoothing in octaves is not enough on its own: at 165 Hz the harmonics near F1
    are ~0.4 octaves apart, wider than any kernel that would still preserve F1, so
    a pitch change moves the fine structure and any correlation-based estimator
    follows the harmonics instead of the formants. Low-quefrency liftering removes
    the harmonic comb by construction, and the cutoff follows each signal's own F0,
    so the source and the output are each analysed on their own terms.
    """
    audio = np.asarray(audio, dtype=np.float64)
    n_fft = 2048
    hop = n_fft // 2
    window = np.hanning(n_fft)
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))

    cutoff = coefficients if coefficients is not None else int(0.75 * sr / max(f0_hz, 50.0))
    cutoff = int(np.clip(cutoff, 8, n_fft // 2 - 1))
    lifter = np.zeros(n_fft)
    lifter[: cutoff + 1] = 1.0
    lifter[-cutoff:] = 1.0

    envelopes = []
    for start_i in range(0, audio.size - n_fft + 1, hop):
        frame = audio[start_i : start_i + n_fft]
        if float(np.sqrt(np.mean(np.square(frame)))) < 1e-5:
            continue  # silence would flatten the average
        log_mag = np.log(np.abs(np.fft.rfft(frame * window)) + 1e-12)
        cepstrum = np.fft.irfft(log_mag, n=n_fft)
        envelopes.append(np.fft.rfft(cepstrum * lifter, n=n_fft).real)
    if not envelopes:
        return np.zeros(n_fft // 2 + 1)
    return np.mean(envelopes, axis=0)


def _log_spectrum(audio: np.ndarray, sr: int, f0_hz: float, coefficients: int | None = None) -> np.ndarray:
    """The liftered envelope, resampled onto a log-frequency grid and detrended."""
    envelope = _cepstral_envelope(audio, sr, f0_hz, coefficients)
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    grid = np.logspace(np.log2(LOG_GRID_LO_HZ), np.log2(LOG_GRID_HI_HZ), LOG_GRID_POINTS, base=2.0)
    resampled = np.interp(grid, freqs, envelope)
    octaves_per_point = np.log2(LOG_GRID_HI_HZ / LOG_GRID_LO_HZ) / (LOG_GRID_POINTS - 1)
    smoothed = _gaussian_smooth(resampled, SMOOTH_OCTAVES, octaves_per_point)
    return smoothed - _gaussian_smooth(smoothed, DETREND_OCTAVES, octaves_per_point)


def measure_band_shift(source: np.ndarray, output: np.ndarray, sr: int, band: str = "full") -> float | None:
    """Ratio by which the spectral envelope moved in `band` (1.15 = up 15%).

    Cross-correlates the two smoothed log-frequency spectra and takes the peak,
    refined by parabolic interpolation so the result is finer than one grid step.
    """
    lo_hz, hi_hz = BANDS[band]
    grid = np.logspace(np.log2(LOG_GRID_LO_HZ), np.log2(LOG_GRID_HI_HZ), LOG_GRID_POINTS, base=2.0)
    octaves_per_point = np.log2(LOG_GRID_HI_HZ / LOG_GRID_LO_HZ) / (LOG_GRID_POINTS - 1)
    mask = (grid >= lo_hz) & (grid <= hi_hz)
    if mask.sum() < 16:
        return None

    a = _log_spectrum(source, sr, measure_pitch_median(source, sr) or 150.0)[mask]
    b = _log_spectrum(output, sr, measure_pitch_median(output, sr) or 150.0)[mask]
    a = a - a.mean()
    b = b - b.mean()
    if not (np.any(a) and np.any(b)):
        return None

    max_lag = int(MAX_SHIFT_OCTAVES / octaves_per_point)
    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.empty(lags.size)
    for i, lag in enumerate(lags):
        # Positive lag: the output envelope sits `lag` points higher in frequency.
        if lag >= 0:
            x, y = a[: a.size - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: b.size + lag]
        denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
        scores[i] = float(np.dot(x, y)) / denom if denom > 0 else -1.0

    peak = int(np.argmax(scores))
    offset = 0.0
    if 0 < peak < scores.size - 1:
        left, mid, right = scores[peak - 1], scores[peak], scores[peak + 1]
        denom = left - 2.0 * mid + right
        if abs(denom) > 1e-12:
            offset = 0.5 * (left - right) / denom
    return float(2.0 ** ((lags[peak] + offset) * octaves_per_point))


def measure_log_spectral_distance(a: np.ndarray, b: np.ndarray, sr: int,
                                  lo_hz: float = 200.0, hi_hz: float = 5000.0) -> float | None:
    """Mean absolute difference between two log spectral envelopes, in dB.

    Used only where the two signals are supposed to be the same (an identity pass,
    or a forward transform followed by its inverse), so any distance is distortion
    the engine introduced. Envelopes rather than raw spectra, because PSOLA does
    not preserve phase or exact harmonic placement and comparing those would score
    an inaudible difference as damage.
    """
    f0_a = measure_pitch_median(a, sr) or 150.0
    f0_b = measure_pitch_median(b, sr) or 150.0
    grid = np.logspace(np.log2(LOG_GRID_LO_HZ), np.log2(LOG_GRID_HI_HZ), LOG_GRID_POINTS, base=2.0)
    mask = (grid >= lo_hz) & (grid <= hi_hz)
    ea = _log_spectrum(a, sr, f0_a, ENVELOPE_ONLY_COEFFICIENTS)[mask]
    eb = _log_spectrum(b, sr, f0_b, ENVELOPE_ONLY_COEFFICIENTS)[mask]
    if not (ea.size and eb.size):
        return None
    # Convert from natural-log magnitude to dB, and remove any overall level
    # difference, which the leveller is entitled to introduce.
    diff = (eb - eb.mean()) - (ea - ea.mean())
    return float(np.mean(np.abs(diff)) * 20.0 / math.log(10.0))


def measure_hnr(audio: np.ndarray, sr: int, floor: float = 60.0) -> float | None:
    """Harmonics-to-noise ratio in dB; the standard acoustic proxy for breathiness."""
    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=float(sr))
    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, float(floor), 0.1, 1.0)
        value = call(harmonicity, "Get mean", 0.0, 0.0)
    except Exception:
        return None
    return None if not math.isfinite(value) else float(value)


def measure_high_band_hnr(audio: np.ndarray, sr: int, cutoff: float = 1500.0) -> float | None:
    """HNR above `cutoff`. Breathy phonation loses periodicity in the upper bands
    first, and full-band HNR is dominated by the strong low harmonics, so the
    full-band figure alone understates the change."""
    spec = np.fft.rfft(np.asarray(audio, dtype=np.float64))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    spec = spec * np.clip((freqs - cutoff) / max(cutoff * 0.5, 1.0), 0.0, 1.0)
    return measure_hnr(np.fft.irfft(spec, n=audio.size), sr, floor=60.0)


def measure_pitch_median(audio: np.ndarray, sr: int) -> float | None:
    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=float(sr))
    try:
        values = np.asarray(sound.to_pitch(time_step=0.01, pitch_floor=50.0, pitch_ceiling=700.0)
                            .selected_array["frequency"], dtype=float)
    except Exception:
        return None
    voiced = values[np.isfinite(values) & (values > 0)]
    return float(np.median(voiced)) if voiced.size else None


def _st(ratio: float) -> float:
    return 12.0 * math.log2(ratio)


# --------------------------------------------------------------------------
# One run through the real API code path
# --------------------------------------------------------------------------
def run(audio: np.ndarray, sr: int, engine: str, mode: str = "explore", protect: bool = False, **params: Any) -> dict[str, Any]:
    """Call the production transform, then measure the result."""
    request = {"engine": engine, "mode": mode, "artifact_protection": protect, **params}
    started = time.perf_counter()
    out, meta = transform_audio(audio, sr, request)
    elapsed = time.perf_counter() - started
    return {"audio": out, "meta": meta, "seconds": elapsed,
            "seconds_per_audio_second": elapsed / (audio.size / sr)}


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def experiment_calibration() -> list[dict[str, Any]]:
    """What the measurement itself is worth.

    Every other number here is a shift estimate, so the first thing to establish
    is the estimator's own error. Pairs of signals are synthesised with a known
    formant ratio and measured; the residual is the noise floor that the engine
    comparisons have to beat to mean anything.
    """
    rows: list[dict[str, Any]] = []
    for vowel_name in ("a", "i", "u", "e"):
        base, _ = signals.vowel(sr=SR, vowel_name=vowel_name)
        for true_scale in (0.85, 0.92, 1.08, 1.15, 1.25):
            moved, _ = signals.vowel(sr=SR, vowel_name=vowel_name, formant_scale=true_scale)
            measured = measure_band_shift(base, moved, SR, "full")
            rows.append({
                "case": f"/{vowel_name}/ formants x{true_scale}",
                "true_scale": true_scale,
                "measured": round(measured, 4) if measured else None,
                "error_pct": round(100.0 * (measured / true_scale - 1.0), 2) if measured else None,
            })
    # A pure pitch change must not register as a formant shift.
    for semitones in (-4.0, 4.0, 6.0):
        base, _ = signals.vowel(sr=SR, vowel_name="a", f0=165.0)
        moved, _ = signals.vowel(sr=SR, vowel_name="a", f0=165.0 * 2.0 ** (semitones / 12.0))
        measured = measure_band_shift(base, moved, SR, "full")
        rows.append({
            "case": f"pitch only {semitones:+.0f} st (truth 1.0)",
            "true_scale": 1.0,
            "measured": round(measured, 4) if measured else None,
            "error_pct": round(100.0 * (measured - 1.0), 2) if measured else None,
        })
    return rows


def experiment_formant_fidelity(profiles: dict[str, Any], scales: list[float]) -> list[dict[str, Any]]:
    """Requested resonance scale vs. the shift actually measured in the envelope.

    This is the experiment that decides whether the 1.08 ceiling can be raised: if
    WORLD tracks the request at 1.15-1.25 while Praat saturates or degrades, the
    ceiling is a property of the Praat method rather than of the task.
    """
    rows: list[dict[str, Any]] = []
    for profile_name, profile in profiles.items():
        # Sustained vowels, which is the condition the estimator was calibrated on.
        # The mixed utterance adds a per-speaker bias of its own that would be read
        # as an engine difference.
        audio, _ = signals.vowel(sr=SR, seconds=1.6, vowel_name="a",
                                 f0=profile["f0"], formant_scale=profile["formant_scale"])
        for engine in ("praat", "world"):
            for scale in scales:
                result = run(audio, SR, engine, resonance_scale=scale, pitch_semitones=0.0,
                             pitch_range_scale=1.0, brightness_db=0.0)
                applied = result["meta"]["effective"]["resonance_scale"]
                measured = measure_band_shift(audio, result["audio"], SR, "full")
                rows.append({
                    "profile": profile_name,
                    "engine": engine,
                    "requested_scale": scale,
                    "applied_scale": round(applied, 4),
                    "clipped": abs(applied - scale) > 1e-4,
                    "measured_shift": round(measured, 4) if measured else None,
                    # Error against what the engine was allowed to apply, so a
                    # clipped request is not scored as an engine failure.
                    "error_vs_applied_pct": round(100.0 * (measured / applied - 1.0), 2) if measured else None,
                    "quality_score": result["meta"]["artifact_audit"]["quality_score"],
                    "flags": result["meta"]["artifact_audit"]["flags"],
                })
    return rows


def experiment_independence(profiles: dict[str, Any]) -> list[dict[str, Any]]:
    """Cross-talk. Praat resamples and then restores pitch, so the two controls are
    coupled by construction; WORLD edits separate streams.

    The shift estimator has a small bias of its own when the pitch changes, so each
    row carries a control: the *same* vowel resynthesised from scratch at the
    shifted F0 with its formants deliberately left alone. Whatever the estimator
    reports for that control is bias, and the corrected column divides it out. A
    corrected drift near zero means the engine really did leave the formants where
    they were.
    """
    rows: list[dict[str, Any]] = []
    semitones = 4.0
    for profile_name, profile in profiles.items():
        audio, _ = signals.vowel(sr=SR, seconds=1.6, vowel_name="a",
                                 f0=profile["f0"], formant_scale=profile["formant_scale"])
        control, _ = signals.vowel(sr=SR, seconds=1.6, vowel_name="a",
                                   f0=profile["f0"] * 2.0 ** (semitones / 12.0),
                                   formant_scale=profile["formant_scale"])
        bias = measure_band_shift(audio, control, SR, "full") or 1.0
        source_pitch = measure_pitch_median(audio, SR)

        for engine in ("praat", "world"):
            pitch_only = run(audio, SR, engine, pitch_semitones=semitones, resonance_scale=1.0,
                             pitch_range_scale=1.0, brightness_db=0.0)
            drift = measure_band_shift(audio, pitch_only["audio"], SR, "full")

            res_only = run(audio, SR, engine, pitch_semitones=0.0, resonance_scale=1.12,
                           pitch_range_scale=1.0, brightness_db=0.0)
            after_pitch = measure_pitch_median(res_only["audio"], SR)
            pitch_drift = _st(after_pitch / source_pitch) if (after_pitch and source_pitch) else None

            rows.append({
                "profile": profile_name,
                "engine": engine,
                "control_bias_pct": round(100.0 * (bias - 1.0), 2),
                "raw_formant_drift_pct": round(100.0 * (drift - 1.0), 2) if drift else None,
                "corrected_formant_drift_pct": round(100.0 * (drift / bias - 1.0), 2) if drift else None,
                "pitch_drift_from_resonance_st": round(pitch_drift, 3) if pitch_drift is not None else None,
                "pitch_only_quality": pitch_only["meta"]["artifact_audit"]["quality_score"],
                "resonance_only_quality": res_only["meta"]["artifact_audit"]["quality_score"],
            })
    return rows


def experiment_per_band(profiles: dict[str, Any]) -> list[dict[str, Any]]:
    """Move F1 alone, then F2/F3 alone. Praat is included to show it cannot.

    Run on a sustained /a/: its F1 sits at ~730 Hz, well clear of both the F0 and
    the band edge, so the low-band estimate is not measuring the source cutoff.
    """
    rows: list[dict[str, Any]] = []
    cases = [
        ("low_only", {"resonance_low_scale": 1.18, "resonance_high_scale": 1.0}),
        ("high_only", {"resonance_low_scale": 1.0, "resonance_high_scale": 1.18}),
        ("opposed", {"resonance_low_scale": 0.88, "resonance_high_scale": 1.18}),
    ]
    for profile_name, profile in profiles.items():
        audio, _ = signals.vowel(sr=SR, seconds=1.6, vowel_name="a",
                                 f0=profile["f0"], formant_scale=profile["formant_scale"])
        for engine in ("praat", "world"):
            for case, params in cases:
                result = run(audio, SR, engine, pitch_semitones=0.0, pitch_range_scale=1.0,
                             brightness_db=0.0, resonance_scale=1.0, **params)
                rows.append({
                    "profile": profile_name,
                    "engine": engine,
                    "case": case,
                    "requested": params,
                    "supports_per_band": result["meta"]["supports_per_band_resonance"],
                    "low_band_shift": round(measure_band_shift(audio, result["audio"], SR, "low") or 0.0, 4),
                    "high_band_shift": round(measure_band_shift(audio, result["audio"], SR, "high") or 0.0, 4),
                    "quality_score": result["meta"]["artifact_audit"]["quality_score"],
                })
    return rows


def experiment_breathiness(profiles: dict[str, Any]) -> list[dict[str, Any]]:
    """HNR against the breathiness control. Monotonic and wide is what we want."""
    rows: list[dict[str, Any]] = []
    for profile_name, profile in profiles.items():
        audio, _ = signals.vowel(sr=SR, seconds=1.4, f0=profile["f0"],
                                 formant_scale=profile["formant_scale"])
        source_hnr = measure_hnr(audio, SR)
        source_high_hnr = measure_high_band_hnr(audio, SR)
        for engine in ("praat", "world"):
            for amount in (-0.6, -0.3, 0.0, 0.3, 0.6, 1.0):
                result = run(audio, SR, engine, breathiness=amount, pitch_semitones=0.0,
                             resonance_scale=1.0, pitch_range_scale=1.0, brightness_db=0.0)
                hnr = measure_hnr(result["audio"], SR)
                high_hnr = measure_high_band_hnr(result["audio"], SR)
                rows.append({
                    "profile": profile_name,
                    "engine": engine,
                    "requested_breathiness": amount,
                    "applied_breathiness": result["meta"]["effective"]["breathiness"],
                    "source_hnr_db": round(source_hnr, 2) if source_hnr is not None else None,
                    "output_hnr_db": round(hnr, 2) if hnr is not None else None,
                    "hnr_change_db": round(hnr - source_hnr, 2) if None not in (hnr, source_hnr) else None,
                    "high_band_hnr_change_db": round(high_hnr - source_high_hnr, 2) if None not in (high_hnr, source_high_hnr) else None,
                    "quality_score": result["meta"]["artifact_audit"]["quality_score"],
                })
    return rows


def experiment_identity(profiles: dict[str, Any]) -> list[dict[str, Any]]:
    """The vocoder's own distortion floor: every control neutral.

    The artifact audit scores both engines at 100 on clean material, so it cannot
    separate them. An identity pass can: whatever the output differs from the
    input by is what the engine costs before any transformation is even asked for.
    """
    rows: list[dict[str, Any]] = []
    for profile_name, profile in profiles.items():
        for label, audio in (
            ("sustained vowel", signals.vowel(sr=SR, seconds=1.4, f0=profile["f0"],
                                              formant_scale=profile["formant_scale"])[0]),
            ("utterance", signals.utterance(sr=SR, **profile)[0]),
        ):
            source_hnr = measure_hnr(audio, SR)
            for engine in ("praat", "world"):
                result = run(audio, SR, engine, pitch_semitones=0.0, resonance_scale=1.0,
                             pitch_range_scale=1.0, brightness_db=0.0, breathiness=0.0)
                hnr = measure_hnr(result["audio"], SR)
                rows.append({
                    "profile": profile_name,
                    "signal": label,
                    "engine": engine,
                    "log_spectral_distance_db": round(measure_log_spectral_distance(audio, result["audio"], SR) or 0.0, 3),
                    "hnr_change_db": round(hnr - source_hnr, 2) if None not in (hnr, source_hnr) else None,
                    "quality_score": result["meta"]["artifact_audit"]["quality_score"],
                })
    return rows


def experiment_roundtrip(profiles: dict[str, Any], scales: list[float]) -> list[dict[str, Any]]:
    """Transform up by `scale`, then back down by `1/scale`, and see what is left.

    This is the direct test of whether a higher formant ceiling is safe. An engine
    that moves formants cleanly returns close to the original; one that is really
    resampling and patching the pitch back on accumulates damage, and the damage
    grows with the scale. Rows where the inverse fell outside the engine's own
    limits are marked, because there the round trip is not a fair comparison --
    that Praat cannot even express the inverse is itself part of the answer.
    """
    rows: list[dict[str, Any]] = []
    for profile_name, profile in profiles.items():
        audio, _ = signals.vowel(sr=SR, seconds=1.4, f0=profile["f0"],
                                 formant_scale=profile["formant_scale"])
        source_hnr = measure_hnr(audio, SR)
        for engine in ("praat", "world"):
            for scale in scales:
                forward = run(audio, SR, engine, resonance_scale=scale, pitch_semitones=0.0,
                              pitch_range_scale=1.0, brightness_db=0.0)
                inverse = run(forward["audio"], SR, engine, resonance_scale=1.0 / scale,
                              pitch_semitones=0.0, pitch_range_scale=1.0, brightness_db=0.0)
                applied_fwd = forward["meta"]["effective"]["resonance_scale"]
                applied_inv = inverse["meta"]["effective"]["resonance_scale"]
                hnr = measure_hnr(inverse["audio"], SR)
                rows.append({
                    "profile": profile_name,
                    "engine": engine,
                    "scale": scale,
                    # The reported effective values are rounded, so compare loosely.
                    "clipped": abs(applied_fwd - scale) > 1e-4 or abs(applied_inv - 1.0 / scale) > 1e-4,
                    "roundtrip_lsd_db": round(measure_log_spectral_distance(audio, inverse["audio"], SR) or 0.0, 3),
                    "roundtrip_hnr_change_db": round(hnr - source_hnr, 2) if None not in (hnr, source_hnr) else None,
                    "residual_shift": round(measure_band_shift(audio, inverse["audio"], SR, "full") or 0.0, 4),
                })
    return rows


def experiment_cost(profiles: dict[str, Any]) -> list[dict[str, Any]]:
    """Wall clock for a realistic full-strength request, with backoff enabled.

    This decides deployability: the backoff loop can run the whole transform up to
    four times, so the per-pass cost is multiplied by up to four in the worst case.
    """
    rows: list[dict[str, Any]] = []
    profile = profiles["mid"]
    for seconds in (1.5, 5.0, 15.0):
        chunk, _ = signals.utterance(sr=SR, **profile)
        reps = int(np.ceil(seconds * SR / chunk.size))
        audio = np.tile(chunk, reps)[: int(seconds * SR)]
        for engine in ("praat", "world"):
            result = run(audio, SR, engine, mode="natural", protect=True, pitch_semitones=2.5,
                         resonance_scale=1.08, pitch_range_scale=1.15, brightness_db=1.0)
            rows.append({
                "engine": engine,
                "audio_seconds": seconds,
                "wall_seconds": round(result["seconds"], 3),
                "realtime_factor": round(result["seconds_per_audio_second"], 3),
                "worst_case_4_pass_s": round(result["seconds"] * 4, 2),
                "backoff_strength": result["meta"]["artifact_backoff_strength"],
                "quality_score": result["meta"]["artifact_audit"]["quality_score"],
            })
    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, list):
                value = ", ".join("-" if v is None else str(v) for v in value) or "-"
            elif isinstance(value, dict):
                value = ", ".join(f"{k}={v}" for k, v in value.items())
            cells.append("-" if value is None or value == "" else str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="One speaker profile and fewer scales.")
    parser.add_argument("--json", type=str, default=None, help="Also write raw results here.")
    args = parser.parse_args(argv)

    if not world_backend.world_available():
        print(f"pyworld unavailable ({world_backend.world_import_error()}); install with `pip install .[world]`.",
              file=sys.stderr)
        return 2

    profiles = {"mid": signals.PROFILES["mid"]} if args.fast else signals.PROFILES
    scales = [1.08, 1.15] if args.fast else [1.04, 1.08, 1.15, 1.20, 1.25]

    results = {
        "calibration": experiment_calibration(),
        "formant_fidelity": experiment_formant_fidelity(profiles, scales),
        "independence": experiment_independence(profiles),
        "per_band": experiment_per_band(profiles),
        "breathiness": experiment_breathiness(profiles),
        "identity": experiment_identity(profiles),
        "roundtrip": experiment_roundtrip(profiles, [1.08, 1.15, 1.25]),
        "cost": experiment_cost(signals.PROFILES),
    }

    print("## 0. Measurement calibration — the estimator's own error\n")
    print(_table(results["calibration"], ["case", "true_scale", "measured", "error_pct"]))
    print("\n## 1. Formant fidelity — requested scale vs. measured shift\n")
    print(_table(results["formant_fidelity"], [
        "profile", "engine", "requested_scale", "applied_scale", "clipped", "measured_shift",
        "error_vs_applied_pct", "quality_score", "flags"]))
    print("\n## 2. Independence — cross-talk between controls\n")
    print(_table(results["independence"], [
        "profile", "engine", "control_bias_pct", "raw_formant_drift_pct",
        "corrected_formant_drift_pct", "pitch_drift_from_resonance_st",
        "pitch_only_quality", "resonance_only_quality"]))
    print("\n## 3. Per-band envelope warping\n")
    print(_table(results["per_band"], [
        "profile", "engine", "case", "requested", "supports_per_band",
        "low_band_shift", "high_band_shift", "quality_score"]))
    print("\n## 4. Breathiness — aperiodicity control vs. HNR\n")
    print(_table(results["breathiness"], [
        "profile", "engine", "requested_breathiness", "applied_breathiness",
        "source_hnr_db", "output_hnr_db", "hnr_change_db", "high_band_hnr_change_db", "quality_score"]))
    print("\n## 5. Identity pass — the vocoder's own distortion floor\n")
    print(_table(results["identity"], [
        "profile", "signal", "engine", "log_spectral_distance_db", "hnr_change_db", "quality_score"]))
    print("\n## 6. Round trip — transform by s, then by 1/s\n")
    print(_table(results["roundtrip"], [
        "profile", "engine", "scale", "clipped", "roundtrip_lsd_db",
        "roundtrip_hnr_change_db", "residual_shift"]))
    print("\n## 7. Cost\n")
    print(_table(results["cost"], [
        "engine", "audio_seconds", "wall_seconds", "realtime_factor", "worst_case_4_pass_s",
        "backoff_strength", "quality_score"]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nRaw results written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
