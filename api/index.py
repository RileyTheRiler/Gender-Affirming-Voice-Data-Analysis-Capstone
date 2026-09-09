from __future__ import annotations

import base64
import io
import math
import wave
from pathlib import Path
from typing import Any, Literal

import numpy as np
import parselmouth
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from parselmouth.praat import call

try:  # Vercel imports this file as a top-level module; pytest imports the package.
    from . import world_backend
except ImportError:  # pragma: no cover
    import world_backend  # type: ignore[no-redef]

app = FastAPI(title="Voice Target Lab API")

MAX_SECONDS = 15.0
ROOT = Path(__file__).resolve().parents[1]

# Vocal-weight tilt: pivot frequency and how far from it the full amount is reached.
WEIGHT_PIVOT_HZ = 1000.0
WEIGHT_SPAN_OCTAVES = 2.0
WEIGHT_BAND_HZ = (60.0, 8000.0)

# Output level correction. The audit band must match the leveller clamp, otherwise
# a gain the leveller itself chose gets reported as an anomaly.
LEVEL_GAIN_MIN = 0.65
LEVEL_GAIN_MAX = 1.50

# How much quality a backoff pass must buy to justify weakening the transformation.
BACKOFF_FIDELITY_WEIGHT = 25.0

# Parameter limits, per engine and mode.
#
# Pitch, intonation range and brightness are held identical across engines so the
# two paths can be compared like for like. Only the two limits that exist because
# of how the engine works differ:
#
# * resonance ceiling - Praat moves formants by resampling, so it has to stay
#   near 1.0 or the resampling becomes audible; WORLD warps the envelope
#   directly, so it is allowed the 15-25% the acoustic literature describes.
# * breathiness - Praat's PSOLA excitation cannot express it at all, so its
#   limits are pinned to zero rather than left unbounded and silently ignored.
PARAM_LIMITS: dict[tuple[str, str], dict[str, tuple[float, float]]] = {
    ("praat", "natural"): {
        "pitch_semitones": (-3.0, 3.0),
        "resonance_scale": (0.96, 1.08),
        "pitch_range_scale": (0.80, 1.30),
        "brightness_db": (-2.0, 2.0),
        "breathiness": (0.0, 0.0),
    },
    ("praat", "explore"): {
        "pitch_semitones": (-6.0, 6.0),
        "resonance_scale": (0.90, 1.15),
        "pitch_range_scale": (0.50, 1.80),
        "brightness_db": (-6.0, 6.0),
        "breathiness": (0.0, 0.0),
    },
    ("world", "natural"): {
        "pitch_semitones": (-3.0, 3.0),
        "resonance_scale": (0.88, 1.15),
        "pitch_range_scale": (0.80, 1.30),
        "brightness_db": (-2.0, 2.0),
        "breathiness": (-0.5, 0.5),
    },
    ("world", "explore"): {
        "pitch_semitones": (-6.0, 6.0),
        "resonance_scale": (0.80, 1.25),
        "pitch_range_scale": (0.50, 1.80),
        "brightness_db": (-6.0, 6.0),
        "breathiness": (-1.0, 1.0),
    },
}


class TransformRequest(BaseModel):
    wav_base64: str = Field(min_length=16)
    pitch_semitones: float = 2.0
    resonance_scale: float = 1.045
    pitch_range_scale: float = 1.10
    brightness_db: float = 0.25
    mode: Literal["natural", "explore"] = "natural"
    artifact_protection: bool = True
    engine: Literal["praat", "world"] = "praat"
    # WORLD-only controls. They are accepted for either engine so the client can
    # keep one parameter set, but the response reports when an engine ignored one.
    breathiness: float = 0.0
    resonance_low_scale: float | None = None
    resonance_high_scale: float | None = None


def _read_wav(payload: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(payload), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sr = wf.getframerate()
            frames = wf.getnframes()
            if channels not in (1, 2):
                raise ValueError("Only mono or stereo WAV files are supported.")
            if sample_width != 2:
                raise ValueError("WAV must use 16-bit PCM samples.")
            if sr < 16000 or sr > 96000:
                raise ValueError("Sample rate must be between 16 and 96 kHz.")
            if frames / sr > MAX_SECONDS + 0.25:
                raise ValueError(f"Recording must be {MAX_SECONDS:.0f} seconds or shorter.")
            raw = wf.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError("Could not read the WAV file.") from exc

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    if audio.size < int(sr * 0.20):
        raise ValueError("Recording is too short.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Recording contains invalid samples.")
    return audio, int(sr)


def _pitch_track(sound: parselmouth.Sound, floor: float, ceiling: float) -> dict[str, Any]:
    """Return robust descriptive pitch-track statistics without modifying audio."""
    empty = {"median_hz": None, "voiced_fraction": 0.0, "p10_hz": None, "p90_hz": None, "jump_p95_st": None}
    try:
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=float(floor), pitch_ceiling=float(ceiling))
        values = np.asarray(pitch.selected_array["frequency"], dtype=float)
    except Exception:
        # The analyser itself failed. Report that separately so the audit does not
        # blame the recording for what is really an analysis problem.
        return {**empty, "analysis_failed": True}

    valid = np.isfinite(values) & (values > 0)
    voiced = values[valid]
    if voiced.size == 0:
        return {**empty, "analysis_failed": False}

    jumps: list[float] = []
    if values.size > 1:
        pair = valid[:-1] & valid[1:]
        a = values[:-1][pair]
        b = values[1:][pair]
        if a.size:
            jumps = np.abs(12.0 * np.log2(b / a)).tolist()

    return {
        "median_hz": float(np.median(voiced)),
        "voiced_fraction": float(np.mean(valid)),
        "p10_hz": float(np.percentile(voiced, 10)),
        "p90_hz": float(np.percentile(voiced, 90)),
        "jump_p95_st": float(np.percentile(jumps, 95)) if jumps else 0.0,
        "analysis_failed": False,
    }


def _adaptive_pitch_bounds(sound: parselmouth.Sound) -> tuple[float, float, dict[str, Any]]:
    """Estimate a speaker-specific analysis range from a permissive first pass."""
    wide = _pitch_track(sound, 50.0, 600.0)
    median = wide["median_hz"]
    if median is None:
        return 60.0, 500.0, wide

    floor = float(np.clip(median * 0.45, 50.0, 120.0))
    ceiling = float(np.clip(median * 3.0, 300.0, 600.0))
    if ceiling < floor * 2.5:
        ceiling = min(600.0, floor * 2.5)
    return floor, ceiling, wide


def _normalize_params(params: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], str, str, bool]:
    """Clip the request to what the chosen engine and mode can actually deliver."""
    mode = str(params.get("mode", "natural")).lower()
    if mode not in {"natural", "explore"}:
        mode = "natural"
    engine = str(params.get("engine", "praat")).lower()
    if engine not in {"praat", "world"}:
        engine = "praat"
    if engine == "world" and not world_backend.world_available():
        engine = "praat"
    protect = bool(params.get("artifact_protection", mode == "natural"))

    resonance = float(params.get("resonance_scale", 1.0))
    low = params.get("resonance_low_scale")
    high = params.get("resonance_high_scale")
    requested = {
        "pitch_semitones": float(params.get("pitch_semitones", 0.0)),
        "resonance_scale": resonance,
        "resonance_low_scale": resonance if low is None else float(low),
        "resonance_high_scale": resonance if high is None else float(high),
        "pitch_range_scale": float(params.get("pitch_range_scale", 1.0)),
        "brightness_db": float(params.get("brightness_db", 0.0)),
        "breathiness": float(params.get("breathiness", 0.0)),
    }

    limits = PARAM_LIMITS[(engine, mode)]
    effective = {key: float(np.clip(requested[key], *bounds)) for key, bounds in limits.items()}
    # The two band scales share the resonance limits; on Praat they collapse onto
    # the single scale it supports, because resampling cannot split the bands.
    band_bounds = limits["resonance_scale"]
    if engine == "world":
        effective["resonance_low_scale"] = float(np.clip(requested["resonance_low_scale"], *band_bounds))
        effective["resonance_high_scale"] = float(np.clip(requested["resonance_high_scale"], *band_bounds))
    else:
        effective["resonance_low_scale"] = effective["resonance_scale"]
        effective["resonance_high_scale"] = effective["resonance_scale"]
    return requested, effective, mode, engine, protect


# Neutral (no-op) value for each control, used when a backoff pass has to pull the
# whole parameter set part-way back toward the untransformed voice.
NEUTRAL = {
    "pitch_semitones": 0.0,
    "resonance_scale": 1.0,
    "resonance_low_scale": 1.0,
    "resonance_high_scale": 1.0,
    "pitch_range_scale": 1.0,
    "brightness_db": 0.0,
    "breathiness": 0.0,
}


def _scale_from_neutral(params: dict[str, float], strength: float) -> dict[str, float]:
    strength = float(np.clip(strength, 0.0, 1.0))
    return {key: neutral + (float(params.get(key, neutral)) - neutral) * strength for key, neutral in NEUTRAL.items()}


def _weight_tilt_gain(freqs: np.ndarray, amount_db: float) -> np.ndarray:
    """Tilt in dB per frequency bin, symmetric in log frequency about the pivot.

    Anchored to hertz rather than to Nyquist so the same slider value means the
    same thing at every sample rate, and shaped over 250 Hz - 4 kHz so the whole
    control travel lands inside the band that carries speech.
    """
    low, high = WEIGHT_BAND_HZ
    banded = np.clip(np.asarray(freqs, dtype=float), low, min(high, max(freqs[-1], low + 1.0)))
    octaves = np.log2(banded / WEIGHT_PIVOT_HZ) / WEIGHT_SPAN_OCTAVES
    return float(amount_db) * np.clip(octaves, -1.0, 1.0)


def _apply_weight_tilt_stft(audio: np.ndarray, sr: int, amount_db: float) -> np.ndarray:
    """Apply a smooth, time-local spectral tilt with overlap-add reconstruction."""
    if abs(amount_db) < 1e-6 or audio.size == 0:
        return audio.copy()

    n_fft = 2048 if sr >= 32000 else 1024
    hop = n_fft // 4
    window = np.sqrt(np.hanning(n_fft) + 1e-12)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    gain = np.power(10.0, _weight_tilt_gain(freqs, amount_db) / 20.0)

    pad = n_fft
    padded = np.pad(np.asarray(audio, dtype=float), (pad, pad))
    out = np.zeros_like(padded)
    weight = np.zeros_like(padded)

    for start in range(0, max(1, len(padded) - n_fft + 1), hop):
        frame = padded[start : start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        spec = np.fft.rfft(frame * window)
        processed = np.fft.irfft(spec * gain, n=n_fft) * window
        end = min(start + n_fft, out.size)
        usable = end - start
        out[start:end] += processed[:usable]
        weight[start:end] += window[:usable] ** 2

    mask = weight > 1e-9
    out[mask] /= weight[mask]
    return out[pad : pad + audio.size]


def _safe_level(audio: np.ndarray, reference_rms: float) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(out)))) if out.size else 0.0
    if reference_rms > 1e-8 and rms > 1e-8:
        out *= float(np.clip(reference_rms / rms, LEVEL_GAIN_MIN, LEVEL_GAIN_MAX))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.96:
        out *= 0.96 / peak
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _encode_wav(audio: np.ndarray, sr: int) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _output_pitch_bounds(floor: float, ceiling: float, params: dict[str, float]) -> tuple[float, float]:
    """Follow the requested shift so an upward transform is not clipped by the source ceiling.

    The band is moved with the pitch median and widened only by the intonation
    scaling, keeping roughly the source's octave width so the tracker does not
    gain room for octave errors.
    """
    ratio = 2.0 ** (float(params["pitch_semitones"]) / 12.0)
    widen = math.sqrt(max(1.0, float(params["pitch_range_scale"])))
    low = float(np.clip(floor * ratio / widen, 40.0, 250.0))
    high = float(np.clip(ceiling * ratio * widen, 300.0, 800.0))
    if high < low * 2.5:
        high = min(800.0, low * 2.5)
    return low, high


def _run_praat_transform(audio: np.ndarray, sr: int, params: dict[str, float], floor: float, ceiling: float, old_median: float) -> np.ndarray:
    sound = parselmouth.Sound(audio, sampling_frequency=float(sr))
    new_median = old_median * (2.0 ** (params["pitch_semitones"] / 12.0))
    try:
        changed = call(sound, "Change gender", float(floor), float(ceiling), params["resonance_scale"], new_median, params["pitch_range_scale"], 1.0)
    except Exception as exc:
        raise ValueError("Praat could not transform this recording. Try recording again with clearer voiced speech.") from exc

    out = np.asarray(changed.values, dtype=np.float64).reshape(-1)
    if out.size > audio.size:
        out = out[: audio.size]
    elif out.size < audio.size:
        out = np.pad(out, (0, audio.size - out.size))
    return out


def _run_world_transform(
    audio: np.ndarray,
    sr: int,
    params: dict[str, float],
    floor: float,
    ceiling: float,
) -> np.ndarray:
    """Resynthesise through WORLD with each stream edited independently.

    Unlike the Praat path, the brightness tilt is folded into the spectral
    envelope the vocoder synthesises from, so it costs no extra STFT pass and
    cannot add overlap-add smearing of its own.
    """
    try:
        frames = world_backend.analyze(audio, sr, floor, ceiling)
    except Exception as exc:
        raise ValueError("WORLD could not analyse this recording. Try recording again with clearer voiced speech.") from exc

    freqs = frames["freqs"]
    out_floor, out_ceiling = _output_pitch_bounds(floor, ceiling, params)
    f0, _ = world_backend.transform_f0(
        frames["f0"],
        params["pitch_semitones"],
        params["pitch_range_scale"],
        out_floor,
        out_ceiling,
    )
    sp = world_backend.warp_spectral_envelope(
        frames["sp"], freqs, params["resonance_low_scale"], params["resonance_high_scale"]
    )
    if abs(params["brightness_db"]) > 1e-6:
        sp = world_backend.apply_envelope_tilt(sp, freqs, _weight_tilt_gain(freqs, params["brightness_db"]))
    ap = world_backend.apply_breathiness(frames["ap"], freqs, params["breathiness"])

    try:
        return world_backend.synthesize(f0, sp, ap, sr, length=audio.size)
    except Exception as exc:
        raise ValueError("WORLD could not resynthesise this recording.") from exc


def _spectral_tilt_db(audio: np.ndarray, sr: int) -> float | None:
    """Energy balance of 1-5 kHz against 50-1000 Hz, the usual "vocal weight" proxy."""
    if audio.size < 512:
        return None
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size))) ** 2
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    low = spectrum[(freqs > 50.0) & (freqs < 1000.0)].sum()
    high = spectrum[(freqs >= 1000.0) & (freqs < 5000.0)].sum()
    if low <= 0.0 or high <= 0.0:
        return None
    return float(10.0 * np.log10(high / low))


def _artifact_audit(source: np.ndarray, output: np.ndarray, sr: int, source_pitch: dict[str, Any], output_pitch: dict[str, Any], requested_shift: float) -> dict[str, Any]:
    flags: list[str] = []
    quality = 100.0

    source_rms = float(np.sqrt(np.mean(np.square(source)))) if source.size else 0.0
    output_rms = float(np.sqrt(np.mean(np.square(output)))) if output.size else 0.0
    rms_ratio = output_rms / max(source_rms, 1e-9)
    clip_fraction = float(np.mean(np.abs(output) >= 0.959)) if output.size else 0.0

    old = source_pitch.get("median_hz")
    new = output_pitch.get("median_hz")
    observed_shift = None
    if old and new:
        observed_shift = float(12.0 * math.log2(new / old))
        pitch_error = abs(observed_shift - requested_shift)
        if pitch_error > 0.75:
            flags.append("pitch target was not reproduced cleanly")
            quality -= min(28.0, 12.0 + pitch_error * 8.0)
    elif output_pitch.get("analysis_failed") or source_pitch.get("analysis_failed"):
        flags.append("pitch analysis could not run on this pass")
        quality -= 30.0
    else:
        flags.append("output pitch tracking became unstable")
        quality -= 30.0

    voiced_drop = float(source_pitch.get("voiced_fraction", 0.0) - output_pitch.get("voiced_fraction", 0.0))
    if voiced_drop > 0.15:
        flags.append("voiced-frame continuity dropped")
        quality -= min(22.0, voiced_drop * 100.0)

    src_jump = float(source_pitch.get("jump_p95_st") or 0.0)
    out_jump = float(output_pitch.get("jump_p95_st") or 0.0)
    if out_jump > max(10.0, src_jump + 4.0):
        flags.append("large frame-to-frame pitch jumps detected")
        quality -= min(20.0, out_jump)

    if clip_fraction > 0.001:
        flags.append("near-clipping samples detected")
        quality -= min(18.0, clip_fraction * 4000.0)

    if not (LEVEL_GAIN_MIN - 0.01 <= rms_ratio <= LEVEL_GAIN_MAX + 0.01):
        flags.append("output level changed unusually")
        quality -= 10.0

    source_tilt = _spectral_tilt_db(source, sr)
    output_tilt = _spectral_tilt_db(output, sr)

    return {
        "quality_score": round(float(np.clip(quality, 0.0, 100.0)), 1),
        "flags": flags,
        "spectral_tilt_source_db": round(source_tilt, 2) if source_tilt is not None else None,
        "spectral_tilt_output_db": round(output_tilt, 2) if output_tilt is not None else None,
        "spectral_tilt_change_db": round(output_tilt - source_tilt, 2) if None not in (source_tilt, output_tilt) else None,
        "clip_fraction": round(clip_fraction, 6),
        "rms_ratio": round(rms_ratio, 3),
        "voiced_fraction_source": round(float(source_pitch.get("voiced_fraction", 0.0)), 3),
        "voiced_fraction_output": round(float(output_pitch.get("voiced_fraction", 0.0)), 3),
        "pitch_jump_p95_source_st": round(src_jump, 2),
        "pitch_jump_p95_output_st": round(out_jump, 2),
        "observed_pitch_shift_semitones": round(observed_shift, 2) if observed_shift is not None else None,
    }


def _backend_label(engine: str, mode: str) -> str:
    family = "WORLD/pyworld" if engine == "world" else "Praat/Parselmouth"
    return f"{family} · {'Natural v2' if mode == 'natural' else 'Explore'}"


def transform_audio(audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    requested, bounded, mode, engine, protect = _normalize_params(params)
    reference_rms = float(np.sqrt(np.mean(np.square(audio))))
    sound = parselmouth.Sound(audio, sampling_frequency=float(sr))
    floor, ceiling, initial_wide = _adaptive_pitch_bounds(sound)
    source_pitch = _pitch_track(sound, floor, ceiling)
    old_median = source_pitch["median_hz"] or initial_wide["median_hz"]
    if old_median is None:
        raise ValueError("A stable pitch could not be estimated. Try a longer voiced sentence with less background noise.")

    strengths = [1.0]
    if mode == "natural" and protect:
        strengths = [1.0, 0.85, 0.70, 0.55]

    best: tuple[np.ndarray, dict[str, float], dict[str, Any], float] | None = None
    best_score = -math.inf
    for strength in strengths:
        effective = _scale_from_neutral(bounded, strength)
        if engine == "world":
            raw = _run_world_transform(audio, sr, effective, floor, ceiling)
        else:
            raw = _run_praat_transform(audio, sr, effective, floor, ceiling, old_median)
            # WORLD folds the tilt into its envelope; Praat needs a separate pass.
            raw = _apply_weight_tilt_stft(raw, sr, effective["brightness_db"])
        out = _safe_level(raw, reference_rms)
        out_floor, out_ceiling = _output_pitch_bounds(floor, ceiling, effective)
        out_sound = parselmouth.Sound(out, sampling_frequency=float(sr))
        out_pitch = _pitch_track(out_sound, out_floor, out_ceiling)
        audit = _artifact_audit(audio, out, sr, source_pitch, out_pitch, effective["pitch_semitones"])

        # A weaker pass always scores better on artifacts simply because it changes
        # less, so charge it for the distance it gives up from what was asked.
        score = audit["quality_score"] - BACKOFF_FIDELITY_WEIGHT * (1.0 - strength)
        if score > best_score:
            best, best_score = (out, effective, audit, strength), score
        if not audit["flags"]:
            best, best_score = (out, effective, audit, strength), math.inf
            break

    if best is None:
        raise ValueError("The transformation produced no usable result.")
    out, effective, audit, applied_strength = best
    out_floor, out_ceiling = _output_pitch_bounds(floor, ceiling, effective)
    output_pitch = _pitch_track(parselmouth.Sound(out, sampling_frequency=float(sr)), out_floor, out_ceiling)
    observed_median = output_pitch["median_hz"]
    observed_shift = None
    if observed_median and old_median:
        observed_shift = 12.0 * math.log2(observed_median / old_median)

    backed_off = applied_strength < 0.999
    return out, {
        "baseline_pitch_median_hz": round(old_median, 2),
        "output_pitch_median_hz": round(observed_median, 2) if observed_median else None,
        "observed_pitch_shift_semitones": round(observed_shift, 2) if observed_shift is not None else None,
        "requested": requested,
        "effective": {k: round(float(v), 5) for k, v in effective.items()},
        "mode": mode,
        "engine": engine,
        "engine_requested": str(params.get("engine", "praat")).lower(),
        "world_available": world_backend.world_available(),
        "supports_breathiness": engine == "world",
        "supports_per_band_resonance": engine == "world",
        "artifact_protection": protect,
        "artifact_backoff_applied": backed_off,
        "artifact_backoff_strength": round(float(applied_strength), 2),
        "artifact_audit": audit,
        "pitch_analysis": {
            "floor_hz": round(floor, 1),
            "ceiling_hz": round(ceiling, 1),
            "output_floor_hz": round(out_floor, 1),
            "output_ceiling_hz": round(out_ceiling, 1),
            "source_voiced_fraction": round(float(source_pitch["voiced_fraction"]), 3),
        },
        "backend": _backend_label(engine, mode),
    }


@app.get("/api")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "backend": "Praat/Parselmouth",
        "engine_version": "natural-v2",
        "max_seconds": MAX_SECONDS,
        "engines": ["praat"] + (["world"] if world_backend.world_available() else []),
        "world_import_error": world_backend.world_import_error(),
    }


@app.post("/api/transform")
def transform(req: TransformRequest) -> dict[str, Any]:
    try:
        raw = base64.b64decode(req.wav_base64, validate=True)
        audio, sr = _read_wav(raw)
        out, metrics = transform_audio(audio, sr, {
            "pitch_semitones": req.pitch_semitones,
            "resonance_scale": req.resonance_scale,
            "pitch_range_scale": req.pitch_range_scale,
            "brightness_db": req.brightness_db,
            "mode": req.mode,
            "artifact_protection": req.artifact_protection,
            "engine": req.engine,
            "breathiness": req.breathiness,
            "resonance_low_scale": req.resonance_low_scale,
            "resonance_high_scale": req.resonance_high_scale,
        })
        encoded = base64.b64encode(_encode_wav(out, sr)).decode("ascii")
        return {"wav_base64": encoded, "sample_rate": sr, "metrics": metrics}
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Transformation failed unexpectedly.") from exc


@app.get("/api/frontend")
def frontend_index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/api/frontend/{filename:path}")
def frontend_asset(filename: str) -> FileResponse:
    allowed = {"app.js", "styles.css", "manifest.webmanifest", "sw.js", "icon-192.png", "icon-512.png"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(ROOT / filename)
