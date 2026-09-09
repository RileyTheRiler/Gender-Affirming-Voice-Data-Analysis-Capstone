from __future__ import annotations

import base64
import io
import math
import wave
from typing import Any

import numpy as np
import parselmouth
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from parselmouth.praat import call

app = FastAPI(title="Voice Target Lab API")

MAX_SECONDS = 15.0


class TransformRequest(BaseModel):
    wav_base64: str = Field(min_length=16)
    pitch_semitones: float = 2.5
    resonance_scale: float = 1.055
    pitch_range_scale: float = 1.15
    brightness_db: float = 0.5


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


def _pitch_median(sound: parselmouth.Sound) -> float | None:
    try:
        pitch = sound.to_pitch(time_step=None, pitch_floor=60.0, pitch_ceiling=500.0)
        value = float(call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz"))
        return value if np.isfinite(value) and value > 0 else None
    except Exception:
        return None


def _apply_brightness(audio: np.ndarray, sr: int, amount_db: float) -> np.ndarray:
    if abs(amount_db) < 1e-6 or audio.size == 0:
        return audio.copy()
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    norm = np.clip(freqs / max(sr / 2.0, 1.0), 0.0, 1.0)
    gain_db = amount_db * (norm - 0.5)
    gain = np.power(10.0, gain_db / 20.0)
    return np.fft.irfft(spectrum * gain, n=audio.size).real


def _safe_level(audio: np.ndarray, reference_rms: float) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(out)))) if out.size else 0.0
    if reference_rms > 1e-8 and rms > 1e-8:
        out *= float(np.clip(reference_rms / rms, 0.5, 2.0))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.98:
        out *= 0.98 / peak
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


def transform_audio(audio: np.ndarray, sr: int, params: dict[str, float]) -> tuple[np.ndarray, dict[str, Any]]:
    pitch_semitones = float(np.clip(params["pitch_semitones"], -6.0, 6.0))
    resonance_scale = float(np.clip(params["resonance_scale"], 0.90, 1.15))
    pitch_range_scale = float(np.clip(params["pitch_range_scale"], 0.50, 1.80))
    brightness_db = float(np.clip(params["brightness_db"], -6.0, 6.0))

    reference_rms = float(np.sqrt(np.mean(np.square(audio))))
    sound = parselmouth.Sound(audio, sampling_frequency=float(sr))
    old_median = _pitch_median(sound)
    if old_median is None:
        raise ValueError("A stable pitch could not be estimated. Try a longer voiced sentence with less background noise.")

    new_median = old_median * (2.0 ** (pitch_semitones / 12.0))
    try:
        changed = call(
            sound,
            "Change gender",
            60.0,
            500.0,
            resonance_scale,
            new_median,
            pitch_range_scale,
            1.0,
        )
    except Exception as exc:
        raise ValueError("Praat could not transform this recording. Try recording again with clearer voiced speech.") from exc

    out = np.asarray(changed.values, dtype=np.float64).reshape(-1)
    if out.size > audio.size:
        out = out[: audio.size]
    elif out.size < audio.size:
        out = np.pad(out, (0, audio.size - out.size))
    out = _apply_brightness(out, sr, brightness_db)
    out = _safe_level(out, reference_rms)

    changed_sound = parselmouth.Sound(out, sampling_frequency=float(sr))
    observed_median = _pitch_median(changed_sound)
    observed_shift = None
    if observed_median and old_median:
        observed_shift = 12.0 * math.log2(observed_median / old_median)

    return out, {
        "baseline_pitch_median_hz": round(old_median, 2),
        "output_pitch_median_hz": round(observed_median, 2) if observed_median else None,
        "observed_pitch_shift_semitones": round(observed_shift, 2) if observed_shift is not None else None,
        "effective": {
            "pitch_semitones": pitch_semitones,
            "resonance_scale": resonance_scale,
            "pitch_range_scale": pitch_range_scale,
            "brightness_db": brightness_db,
        },
        "backend": "Praat/Parselmouth",
    }


@app.get("/api")
def health() -> dict[str, Any]:
    return {"ok": True, "backend": "Praat/Parselmouth", "max_seconds": MAX_SECONDS}


@app.post("/api/transform")
def transform(req: TransformRequest) -> dict[str, Any]:
    try:
        raw = base64.b64decode(req.wav_base64, validate=True)
        audio, sr = _read_wav(raw)
        out, metrics = transform_audio(
            audio,
            sr,
            {
                "pitch_semitones": req.pitch_semitones,
                "resonance_scale": req.resonance_scale,
                "pitch_range_scale": req.pitch_range_scale,
                "brightness_db": req.brightness_db,
            },
        )
        encoded = base64.b64encode(_encode_wav(out, sr)).decode("ascii")
        return {"wav_base64": encoded, "sample_rate": sr, "metrics": metrics}
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Transformation failed unexpectedly.") from exc
