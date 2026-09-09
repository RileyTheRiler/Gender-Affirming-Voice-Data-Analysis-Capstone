import numpy as np
import parselmouth
import pytest
from pydantic import ValidationError

from api.index import (
    LEVEL_GAIN_MAX,
    LEVEL_GAIN_MIN,
    MAX_WAV_BASE64_CHARS,
    TransformRequest,
    _adaptive_pitch_bounds,
    _apply_weight_tilt_stft,
    _normalize_params,
    _output_pitch_bounds,
    _scale_from_neutral,
    _spectral_tilt_db,
    _weight_tilt_gain,
    transform_audio,
)


def voiced(sr=24000, seconds=1.2, f0=180.0):
    t = np.arange(int(sr * seconds)) / sr
    y = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 8))
    y *= 0.55 + 0.45 * np.sin(np.pi * np.clip(t / seconds, 0, 1))
    return (0.15 * y).astype(float), sr


def test_transform_returns_same_length_and_finite_natural():
    y, sr = voiced()
    out, meta = transform_audio(
        y,
        sr,
        dict(
            pitch_semitones=2.0,
            resonance_scale=1.04,
            pitch_range_scale=1.1,
            brightness_db=0.5,
            mode="natural",
            artifact_protection=True,
        ),
    )
    assert len(out) == len(y)
    assert np.isfinite(out).all()
    assert meta["backend"].startswith("Praat/Parselmouth")
    assert meta["mode"] == "natural"
    assert 0 <= meta["artifact_audit"]["quality_score"] <= 100


def test_request_model_rejects_oversized_base64_before_decode():
    with pytest.raises(ValidationError):
        TransformRequest(wav_base64="A" * (MAX_WAV_BASE64_CHARS + 1))


def test_15_second_24khz_pcm_payload_fits_request_cap():
    wav_bytes = 44 + int(15 * 24000) * 2
    base64_chars = 4 * ((wav_bytes + 2) // 3)
    assert base64_chars < MAX_WAV_BASE64_CHARS


def test_natural_controls_are_conservatively_clipped():
    _, effective, mode, _engine, protect = _normalize_params(
        dict(pitch_semitones=99, resonance_scale=9, pitch_range_scale=9, brightness_db=99, mode="natural")
    )
    assert mode == "natural"
    assert protect is True
    assert effective["pitch_semitones"] == 3.0
    assert effective["resonance_scale"] == 1.08
    assert effective["pitch_range_scale"] == 1.3
    assert effective["brightness_db"] == 2.0


def test_explore_preserves_wider_limits():
    _, effective, mode, _engine, _ = _normalize_params(
        dict(pitch_semitones=99, resonance_scale=9, pitch_range_scale=9, brightness_db=99, mode="explore")
    )
    assert mode == "explore"
    assert effective["pitch_semitones"] == 6.0
    assert effective["resonance_scale"] == 1.15
    assert effective["pitch_range_scale"] == 1.8
    assert effective["brightness_db"] == 6.0


def test_scale_from_neutral_backs_off_all_dimensions():
    p = dict(pitch_semitones=3.0, resonance_scale=1.08, pitch_range_scale=1.3, brightness_db=2.0)
    q = _scale_from_neutral(p, 0.5)
    assert q["pitch_semitones"] == 1.5
    assert abs(q["resonance_scale"] - 1.04) < 1e-9
    assert abs(q["pitch_range_scale"] - 1.15) < 1e-9
    assert q["brightness_db"] == 1.0


def test_weight_tilt_zero_is_exact_copy():
    y, sr = voiced()
    out = _apply_weight_tilt_stft(y, sr, 0.0)
    assert np.array_equal(out, y)


def test_weight_tilt_preserves_length_and_finite():
    y, sr = voiced()
    out = _apply_weight_tilt_stft(y, sr, 1.5)
    assert len(out) == len(y)
    assert np.isfinite(out).all()


def test_weight_tilt_is_anchored_to_hertz_not_nyquist():
    """The same slider value must mean the same thing at every sample rate."""
    freqs = np.array([250.0, 1000.0, 4000.0])
    at_24k = _weight_tilt_gain(np.append(freqs, 12000.0), 2.0)[:3]
    at_48k = _weight_tilt_gain(np.append(freqs, 24000.0), 2.0)[:3]
    assert np.allclose(at_24k, at_48k)
    assert at_48k[0] == pytest.approx(-2.0)
    assert at_48k[1] == pytest.approx(0.0)
    assert at_48k[2] == pytest.approx(2.0)


def test_weight_tilt_moves_speech_band_energy_audibly():
    y, sr = voiced()
    before = _spectral_tilt_db(y, sr)
    lighter = _spectral_tilt_db(_apply_weight_tilt_stft(y, sr, 2.0), sr)
    heavier = _spectral_tilt_db(_apply_weight_tilt_stft(y, sr, -2.0), sr)
    assert lighter - before > 1.0
    assert before - heavier > 1.0


def test_level_audit_band_matches_leveller_clamp():
    assert LEVEL_GAIN_MIN < 1.0 < LEVEL_GAIN_MAX


def test_output_pitch_bounds_follow_an_upward_shift():
    floor, ceiling = 80.0, 300.0
    up = _output_pitch_bounds(floor, ceiling, {"pitch_semitones": 6.0, "pitch_range_scale": 1.0})
    assert up[1] > ceiling
    assert up[0] > floor
    flat = _output_pitch_bounds(floor, ceiling, {"pitch_semitones": 0.0, "pitch_range_scale": 1.0})
    assert flat == pytest.approx((floor, ceiling))


def test_adaptive_pitch_bounds_follow_speaker():
    y, sr = voiced(f0=180.0)
    floor, ceiling, wide = _adaptive_pitch_bounds(parselmouth.Sound(y, sampling_frequency=sr))
    assert wide["median_hz"] is not None
    assert 50 <= floor <= 120
    assert 300 <= ceiling <= 600
    assert floor < 180 < ceiling


def test_backoff_prefers_the_requested_strength_when_quality_is_close():
    y, sr = voiced()
    _, meta = transform_audio(
        y,
        sr,
        dict(
            pitch_semitones=3.0,
            resonance_scale=1.08,
            pitch_range_scale=1.3,
            brightness_db=2.0,
            mode="natural",
            artifact_protection=True,
        ),
    )
    assert meta["artifact_backoff_strength"] == 1.0
    assert meta["artifact_audit"]["spectral_tilt_change_db"] is not None
