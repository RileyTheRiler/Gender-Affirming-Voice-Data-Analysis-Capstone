import numpy as np
import parselmouth

from api.index import (
    _adaptive_pitch_bounds,
    _apply_brightness_stft,
    _normalize_params,
    _scale_from_neutral,
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


def test_natural_controls_are_conservatively_clipped():
    _, effective, mode, protect = _normalize_params(
        dict(pitch_semitones=99, resonance_scale=9, pitch_range_scale=9, brightness_db=99, mode="natural")
    )
    assert mode == "natural"
    assert protect is True
    assert effective["pitch_semitones"] == 3.0
    assert effective["resonance_scale"] == 1.08
    assert effective["pitch_range_scale"] == 1.3
    assert effective["brightness_db"] == 2.0


def test_explore_preserves_wider_limits():
    _, effective, mode, _ = _normalize_params(
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


def test_brightness_zero_is_exact_copy():
    y, sr = voiced()
    out = _apply_brightness_stft(y, sr, 0.0)
    assert np.array_equal(out, y)


def test_brightness_stft_preserves_length_and_finite():
    y, sr = voiced()
    out = _apply_brightness_stft(y, sr, 1.5)
    assert len(out) == len(y)
    assert np.isfinite(out).all()


def test_adaptive_pitch_bounds_follow_speaker():
    y, sr = voiced(f0=180.0)
    floor, ceiling, wide = _adaptive_pitch_bounds(parselmouth.Sound(y, sampling_frequency=sr))
    assert wide["median_hz"] is not None
    assert 50 <= floor <= 120
    assert 300 <= ceiling <= 600
    assert floor < 180 < ceiling
