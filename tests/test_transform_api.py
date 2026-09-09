import numpy as np
from api.index import transform_audio


def voiced(sr=24000, seconds=1.2, f0=180.0):
    t = np.arange(int(sr * seconds)) / sr
    y = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 8))
    y *= 0.55 + 0.45 * np.sin(np.pi * np.clip(t / seconds, 0, 1))
    return (0.15 * y).astype(float), sr


def test_transform_returns_same_length_and_finite():
    y, sr = voiced()
    out, meta = transform_audio(y, sr, dict(pitch_semitones=2.0, resonance_scale=1.04, pitch_range_scale=1.1, brightness_db=0.5))
    assert len(out) == len(y)
    assert np.isfinite(out).all()
    assert meta['backend'] == 'Praat/Parselmouth'


def test_controls_are_clipped():
    y, sr = voiced()
    _, meta = transform_audio(y, sr, dict(pitch_semitones=99, resonance_scale=9, pitch_range_scale=9, brightness_db=99))
    e = meta['effective']
    assert e['pitch_semitones'] == 6.0
    assert e['resonance_scale'] == 1.15
    assert e['pitch_range_scale'] == 1.8
    assert e['brightness_db'] == 6.0
