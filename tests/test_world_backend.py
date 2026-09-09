import numpy as np
import pytest

from api import world_backend as wb
from api.index import PARAM_LIMITS, _normalize_params, _scale_from_neutral, transform_audio

pytestmark = pytest.mark.skipif(not wb.world_available(), reason=f"pyworld unavailable: {wb.world_import_error()}")

SR = 24000


def voiced(sr=SR, seconds=1.2, f0=180.0):
    t = np.arange(int(sr * seconds)) / sr
    y = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 8))
    y *= 0.55 + 0.45 * np.sin(np.pi * np.clip(t / seconds, 0, 1))
    return (0.15 * y).astype(float), sr


# --------------------------------------------------------------------------
# Envelope warping
# --------------------------------------------------------------------------
def test_scale_curve_reaches_each_band_target():
    freqs = np.fft.rfftfreq(1024, 1.0 / SR)
    curve = wb.formant_scale_curve(freqs, 1.20, 1.00)
    at = lambda hz: curve[int(np.argmin(np.abs(freqs - hz)))]  # noqa: E731
    assert at(400) == pytest.approx(1.20, abs=0.01)  # F1 territory
    assert at(2500) == pytest.approx(1.00, abs=0.01)  # F2/F3 territory
    assert at(1150) == pytest.approx(1.10, abs=0.03)  # halfway at the crossover


def test_scale_curve_relaxes_to_identity_at_nyquist():
    freqs = np.fft.rfftfreq(1024, 1.0 / SR)
    curve = wb.formant_scale_curve(freqs, 1.25, 1.25)
    assert curve[-1] == pytest.approx(1.0, abs=1e-6)


def test_warp_axis_is_strictly_monotonic_at_the_limits():
    freqs = np.fft.rfftfreq(1024, 1.0 / SR)
    for low, high in ((1.25, 0.80), (0.80, 1.25), (1.25, 1.25), (0.80, 0.80)):
        axis = wb._warp_axis(freqs, low, high)
        assert np.all(np.diff(axis) > 0), f"folded warp for {low}/{high}"


def test_unit_warp_is_an_exact_copy():
    sp = np.abs(np.random.default_rng(0).standard_normal((5, 513))) + 0.1
    freqs = np.fft.rfftfreq(1024, 1.0 / SR)
    assert np.array_equal(wb.warp_spectral_envelope(sp, freqs, 1.0, 1.0), sp)


def test_warp_moves_a_peak_to_the_requested_place():
    freqs = np.fft.rfftfreq(1024, 1.0 / SR)
    sp = np.exp(-0.5 * ((freqs - 700.0) / 60.0) ** 2)[None, :] + 1e-6
    warped = wb.warp_spectral_envelope(sp, freqs, 1.15, 1.15)
    assert freqs[int(np.argmax(warped[0]))] == pytest.approx(700.0 * 1.15, rel=0.03)


def test_low_band_warp_leaves_the_high_band_alone():
    freqs = np.fft.rfftfreq(1024, 1.0 / SR)
    sp = (np.exp(-0.5 * ((freqs - 600.0) / 60.0) ** 2)
          + np.exp(-0.5 * ((freqs - 2600.0) / 90.0) ** 2))[None, :] + 1e-6
    warped = wb.warp_spectral_envelope(sp, freqs, 1.18, 1.0)
    high = freqs > 1800.0
    assert freqs[high][int(np.argmax(warped[0][high]))] == pytest.approx(2600.0, rel=0.03)
    low = freqs < 1500.0
    assert freqs[low][int(np.argmax(warped[0][low]))] == pytest.approx(600.0 * 1.18, rel=0.05)


# --------------------------------------------------------------------------
# Breathiness
# --------------------------------------------------------------------------
def test_breathiness_zero_is_an_exact_copy():
    ap = np.full((4, 513), 0.05)
    assert np.array_equal(wb.apply_breathiness(ap, np.fft.rfftfreq(1024, 1.0 / SR), 0.0), ap)


def test_breathiness_adds_noise_to_a_band_that_had_none():
    """The reason for a blend rather than a gain: a power law cannot lift a band
    that starts at zero, which is exactly the clear voice the control is for."""
    freqs = np.fft.rfftfreq(1024, 1.0 / SR)
    ap = np.full((2, freqs.size), 1e-6)
    out = wb.apply_breathiness(ap, freqs, 1.0)
    assert out.max() > 0.2


def test_breathiness_is_monotonic_and_stays_in_range():
    freqs = np.fft.rfftfreq(1024, 1.0 / SR)
    ap = np.full((2, freqs.size), 0.1)
    previous = None
    for amount in (-1.0, -0.5, 0.0, 0.5, 1.0):
        out = wb.apply_breathiness(ap, freqs, amount)
        assert np.all(out > 0.0) and np.all(out < 1.0)
        mean = float(out.mean())
        if previous is not None:
            assert mean > previous
        previous = mean


def test_breathiness_weights_the_upper_bands_more():
    freqs = np.fft.rfftfreq(1024, 1.0 / SR)
    ap = np.full((1, freqs.size), 0.05)
    out = wb.apply_breathiness(ap, freqs, 1.0)[0]
    assert out[freqs > 4000.0].mean() > out[freqs < 500.0].mean()


# --------------------------------------------------------------------------
# F0 contour
# --------------------------------------------------------------------------
def test_f0_shift_moves_the_median_and_keeps_unvoiced_frames_unvoiced():
    f0 = np.array([0.0, 100.0, 200.0, 0.0, 150.0])
    out, median = wb.transform_f0(f0, 12.0, 1.0, 40.0, 800.0)
    assert median == pytest.approx(150.0, rel=0.01)
    assert np.array_equal(out == 0.0, f0 == 0.0)
    assert np.median(out[out > 0]) == pytest.approx(300.0, rel=0.01)


def test_f0_range_scaling_is_symmetric_in_semitones():
    """Praat scales the range linearly in hertz, which stretches the top of the
    contour further than the bottom; doing it in log hertz keeps it symmetric."""
    median = 150.0
    f0 = np.array([median / 2, median, median * 2])
    out, _ = wb.transform_f0(f0, 0.0, 2.0, 10.0, 2000.0)
    assert out[0] == pytest.approx(median / 4, rel=0.01)
    assert out[2] == pytest.approx(median * 4, rel=0.01)


def test_f0_is_clamped_to_the_output_band():
    f0 = np.array([100.0, 400.0])
    out, _ = wb.transform_f0(f0, 24.0, 1.0, 60.0, 500.0)
    assert out.max() <= 500.0


def test_all_unvoiced_input_is_returned_unchanged():
    f0 = np.zeros(10)
    out, median = wb.transform_f0(f0, 5.0, 1.5, 60.0, 500.0)
    assert median is None
    assert np.array_equal(out, f0)


# --------------------------------------------------------------------------
# End-to-end through the API
# --------------------------------------------------------------------------
def test_world_engine_round_trips_through_transform_audio():
    y, sr = voiced()
    out, meta = transform_audio(y, sr, dict(
        engine="world", mode="natural", pitch_semitones=2.0, resonance_scale=1.10,
        pitch_range_scale=1.1, brightness_db=0.5, breathiness=0.3, artifact_protection=True))
    assert len(out) == len(y)
    assert np.isfinite(out).all()
    assert meta["engine"] == "world"
    assert meta["backend"].startswith("WORLD")
    assert meta["supports_breathiness"] and meta["supports_per_band_resonance"]


def test_world_natural_allows_a_higher_formant_ceiling_than_praat():
    """The whole point of the spike: Praat's 1.08 is a limit of resampling, not
    of the task, so WORLD is allowed the range the acoustic literature describes."""
    assert PARAM_LIMITS[("world", "natural")]["resonance_scale"][1] > PARAM_LIMITS[("praat", "natural")]["resonance_scale"][1]


def test_praat_pins_breathiness_to_zero_rather_than_ignoring_it():
    _, effective, _, engine, _ = _normalize_params(dict(engine="praat", breathiness=0.8))
    assert engine == "praat"
    assert effective["breathiness"] == 0.0


def test_praat_collapses_the_two_band_scales_onto_one():
    _, effective, _, _, _ = _normalize_params(
        dict(engine="praat", mode="explore", resonance_scale=1.05,
             resonance_low_scale=1.15, resonance_high_scale=0.95))
    assert effective["resonance_low_scale"] == effective["resonance_high_scale"] == effective["resonance_scale"]


def test_band_scales_default_to_the_overall_scale():
    requested, effective, _, _, _ = _normalize_params(dict(engine="world", resonance_scale=1.10))
    assert requested["resonance_low_scale"] == requested["resonance_high_scale"] == 1.10
    assert effective["resonance_low_scale"] == effective["resonance_high_scale"] == 1.10


def test_unknown_engine_falls_back_to_praat():
    _, _, _, engine, _ = _normalize_params(dict(engine="wishful"))
    assert engine == "praat"


def test_backoff_scales_every_world_control_including_the_new_ones():
    params = dict(pitch_semitones=4.0, resonance_scale=1.20, resonance_low_scale=1.20,
                  resonance_high_scale=0.90, pitch_range_scale=1.4, brightness_db=2.0, breathiness=0.8)
    half = _scale_from_neutral(params, 0.5)
    assert half["breathiness"] == pytest.approx(0.4)
    assert half["resonance_low_scale"] == pytest.approx(1.10)
    assert half["resonance_high_scale"] == pytest.approx(0.95)
