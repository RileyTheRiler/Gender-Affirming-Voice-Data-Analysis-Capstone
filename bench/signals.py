"""Synthetic speech-like test signals with known ground truth.

The repository carries no speech recordings, so the benchmark synthesises its own
through an explicit source-filter model. That is a real limitation -- it measures
parameter fidelity and artifact behaviour, not perceptual naturalness -- but it
buys something a real recording cannot: the formant frequencies are *known*, so
"did the formants move by the requested factor" is a measurement rather than an
estimate on top of an estimate.
"""

from __future__ import annotations

import numpy as np

# (F1, F2, F3, F4) in Hz with bandwidths, roughly cardinal vowels for a speaker
# with a ~17 cm vocal tract. Individual runs rescale these per speaker profile.
VOWELS: dict[str, tuple[tuple[float, float], ...]] = {
    "a": ((730, 90), (1090, 110), (2440, 170), (3500, 250)),
    "i": ((270, 70), (2290, 110), (3010, 180), (3700, 250)),
    "u": ((300, 80), (870, 100), (2240, 170), (3400, 250)),
    "e": ((530, 85), (1840, 110), (2480, 170), (3600, 250)),
}


def _glottal_source(n: int, sr: int, f0_hz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Harmonic pulse train with a -12 dB/octave rolloff and a little jitter."""
    phase = np.cumsum(2.0 * np.pi * f0_hz / sr)
    src = np.zeros(n)
    max_harmonic = int(sr / 2.0 / max(float(np.max(f0_hz)), 1.0))
    for k in range(1, max(2, max_harmonic)):
        src += (1.0 / k**1.8) * np.sin(k * phase)
    src += 0.004 * rng.standard_normal(n) * float(np.std(src))  # aspiration floor
    return src


def _formant_filter(x: np.ndarray, sr: int, formants: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Cascade of analog resonators, applied in the frequency domain."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    h = np.ones_like(spec)
    for fc, bw in formants:
        h = h * (fc**2) / (fc**2 - freqs**2 + 1j * freqs * bw)
    return np.fft.irfft(spec * h, n=x.size)


def vowel(
    sr: int = 24000,
    seconds: float = 1.4,
    f0: float = 150.0,
    vowel_name: str = "a",
    formant_scale: float = 1.0,
    excursion_st: float = 3.5,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """A single sustained vowel with a moving F0. Returns the audio and its truth."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    contour = f0 * np.power(2.0, (excursion_st / 12.0) * np.sin(2.0 * np.pi * 0.8 * t))
    formants = tuple((fc * formant_scale, bw) for fc, bw in VOWELS[vowel_name])
    y = _formant_filter(_glottal_source(n, sr, contour, rng), sr, formants)
    y *= 0.22 / (np.max(np.abs(y)) + 1e-12)
    truth = {"f0_median_hz": float(np.median(contour))}
    truth.update({f"F{i + 1}_hz": fc for i, (fc, _) in enumerate(formants)})
    return y, truth


def utterance(
    sr: int = 24000,
    f0: float = 150.0,
    formant_scale: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Vowels, a fricative and short pauses, so voiced-fraction metrics mean something."""
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for i, name in enumerate(("a", "i", "u", "e")):
        seg, _ = vowel(sr=sr, seconds=0.55, f0=f0 * (1.0 + 0.04 * i), vowel_name=name,
                       formant_scale=formant_scale, excursion_st=2.5, seed=seed + i)
        parts.append(seg)
        if i == 1:  # a /s/-like fricative between the second and third vowel
            noise = rng.standard_normal(int(sr * 0.14))
            hp = np.fft.rfft(noise)
            freqs = np.fft.rfftfreq(noise.size, 1.0 / sr)
            hp *= np.clip((freqs - 3000.0) / 3000.0, 0.0, 1.0)
            parts.append(0.06 * np.fft.irfft(hp, n=noise.size))
        parts.append(np.zeros(int(sr * 0.06)))
    y = np.concatenate(parts)
    y *= 0.22 / (np.max(np.abs(y)) + 1e-12)
    truth = {"f0_median_hz": float(f0), **{f"F{i + 1}_hz": fc * formant_scale for i, (fc, _) in enumerate(VOWELS["a"])}}
    return y, truth


# Speaker profiles: a lower-pitched, longer-tract voice and a higher-pitched,
# shorter-tract one, so results are not an artifact of one starting point.
PROFILES = {
    "low_f0_long_tract": {"f0": 118.0, "formant_scale": 0.92},
    "mid": {"f0": 165.0, "formant_scale": 1.0},
    "high_f0_short_tract": {"f0": 212.0, "formant_scale": 1.14},
}
