# WORLD vocoder backend — spike results

**Status: spike, behind a flag. Not the default, and not ready to be.**

The Praat path (`Change gender`) shifts formants by resampling the whole signal and
then restores the pitch with PSOLA. Three consequences follow from that one design
choice: resonance and pitch are coupled, the formant scale has to stay near 1.0 or
the resampling becomes audible, and the excitation is whatever PSOLA leaves behind,
so breathiness cannot be expressed at all.

WORLD decomposes speech into F0, a spectral envelope (CheapTrick) and band
aperiodicity (D4C), and resynthesises from the three edited streams. This spike
implements that path and measures it against the current one.

Reproduce with:

```bash
pip install -e '.[world]' pytest
python -m bench.compare_backends --json results.json
```

Full output: [`WORLD_SPIKE_RESULTS.md`](WORLD_SPIKE_RESULTS.md).

---

## What the measurements say

### 1. The 1.08 formant ceiling is a property of the Praat method, not of the task

Praat saturates. Asked for 1.20 or 1.25 it clips to 1.15 and the measured shift stops
moving — three identical rows, because it is literally the same transform. WORLD
tracks the request all the way to 1.25, with an error inside the estimator's own
±2–4% noise floor:

| requested | Praat applied → measured | WORLD applied → measured |
| --- | --- | --- |
| 1.08 | 1.08 → 1.021 | 1.08 → 1.034 |
| 1.15 | 1.15 → 1.107 | 1.15 → 1.104 |
| 1.20 | **1.15** → 1.107 (clipped) | 1.20 → 1.166 |
| 1.25 | **1.15** → 1.107 (clipped) | 1.25 → 1.234 |

(mid profile; the other two profiles behave the same way — see the full results.)

More importantly, WORLD does not degrade as it goes up. The round-trip test —
transform by *s*, then by *1/s*, and see what is left of the original — shows Praat
losing 3.7–5.5 dB of HNR at every scale, while WORLD stays within ±0.6 dB even at
1.25. Praat cannot even *express* the inverse above 1.11, because 1/1.15 falls below
its own 0.90 floor.

**So the ceiling can be raised — but on WORLD, not on Praat.** This spike sets WORLD's
natural-mode ceiling at 1.15 and explore at 1.25, and leaves Praat's at 1.08/1.15.

### 2. Breathiness works, and is genuinely new

Aperiodicity control produces a clean monotonic HNR response, roughly −12.5 dB across
the full positive range, weighted toward the upper bands as breathy phonation is:

| breathiness | 0.0 | 0.3 | 0.6 | 1.0 |
| --- | --- | --- | --- | --- |
| HNR change (dB) | −0.08 | −4.8 | −8.8 | −12.5 |
| high-band HNR change (dB) | −0.2 | −2.1 | −3.8 | −5.8 |

Praat is flat across the same sweep, as expected — it has no such control, and the
parameter table now pins its limits to zero rather than accepting the value and
silently discarding it.

One honest caveat: *negative* breathiness barely moves on these test signals
(+0.1 dB), because a synthetic source is already nearly noiseless and there is
nothing to remove. Whether "less breathy" is useful has to be checked on a real
breathy voice.

The implementation is a blend toward (or away from) full noise, not a gain. A
multiplicative or power-law control can only scale noise a band already has, so it
does nothing for a clear voice — which is exactly the case a breathiness control
exists for. This was the first implementation and the benchmark caught it: max HNR
change was −1.1 dB instead of −12.5 dB.

### 3. Per-band warping works, and Praat cannot do it at all

Moving F1 while leaving F2/F3 alone, and the reverse:

| case | Praat (low / high band) | WORLD (low / high band) |
| --- | --- | --- |
| F1 only ×1.18 | 1.00 / 0.95 | **1.06 / 0.95** |
| F2–F3 only ×1.18 | 1.00 / 0.95 | 1.00 / **1.13** |
| opposed (0.88 / 1.18) | 1.00 / 0.95 | **0.96 / 1.12** |

Praat's three rows are identical because it receives the same single scale in all
three cases; the split is not something it can represent. WORLD separates the bands
cleanly, though the realised split is compressed relative to the request (a requested
1.18 in one band reads ~1.06–1.13), partly the estimator's band-limited error and
partly the crossover's smooth blend deliberately not being a brick wall.

### 4. Pitch/formant independence: not demonstrated either way

This is the claim the benchmark **failed to settle**. After correcting for the
estimator's own bias, both engines show a residual −4% to −7% apparent formant drift
under a pitch-only change, with no consistent gap between them — including for WORLD,
which by construction cannot move the envelope when only F0 changes. That is a
measurement artifact, not a finding. The theoretical argument for independence still
holds, and the reverse direction is clean (WORLD's pitch drift under a resonance-only
change is ~0.001 st vs Praat's ~0.10 st), but the forward direction needs a better
estimator or real speech before anyone should quote a number for it.

### 5. Cost is the real deployment blocker

WORLD runs ~10× slower than Praat: 0.19× realtime against 0.019×.

| audio | Praat | WORLD | WORLD, worst-case 4-pass backoff |
| --- | --- | --- | --- |
| 5 s | 0.10 s | 0.96 s | 3.9 s |
| 15 s | 0.25 s | 2.86 s | **11.4 s** |

The artifact-protection loop can run the whole transform up to four times. At the
15-second limit that is 11.4 s inside a Vercel function — over the 10 s default for
Hobby-tier Python functions. Before WORLD could become the default, at least one of:
raise the function timeout, cap WORLD to shorter clips, use `dio` instead of
`harvest` for F0 (roughly 2× faster, less accurate), or make the backoff loop reuse
one analysis pass across strengths rather than re-analysing each time. The last is
the obvious win — the analysis is the expensive half and it does not depend on the
parameters at all.

### 6. One result that goes against WORLD

On the low-F0 sustained vowel, WORLD's identity pass has a *higher* log-spectral
distance than Praat's (4.59 dB vs 2.80 dB) — CheapTrick's envelope smoothing costs
more spectral detail than PSOLA does at low F0 on a stationary signal. WORLD wins the
same comparison on every other signal and profile, and wins on HNR everywhere
(−0.1 dB vs −1.4 to −6.8 dB), but the exception is real and worth listening to before
the ceiling is raised for low-pitched voices specifically.

---

## What this does not measure

The repository contains no speech recordings, so the benchmark synthesises its own
through an explicit source-filter model. That buys known ground truth — the formant
frequencies are set, not estimated — at the cost of realism:

- **No perceptual claim whatsoever.** Nothing here says WORLD *sounds* better. HNR
  and log-spectral distance are proxies. The 1.15–1.25 ceiling is justified by the
  acoustics tracking the request without measurable damage, not by anyone listening.
- **Synthetic excitation.** No creak, no jitter beyond a token amount, no voicing
  irregularity, no room. These are the conditions vocoders find easiest.
- **The existing artifact audit scored 100 on every single run**, both engines, every
  parameter setting. It cannot separate the two paths on this material, which is why
  the identity and round-trip tests were added. That is also a finding about the
  audit: it is tuned for catching gross failures, not for grading quality.
- **A silent skip is not a pass.** The first CI run on this branch went green with
  `12 passed, 21 skipped` — pyworld had built but could not import, and the tests
  skipped themselves rather than failing. CI now asserts the backend imports before
  running the suite, so this cannot read as green again.
- **The measurement had to be built and validated first.** Praat's Burg formant
  tracker was tried and rejected — it inserts spurious poles and renumbers formants
  on these signals. Section 0 of the results is the replacement estimator measured
  against ground truth, and every other number should be read against that ±2–4%
  noise floor.

## Recommendation

Keep it behind the flag. It has earned a real listening test on recorded speech —
that is the next step, and it is the one that decides this, not another benchmark
round. Before it could become the default:

1. Listening comparison on real recordings, including a low-pitched voice (§6).
2. Fix the analysis cost by caching the WORLD decomposition across backoff passes (§5).
3. A better independence measurement, or drop the claim (§4).
4. ~~Decide whether `pyworld` can be built in the deployment environment~~ —
   **settled, and it is the strongest argument against adopting pyworld as it
   stands.** It builds from sdist on a clean Ubuntu runner with Python 3.12 in about
   17 s, so the toolchain is not the problem. The import is: pyworld 0.3.5 calls
   `pkg_resources.get_distribution()` at import time, and `pkg_resources` is gone
   twice over — Python 3.12 no longer ships setuptools at all, and setuptools itself
   removed `pkg_resources` in 82.0.0 (present through 81.0.0). Both times pyworld
   compiles and installs cleanly and is then unimportable, which fails far more
   confusingly than a build error.

   The `world` extra therefore carries `setuptools>=68,<82` as a *runtime*
   dependency. That pin holds an unrelated build tool back for the sake of one
   import line in an unmaintained package, and it will age badly. Before adopting
   WORLD permanently, prefer a maintained fork or vendoring the two WORLD entry
   points over carrying this pin.

## Using it

```jsonc
// POST /api/transform
{
  "engine": "world",              // default "praat"
  "mode": "natural",              // or "explore" for wider limits
  "resonance_low_scale": 1.12,    // F1 — optional, defaults to resonance_scale
  "resonance_high_scale": 1.02,   // F2/F3 — optional, defaults to resonance_scale
  "breathiness": 0.3              // -1..1, WORLD only
}
```

`GET /api` reports which engines are available. An unknown engine, or `world` when
`pyworld` is missing, falls back to Praat and says so in the response metadata
(`engine_requested` vs `engine`).
