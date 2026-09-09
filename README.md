# Voice Target Lab Mobile

A phone-first version of Voice Target Lab for recording a short baseline utterance, manipulating personally selected acoustic parameters, and A/B listening to the transformed version.

## Current controls

- Pitch median offset in semitones
- Formant/resonance scale
- Pitch-range / intonation-range scale
- Experimental brightness / spectral-tilt adjustment
- Breathiness / aperiodicity (WORLD engine only, experimental)
- Separate F1 and F2/F3 resonance scales (WORLD engine only, experimental)

The app does **not** calculate a gender, femininity, masculinity, attractiveness, or passing score.

## Architecture

- Static HTML/CSS/JavaScript frontend designed for Android Chrome
- Browser microphone capture via `MediaRecorder`
- Browser converts recordings to mono 16-bit WAV
- `/api/transform` is a Vercel Python/FastAPI function
- Acoustic resynthesis uses `praat-parselmouth` and Praat's source-filter/PSOLA transformation
- An experimental WORLD-vocoder engine is available behind `"engine": "world"`; see
  [`docs/WORLD_SPIKE.md`](docs/WORLD_SPIKE.md) for what it does and what was measured
- No database and no application-level server storage

Because processing happens in a Vercel Function, audio is transmitted to the deployment when **Generate modified voice** is pressed. The application code does not persist it server-side.

## Vercel

This repository is laid out for Vercel:

- `index.html`, `app.js`, `styles.css` — frontend
- `api/index.py` — FastAPI Python function
- `pyproject.toml` — Python 3.12 dependencies (`.[world]` adds the optional WORLD engine)
- `bench/` — backend comparison harness, not part of the deployed function
- `vercel.json` — security headers

Import the GitHub repository into Vercel and deploy. HTTPS is required for browser microphone access and Vercel provides it automatically.

## Android

Open the Vercel URL in Chrome, allow microphone access, then use Chrome's **Add to Home screen / Install app** option if desired.

## Limits

- Maximum recording length: 15 seconds
- Best results: clear voiced speech, limited background noise
- Brightness remains experimental
- The WORLD engine is a spike: it is not the default, it has had no listening test,
  and it runs about 10x slower than the Praat path
- This is a personal/research prototype, not a validated clinical device
