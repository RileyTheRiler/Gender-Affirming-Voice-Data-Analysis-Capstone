# Voice Target Lab Mobile — Critic Review

## Score: 9.1 / 10

This is a release-engineering score, not a clinical-validation score.

### Strong points

- Mobile workflow is much clearer than the desktop research UI: baseline → parameters → generate → A/B.
- Reuses Praat/Parselmouth for pitch median, formant/resonance scaling, and pitch-range scaling rather than substituting a weaker browser-only approximation.
- Explicitly separates pitch, resonance, intonation range, and experimental brightness.
- Does not calculate a gender or passing score.
- Browser normalizes imported/recorded audio to mono 16-bit WAV before upload.
- API clips parameter ranges and rejects overlong or malformed audio.
- Synthetic round-trip test produced the requested +2.00 semitone pitch shift.
- PWA manifest and Android-size icons are present.
- Microphone use is limited to the deployment origin by `Permissions-Policy`.
- GitHub Actions provides a Linux dependency/transform test path.

### Remaining weaknesses

1. **Production Vercel deployment is not yet verified.** The code is structured correctly, but the actual Vercel build/function has not been observed running.
2. **No physical Android-device test yet.** MediaRecorder/AudioContext behavior can vary by Chrome version and handset audio routing.
3. **Audio is not fully local.** Pressing Generate sends the short WAV to the Vercel function. The app does not store it, but network transport is still involved.
4. **Brightness remains exploratory.** It is an FFT spectral-tilt adjustment, not a validated perceptual "vocal weight" control.
5. **No saved target-candidate history yet.** The mobile playground currently optimizes rapid A/B exploration rather than the desktop app's longitudinal research workflow.

### Recommended next iteration

- Verify Vercel production build and `/api/transform` on deployment.
- Test microphone recording and A/B playback on the user's Android device.
- Add local target candidate slots (A/B/C) with settings snapshots.
- Add an optional download/share button for the modified WAV.
- Later, consider an all-client/WASM transformation path if eliminating server-side audio transit becomes a priority.
