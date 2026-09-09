const $ = id => document.getElementById(id);
const els = {
  recordBtn: $('recordBtn'), fileInput: $('fileInput'), recordStatus: $('recordStatus'),
  baselineAudio: $('baselineAudio'), modifiedAudio: $('modifiedAudio'),
  generateBtn: $('generateBtn'), generateStatus: $('generateStatus'),
  playOriginal: $('playOriginal'), playModified: $('playModified'), metrics: $('metrics'),
  pitch: $('pitch'), resonance: $('resonance'), range: $('range'), brightness: $('brightness'),
  pitchOut: $('pitchOut'), resonanceOut: $('resonanceOut'), rangeOut: $('rangeOut'), brightnessOut: $('brightnessOut')
};

let recorder = null;
let stream = null;
let chunks = [];
let baselineWav = null;
let baselineUrl = null;
let modifiedUrl = null;
let autoStopTimer = null;

const saved = JSON.parse(localStorage.getItem('voiceTargetSettings') || '{}');
for (const key of ['pitch','resonance','range','brightness']) if (saved[key] != null) els[key].value = saved[key];

function fmtSigned(n, digits=1) { const v = Number(n); return `${v > 0 ? '+' : ''}${v.toFixed(digits)}`; }
function refreshLabels() {
  els.pitchOut.textContent = `${fmtSigned(els.pitch.value)} ST`;
  els.resonanceOut.textContent = `${Number(els.resonance.value).toFixed(3)}×`;
  els.rangeOut.textContent = `${Number(els.range.value).toFixed(2)}×`;
  els.brightnessOut.textContent = `${fmtSigned(els.brightness.value)} dB`;
  localStorage.setItem('voiceTargetSettings', JSON.stringify({ pitch: els.pitch.value, resonance: els.resonance.value, range: els.range.value, brightness: els.brightness.value }));
}
for (const el of [els.pitch, els.resonance, els.range, els.brightness]) el.addEventListener('input', refreshLabels);
refreshLabels();

function base64FromArrayBuffer(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  return btoa(binary);
}

function blobFromBase64(base64, type='audio/wav') {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type });
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  write(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); write(8, 'WAVE');
  write(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, 'data'); view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

async function audioBlobToWav(blob) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const audioBuffer = await ctx.decodeAudioData(await blob.arrayBuffer());
    const frames = audioBuffer.length;
    const mono = new Float32Array(frames);
    for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
      const data = audioBuffer.getChannelData(ch);
      for (let i = 0; i < frames; i++) mono[i] += data[i] / audioBuffer.numberOfChannels;
    }
    if (audioBuffer.duration > 15.25) throw new Error('Recording must be 15 seconds or shorter.');
    return encodeWav(mono, audioBuffer.sampleRate);
  } finally { await ctx.close(); }
}

function setBaseline(wavBlob, label='Baseline ready.') {
  baselineWav = wavBlob;
  if (baselineUrl) URL.revokeObjectURL(baselineUrl);
  baselineUrl = URL.createObjectURL(wavBlob);
  els.baselineAudio.src = baselineUrl;
  els.generateBtn.disabled = false;
  els.playOriginal.disabled = false;
  els.recordStatus.textContent = label;
  els.generateStatus.textContent = 'Adjust the sliders, then generate a modified version.';
  clearModified();
}

function clearModified() {
  if (modifiedUrl) URL.revokeObjectURL(modifiedUrl);
  modifiedUrl = null;
  els.modifiedAudio.removeAttribute('src'); els.modifiedAudio.load();
  els.playModified.disabled = true;
  els.metrics.classList.add('hidden');
}

els.recordBtn.addEventListener('click', async () => {
  if (recorder?.state === 'recording') { recorder.stop(); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const preferred = ['audio/webm;codecs=opus','audio/webm','audio/mp4'];
    const mimeType = preferred.find(t => MediaRecorder.isTypeSupported(t));
    recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = async () => {
      clearTimeout(autoStopTimer);
      els.recordBtn.textContent = '● Record'; els.recordBtn.classList.remove('recording');
      stream?.getTracks().forEach(t => t.stop());
      els.recordStatus.textContent = 'Converting recording…';
      try { setBaseline(await audioBlobToWav(new Blob(chunks, { type: recorder.mimeType })), 'Baseline recorded.'); }
      catch (err) { els.recordStatus.textContent = err.message || 'Could not decode this recording.'; }
    };
    recorder.start();
    els.recordBtn.textContent = '■ Stop'; els.recordBtn.classList.add('recording');
    els.recordStatus.textContent = 'Recording… speak naturally. Auto-stops at 12 seconds.';
    autoStopTimer = setTimeout(() => recorder?.state === 'recording' && recorder.stop(), 12000);
  } catch (err) {
    els.recordStatus.textContent = 'Microphone access failed. Allow microphone permission in Chrome and try again.';
  }
});

els.fileInput.addEventListener('change', async e => {
  const file = e.target.files?.[0]; if (!file) return;
  els.recordStatus.textContent = 'Importing audio…';
  try { setBaseline(await audioBlobToWav(file), `Imported ${file.name}.`); }
  catch (err) { els.recordStatus.textContent = err.message || 'Could not read that audio file.'; }
  e.target.value = '';
});

els.generateBtn.addEventListener('click', async () => {
  if (!baselineWav) return;
  els.generateBtn.disabled = true; els.generateStatus.textContent = 'Generating with Praat…'; clearModified();
  try {
    const payload = {
      wav_base64: base64FromArrayBuffer(await baselineWav.arrayBuffer()),
      pitch_semitones: Number(els.pitch.value),
      resonance_scale: Number(els.resonance.value),
      pitch_range_scale: Number(els.range.value),
      brightness_db: Number(els.brightness.value)
    };
    const response = await fetch('/api/transform', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Transformation failed.');
    const blob = blobFromBase64(data.wav_base64);
    modifiedUrl = URL.createObjectURL(blob);
    els.modifiedAudio.src = modifiedUrl; els.playModified.disabled = false;
    const m = data.metrics || {};
    els.metrics.innerHTML = `
      <div class="metric"><strong>${m.baseline_pitch_median_hz ?? '—'} Hz</strong><small>baseline median F0</small></div>
      <div class="metric"><strong>${m.output_pitch_median_hz ?? '—'} Hz</strong><small>modified median F0</small></div>
      <div class="metric"><strong>${m.observed_pitch_shift_semitones != null ? fmtSigned(m.observed_pitch_shift_semitones, 2) + ' ST' : '—'}</strong><small>observed pitch shift</small></div>
      <div class="metric"><strong>${m.backend || 'Praat'}</strong><small>processing backend</small></div>`;
    els.metrics.classList.remove('hidden');
    els.generateStatus.textContent = 'Modified voice ready. A/B it below, then change the sliders and regenerate.';
  } catch (err) {
    els.generateStatus.textContent = err.message || 'Could not generate the modified voice.';
  } finally { els.generateBtn.disabled = !baselineWav; }
});

function play(audio) { audio.currentTime = 0; audio.play().catch(() => {}); }
els.playOriginal.addEventListener('click', () => play(els.baselineAudio));
els.playModified.addEventListener('click', () => play(els.modifiedAudio));

document.querySelectorAll('[data-preset]').forEach(btn => btn.addEventListener('click', () => {
  const p = btn.dataset.preset;
  if (p === 'pitch') Object.assign(els.pitch, {value: 3}), Object.assign(els.resonance, {value: 1}), Object.assign(els.range, {value: 1}), Object.assign(els.brightness, {value: 0});
  if (p === 'resonance') Object.assign(els.pitch, {value: 0}), Object.assign(els.resonance, {value: 1.06}), Object.assign(els.range, {value: 1}), Object.assign(els.brightness, {value: 0});
  if (p === 'combined') Object.assign(els.pitch, {value: 2.5}), Object.assign(els.resonance, {value: 1.055}), Object.assign(els.range, {value: 1.15}), Object.assign(els.brightness, {value: .5});
  if (p === 'reset') Object.assign(els.pitch, {value: 0}), Object.assign(els.resonance, {value: 1}), Object.assign(els.range, {value: 1}), Object.assign(els.brightness, {value: 0});
  refreshLabels();
}));

if ('serviceWorker' in navigator) window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js').catch(() => {}));
