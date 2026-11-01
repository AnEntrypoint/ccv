// Acoustic features of one utterance from 16 kHz mono Int16 PCM: loudness, pitch, pitch variation and how much of it is voiced.
// No model and no network: plain arithmetic, so it adds little to the reply path.
const SR = 16000;
const RATE = 8000; // analysis rate: pitch sits well below 4 kHz, so half the samples are enough
const WIN = 240; // 30 ms at RATE
const HOP = 80; // 10 ms step at RATE
const LAG_MIN = Math.floor(RATE / 350); // highest pitch searched: 350 Hz
const LAG_MAX = Math.ceil(RATE / 70); // lowest pitch searched: 70 Hz
const SPEECH_BELOW_PEAK_DB = 25; // frames this far under the loudest frame are not speech
const SILENCE_DB = -60; // absolute floor under which a frame is silence
const PITCH_CLARITY = 0.6; // normalized autocorrelation a frame needs to count as voiced
const OCTAVE_SAFE = 0.9; // a peak must score this share of the best, so octave-down errors do not win

const round = (x, digits = 0) => (x === null ? null : Number(x.toFixed(digits)));
const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

// Halves the rate by averaging sample pairs; enough for pitch, and it keeps the analysis cheap.
function toAnalysisRate(pcm) {
  const out = new Float64Array(Math.floor(pcm.length / 2));
  for (let i = 0; i < out.length; i++) out[i] = (pcm[2 * i] + pcm[2 * i + 1]) / 2;
  return out;
}

function frameDb(x, off) {
  let e = 0;
  for (let i = 0; i < WIN; i++) e += x[off + i] * x[off + i];
  return 10 * Math.log10(e / WIN / (32768 * 32768) + 1e-12);
}

// Pitch of one frame: the first local autocorrelation peak that is close to the best one, refined between lags.
function pitchOf(x, off) {
  const corr = new Float64Array(LAG_MAX + 2);
  let best = 0;
  for (let lag = LAG_MIN - 1; lag <= LAG_MAX + 1; lag++) {
    let num = 0, e0 = 0, e1 = 0;
    for (let i = 0; i + lag < WIN; i++) {
      const a = x[off + i], b = x[off + i + lag];
      num += a * b;
      e0 += a * a;
      e1 += b * b;
    }
    corr[lag] = num / Math.sqrt(e0 * e1 + 1e-9);
    if (lag >= LAG_MIN && lag <= LAG_MAX && corr[lag] > best) best = corr[lag];
  }
  if (best < PITCH_CLARITY) return null;
  for (let lag = LAG_MIN; lag <= LAG_MAX; lag++) {
    const peak = corr[lag] >= OCTAVE_SAFE * best && corr[lag] >= corr[lag - 1] && corr[lag] >= corr[lag + 1];
    if (!peak) continue;
    const curve = corr[lag - 1] - 2 * corr[lag] + corr[lag + 1];
    const offset = curve < 0 ? (0.5 * (corr[lag - 1] - corr[lag + 1])) / curve : 0;
    return RATE / (lag + offset);
  }
  return null;
}

export function prosody(pcm) {
  const durationMs = Math.round((pcm.length * 1000) / SR);
  const x = toAnalysisRate(pcm);
  const frames = [];
  for (let off = 0; off + WIN <= x.length; off += HOP) frames.push({ off, db: frameDb(x, off) });
  const empty = { durationMs, speechMs: null, levelDb: null, peakDb: null, f0Hz: null, f0SpreadSt: null, voicedShare: 0 };
  if (!frames.length) return empty;
  const peakDb = Math.max(...frames.map((f) => f.db));
  if (peakDb < SILENCE_DB) return { ...empty, peakDb: round(peakDb, 1) };
  const speech = frames.filter((f) => f.db >= Math.max(peakDb - SPEECH_BELOW_PEAK_DB, SILENCE_DB));
  const hz = [];
  for (const f of speech) {
    const p = pitchOf(x, f.off);
    if (p !== null) hz.push(p);
  }
  const semitones = hz.map((h) => 12 * Math.log2(h));
  const spread = semitones.length >= 3 ? Math.sqrt(mean(semitones.map((s) => (s - mean(semitones)) ** 2))) : null;
  // Speech span: first to last speech frame, so trailing silence the recorder adds is not counted in the cadence.
  const speechMs = Math.round(((speech.at(-1).off + WIN - speech[0].off) * 1000) / RATE);
  return {
    durationMs,
    speechMs,
    levelDb: round(mean(speech.map((f) => f.db)), 1),
    peakDb: round(peakDb, 1),
    f0Hz: hz.length ? round(median(hz)) : null,
    f0SpreadSt: round(spread, 1),
    voicedShare: round(hz.length / frames.length, 2),
  };
}

// Tone words for channel metadata: how this utterance sits against the same speaker's recent utterances (past).
// Until a speaker has history, the typical speech profile stands in for it, so early utterances already read.
// Each axis is judged on its own: a utterance with no clear pitch can still be loud. Nothing standing out is neutral.
const LOUD_DB = 4;
const PITCH_RATIO = 1.15;
const SPREAD_ST = 1.5;
const RATE_RATIO = 1.25;
const TYPICAL_SPEECH = { levelDb: -26.4, f0Hz: 122, f0SpreadSt: 2.5, wordsPerSec: 3.0 }; // medians of recorded Discord utterances
const SEED_COUNT = 3;
export function toneWords(p, past) {
  const history = [...Array(SEED_COUNT).fill(TYPICAL_SPEECH), ...past];
  const words = [];
  if (p.levelDb !== null) {
    const level = p.levelDb - median(history.filter((q) => q.levelDb !== null).map((q) => q.levelDb));
    if (level >= LOUD_DB) words.push('loud');
    else if (level <= -LOUD_DB) words.push('quiet');
  }
  if (p.f0Hz !== null) {
    const pitch = p.f0Hz / median(history.filter((q) => q.f0Hz !== null).map((q) => q.f0Hz));
    if (pitch >= PITCH_RATIO) words.push('high');
    else if (pitch <= 1 / PITCH_RATIO) words.push('low');
  }
  if (p.f0SpreadSt !== null) {
    const spread = p.f0SpreadSt - median(history.filter((q) => q.f0SpreadSt !== null).map((q) => q.f0SpreadSt));
    if (spread >= SPREAD_ST) words.push('animated');
    else if (spread <= -SPREAD_ST) words.push('flat');
  }
  if (p.wordsPerSec) {
    const rate = p.wordsPerSec / median(history.filter((q) => q.wordsPerSec).map((q) => q.wordsPerSec));
    if (rate >= RATE_RATIO) words.push('fast');
    else if (rate <= 1 / RATE_RATIO) words.push('slow');
  }
  return words.length ? words.join(' ') : 'neutral';
}
