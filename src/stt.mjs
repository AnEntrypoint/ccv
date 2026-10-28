// Parakeet-TDT (NeMo) speech-to-text on onnxruntime-node: 16 kHz mono PCM -> log-mel (128 bins)
// -> FastConformer encoder -> TDT greedy decoder. Shapes match the Underdog Bark Clean ONNX export.
import * as ort from 'onnxruntime-node';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const SR = 16000;
const N_FFT = 512;
const WIN = 400;
const HOP = 160;
const N_MELS = 128;
const PREEMPH = 0.97;
const LOG_GUARD = 2 ** -24;
const DUR_BINS = 5;

// Slaney mel scale: linear below 1 kHz, logarithmic above.
const hz2mel = (f) => (f >= 1000 ? 15 + (27 * Math.log(f / 1000)) / Math.log(6.4) : (3 * f) / 200);
const mel2hz = (m) => (m >= 15 ? 1000 * Math.exp((Math.log(6.4) / 27) * (m - 15)) : (200 * m) / 3);

// Slaney-scaled mel filterbank, area-normalised, as NeMo's preprocessor builds it.
function melFilterbank() {
  const nBins = N_FFT / 2 + 1;
  const fftFreqs = Array.from({ length: nBins }, (_, i) => (i * SR) / N_FFT);
  const mMin = hz2mel(0);
  const mMax = hz2mel(SR / 2);
  const pts = Array.from({ length: N_MELS + 2 }, (_, i) => mel2hz(mMin + ((mMax - mMin) * i) / (N_MELS + 1)));
  const fb = new Float32Array(N_MELS * nBins);
  for (let m = 0; m < N_MELS; m++) {
    const lo = pts[m], mid = pts[m + 1], hi = pts[m + 2];
    const enorm = 2 / (hi - lo);
    for (let k = 0; k < nBins; k++) {
      const f = fftFreqs[k];
      const down = (f - lo) / (mid - lo);
      const up = (hi - f) / (hi - mid);
      fb[m * nBins + k] = Math.max(0, Math.min(down, up)) * enorm;
    }
  }
  return fb;
}

const HANN = Float64Array.from({ length: WIN }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / WIN));
const MEL_FB = melFilterbank();

// In-place iterative radix-2 FFT on separate real/imag arrays.
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < len / 2; j++) {
        const a = i + j, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

// Log-mel features shaped [N_MELS, frames], per-feature normalised over time.
export function logMel(pcm) {
  const x = new Float64Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) x[i] = pcm[i] - (i ? PREEMPH * pcm[i - 1] : 0);
  const pad = N_FFT / 2;
  const padded = new Float64Array(x.length + 2 * pad);
  for (let i = 0; i < pad; i++) padded[pad - 1 - i] = x[Math.min(i + 1, x.length - 1)];
  padded.set(x, pad);
  for (let i = 0; i < pad; i++) padded[pad + x.length + i] = x[Math.max(x.length - 2 - i, 0)];
  const frames = Math.floor((padded.length - N_FFT) / HOP) + 1;
  const nBins = N_FFT / 2 + 1;
  const mel = new Float32Array(N_MELS * frames);
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
  for (let f = 0; f < frames; f++) {
    re.fill(0); im.fill(0);
    const off = f * HOP + (N_FFT - WIN) / 2;
    for (let n = 0; n < WIN; n++) re[n + (N_FFT - WIN) / 2] = padded[off + n] * HANN[n];
    fft(re, im);
    const power = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      for (let k = 0; k < nBins; k++) s += MEL_FB[m * nBins + k] * power[k];
      mel[m * frames + f] = Math.log(s + LOG_GUARD);
    }
  }
  for (let m = 0; m < N_MELS; m++) {
    let mean = 0;
    for (let f = 0; f < frames; f++) mean += mel[m * frames + f];
    mean /= frames;
    let v = 0;
    for (let f = 0; f < frames; f++) v += (mel[m * frames + f] - mean) ** 2;
    const std = Math.sqrt(v / Math.max(frames - 1, 1)) + 1e-5;
    for (let f = 0; f < frames; f++) mel[m * frames + f] = (mel[m * frames + f] - mean) / std;
  }
  return { mel, frames };
}

export async function loadStt(modelDir) {
  const opts = { executionProviders: ['cpu'], graphOptimizationLevel: 'all' };
  const [encoder, decoder, joiner] = await Promise.all([
    ort.InferenceSession.create(path.join(modelDir, 'encoder.int8.onnx'), opts),
    ort.InferenceSession.create(path.join(modelDir, 'decoder.int8.onnx'), opts),
    ort.InferenceSession.create(path.join(modelDir, 'joiner.int8.onnx'), opts),
  ]);
  const vocab = readFileSync(path.join(modelDir, 'tokens.txt'), 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((line) => line.slice(0, line.lastIndexOf(' ')));
  const blank = vocab.indexOf('<blk>');
  return { encoder, decoder, joiner, vocab, blank };
}

async function decodeTdt(model, enc, encFrames) {
  const { decoder, joiner, vocab, blank } = model;
  const H = 640;
  const tokens = [];
  let state1 = new Float32Array(2 * H);
  let state2 = new Float32Array(2 * H);
  const runDecoder = async (token, s1, s2) => {
    const out = await decoder.run({
      targets: new ort.Tensor('int32', Int32Array.from([token]), [1, 1]),
      target_length: new ort.Tensor('int32', Int32Array.from([1]), [1]),
      'states.1': new ort.Tensor('float32', s1, [2, 1, H]),
      'states.2': new ort.Tensor('float32', s2, [2, 1, H]),
    });
    return { dec: out.outputs.data.slice(0, H), s1: out.states.data, s2: out['states.2_out'].data };
  };
  let cur = await runDecoder(blank, state1, state2);
  let t = 0;
  let emittedThisFrame = 0;
  while (t < encFrames) {
    const frame = new Float32Array(1024);
    for (let c = 0; c < 1024; c++) frame[c] = enc[c * encFrames + t];
    const out = await joiner.run({
      encoder_outputs: new ort.Tensor('float32', frame, [1, 1024, 1]),
      decoder_outputs: new ort.Tensor('float32', cur.dec, [1, H, 1]),
    });
    const logits = out.outputs.data;
    const nTok = vocab.length;
    let best = 0;
    for (let i = 1; i < nTok; i++) if (logits[i] > logits[best]) best = i;
    let dur = 0;
    for (let d = 1; d < DUR_BINS; d++) if (logits[nTok + d] > logits[nTok + dur]) dur = d;
    if (best !== blank && emittedThisFrame < 10) {
      tokens.push(best);
      cur = await runDecoder(best, cur.s1, cur.s2);
      emittedThisFrame++;
    }
    const step = dur > 0 ? dur : best === blank ? 1 : 0;
    if (step > 0) { t += step; emittedThisFrame = 0; }
    else if (emittedThisFrame >= 10) { t += 1; emittedThisFrame = 0; }
  }
  return tokens.map((id) => vocab[id]).join('').replace(/▁/g, ' ').trim();
}

export async function transcribePcm16k(model, pcm) {
  const { mel, frames } = logMel(pcm);
  const T = frames;
  const encIn = await model.encoder.run({
    audio_signal: new ort.Tensor('float32', mel, [1, N_MELS, T]),
    length: new ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1]),
  });
  const encFrames = Number(encIn.encoded_lengths.data[0]);
  return decodeTdt(model, encIn.outputs.data, encFrames);
}

