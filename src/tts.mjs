// Paradee TTS on onnxruntime-node, matching the paradee-web pipeline: sentence split, phonemize,
// vocabulary ids, ONNX. synthesizeStream yields each sentence's PCM as soon as it is ready.
import * as ort from 'onnxruntime-node';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { phonemize } from './phonemize.mjs';

export const SAMPLE_RATE = 24000;
const MAX_PHONEMES = 510;

// Same sentence boundaries as paradee-web: after . ! ? or … followed by whitespace, and at newlines.
export function splitSentences(text) {
  return text.trim().split(/(?<=[.!?…])\s+|\n+/).map((s) => s.trim()).filter(Boolean);
}

export async function loadParadee(modelDir) {
  const session = await ort.InferenceSession.create(path.join(modelDir, 'paradee_int8.onnx'), {
    executionProviders: ['cpu'],
    intraOpNumThreads: Number(process.env.PARADEE_THREADS ?? 4),
    interOpNumThreads: 1,
  });
  const cfg = JSON.parse(readFileSync(path.join(modelDir, 'config.json'), 'utf8'));
  return { session, vocab: new Map(Object.entries(cfg.vocab)) };
}

// Phoneme characters -> ids, unknown characters skipped, wrapped in the pad token (0).
function tokenIds(vocab, phonemes) {
  const ids = [0];
  for (const ch of phonemes) {
    if (ids.length > MAX_PHONEMES) break;
    const id = vocab.get(ch);
    if (id !== undefined) ids.push(id);
  }
  ids.push(0);
  return ids;
}

async function runSentence(model, phonemes, speed) {
  const ids = tokenIds(model.vocab, phonemes);
  const out = await model.session.run({
    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    speed: new ort.Tensor('float32', Float32Array.from([speed]), [1]),
  });
  return out[model.session.outputNames[0]].data;
}

function toPcm16(float) {
  const pcm = new Int16Array(float.length);
  for (let i = 0; i < float.length; i++) pcm[i] = Math.round(Math.max(-1, Math.min(1, float[i])) * 32767);
  return pcm;
}

// Yields one Int16Array (24 kHz) per sentence, so playback can start before the text is finished.
export async function* synthesizeStream(model, text, speed = 1.0) {
  for (const sentence of splitSentences(text)) {
    const phonemes = await phonemize(sentence);
    if (!phonemes) continue;
    yield toPcm16(await runSentence(model, phonemes, speed));
  }
}

export async function synthesize(model, text, speed = 1.0) {
  const parts = [];
  for await (const pcm of synthesizeStream(model, text, speed)) parts.push(pcm);
  const out = new Int16Array(parts.reduce((n, p) => n + p.length, 0));
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}
