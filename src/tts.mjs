// Paradee TTS on onnxruntime-node, matching the paradee-web pipeline: sentence split, phonemize,
// vocabulary ids, ONNX. synthesizeStream yields each sentence's PCM as soon as it is ready.
import * as ort from 'onnxruntime-node';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { phonemize } from './phonemize.mjs';

export const SAMPLE_RATE = 24000;
const MAX_PHONEMES = 510;
const CHATTERBOX_URL = 'http://127.0.0.1:8765';

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

export async function loadTts(modelDir) {
  const engine = process.env.TTS || 'paradee';
  if (engine === 'chatterbox') return { chatterboxUrl: process.env.CHATTERBOX_URL || CHATTERBOX_URL };
  if (engine === 'paradee') return loadParadee(modelDir);
  throw new Error(`unknown TTS engine: ${engine}`);
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

async function* chatterboxBlocks(baseUrl, sentence) {
  const url = new URL('/tts', baseUrl);
  url.searchParams.set('text', sentence);
  url.searchParams.set('encoding', 'int16');
  const res = await fetch(url);
  if (!res.ok) throw new Error(`chatterbox ${res.status}`);
  const rate = res.headers.get('x-sample-rate');
  const encoding = res.headers.get('x-encoding');
  if (rate !== String(SAMPLE_RATE) || encoding !== 'int16le') {
    throw new Error(`unexpected chatterbox audio: ${encoding} at ${rate} Hz`);
  }
  let carry = Buffer.alloc(0);
  for await (const block of res.body) {
    const bytes = Buffer.concat([carry, block]);
    const whole = bytes.length - (bytes.length % 2);
    carry = bytes.subarray(whole);
    yield new Int16Array(new Uint8Array(bytes.subarray(0, whole)).buffer);
  }
}

// Yields one Int16Array (24 kHz) per sentence, so playback can start before the text is finished.
export async function* synthesizeStream(model, text, speed = 1.0) {
  if (model.chatterboxUrl) {
    for (const sentence of splitSentences(text)) yield* chatterboxBlocks(model.chatterboxUrl, sentence);
    return;
  }
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
