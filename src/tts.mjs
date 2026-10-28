// Paradee TTS on onnxruntime-node: text -> misaki lexicon phonemes -> ONNX -> 24 kHz PCM16.
// Same algorithm as the C++ server it replaces: word lookup (espeak-ng binary for out-of-vocabulary
// words when ESPEAK_NG is set), longest-match phoneme ids, 0.2 s between sentences.
import * as ort from 'onnxruntime-node';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

export const SAMPLE_RATE = 24000;
const MAX_PHONEMES = 510;
const PUNCT = new Set([',', '.', '!', '?', ';', ':', '—', '…']);
const ONES = ['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen'];
const TENS = ['', '', 'twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'];

// Cardinal words for non-negative integers below one billion, e.g. 2015 -> two thousand fifteen.
export function numberToWords(n) {
  if (n < 20) return ONES[n];
  if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? ' ' + ONES[n % 10] : '');
  if (n < 1000) return ONES[Math.floor(n / 100)] + ' hundred' + (n % 100 ? ' ' + numberToWords(n % 100) : '');
  for (const [unit, name] of [[1e9, 'billion'], [1e6, 'million'], [1e3, 'thousand']]) {
    if (n >= unit) return numberToWords(Math.floor(n / unit)) + ' ' + name + (n % unit ? ' ' + numberToWords(n % unit) : '');
  }
  return String(n);
}

function loadLexicon(files) {
  const words = new Map();
  for (const file of files) {
    for (const [word, value] of Object.entries(JSON.parse(readFileSync(file, 'utf8')))) {
      const ps = typeof value === 'string' ? value : value?.DEFAULT;
      if (typeof ps === 'string' && !words.has(word)) words.set(word, ps);
    }
  }
  return words;
}

function lookup(words, word) {
  return words.get(word) ?? words.get(word.toLowerCase());
}

const espeakCache = new Map();
// Out-of-vocabulary words go to espeak-ng when ESPEAK_NG names the binary; otherwise they are dropped.
function espeakWord(word) {
  if (espeakCache.has(word)) return espeakCache.get(word);
  let out = '';
  const bin = process.env.ESPEAK_NG;
  if (bin) {
    try {
      out = execFileSync(bin, ['-q', '-v', 'en-us', '--ipa=3', word], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
    } catch {
      out = '';
    }
    out = out.replace(/‍/g, '').replace(/\s+/g, '');
  }
  espeakCache.set(word, out);
  return out;
}

// Words go through the lexicon (espeak for unknowns). Digit runs are spelled out first, and
// punctuation is kept as its own token so the model gets the same pauses and intonation cues.
function phonemizeSentence(words, sentence) {
  const expanded = sentence.replace(/\d+/g, (d) => (d.length > 12 ? d.split('').join(' ') : numberToWords(Number(d))));
  const out = [];
  const pushWord = (word) => {
    let ps = lookup(words, word);
    if (ps === undefined) {
      const letters = word.replace(/'/g, '');
      if (letters) ps = espeakWord(letters);
    }
    if (ps) out.push(ps);
  };
  let word = '';
  for (const ch of expanded) {
    if (/[A-Za-z']/.test(ch)) { word += ch; continue; }
    if (word) { pushWord(word); word = ''; }
    if (PUNCT.has(ch)) {
      if (out.length) out[out.length - 1] += ch;
      else out.push(ch);
    }
  }
  if (word) pushWord(word);
  return out.join(' ');
}

// Splits on newlines, and after . ! ? followed by whitespace.
function splitSentences(text) {
  const out = [];
  let cur = '';
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '\n') { if (cur) out.push(cur); cur = ''; continue; }
    cur += c;
    if ((c === '.' || c === '!' || c === '?') && i + 1 < text.length && /\s/.test(text[i + 1])) {
      out.push(cur);
      cur = '';
    }
  }
  if (cur) out.push(cur);
  return out;
}

export async function loadParadee(modelDir, misakiDir) {
  const session = await ort.InferenceSession.create(path.join(modelDir, 'paradee_int8.onnx'), {
    executionProviders: ['cpu'],
    intraOpNumThreads: Number(process.env.PARADEE_THREADS ?? 4),
    interOpNumThreads: 1,
  });
  const cfg = JSON.parse(readFileSync(path.join(modelDir, 'config.json'), 'utf8'));
  const vocab = new Map(Object.entries(cfg.vocab));
  const words = loadLexicon([path.join(misakiDir, 'us_gold.json'), path.join(misakiDir, 'us_silver.json')]);
  return { session, vocab, words };
}

// Phoneme string -> ids. Matches the longest vocab key (up to 4 UTF-16 units) at each position.
function phonemeIds(vocab, phonemes) {
  const ids = [0];
  for (let i = 0; i < phonemes.length && ids.length < MAX_PHONEMES + 1; ) {
    let matched = false;
    for (let len = Math.min(4, phonemes.length - i); len >= 1 && !matched; len--) {
      const id = vocab.get(phonemes.substr(i, len));
      if (id !== undefined) { ids.push(id); i += len; matched = true; }
    }
    if (!matched) i++;
  }
  ids.push(0);
  return ids;
}

async function runPhonemes(model, phonemes, speed) {
  const ids = phonemeIds(model.vocab, phonemes);
  const out = await model.session.run({
    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    speed: new ort.Tensor('float32', Float32Array.from([speed]), [1]),
  });
  return out[model.session.outputNames[0]].data;
}

// Returns Int16Array PCM at 24 kHz for the whole text.
export async function synthesize(model, text, speed = 1.0) {
  const chunks = [];
  let total = 0;
  for (const sentence of splitSentences(text)) {
    const ps = phonemizeSentence(model.words, sentence);
    if (!ps) continue;
    const float = await runPhonemes(model, ps, speed);
    const pcm = new Int16Array(float.length);
    for (let i = 0; i < float.length; i++) {
      const c = Math.max(-1, Math.min(1, float[i]));
      pcm[i] = Math.round(c * 32767);
    }
    chunks.push(pcm);
    total += pcm.length;
  }
  const out = new Int16Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}
