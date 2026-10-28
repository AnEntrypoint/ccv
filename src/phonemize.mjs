// Text -> Paradee phoneme string, ported from the paradee-web pipeline: text normalisation,
// punctuation-aware splitting, espeak-ng IPA (en-us) per text run, then IPA fixups.
import createEspeak from '@echogarden/espeak-ng-emscripten';

let espeakPromise = null;
function espeak() {
  espeakPromise ??= createEspeak().then((m) => {
    const worker = new m.eSpeakNGWorker();
    worker.set_voice('en-us');
    return worker;
  });
  return espeakPromise;
}

async function ipaLines(text) {
  const worker = await espeak();
  const ipa = worker.synthesize_ipa(text).ipa ?? '';
  return ipa.split('\n').filter((line) => line.length > 0);
}

const ONES_TENS = ['', 'oh', 'twenty'];

function yearOrTime(e) {
  if (e.includes('.')) return e;
  if (e.includes(':')) {
    const [h, m] = e.split(':').map(Number);
    return m === 0 ? `${h} o'clock` : m < 10 ? `${h} oh ${m}` : `${h} ${m}`;
  }
  const t = parseInt(e.slice(0, 4), 10);
  if (t < 1100 || t % 1000 < 10) return e;
  const n = e.slice(0, 2);
  const r = parseInt(e.slice(2, 4), 10);
  const i = e.endsWith('s') ? 's' : '';
  if (t % 1000 >= 100 && t % 1000 <= 999) {
    if (r === 0) return `${n} hundred${i}`;
    if (r < 10) return `${n} oh ${r}${i}`;
  }
  return `${n} ${r}${i}`;
}

function currency(e) {
  const word = e[0] === '$' ? 'dollar' : 'pound';
  if (isNaN(Number(e.slice(1)))) return `${e.slice(1)} ${word}s`;
  if (!e.includes('.')) {
    const s = e.slice(1) === '1' ? '' : 's';
    return `${e.slice(1)} ${word}${s}`;
  }
  const [whole, frac] = e.slice(1).split('.');
  const cents = parseInt(frac.padEnd(2, '0'), 10);
  const unit = e[0] === '$' ? (cents === 1 ? 'cent' : 'cents') : cents === 1 ? 'penny' : 'pence';
  return `${whole} ${word}${whole === '1' ? '' : 's'} and ${cents} ${unit}`;
}

function decimal(e) {
  const [whole, frac] = e.split('.');
  return `${whole} point ${frac.split('').join(' ')}`;
}

// Text normalisation (paradee-web lm): quotes, CJK punctuation, titles, years, currency, decimals.
export function normalize(input) {
  return input
    .replace(/[`']/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/\(/g, '«')
    .replace(/\)/g, '»')
    .replace(/、/g, ', ')
    .replace(/。/g, '. ')
    .replace(/！/g, '! ')
    .replace(/，/g, ', ')
    .replace(/：/g, ': ')
    .replace(/；/g, '; ')
    .replace(/？/g, '? ')
    .replace(/[^\S \n]/g, ' ')
    .replace(/  +/, ' ')
    .replace(/(?<=\n) +(?=\n)/g, '')
    .replace(/\bD[Rr]\.(?= [A-Z])/g, 'Doctor')
    .replace(/\b(?:Mr\.|MR\.(?= [A-Z]))/g, 'Mister')
    .replace(/\b(?:Ms\.|MS\.(?= [A-Z]))/g, 'Miss')
    .replace(/\b(?:Mrs\.|MRS\.(?= [A-Z]))/g, 'Mrs')
    .replace(/\betc\.(?! [A-Z])/gi, 'etc')
    .replace(/\b(y)eah?\b/gi, "$1e'a")
    .replace(/\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)/g, yearOrTime)
    .replace(/(?<=\d),(?=\d)/g, '')
    .replace(/[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b/gi, currency)
    .replace(/\d*\.\d+/g, decimal)
    .replace(/(?<=\d)-(?=\d)/g, ' to ')
    .replace(/(?<=\d)S/g, ' S')
    .replace(/(?<=[BCDFGHJ-NP-TV-Z])'?s\b/g, "'S")
    .replace(/(?<=X')S\b/g, 's')
    .replace(/(?:[A-Za-z]\.){2,} [a-z]/g, (e) => e.replace(/\./g, '-'))
    .replace(/(?<=[A-Z])\.(?=[A-Z])/gi, '-')
    .trim();
}

const PUNCT_CLASS = ';:,.!?—…"“”(){}[]«»'.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const PUNCT_RUN = new RegExp(`(\\s*[${PUNCT_CLASS}]+\\s*)+`, 'g');

// Splits into alternating text runs and punctuation runs, keeping both.
function splitPunctuation(text) {
  const parts = [];
  let last = 0;
  for (const m of text.matchAll(PUNCT_RUN)) {
    if (last < m.index) parts.push({ match: false, text: text.slice(last, m.index) });
    if (m[0].length > 0) parts.push({ match: true, text: m[0] });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ match: false, text: text.slice(last) });
  return parts;
}

// IPA substitutions applied to the joined phoneme string (paradee-web zd).
const IPA_PAIRS = [
  ['ʔˌn̩', 'ʔn'], ['ʔn̩', 'ʔn'], ['aɪ', 'I'], ['aʊ', 'W'],
  ['dʒ', 'ʤ'], ['eɪ', 'A'], ['e', 'A'], ['tʃ', 'ʧ'], ['ɔɪ', 'Y'],
  ['ʲo', 'jo'], ['ʲə', 'jə'], ['ʲ', ''], ['ɚ', 'əɹ'], ['r', 'ɹ'],
  ['x', 'k'], ['ç', 'k'], ['ɬ', 'l'], ['̃', ''],
];

function ipaSubstitute(e) {
  let s = e;
  for (const [from, to] of IPA_PAIRS) s = s.split(from).join(to);
  s = s.replace(/(\S)̩/gu, 'ᵊ$1').replace(/̩/g, '');
  s = s.replace(/əl(?=[\s;:,.!?—…"“”()]|$)/gu, 'ᵊl');
  s = s.replace(/ɐ(?![\s;:,.!?—…"“”()]|$)/gu, 'ə');
  s = s.split('oʊ').join('O').split('ɜːɹ').join('ɜɹ').split('ɜː').join('ɜɹ')
    .split('ɪə').join('iə').split('ː').join('');
  s = s.split('o').join('ɔ');
  return s.split('ɾ').join('T').split('ʔ').join('t');
}

// Fixups applied after the IPA is joined (paradee-web fm).
function phonemeFixups(e) {
  let t = e
    .replace(/kəsekˈoːɹoʊ/g, 'kˈoʊkəɹoʊ')
    .replace(/kəsekˈoːɹəʊ/g, 'kˈəʊkəɹəʊ')
    .replace(/ʝ/g, 'j').replace(/r/g, 'ɹ').replace(/x/g, 'k').replace(/ɬ/g, 'l')
    .replace(/(?<=[a-zɹ])(?=hˈʌndɹəd)/g, ' ')
    .replace(/ z(?=[;:,.!?—…"“” ]|$)/g, 'z');
  t = t.replace(/(?<=nˈaɪn)ti(?!ː)/g, 'di');
  return t.trim();
}

// Full text -> phonemes (paradee-web pm/mm + zd): what the model's vocabulary is mapped from.
export async function phonemize(text) {
  if (!text || !text.trim()) return '';
  const normalized = normalize(text);
  const pieces = await Promise.all(
    splitPunctuation(normalized).map(async (p) => (p.match ? p.text : (await ipaLines(p.text)).join(' '))),
  );
  return ipaSubstitute(phonemeFixups(pieces.join('')));
}
