// Voice host: joins the LiveKit room, transcribes every remote mic and screen-audio track with
// Underdog Bark (STT), speaks replies with Paradee (TTS), and serves the browser page.
// Endpointing is energy-based with debounce only: an utterance ends after END_SILENCE_MS of quiet,
// and never before it, so nothing is force-sent. Barge-in cuts playback when the raw level stays loud.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AccessToken } from 'livekit-server-sdk';
import { AudioSource, AudioFrame, AudioStream, LocalAudioTrack, Room, RoomEvent, TrackKind, TrackPublishOptions, TrackSource } from '@livekit/rtc-node';
import { loadStt, transcribePcm16k } from './stt.mjs';
import { loadParadee, synthesizeStream, SAMPLE_RATE as TTS_RATE } from './tts.mjs';
import { modelDir } from './install.mjs';

const WEB = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'stack', 'web');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript' };

const STT_RATE = 16000;
const FRAME_MS = 20;
const CAPTURE_RATE = 48000;
const CAPTURE_FRAME_MS = 10;
const DECIMATE = CAPTURE_RATE / STT_RATE;
const PER_TTS_FRAME = (TTS_RATE * FRAME_MS) / 1000;
const ENERGY_DBFS = Number(process.env.ENERGY_DBFS ?? -40);
const ENERGY_CONTINUE_DBFS = Number(process.env.ENERGY_CONTINUE_DBFS ?? -46);
const BARGE_DBFS = Number(process.env.BARGE_DBFS ?? -30);
const BARGE_FRAMES = Number(process.env.BARGE_MS ?? 250) / CAPTURE_FRAME_MS;
// Barge-in must beat our own playback by this many dB; a microphone that hears the
// speaker (or a remote-desktop loopback) would otherwise cancel every reply on onset.
const ECHO_MARGIN_DB = Number(process.env.ECHO_MARGIN_DB ?? 8);
const PLAYBACK_REF_MS = 400;
const MIN_SPEECH_FRAMES = Number(process.env.MIN_SPEECH_MS ?? 250) / CAPTURE_FRAME_MS;
const END_SILENCE_FRAMES = Number(process.env.END_SILENCE_MS ?? 800) / CAPTURE_FRAME_MS;
const PRE_ROLL_FRAMES = 20;
const ECHO_TAIL_MS = 800;

// Windowed-sinc low-pass (7.2 kHz cutoff at 48 kHz, Hann window, 96 taps) so energy above 8 kHz
// cannot fold into the speech band when decimating by 3.
const LP_TAPS = 96;
const LP_COEFFS = (() => {
  const fc = 7200 / CAPTURE_RATE;
  const taps = new Float64Array(LP_TAPS);
  let sum = 0;
  for (let n = 0; n < LP_TAPS; n++) {
    const m = n - (LP_TAPS - 1) / 2;
    const sinc = m === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * m) / (Math.PI * m);
    taps[n] = sinc * (0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (LP_TAPS - 1)));
    sum += taps[n];
  }
  for (let n = 0; n < LP_TAPS; n++) taps[n] /= sum;
  return taps;
})();

// Per-track decimator: each track keeps its own filter history.
function makeDecimator() {
  const history = new Float64Array(LP_TAPS);
  let pos = 0;
  return (pcm48) => {
    const out = new Int16Array(pcm48.length / DECIMATE);
    let o = 0;
    for (let i = 0; i < pcm48.length; i++) {
      history[pos] = pcm48[i];
      if (i % DECIMATE === DECIMATE - 1) {
        let acc = 0;
        for (let k = 0; k < LP_TAPS; k++) acc += LP_COEFFS[k] * history[(pos - k + LP_TAPS) % LP_TAPS];
        out[o++] = Math.max(-32768, Math.min(32767, Math.round(acc)));
      }
      pos = (pos + 1) % LP_TAPS;
    }
    return out;
  };
}

// Level of one frame we are about to send, in dBFS on the 16-bit scale.
function pcmDbfs(pcm16) {
  if (pcm16.length === 0) return -Infinity;
  let sum = 0;
  for (let i = 0; i < pcm16.length; i++) sum += pcm16[i] * pcm16[i];
  const rms = Math.sqrt(sum / pcm16.length) / 32768;
  return rms > 0 ? 20 * Math.log10(rms) : -Infinity;
}

function rawDbfs(pcm48) {
  let sum = 0;
  for (let i = 0; i < pcm48.length; i++) sum += pcm48[i] * pcm48[i];
  const rms = Math.sqrt(sum / pcm48.length) / 32768;
  return rms > 0 ? 20 * Math.log10(rms) : -Infinity;
}

export async function startVoiceHost({ home, livekit, port, onHeard, log = console.log }) {
  const [stt, tts] = await Promise.all([
    loadStt(modelDir(home, 'stt')),
    loadParadee(modelDir(home, 'tts')),
  ]);
  log('speech models loaded');

  let speakingUntil = 0;
  let playbackId = 0;
  let source;
  let sttChain = Promise.resolve();

  // Levels of the audio just sent, so barge-in can be told apart from the microphone
  // hearing our own playback through speakers or a remote desktop loopback.
  const playbackLevels = [];
  function notePlaybackLevel(db) {
    const now = Date.now();
    playbackLevels.push({ t: now, db });
    while (playbackLevels.length > 0 && now - playbackLevels[0].t > PLAYBACK_REF_MS) playbackLevels.shift();
  }
  function playbackRefDbfs() {
    const now = Date.now();
    let ref = -Infinity;
    for (const entry of playbackLevels) if (now - entry.t <= PLAYBACK_REF_MS && entry.db > ref) ref = entry.db;
    return ref;
  }

  // Cancels the current reply: drops queued audio and stops the speak loop.
  function interruptPlayback() {
    playbackId++;
    speakingUntil = 0;
    playbackLevels.length = 0;
    source?.clearQueue();
  }

  // Streams the reply sentence by sentence; the next sentence is synthesized while this one plays.
  async function speak(text) {
    if (!source) throw new Error('not connected to a room yet');
    const id = ++playbackId;
    const it = synthesizeStream(tts, text)[Symbol.asyncIterator]();
    let next = it.next();
    for (;;) {
      const { value, done } = await next;
      if (done || id !== playbackId) break;
      next = it.next();
      next.catch(() => {});
      speakingUntil = Math.max(speakingUntil, Date.now()) + (value.length / TTS_RATE) * 1000;
      for (let i = 0; i < value.length; i += PER_TTS_FRAME) {
        if (id !== playbackId) break;
        const chunk = value.subarray(i, i + PER_TTS_FRAME);
        const padded = chunk.length === PER_TTS_FRAME ? chunk : Object.assign(new Int16Array(PER_TTS_FRAME), chunk);
        notePlaybackLevel(pcmDbfs(padded));
        await source.captureFrame(new AudioFrame(padded, TTS_RATE, 1, PER_TTS_FRAME));
      }
    }
    if (id !== playbackId) it.return?.();
    if (id === playbackId) speakingUntil = Date.now() + ECHO_TAIL_MS;
  }

  function transcribeQueued(pcm) {
    const job = sttChain.then(() => transcribePcm16k(stt, pcm));
    sttChain = job.catch(() => {});
    return job;
  }

  async function handleUtterance(frames, identity) {
    const total = frames.reduce((n, f) => n + f.length, 0);
    const pcm = new Int16Array(total);
    let off = 0;
    for (const f of frames) { pcm.set(f, off); off += f.length; }
    try {
      const heard = (await transcribeQueued(pcm)).trim();
      if (!heard) return;
      log(`[${identity}] heard: ${heard}`);
      onHeard?.(heard, identity);
    } catch (err) {
      log(`utterance failed: ${err.message}`);
    }
  }

  async function listen(track, identity) {
    const stream = new AudioStream(track, { sampleRate: CAPTURE_RATE, numChannels: 1, frameSizeMs: CAPTURE_FRAME_MS });
    const decimate = makeDecimator();
    const preRoll = [];
    let utter = null;
    let silent = 0;
    let voiced = 0;
    let bargeCount = 0;
    for await (const frame of stream) {
      const pcm16k = decimate(frame.data);
      const db = rawDbfs(frame.data);
      if (Date.now() < speakingUntil) {
        const bargeGate = Math.max(BARGE_DBFS, playbackRefDbfs() + ECHO_MARGIN_DB);
        bargeCount = db >= bargeGate ? bargeCount + 1 : 0;
        if (bargeCount < BARGE_FRAMES) {
          preRoll.push(pcm16k);
          if (preRoll.length > PRE_ROLL_FRAMES) preRoll.shift();
          continue;
        }
        bargeCount = 0;
        interruptPlayback();
        log(`[${identity}] barge-in: playback interrupted`);
      }
      if (utter) {
        utter.push(pcm16k);
        silent = db >= ENERGY_CONTINUE_DBFS ? 0 : silent + 1;
        if (db >= ENERGY_DBFS) voiced++;
        if (silent >= END_SILENCE_FRAMES) {
          const done = utter;
          utter = null;
          if (voiced >= MIN_SPEECH_FRAMES) handleUtterance(done, identity);
          voiced = 0;
        }
      } else if (db >= ENERGY_DBFS) {
        utter = [...preRoll, pcm16k];
        silent = 0;
        voiced = 1;
      }
      preRoll.push(pcm16k);
      if (preRoll.length > PRE_ROLL_FRAMES) preRoll.shift();
    }
  }

  const room = new Room();
  room.on(RoomEvent.TrackSubscribed, (track, _pub, participant) => {
    if (track.kind === TrackKind.KIND_AUDIO && participant.identity !== livekit.identity) listen(track, participant.identity);
  });
  const at = new AccessToken(livekit.key, livekit.secret, { identity: livekit.identity });
  at.addGrant({ roomJoin: true, room: livekit.room, canPublish: true, canSubscribe: true });
  await room.connect(livekit.url, await at.toJwt(), { autoSubscribe: true });
  log(`joined ${livekit.room} as ${livekit.identity}`);

  source = new AudioSource(TTS_RATE, 1);
  const track = LocalAudioTrack.createAudioTrack('voice-host', source);
  await room.localParticipant.publishTrack(track, new TrackPublishOptions({ source: TrackSource.SOURCE_MICROPHONE }));

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (req.method === 'GET' && url.pathname === '/token') {
      const identity = url.searchParams.get('identity') ?? 'browser';
      const browser = new AccessToken(livekit.key, livekit.secret, { identity });
      browser.addGrant({ roomJoin: true, room: livekit.room, canPublish: true, canSubscribe: true });
      res.setHeader('content-type', 'application/json');
      return res.end(JSON.stringify({ url: livekit.url, token: await browser.toJwt() }));
    }
    if (req.method === 'GET' && url.pathname === '/health') return res.end('ok');
    if (req.method === 'POST' && url.pathname === '/speak') {
      let body = '';
      for await (const chunk of req) body += chunk;
      try {
        const { text } = JSON.parse(body || '{}');
        if (!text) { res.statusCode = 400; return res.end('{"error":"text required"}'); }
        await speak(text);
        return res.end('{"ok":true}');
      } catch (err) {
        res.statusCode = 500;
        return res.end(JSON.stringify({ error: err.message }));
      }
    }
    if (req.method === 'GET' && (url.pathname === '/' || url.pathname.endsWith('.js'))) {
      const file = url.pathname === '/' ? 'index.html' : path.basename(url.pathname);
      try {
        res.setHeader('content-type', TYPES[path.extname(file)] ?? 'application/octet-stream');
        return res.end(await readFile(path.join(WEB, file)));
      } catch { /* fall through to 404 */ }
    }
    res.statusCode = 404;
    res.end();
  });
  await new Promise((resolve) => server.listen(port, '127.0.0.1', resolve));
  log(`voice page and /speak on http://127.0.0.1:${port}`);

  return {
    url: `http://127.0.0.1:${port}/`,
    speak,
    async close() {
      interruptPlayback();
      server.close();
      await room.disconnect();
    },
  };
}
