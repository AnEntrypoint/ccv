// Browser side of the voice loop: publishes the mic (receiver path) and plays the host's track (responder path).
import { Room, RoomEvent, Track } from 'https://esm.sh/livekit-client@2';

const $ = (id) => document.getElementById(id);
const setStatus = (s) => { $('status').textContent = s; };
const TICK = 50, HOLD = 150, RESUME = 1200, REPORT_MS = 500;
const PLAY_LOUD = -50, SPEECH_FLOOR = -42, CAL_MS = 350, FLOOR_MARGIN = 6, FLOOR_DECAY = 0.05, MIC_ECHO_DBFS = -55;
const buf = new Float32Array(2048);
let room, micPub, analyserCtx, micAnalyser, tickTimer;
const playAnalysers = new Map();
const playStreams = new Map();
const playGains = new Map();
let overSince = 0, interrupted = false, playing = false, micSuppressed = false, echoDb = -Infinity, calUntil = 0, interruptedAt = 0, reportAt = 0;

function ctx() {
  if (!analyserCtx) analyserCtx = new AudioContext();
  if (analyserCtx.state === 'suspended') analyserCtx.resume();
  return analyserCtx;
}

function analyserFor(stream) {
  const a = ctx().createAnalyser();
  a.fftSize = 2048;
  ctx().createMediaStreamSource(stream).connect(a);
  return a;
}

// The element's own output, which is what the user actually hears. Tapping el.srcObject instead
// reads silence in this client, so a reply could not be measured and barge-in never armed.
// Routing the element through the graph also moves silencing here: el.muted no longer reaches it,
// so cutting a reply dead has to be done on the gain.
function analyserForElement(el) {
  const a = ctx().createAnalyser();
  a.fftSize = 2048;
  const gain = ctx().createGain();
  const src = ctx().createMediaElementSource(el);
  src.connect(a);
  src.connect(gain);
  gain.connect(ctx().destination);
  playGains.set(el, gain);
  return a;
}

function silence(on) {
  for (const g of playGains.values()) g.gain.setTargetAtTime(on ? 0 : 1, ctx().currentTime, 0.01);
}

function micTrack() {
  return room.localParticipant.getTrackPublication(Track.Source.Microphone)?.track?.mediaStreamTrack
    || micPub?.track?.mediaStreamTrack || null;
}

function rmsDb(analyser) {
  analyser.getFloatTimeDomainData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
  const rms = Math.sqrt(sum / buf.length);
  return rms > 0 ? 20 * Math.log10(rms) : -Infinity;
}

function setMicSending(on) {
  const pub = room?.localParticipant.getTrackPublication(Track.Source.Microphone)?.track;
  pub?.[on ? 'unmute' : 'mute']();
}

function interrupt() {
  if (interrupted) return;
  interrupted = true;
  overSince = 0;
  interruptedAt = performance.now();
  for (const el of $('audio').querySelectorAll('audio')) el.muted = true;
  silence(true);
  // Whatever the user says next is the point of the interrupt, so it has to reach the host.
  setMicSending(true);
  fetch('/interrupt', { method: 'POST' }).catch(() => {});
}

// Unmute on the clock, not on a level: a muted element reads as silent, so measuring whether the
// host has stopped would answer its own question and unmute straight into the buffered tail.
function restore() {
  interrupted = false;
  for (const el of $('audio').querySelectorAll('audio')) el.muted = false;
  silence(false);
}

function report(state, micDb, playbackDb, gate, streamDb) {
  const r = (d) => (Number.isFinite(d) ? Math.round(d) : -100);
  fetch('/levels', { method: 'POST', body: JSON.stringify({ b: 3, s: state, m: r(micDb), p: r(playbackDb), ps: r(streamDb), g: r(gate) }) }).catch(() => {});
}

function tick() {
  const now = performance.now();
  let playbackDb = -Infinity;
  for (const a of playAnalysers.values()) playbackDb = Math.max(playbackDb, rmsDb(a));
  const micDb = micAnalyser ? rmsDb(micAnalyser) : -Infinity;
  let gate = -Infinity;
  if (playbackDb > PLAY_LOUD) {
    // The gate is what the mic hears while the user is silent: echo-cancelled silence on a
    // headset, or near our own level on a speaker, so it is measured rather than assumed.
    if (!playing) { playing = true; echoDb = -Infinity; calUntil = now + CAL_MS; }
    if (now < calUntil) echoDb = Math.max(echoDb, micDb);
    else {
      gate = Math.max(SPEECH_FLOOR, echoDb + FLOOR_MARGIN);
      if (micDb > gate) {
        if (!overSince) overSince = now;
        if (now - overSince >= HOLD) interrupt();
      } else {
        // Rising tracks echo peaks; the slow decay forgets a one-off louder moment.
        echoDb = Math.max(micDb, echoDb - FLOOR_DECAY);
        overSince = 0;
      }
    }
  } else {
    playing = false;
    overSince = 0;
  }
  // A mic that hears the reply sends it back to the host, which transcribes our own words and
  // answers them: stop sending while we play, but only when this room really does echo, so a
  // headset user keeps every word of the sentence they speak over.
  const suppress = playing && echoDb > MIC_ECHO_DBFS;
  if (suppress !== micSuppressed) { micSuppressed = suppress; setMicSending(!suppress); }
  if (interrupted && now - interruptedAt >= RESUME) restore();
  if (now - reportAt >= REPORT_MS) {
    reportAt = now;
    let streamDb = -Infinity;
    for (const a of playStreams.values()) streamDb = Math.max(streamDb, rmsDb(a));
    report(interrupted ? 'interrupted' : playing ? 'playing' : playAnalysers.size ? 'idle' : 'no-mic', micDb, playbackDb, gate, streamDb);
  }
}

function teardown() {
  clearInterval(tickTimer);
  tickTimer = 0;
  micAnalyser = null;
  micPub = null;
  playAnalysers.clear();
  playStreams.clear();
  playGains.clear();
  overSince = 0;
  interrupted = false;
  interruptedAt = 0;
  micSuppressed = false;
}

async function join() {
  $('join').disabled = true;
  setStatus('connecting');
  ctx();
  const identity = `browser-${Math.random().toString(36).slice(2, 7)}`;
  const { url, token } = await (await fetch(`/token?identity=${identity}`)).json();
  // Browser echo cancellation removes the host's own voice from the mic, so barge-in hears only the user.
  // Noise suppression and gain stay off so the host sees the mic as captured.
  room = new Room({ audioCaptureDefaults: { echoCancellation: true, noiseSuppression: false, autoGainControl: false } });
  room.on(RoomEvent.TrackSubscribed, (track, pub, participant) => {
    if (track.kind !== Track.Kind.Audio) return;
    const el = track.attach();
    el.dataset.from = participant.identity;
    el.muted = interrupted;
    $('audio').appendChild(el);
    // The element's own output is what the speakers get, and it is the only signal that
    // matches what the user hears: el.srcObject reads silence in this client.
    playAnalysers.set(el, analyserForElement(el));
    try { if (el.srcObject) playStreams.set(el, analyserFor(el.srcObject)); } catch { /* not measured */ }
    setStatus(`hearing ${participant.identity}`);
  });
  room.on(RoomEvent.Disconnected, () => { teardown(); setStatus('disconnected'); $('join').disabled = false; $('leave').disabled = true; $('interrupt').disabled = true; });
  await room.connect(url, token);
  $('leave').disabled = false;
  $('share').disabled = false;
  $('interrupt').disabled = false;
  try {
    micPub = await room.localParticipant.setMicrophoneEnabled(true);
    setStatus(`connected as ${identity}, mic on`);
  } catch {
    setStatus(`connected as ${identity}, no mic (listen only)`);
  }
  // Mic is measured only; connecting it onward would play the user back to themselves.
  const track = micTrack();
  if (track) micAnalyser = analyserFor(new MediaStream([track]));
  tickTimer = setInterval(tick, TICK);
}

// Publishes the audio of a chosen screen/tab as its own track so the host transcribes it like speech.
async function shareAudio() {
  if (!room) return;
  const display = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
  const audio = display.getAudioTracks()[0];
  if (!audio) {
    display.getTracks().forEach((t) => t.stop());
    throw new Error('the shared screen has no audio; tick "share audio" in the picker');
  }
  display.getVideoTracks().forEach((t) => t.stop());
  await room.localParticipant.publishTrack(audio, { source: Track.Source.ScreenShareAudio, name: 'screen-audio' });
  audio.addEventListener('ended', () => setStatus('screen audio stopped'));
  $('share').disabled = true;
  setStatus('sharing screen audio to host');
}

async function leave() {
  await room?.disconnect();
  teardown();
  $('audio').replaceChildren();
  $('share').disabled = true;
  $('interrupt').disabled = true;
}

// Opened with ?autojoin there is no gesture yet, and a suspended context measures silence in both
// directions: the first click or key anywhere arms it. The Interrupt button works regardless.
for (const ev of ['pointerdown', 'keydown']) document.addEventListener(ev, () => { if (analyserCtx?.state === 'suspended') analyserCtx.resume(); });

$('join').onclick = () => join().catch((err) => { teardown(); setStatus(`error: ${err.message}`); $('join').disabled = false; });
$('leave').onclick = leave;
$('share').onclick = () => shareAudio().catch((err) => setStatus(`share failed: ${err.message}`));
$('interrupt').onclick = interrupt;
if (new URLSearchParams(location.search).has('autojoin')) join().catch((err) => { teardown(); setStatus(`error: ${err.message}`); });
