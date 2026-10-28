// Browser side of the voice loop: publishes the mic (receiver path) and plays the host's track (responder path).
import { Room, RoomEvent, Track } from 'https://esm.sh/livekit-client@2';

const $ = (id) => document.getElementById(id);
const setStatus = (s) => { $('status').textContent = s; };
const TICK = 50, HOLD = 150, RESUME = 700;
const PLAY_LOUD = -50, SPEECH_FLOOR = -42, CAL_MS = 350, FLOOR_MARGIN = 9, FLOOR_DECAY = 0.05;
const buf = new Float32Array(2048);
let room, micPub, analyserCtx, micAnalyser, tickTimer;
const playAnalysers = new Map();
let overSince = 0, quietSince = 0, interrupted = false, playing = false, echoDb = -Infinity, calUntil = 0;

function analyserFor(stream) {
  const a = analyserCtx.createAnalyser();
  a.fftSize = 2048;
  analyserCtx.createMediaStreamSource(stream).connect(a);
  return a;
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

function interrupt() {
  if (interrupted) return;
  interrupted = true;
  overSince = 0;
  quietSince = 0;
  for (const el of $('audio').querySelectorAll('audio')) el.muted = true;
  fetch('/interrupt', { method: 'POST' }).catch(() => {});
}

function restore() {
  interrupted = false;
  quietSince = 0;
  for (const el of $('audio').querySelectorAll('audio')) el.muted = false;
}

function tick() {
  const now = performance.now();
  let playbackDb = -Infinity;
  for (const a of playAnalysers.values()) playbackDb = Math.max(playbackDb, rmsDb(a));
  const micDb = micAnalyser ? rmsDb(micAnalyser) : -Infinity;
  if (playbackDb > PLAY_LOUD) {
    // The gate is what the mic hears while the user is silent: echo-cancelled silence on a
    // headset, or near our own level on a speaker, so it is measured rather than assumed.
    if (!playing) { playing = true; echoDb = -Infinity; calUntil = now + CAL_MS; }
    if (now < calUntil) echoDb = Math.max(echoDb, micDb);
    else if (micDb > Math.max(SPEECH_FLOOR, echoDb + FLOOR_MARGIN)) {
      if (!overSince) overSince = now;
      if (now - overSince >= HOLD) interrupt();
    } else {
      // Rising tracks echo peaks; the slow decay forgets a one-off louder moment.
      echoDb = Math.max(micDb, echoDb - FLOOR_DECAY);
      overSince = 0;
    }
  } else {
    playing = false;
    overSince = 0;
  }
  if (interrupted) {
    if (playbackDb > PLAY_LOUD) quietSince = 0;
    else if (!quietSince) quietSince = now;
    if (quietSince && now - quietSince >= RESUME) restore();
  }
}

function teardown() {
  clearInterval(tickTimer);
  tickTimer = 0;
  analyserCtx?.close();
  analyserCtx = null;
  micAnalyser = null;
  micPub = null;
  playAnalysers.clear();
  overSince = 0;
  quietSince = 0;
  interrupted = false;
}

async function join() {
  $('join').disabled = true;
  setStatus('connecting');
  analyserCtx = new AudioContext();
  analyserCtx.resume();
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
    // Tap el.srcObject, not the element: createMediaElementSource would reroute its audio and silence the speaker.
    playAnalysers.set(el, analyserFor(el.srcObject));
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
