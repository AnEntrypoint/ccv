// Browser side of the voice loop: publishes the mic (receiver path) and plays the host's track (responder path).
import { Room, RoomEvent, Track } from 'https://esm.sh/livekit-client@2';

const $ = (id) => document.getElementById(id);
const setStatus = (s) => { $('status').textContent = s; };
let room;

async function join() {
  $('join').disabled = true;
  setStatus('connecting');
  const identity = `browser-${Math.random().toString(36).slice(2, 7)}`;
  const { url, token } = await (await fetch(`/token?identity=${identity}`)).json();
  // Browser echo cancellation removes the host's own voice from the mic, so barge-in hears only the user.
  // Noise suppression and gain stay off so the host sees the mic as captured.
  room = new Room({ audioCaptureDefaults: { echoCancellation: true, noiseSuppression: false, autoGainControl: false } });
  room.on(RoomEvent.TrackSubscribed, (track, pub, participant) => {
    if (track.kind !== Track.Kind.Audio) return;
    const el = track.attach();
    el.dataset.from = participant.identity;
    $('audio').appendChild(el);
    setStatus(`hearing ${participant.identity}`);
  });
  room.on(RoomEvent.Disconnected, () => { setStatus('disconnected'); $('join').disabled = false; $('leave').disabled = true; });
  await room.connect(url, token);
  $('leave').disabled = false;
  $('share').disabled = false;
  try {
    await room.localParticipant.setMicrophoneEnabled(true);
    setStatus(`connected as ${identity}, mic on`);
  } catch {
    setStatus(`connected as ${identity}, no mic (listen only)`);
  }
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
  $('audio').replaceChildren();
  $('share').disabled = true;
}

$('join').onclick = () => join().catch((err) => { setStatus(`error: ${err.message}`); $('join').disabled = false; });
$('leave').onclick = leave;
$('share').onclick = () => shareAudio().catch((err) => setStatus(`share failed: ${err.message}`));
if (new URLSearchParams(location.search).has('autojoin')) join().catch((err) => setStatus(`error: ${err.message}`));
