# Voice host

`src/host.mjs` joins a LiveKit room, transcribes remote audio tracks with the STT model, speaks replies through the selected TTS engine, and serves the browser page from `stack/web`.

Endpointing is energy-based with debounce only: an utterance ends after `END_SILENCE_MS` of quiet and never before it. Quiet is measured against the room's own floor; absolute thresholds are floors on that floor, so a noisy room still closes an utterance. Barge-in cuts playback when the raw level stays loud.

`ECHO_MARGIN_DB` keeps a microphone that hears our own playback through speakers or a remote-desktop loopback from cancelling every reply on onset: the barge gate is the louder of `BARGE_DBFS` and the recent playback level plus this margin. `MAX_UTTERANCE_FRAMES` bounds a capture that never drops below the continue threshold; without it an utterance would stay open forever and swallow every reply. `makeFloor` tracks the quietest level of the last `FLOOR_FRAMES` frames and substitutes `SILENCE_DBFS` for digital silence, so `-Infinity` never collapses the gates back to the absolute thresholds.

The decimator is a 96-tap windowed-sinc low-pass with a 7.2 kHz cutoff at 48 kHz and a Hann window; it prevents energy above 8 kHz folding into the speech band when decimating by 3. Each track keeps its own filter history. `pcmDbfs` reports a frame's level in dBFS on the 16-bit scale.

`playbackLevels` records levels of audio just sent so barge-in can be told apart from the microphone hearing our own playback. `interruptPlayback` increments the playback id, clears queued audio and stops the speak loop. `speak` streams the reply sentence by sentence and synthesizes the next sentence while the current one plays; it copies each frame into a fresh `Int16Array` so the published frame contains only that frame's samples.

`POST /levels` reports what the page measures about its own mic and playback into the host log. `POST /interrupt` lets a client that can silence its own playback request an interrupt directly; the energy gate cannot be cleared by a normal microphone while our own voice plays.

RNNoise denoising runs on every audio thread before voice activity detection and speech-to-text; `src/denoise.mjs` owns that state per caller.
