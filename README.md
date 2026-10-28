# ccv

Realtime voice for Claude Code: a LiveKit room you join from the browser, speech-to-text (Underdog Bark 3.0 Clean), and text-to-speech (Paradee). Runs as an MCP server through `npx ccv`.

## Use

Add the server to Claude Code (`.mcp.json` in the project, or `claude mcp add ccv -- npx -y ccv`), then from the session:

1. `voice_setup`: installs LiveKit and downloads the models in the background (about 1 GB once).
2. `voice_start`: starts LiveKit (dev mode) and the voice host. Open the printed page (http://127.0.0.1:8800/) and click join.
3. `voice_heard` lists what was said; `voice_speak` replies. Long replies stream sentence by sentence.
4. `voice_status` shows setup progress and recent logs; `voice_stop` shuts the stack down.

## Configuration

`.env` in the current directory is read at start. Process environment variables win over it.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | none | Faster Hugging Face downloads (sent as an Authorization header) |
| `LIVEKIT_URL` | started locally | Use an existing LiveKit server instead of starting one |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | `devkey` / `secret` | Keys for the room |
| `LK_ROOM` | `lk-voice` | Room name |
| `CCV_PORT` | `8800` | Voice page and host port (127.0.0.1) |
| `CCV_HOME` | `./.ccv` | Where binaries and models live |
| `END_SILENCE_MS` | `800` | Quiet time that ends an utterance (debounce only, nothing is force-sent) |
| `MIN_SPEECH_MS` | `250` | Shorter bursts (clicks, knocks) are dropped |
| `BARGE_DBFS` / `BARGE_MS` | `-30` / `250` | Level and duration that interrupt playback |

## Platforms

- LiveKit: Linux amd64/arm64 and Windows amd64/arm64 use the release archive, checked against its SHA-256 checksums. macOS uses `brew install livekit`, with Docker (`livekit/livekit-server`) as a fallback.
- Speech runs in Node with onnxruntime-node and an espeak-ng WebAssembly build. No Python and no native build step.

## Models

- STT: `velddev/Underdog-Bark-3.0-Clean-sherpa-onnx` (int8 encoder, decoder, joiner, tokens).
- TTS: `sahilmahendrakar/Paradee-8M-v1.1` (`onnx/paradee_int8.onnx`, `config.json`).

## Status

Verified locally: models download with resume and checksum, LiveKit installs and passes its checksum, MCP tools list and start/stop, the host joins the room and serves the page. Not yet verified: a live conversation in the room (barge-in, long-form speech).
