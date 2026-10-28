# ccv

Realtime voice for Claude Code: a LiveKit room you join from the browser, speech-to-text (Underdog Bark 3.0 Clean), and text-to-speech (Paradee). Runs as an MCP server through `npx -y github:AnEntrypoint/ccv`.

## Use

Add the server to Claude Code (`.mcp.json` in the project, or `claude mcp add ccv -- npx -y github:AnEntrypoint/ccv`).

Start Claude Code with the channel flag, so speech arrives in the session on its own:

    npx -y --package github:AnEntrypoint/ccv ccv-claude

That runs `claude --dangerously-load-development-channels server:ccv`, passing through any arguments. It exists because the name after `server:` is matched exactly and a stray character in a hand-typed flag silently matches no server; the equivalent by hand is:

    claude --dangerously-load-development-channels server:ccv

The name after `server:` must match the MCP server name exactly (`ccv`). A stray trailing character — including the carriage return a paste can carry — silently matches no server, the flag is ignored, and no speech is ever delivered. Custom channels are not on the approved allowlist, so `--channels` alone is not enough and the startup confirmation dialog has to be accepted. Nothing in the voice stack depends on this flag: a session that receives nothing can call `voice_listen`, which blocks until the user speaks.

Then from the session:

1. `voice_setup`: installs LiveKit and downloads the models in the background (about 1 GB once).
2. `voice_start`: starts LiveKit (dev mode) and the voice host. Open the printed page (http://127.0.0.1:8800/) and click join.
3. `voice_heard` lists what was said; `voice_listen` waits for the next thing said; `voice_speak` replies. Long replies stream sentence by sentence.
4. `voice_status` shows setup progress and recent logs; `voice_stop` shuts the stack down.

Every utterance is also appended to `<CCV_HOME>/heard.jsonl` as one JSON object per line
(`at`, `from`, `text`). Clients that do not deliver MCP channel notifications can follow that file instead.
When a spoken reply finishes playing, the server sends the same channel notification with `event: done`.
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
| `ECHO_MARGIN_DB` | `8` | dB a barge-in must beat our own playback by, so a mic that hears the speaker cannot cancel every reply |

## Platforms

- LiveKit: Linux amd64/arm64 and Windows amd64/arm64 use the release archive, checked against its SHA-256 checksums. macOS uses `brew install livekit`, with Docker (`livekit/livekit-server`) as a fallback.
- Speech runs in Node with onnxruntime-node and an espeak-ng WebAssembly build. No Python and no native build step.

## Models

- STT: `velddev/Underdog-Bark-3.0-Clean-sherpa-onnx` (int8 encoder, decoder, joiner, tokens).
- TTS: `sahilmahendrakar/Paradee-8M-v1.1` (`onnx/paradee_int8.onnx`, `config.json`).

## Status

Verified locally: models download with resume and checksum, LiveKit installs and passes its checksum, MCP tools list and start/stop, the host joins the room and serves the page, and a live conversation across two browser clients — long-form speech, sentence-by-sentence streaming, and barge-in interrupting a reply. Channel delivery into the session is the client's to grant: the flag above, the accepted dialog, and a session that runs long enough to register the handler.
