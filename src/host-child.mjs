// Runs the voice host in a child process so native library output on stdout cannot corrupt the
// MCP stdio stream. Parent sends {type:'speak', id, text} and receives {type:'ready'|'heard'|'spoke'|'error'|'log'}.
// Exits when the parent's IPC channel closes, so a dead MCP server never leaves the host holding its port.
import { startVoiceHost } from './host.mjs';

const config = JSON.parse(process.env.CCV_HOST_CONFIG);
const send = (msg) => process.send?.(msg);
const log = (line) => send({ type: 'log', line });

let host;
const shutdown = async () => {
  await host?.close().catch(() => {});
  process.exit(0);
};

try {
  host = await startVoiceHost({
    home: config.home,
    livekit: config.livekit,
    port: config.port,
    log,
    onHeard: (text, identity) => send({ type: 'heard', text, from: identity, at: Date.now() }),
  });
  send({ type: 'ready', url: host.url });

  process.on('disconnect', shutdown);
  process.on('message', async (msg) => {
    if (msg.type === 'speak') {
      try {
        await host.speak(msg.text);
        send({ type: 'spoke', id: msg.id });
      } catch (err) {
        send({ type: 'spoke', id: msg.id, error: err.message });
      }
    }
    if (msg.type === 'stop') await shutdown();
  });
} catch (err) {
  send({ type: 'error', message: err.message });
  process.exit(1);
}
