// MCP stdio server for the ccv voice stack. Tools:
//   voice_setup   install LiveKit and download models in the background (returns at once)
//   voice_start   wait for setup, start LiveKit (unless LIVEKIT_URL is set) and the voice host
//   voice_stop    stop the host and the LiveKit server this process started
//   voice_status  setup progress, host state, recent log lines
//   voice_speak   say text in the room (streams per sentence)
//   voice_heard   recent utterances the host transcribed
import { mkdirSync, appendFileSync, existsSync, readdirSync, readFileSync, statSync, unlinkSync, writeFileSync } from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fork, spawn } from 'node:child_process';
import { ensureLiveKit, ensureModels, modelsReady } from './install.mjs';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { SubscribeRequestSchema, UnsubscribeRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { loadDotEnv } from './env.mjs';
const HEARD_KEEP = 50;
const LOG_KEEP = 200;
const LIVEKIT_PORT = 7880;
const HOST_CHILD = fileURLToPath(new URL('./host-child.mjs', import.meta.url));
const HEARD_URI = 'ccv://heard';
const subscriptions = new Set();

// Pushes each utterance into the Claude session as a channel event as it arrives; set once the MCP server exists.
let notifyHeard = () => {};

const state = {
  home: '',
  setup: { status: 'idle', error: null, promise: null, detail: '' },
  install: null,
  livekit: null,
  host: null,
  heard: [],
  logs: [],
};

function log(line) {
  const stamped = `${new Date().toISOString()} ${line}`;
  state.logs.push(stamped);
  if (state.setup.status === 'running') state.setup.detail = line;
  if (state.logs.length > LOG_KEEP) state.logs.shift();
  if (state.home) appendFileSync(path.join(state.home, 'ccv.log'), `${stamped}\n`);
  console.error(stamped);
}

// Child PIDs survive only in this file, so a server killed without a clean stop cannot orphan them.
function childrenFile() {
  return path.join(state.home, 'children.json');
}

function recordChildren() {
  writeFileSync(childrenFile(), JSON.stringify({ livekit: state.livekit?.pid ?? null, host: state.host?.child.pid ?? null }));
}

// Kills children a previous server left behind, identified by the PIDs it recorded.
function reapStaleChildren() {
  if (!existsSync(childrenFile())) return;
  for (const [name, pid] of Object.entries(JSON.parse(readFileSync(childrenFile(), 'utf8')))) {
    if (!pid) continue;
    try {
      process.kill(pid, 'SIGTERM');
      log(`reaped stale ${name} pid ${pid} from an earlier server`);
    } catch { /* already gone */ }
  }
  unlinkSync(childrenFile());
}

// Debug and info SDK and server chatter: kept in livekit.log for debugging, out of the status output.
function noisy(line) {
  return /\t(DEBUG|INFO)\t/.test(line) || line.includes('"name":"lk-rtc"') || line.startsWith('Nvidia Decoder');
}

function quietLog(source, line) {
  if (state.home) appendFileSync(path.join(state.home, 'livekit.log'), `${source}: ${line}\n`);
}

function text(message) {
  return { content: [{ type: 'text', text: message }] };
}

// Starts the install once; later calls share its promise.
function ensureSetup() {
  if (state.setup.promise) return state.setup.promise;
  state.setup = { status: 'running', error: null, promise: null, detail: 'starting' };
  state.setup.promise = (async () => {
    try {
      if (!modelsReady(state.home)) await ensureModels(state.home, log);
      if (!process.env.LIVEKIT_URL) state.install = await ensureLiveKit(state.home, log);
      state.setup.status = 'done';
      log('setup complete');
    } catch (err) {
      state.setup.status = 'error';
      state.setup.error = err.message;
      state.setup.promise = null;
      log(`setup failed: ${err.message}`);
      throw err;
    }
  })();
  return state.setup.promise;
}

// Bytes written so far for each model download still in progress.
function downloadProgress() {
  const root = path.join(state.home, 'models');
  if (!existsSync(root)) return [];
  const parts = [];
  for (const dir of readdirSync(root)) {
    for (const f of readdirSync(path.join(root, dir))) {
      if (f.endsWith('.part')) parts.push(`${dir}/${f.slice(0, -5)}: ${(statSync(path.join(root, dir, f)).size / 1e6).toFixed(0)} MB`);
    }
  }
  return parts;
}

// Opens the voice page in the default browser. Failure is ignored: the URL is also returned to the caller.
function openPage(url) {
  if (process.env.CCV_NO_BROWSER) return;
  const [cmd, args] = process.platform === 'darwin' ? ['open', [url]]
    : process.platform === 'win32' ? ['cmd', ['/c', 'start', '', url]]
    : ['xdg-open', [url]];
  const child = spawn(cmd, args, { stdio: 'ignore', detached: true });
  child.on('error', () => {});
  child.unref();
}

function portOpen(port) {
  return new Promise((resolve) => {
    const sock = net.connect(port, '127.0.0.1');
    sock.once('connect', () => { sock.destroy(); resolve(true); });
    sock.once('error', () => resolve(false));
  });
}

async function waitForPort(port, ms) {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    if (await portOpen(port)) return true;
    await new Promise((r) => setTimeout(r, 200));
  }
  return false;
}

// Starts LiveKit in dev mode. Docker is used when the binary install fell back to a container.
async function startLiveKit(install) {
  const child = install.docker
    ? spawn('docker', ['run', '--rm', '-p', '7880:7880', '-p', '7881:7881', '-p', '7882:7882/udp', 'livekit/livekit-server', '--dev', '--bind', '0.0.0.0'], { stdio: ['ignore', 'pipe', 'pipe'] })
    : spawn(install.cmd, ['--dev', '--bind', '127.0.0.1'], { stdio: ['ignore', 'pipe', 'pipe'] });
  const relay = (d) => String(d).split(/\r?\n/).filter(Boolean).forEach((l) => noisy(l) ? quietLog('livekit', l) : log(`livekit: ${l}`));
  child.stdout.on('data', relay);
  child.stderr.on('data', relay);
  child.on('exit', (code) => { log(`livekit exited (${code})`); state.livekit = null; recordChildren(); });
  if (!(await waitForPort(LIVEKIT_PORT, 15000))) {
    child.kill();
    throw new Error(`LiveKit did not open port ${LIVEKIT_PORT}`);
  }
  return child;
}

// Forks the voice host and resolves once it has joined the room and serves the page.
function startHostChild(config) {
  return new Promise((resolve, reject) => {
    const child = fork(HOST_CHILD, [], {
      env: { ...process.env, CCV_HOST_CONFIG: JSON.stringify(config) },
      stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
    });
    const relay = (d) => String(d).split(/\r?\n/).filter(Boolean).forEach((l) => noisy(l) ? quietLog('host', l) : log(`host: ${l}`));
    child.stdout.on('data', relay);
    child.stderr.on('data', relay);
    recordChildren();
    const pendingSpeech = new Map();
    let nextId = 0;
    child.on('message', (msg) => {
      if (msg.type === 'log') (noisy(msg.line) ? quietLog('host', msg.line) : log(`host: ${msg.line}`));
      if (msg.type === 'ready') resolve({ child, url: msg.url, speak: (t) => speakVia(child, pendingSpeech, () => nextId++, t) });
      if (msg.type === 'error') reject(new Error(msg.message));
      if (msg.type === 'heard') {
        const utterance = { at: msg.at, from: msg.from, text: msg.text };
        state.heard.push(utterance);
        if (state.heard.length > HEARD_KEEP) state.heard.shift();
        notifyHeard(utterance);
      }
      if (msg.type === 'spoke') {
        const done = pendingSpeech.get(msg.id);
        pendingSpeech.delete(msg.id);
        if (done) msg.error ? done.reject(new Error(msg.error)) : done.resolve();
      }
    });
    child.on('exit', (code) => {
      log(`host exited (${code})`);
      state.host = null;
      recordChildren();
      reject(new Error(`host exited (${code}) before ready`));
    });
  });
}

function speakVia(child, pending, nextId, spoken) {
  const id = nextId();
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    child.send({ type: 'speak', id, text: spoken });
  });
}

async function start() {
  if (state.host) return `already running at ${state.host.url}`;
  await ensureSetup();
  const external = process.env.LIVEKIT_URL;
  const livekit = {
    url: external ?? `ws://127.0.0.1:${LIVEKIT_PORT}`,
    key: process.env.LIVEKIT_API_KEY ?? 'devkey',
    secret: process.env.LIVEKIT_API_SECRET ?? 'secret',
    room: process.env.LK_ROOM ?? 'lk-voice',
    identity: process.env.LK_IDENTITY ?? 'voice-host',
  };
  if (!external) {
    if (await portOpen(LIVEKIT_PORT)) {
      throw new Error(`port ${LIVEKIT_PORT} is already in use; stop that server or set LIVEKIT_URL to use it`);
    }
    state.livekit = await startLiveKit(state.install);
    recordChildren();
  }
  const port = Number(process.env.CCV_PORT ?? 8800);
  if (await portOpen(port)) throw new Error(`port ${port} is already in use; set CCV_PORT to another port`);
  const host = await startHostChild({ home: state.home, livekit, port });
  state.host = host;
  openPage(host.url);
  return `voice host running. Open ${host.url} and click join.`;
}

async function stop() {
  if (state.host) {
    state.host.child.send({ type: 'stop' });
    state.host = null;
  }
  state.livekit?.kill();
  state.livekit = null;
  recordChildren();
  return 'voice stack stopped';
}

function status() {
  return JSON.stringify({
    setup: state.setup.status,
    setupDetail: state.setup.detail,
    setupError: state.setup.error,
    downloading: downloadProgress(),
    modelsReady: state.home ? modelsReady(state.home) : false,
    livekit: state.livekit ? 'running (started by ccv)' : process.env.LIVEKIT_URL ? 'external' : 'stopped',
    host: state.host ? state.host.url : 'stopped',
    heard: state.heard.length,
    logs: state.logs.slice(-20),
  }, null, 2);
}

export async function main() {
  const cwd = process.cwd();
  const loaded = loadDotEnv(cwd);
  state.home = process.env.CCV_HOME ?? path.join(cwd, '.ccv');
  mkdirSync(state.home, { recursive: true });
  reapStaleChildren();
  if (loaded.length) log(`loaded from .env: ${loaded.join(', ')}`);
  ensureSetup().catch(() => {});

  const server = new McpServer(
    { name: 'ccv', version: '0.1.0' },
    {
      capabilities: { logging: {}, resources: { subscribe: true }, experimental: { 'claude/channel': {} } },
      instructions: 'Voice utterances arrive as <channel source="ccv" from="..." at="..."> and are spoken by the user in the voice room. Reply with voice_speak.',
    },
  );
  server.registerResource(
    'heard',
    HEARD_URI,
    { description: 'Recent voice utterances. Subscribe for resources/updated on each new utterance; read for the recent list.', mimeType: 'application/json' },
    async (uri) => ({ contents: [{ uri: uri.href, mimeType: 'application/json', text: JSON.stringify(state.heard, null, 2) }] }),
  );
  server.server.setRequestHandler(SubscribeRequestSchema, async ({ params }) => {
    subscriptions.add(params.uri);
    return {};
  });
  server.server.setRequestHandler(UnsubscribeRequestSchema, async ({ params }) => {
    subscriptions.delete(params.uri);
    return {};
  });
  notifyHeard = (utterance) => {
    // Standard resource notification: clients subscribed to ccv://heard re-read the list.
    if (subscriptions.has(HEARD_URI)) server.server.sendResourceUpdated({ uri: HEARD_URI }).catch((err) => log(`resource notify failed: ${err.message}`));
    // Standard logging notification: any MCP client that renders logs gets it.
    server.sendLoggingMessage({ level: 'notice', logger: 'ccv', data: utterance }).catch((err) => log(`logging notify failed: ${err.message}`));
    // Claude Code channel push: clients that do not declare claude/channel ignore it.
    server.server.notification({
      method: 'notifications/claude/channel',
      params: { content: utterance.text, meta: { from: utterance.from, at: String(utterance.at) } },
    }).then(() => log(`channel notify sent: ${utterance.text}`), (err) => log(`channel notify failed: ${err.message}`));
  };

  server.registerTool('voice_setup', {
    description: 'Install LiveKit and download the speech models in the background. Runs automatically at launch; use this only to retry after an error.',
    inputSchema: {},
  }, async () => {
    ensureSetup().catch(() => {});
    return text('setup started in the background; poll voice_status');
  });

  server.registerTool('voice_start', {
    description: 'Start LiveKit and the voice host, waiting for setup if it is still running, and open the voice page in the browser.',
    inputSchema: {},
  }, async () => text(await start()));

  server.registerTool('voice_stop', {
    description: 'Stop the voice host and the LiveKit server this process started.',
    inputSchema: {},
  }, async () => text(await stop()));

  server.registerTool('voice_status', {
    description: 'Setup progress, host and LiveKit state, and recent log lines.',
    inputSchema: {},
  }, async () => text(status()));

  server.registerTool('voice_speak', {
    description: 'Say text in the voice room. Long text streams sentence by sentence.',
    inputSchema: { text: z.string().min(1) },
  }, async ({ text: spoken }) => {
    if (!state.host) return text('voice host is not running; call voice_start');
    await state.host.speak(spoken);
    return text('spoken');
  });

  server.registerTool('voice_heard', {
    description: 'Recent utterances transcribed from the room, oldest first.',
    inputSchema: { since: z.number().optional().describe('only utterances after this epoch ms') },
  }, async ({ since = 0 }) => text(JSON.stringify(state.heard.filter((u) => u.at > since), null, 2)));

  await server.connect(new StdioServerTransport());
  log('ccv MCP server ready');

  const shutdown = async () => { await stop().catch(() => {}); process.exit(0); };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
  process.on('SIGHUP', shutdown);
  process.stdin.on('end', shutdown);
}
