// Installs the stack's public binaries and models into the ccv home (./.ccv by default).
// LiveKit: release archive checked against the published checksums (Linux, Windows); Homebrew on macOS.
// Models: Hugging Face, authenticated with HF_TOKEN from .env through an Authorization header.
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, chmodSync, statSync } from 'node:fs';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';

export const LIVEKIT_VERSION = '1.13.11';
const LIVEKIT_BASE = `https://github.com/livekit/livekit/releases/download/v${LIVEKIT_VERSION}`;

const MODEL_FILES = [
  { repo: 'velddev/Underdog-Bark-3.0-Clean-sherpa-onnx', file: 'encoder.int8.onnx', dir: 'stt',
    sha256: 'dd8b0275d775ea664f0306764f80a6d9b357ed7d7a1fc15e8269b08b14e89c47' },
  { repo: 'velddev/Underdog-Bark-3.0-Clean-sherpa-onnx', file: 'decoder.int8.onnx', dir: 'stt' },
  { repo: 'velddev/Underdog-Bark-3.0-Clean-sherpa-onnx', file: 'joiner.int8.onnx', dir: 'stt' },
  { repo: 'velddev/Underdog-Bark-3.0-Clean-sherpa-onnx', file: 'tokens.txt', dir: 'stt' },
  { repo: 'sahilmahendrakar/Paradee-8M-v1.1', file: 'onnx/paradee_int8.onnx', as: 'paradee_int8.onnx', dir: 'tts' },
  { repo: 'sahilmahendrakar/Paradee-8M-v1.1', file: 'config.json', dir: 'tts' },
];

export function platformKey() {
  return `${process.platform}-${process.arch}`;
}

function livekitArchive() {
  const map = {
    'linux-x64': `livekit_${LIVEKIT_VERSION}_linux_amd64.tar.gz`,
    'linux-arm64': `livekit_${LIVEKIT_VERSION}_linux_arm64.tar.gz`,
    'win32-x64': `livekit_${LIVEKIT_VERSION}_windows_amd64.zip`,
    'win32-arm64': `livekit_${LIVEKIT_VERSION}_windows_arm64.zip`,
  };
  return map[platformKey()] ?? null;
}

export function livekitBinary(home) {
  return path.join(home, 'bin', process.platform === 'win32' ? 'livekit-server.exe' : 'livekit-server');
}

function sha256(file) {
  return createHash('sha256').update(readFileSync(file)).digest('hex');
}

async function download(url, dest, headers = {}) {
  const part = `${dest}.part`;
  const have = existsSync(part) ? statSync(part).size : 0;
  const res = await fetch(url, { headers: { ...headers, ...(have ? { range: `bytes=${have}-` } : {}) }, redirect: 'follow' });
  if (!res.ok) throw new Error(`GET ${new URL(url).pathname} -> ${res.status}`);
  const resumed = res.status === 206;
  await pipeline(Readable.fromWeb(res.body), createWriteStream(part, { flags: resumed ? 'a' : 'w' }));
  renameSync(part, dest);
}

// Runs a command and resolves with its exit code; output goes to the log callback line by line.
function run(cmd, args, log) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    const relay = (chunk) => String(chunk).split(/\r?\n/).filter(Boolean).forEach((l) => log(l));
    child.stdout.on('data', relay);
    child.stderr.on('data', relay);
    child.on('error', (err) => { log(err.message); resolve(-1); });
    child.on('close', resolve);
  });
}

// Installs LiveKit server. Returns the command that starts it, or throws with what to do by hand.
export async function ensureLiveKit(home, log) {
  const bin = livekitBinary(home);
  if (existsSync(bin)) return { cmd: bin, docker: false };

  if (process.platform === 'darwin') {
    log('macOS: installing livekit with Homebrew');
    const code = await run('brew', ['install', 'livekit'], log);
    if (code === 0) return { cmd: 'livekit-server', docker: false };
    if (await run('docker', ['info'], () => {}) === 0) {
      log('Homebrew failed; falling back to the livekit/livekit-server Docker image');
      return { cmd: null, docker: true };
    }
    throw new Error('install Homebrew (https://brew.sh) then run: brew install livekit, or install Docker');
  }

  const asset = livekitArchive();
  if (!asset) throw new Error(`no LiveKit archive for ${platformKey()}; install livekit-server by hand`);
  const dir = path.join(home, 'bin');
  mkdirSync(dir, { recursive: true });
  const archive = path.join(dir, asset);
  const sums = await (await fetch(`${LIVEKIT_BASE}/checksums.txt`)).text();
  const expected = sums.split('\n').find((l) => l.endsWith(asset))?.split(/\s+/)[0];
  log(`downloading ${asset}`);
  await download(`${LIVEKIT_BASE}/${asset}`, archive);
  if (!expected || sha256(archive) !== expected) throw new Error(`checksum mismatch for ${asset}`);

  const code = await run('tar', ['-xf', archive, '-C', dir], log);
  if (code !== 0) throw new Error(`could not extract ${asset}`);
  rmSync(archive, { force: true });
  if (process.platform !== 'win32') chmodSync(bin, 0o755);
  log(`livekit-server ${LIVEKIT_VERSION} installed`);
  return { cmd: bin, docker: false };
}

// Downloads every model file that is not already present. Existing files are kept.
export async function ensureModels(home, log) {
  const token = process.env.HF_TOKEN ?? '';
  const headers = token ? { authorization: `Bearer ${token}` } : {};
  if (token) log('using HF_TOKEN for faster downloads');
  for (const m of MODEL_FILES) {
    const dir = path.join(home, 'models', m.dir);
    const dest = path.join(dir, m.as ?? path.basename(m.file));
    if (existsSync(dest)) continue;
    mkdirSync(dir, { recursive: true });
    log(`downloading ${m.repo}/${m.file}`);
    await download(`https://huggingface.co/${m.repo}/resolve/main/${m.file}`, dest, headers);
    if (m.sha256 && sha256(dest) !== m.sha256) {
      rmSync(dest, { force: true });
      throw new Error(`checksum mismatch for ${m.file}`);
    }
  }
  log('models ready');
  return path.join(home, 'models');
}

export function modelsReady(home) {
  return MODEL_FILES.every((m) => existsSync(path.join(home, 'models', m.dir, m.as ?? path.basename(m.file))));
}

export function modelDir(home, dir) {
  return path.join(home, 'models', dir);
}
