// RNNoise noise suppression, applied to every audio thread before voice activity detection and
// speech-to-text. Works on mono 48 kHz Int16 PCM in RNNoise's native 480-sample (10 ms) frames.
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const createModule = require('@jitsi/rnnoise-wasm/dist/rnnoise-sync.js');

export const DENOISE_FRAME = 480;

let modulePromise = null;
const loadModule = () => (modulePromise ??= (async () => {
  const factory = createModule.default ?? createModule;
  return factory();
})());

// Each caller gets its own RNNoise state, because the model keeps history between frames.
export async function createDenoiser() {
  const m = await loadModule();
  const state = m._rnnoise_create(0);
  const ptr = m._malloc(DENOISE_FRAME * 4);
  return {
    // Returns the denoised PCM and RNNoise's per-frame voice-activity probability (0..1).
    processFrames(pcm) {
      const out = new Int16Array(pcm.length);
      const vad = new Float32Array(Math.ceil(pcm.length / DENOISE_FRAME));
      for (let f = 0, off = 0; off < pcm.length; f++, off += DENOISE_FRAME) {
        const n = Math.min(DENOISE_FRAME, pcm.length - off);
        const heap = new Float32Array(m.HEAPF32.buffer, ptr, DENOISE_FRAME);
        heap.fill(0);
        for (let i = 0; i < n; i++) heap[i] = pcm[off + i];
        vad[f] = m._rnnoise_process_frame(state, ptr, ptr);
        for (let i = 0; i < n; i++) out[off + i] = Math.max(-32768, Math.min(32767, Math.round(heap[i])));
      }
      return { pcm: out, vad };
    },
    process(pcm) {
      return this.processFrames(pcm).pcm;
    },
    close() {
      m._rnnoise_destroy(state);
      m._free(ptr);
    },
  };
}
