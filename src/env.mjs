import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

// Loads ./.env from the current directory only. Variables already set in the
// process environment win. Values are never logged, only the key names loaded.
export function loadDotEnv(cwd = process.cwd()) {
  const file = path.join(cwd, '.env');
  if (!existsSync(file)) return [];
  const loaded = [];
  for (const raw of readFileSync(file, 'utf8').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq < 1) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (process.env[key] === undefined) {
      process.env[key] = value;
      loaded.push(key);
    }
  }
  return loaded;
}
