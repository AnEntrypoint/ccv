#!/usr/bin/env node
// Launches Claude Code with ccv's channel flag, so speech arrives in the session on its own.
// The name after "server:" is matched exactly and a stray character in it silently matches no
// server, so the flag is assembled here rather than left to be typed by hand.
import { spawn } from 'node:child_process';

const server = process.env.CCV_SERVER_NAME ?? 'ccv';
const args = ['--dangerously-load-development-channels', `server:${server}`, ...process.argv.slice(2)];

// Reaching claude.cmd on Windows needs a shell, and a shell re-splits the command line, so every
// argument is quoted for cmd. Elsewhere spawn passes the arguments through untouched.
const child = process.platform === 'win32'
  ? spawn(process.env.ComSpec ?? 'cmd.exe', ['/d', '/c', `claude ${args.map((a) => `"${a.replace(/"/g, '""')}"`).join(' ')}`], { stdio: 'inherit' })
  : spawn('claude', args, { stdio: 'inherit' });

child.on('error', (err) => {
  console.error(`ccv-claude: could not run claude: ${err.message}`);
  process.exit(1);
});
child.on('exit', (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code ?? 0);
});
