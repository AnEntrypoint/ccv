#!/usr/bin/env node
import { main } from '../src/server.mjs';

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
