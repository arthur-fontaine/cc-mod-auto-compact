// Sends sample decisions to the configured provider (a cloud System One endpoint, or Spisev
// already served by llama-server) with the mod's own request, and prints what the mod would do.
// Reads the same environment variables as the mod (AUTO_COMPACT_*), and a .env file.
import { readFileSync } from 'node:fs'

import { decide, describe, headers, resolveConfig, SPISEV } from '../hooks/policy.js'
import { clipStateV2, requestV2 } from '../hooks/task.js'

try {
  for (const line of readFileSync(new URL('../.env', import.meta.url), 'utf8').split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/)
    if (m && process.env[m[1]] === undefined) process.env[m[1]] = m[2].replace(/^(['"])(.*)\1$/, '$2')
  }
} catch {
  // No .env file.
}

const env = process.env
const config = resolveConfig({
  provider: env.AUTO_COMPACT_PROVIDER, endpoint: env.AUTO_COMPACT_ENDPOINT, model: env.AUTO_COMPACT_MODEL,
  auth: env.AUTO_COMPACT_AUTH, apiKey: env.AUTO_COMPACT_API_KEY, mode: env.AUTO_COMPACT_MODE,
  minConfidence: env.AUTO_COMPACT_MIN_CONFIDENCE,
})
if (!config.provider || config.missing.length) {
  console.error('Set AUTO_COMPACT_PROVIDER=spisev, or ' + (config.missing.join(', ') || 'AUTO_COMPACT_ENDPOINT, AUTO_COMPACT_MODEL, AUTO_COMPACT_API_KEY') + ', in the environment or in .env')
  process.exit(1)
}

const MEMORY = [
  '0+4|task: --verbose flag for the CLI [done]; user: "add a --verbose flag"; src/cli.py',
  '4+1|user: thanks, now document it in the README',
  '5+1|= [done] agent added a "Verbose output" section to README.md',
].join('\n')
const LAST_AGENT_MESSAGE = 'Added a "Verbose output" section to README.md, with an example run.'

// [new message, the category a person would pick]
const SAMPLES = [
  ['looks good, also mention it in --help', 'continuation'],
  ['the flag should also print timings, can you add that?', 'follow_up'],
  ['now set up GitHub Actions CI for this repo', 'related'],
  ['unrelated: explain how a B-tree works', 'independent'],
  ['back to the CLI flag: rename it to --debug', 'return'],
]

let agree = 0
for (const [message, expected] of SAMPLES) {
  const model = config.provider === 'spisev' ? SPISEV.alias : config.model
  const started = Date.now()
  const response = await fetch(config.endpoint, {
    method: 'POST', headers: headers(config), body: JSON.stringify(requestV2(clipStateV2(MEMORY, LAST_AGENT_MESSAGE, message), model)),
  })
  if (!response.ok) {
    console.error('HTTP ' + response.status + ' from ' + config.endpoint)
    process.exit(1)
  }
  const decision = decide(await response.json(), config)
  agree += decision.category === expected
  console.log(`${String(Date.now() - started).padStart(5)} ms  ${describe(decision).padEnd(34)} expected ${expected.padEnd(13)} ${message}`)
}
console.log(`${agree}/${SAMPLES.length} as a person would pick`)
