// Pure decision logic: no `$`, so it can be unit tested on its own.
// The category -> action table is training/pipeline/policy.py's (tests check both agree).

import { CATEGORIES, QUESTION_ID } from './task.js'

export { CATEGORIES }

export const POLICY = {
  continuation: 'keep',
  follow_up: 'keep',
  related: 'compact_view',
  independent: 'clear',
  return: 'compact_branch',
}

// False positives destroy context: these never act, whatever the confidence.
export const NEVER_ACT = new Set(['continuation', 'follow_up'])

export const MODES = ['off', 'suggest', 'auto']

export const DEFAULTS = { mode: 'suggest', minConfidence: 0.85, timeoutMs: 4000 }

// Spisev, the local provider: spisev-alpha, the distilled proxy of Mammoth, served by llama.cpp's llama-server.
export const SPISEV = {
  model: 'arthur-fontaine/spisev-alpha-GGUF:Q8_0',
  sizeLabel: 'about 1.9 GB',
  endpoint: 'http://127.0.0.1:8766/v1/systemone',
  servers: ['llama-server', 'llama'],
  alias: 'spisev-alpha',
  minBuild: 11361,
}

// Cloud System One endpoints. The key is read from AUTO_COMPACT_API_KEY, never stored.
export const PRESETS = {
  'Mammoth (Baseten)': {
    endpoint: 'https://inference.baseten.co/v1/systemone',
    model: 'decision-models/baseten-mammoth-27b-latest',
    auth: 'Api-Key',
  },
  'Jev (OpenCode Zen)': { endpoint: 'https://opencode.ai/zen/v1/systemone', model: 'jev-1.13', auth: 'Bearer' },
}

export function action(category) {
  return POLICY[category] ?? 'keep'
}

function pick(...values) {
  return values.find((v) => v !== undefined && v !== null && v !== '')
}

function toNumber(value, fallback) {
  const n = typeof value === 'number' ? value : Number(value)
  return value !== undefined && value !== null && value !== '' && Number.isFinite(n) ? n : fallback
}

export function localPort(endpoint) {
  const m = /^http:\/\/(?:127\.0\.0\.1|localhost):(\d+)\/v1\/systemone$/.exec(endpoint ?? '')
  return m ? Number(m[1]) : null
}

// `env` holds the AUTO_COMPACT_* values; `stored` what /auto-compact setup and /auto-compact mode saved (never a key).
// No default provider: with neither, the mod does nothing.
export function resolveConfig(env = {}, stored = {}) {
  const provider = [env.provider, env.endpoint && 'systemone', stored.provider].find((p) => ['spisev', 'systemone'].includes(p))
  const saved = stored.provider === provider ? stored : {}
  const spisev = provider === 'spisev'
  const config = {
    provider,
    mode: MODES.includes(env.mode) ? env.mode : MODES.includes(stored.mode) ? stored.mode : DEFAULTS.mode,
    minConfidence: toNumber(env.minConfidence, DEFAULTS.minConfidence),
    timeoutMs: toNumber(env.timeoutMs, DEFAULTS.timeoutMs),
    endpoint: pick(env.endpoint, saved.endpoint, spisev ? SPISEV.endpoint : undefined),
    model: pick(env.model, saved.model, spisev ? SPISEV.model : undefined),
    auth: pick(env.auth, saved.auth, 'Bearer'),
    apiKey: spisev ? undefined : pick(env.apiKey),
    llamaServer: spisev ? pick(env.llamaServer, saved.llamaServer) : undefined,
    problems: [],
    missing: [],
  }
  if (spisev) {
    config.port = localPort(config.endpoint)
    if (!config.port) config.problems.push('AUTO_COMPACT_ENDPOINT is ' + config.endpoint + ', not a local URL such as ' + SPISEV.endpoint)
  } else if (provider) {
    if (!config.endpoint) config.missing.push('AUTO_COMPACT_ENDPOINT')
    if (!config.model) config.missing.push('AUTO_COMPACT_MODEL')
    if (!config.apiKey) config.missing.push('AUTO_COMPACT_API_KEY')
  }
  return config
}

export function headers(config) {
  const h = { 'Content-Type': 'application/json' }
  if (config.apiKey) h.Authorization = (config.auth === 'Api-Key' ? 'Api-Key ' : 'Bearer ') + config.apiKey
  return h
}

// What to do with a provider's answer. `act` is true only for an action this mode may take
// without asking (auto, confident enough); `suggest` is true when the user should be offered it.
export function decide(response, config) {
  const answer = (response?.answers ?? response?.result?.answers)?.[QUESTION_ID]
  if (!answer || answer.type !== 'choice' || !CATEGORIES.includes(answer.choice)) {
    return { category: null, confidence: null, action: 'keep', act: false, suggest: false, reason: 'no answer' }
  }
  const category = answer.choice
  const confidence = Number(answer.probabilities?.[category] ?? answer.confidence ?? 0)
  const wanted = action(category)
  const base = { category, confidence, action: wanted, probabilities: answer.probabilities ?? {} }
  if (NEVER_ACT.has(category) || wanted === 'keep') return { ...base, action: 'keep', act: false, suggest: false, reason: 'keep' }
  if (config.mode === 'off') return { ...base, act: false, suggest: false, reason: 'off' }
  if (config.mode === 'auto' && confidence >= config.minConfidence) return { ...base, act: true, suggest: false, reason: 'auto' }
  if (config.mode === 'auto') return { ...base, act: false, suggest: false, reason: 'low confidence' }
  return { ...base, act: false, suggest: true, reason: 'suggest' }
}

const pct = (x) => Math.round(x * 100) + '%'
const VERB = { compact_view: 'compact', compact_branch: 'compact', clear: 'clear' }
const DONE = { compact_view: 'compacted', compact_branch: 'compacted', clear: 'cleared' }

// The footer label, like auto-effort's `effort high · 90%`: `context kept · 92%`,
// `compact? related · 71%`, `cleared · independent · 91%`.
export function describe(decision, done = false) {
  if (!decision?.category) return 'context kept (' + (decision?.reason ?? 'no answer') + ')'
  if (decision.action === 'keep') return 'context kept · ' + pct(decision.confidence)
  if (done) return DONE[decision.action] + ' · ' + decision.category + ' · ' + pct(decision.confidence)
  return VERB[decision.action] + '? ' + decision.category + ' · ' + pct(decision.confidence)
}

// The debug log line: `[auto-compact] <category> · <confidence> → <action>`.
export function logLine(decision, outcome) {
  return '[auto-compact] ' + (decision.category ?? 'none') + ' · ' +
    (decision.confidence === null || decision.confidence === undefined ? 'n/a' : decision.confidence.toFixed(2)) +
    ' → ' + outcome
}

export function compactInstructions(decision, viewText, prompt) {
  if (decision.action === 'compact_branch') {
    return 'The next message returns to an earlier task of this conversation: keep that task, its files, ' +
      'decisions and open questions in full, and summarize the rest briefly. The next message:\n' + prompt +
      '\n\nThe conversation, one summary line per span:\n' + viewText
  }
  return 'The next message starts a related task: keep the general picture of the project (repo, files, ' +
    'decisions, conventions) these lines describe, and drop the details of the finished work.\n' + viewText
}
