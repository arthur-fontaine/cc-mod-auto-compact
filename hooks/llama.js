// The Spisev provider: spisev-alpha, the distilled classifier served by llama.cpp's llama-server.
// Pure helpers, from auto-effort's nisev.js; register.js makes the calls.

import { SPISEV } from './policy.js'

// `llama` is llama.cpp's unified CLI, whose server is `llama serve`; it takes the same flags.
export function serverCommand(binary) {
  return /(^|\/)llama$/.test(binary) ? [binary, 'serve'] : [binary]
}

export function serverArgs(config, binary) {
  const source = /\.gguf$/i.test(config.model) ? ['-m', config.model] : ['-hf', config.model]
  return [
    ...serverCommand(binary), ...source,
    '--host', '127.0.0.1', '--port', String(config.port),
    '--alias', SPISEV.alias,
    // States are at most 12k tokens; one slot keeps the prompt cache for the session's prefix.
    '-c', '16384', '-np', '1',
  ]
}

export function parseBuild(output) {
  const m = output.match(/build (\d+)/)
  return m ? Number(m[1]) : null
}

export function parseServerOutput(text) {
  if (/listening on/i.test(text)) return { ready: true }
  const percent = /download/i.test(text) && text.match(/(\d{1,3}(?:\.\d+)?)%/)
  return percent ? { percent: Math.round(Number(percent[1])) } : {}
}

export function servesOurModel(text) {
  try {
    return (JSON.parse(text).data ?? []).some((m) => m.id === SPISEV.alias)
  } catch {
    return false
  }
}
