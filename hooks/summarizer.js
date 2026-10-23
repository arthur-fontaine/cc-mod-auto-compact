// The mod's summarizer: the same request as training/pipeline/summarizer.py, sent with
// $.model.complete through the user's own session. System prompt = prompts/compact.md, one
// user message (a leaf, a merge, or a "too long" retry written into one message), the model
// pinned by full id (never the `haiku` alias), the same effort and output cap.

import { LIMIT, RETRIES, shortest } from './tree.js'
import { utf8Len } from './text.js'
import { retryPrompt } from './tree.js'

export const SUMMARIZER = { model: 'claude-haiku-5-5', effort: 'low', maxTokens: 4096 }
export const PROMPT_PATH = 'prompts/compact.md'

export function summaryRequest(prompt, system, settings = SUMMARIZER) {
  return { model: settings.model, system, prompt, effort: settings.effort, maxTokens: settings.maxTokens }
}

// One node: (line, calls). `complete` is $.model.complete; a usage limit or an API error throws.
export async function summarize(complete, prompt, system, settings = SUMMARIZER) {
  const lines = []
  let ask = prompt
  let calls = 0
  for (let attempt = 0; attempt <= RETRIES; attempt++) {
    const r = await complete(summaryRequest(ask, system, settings))
    calls++
    if (!r.isAnswered) {
      const error = new Error('summary call failed: ' + r.reason + (r.status ? ' ' + r.status : '') +
        (r.error ? ' ' + r.error : ''))
      error.isLimit = r.status === 429 || r.error === 'rate_limit'
      throw error
    }
    const line = (r.text ?? '').trim()
    if (line) {
      lines.push(line)
      if (utf8Len(line) <= LIMIT) break
      ask = retryPrompt(prompt, line)
    }
  }
  if (!lines.length) throw new Error('the summarizer returned no text')
  return { line: shortest(lines), calls }
}
