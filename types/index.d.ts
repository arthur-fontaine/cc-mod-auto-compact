export type AutoCompactLabel = string | null

// What the debug pane shows of the last prompt the mod looked at.
export type AutoCompactTrace = {
  at: number
  // As sent to the decision model (clipped), or the cleaned prompt when nothing was sent.
  message: string
  // The summary lines sent as `conversation`, one per line; null when nothing was sent.
  memory: string | null
  // Each line of the view before clipping: its text verbatim, both halves joined, or a summary, with
  // what the summary was made from (the message, name null, or both halves by name; clipped).
  lines: {
    name: string
    source: 'raw' | 'joined' | 'summary'
    bytes: number
    from: { name: string | null; text: string }[] | null
  }[] | null
  messages: number | null
  decision: {
    category: string | null
    confidence: number | null
    action: string
    reason: string
    probabilities: Record<string, number>
  } | null
  outcome: string
  latencyMs: number | null
}

declare module 'claude-code' {
  interface PluginState {
    'auto-compact': { label: AutoCompactLabel; trace: Shaped<AutoCompactTrace | null> }
  }
}
