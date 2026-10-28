export type AutoCompactLabel = string | null

// What the debug pane shows of the last prompt the mod looked at.
export type AutoCompactTrace = {
  at: number
  // As sent to the decision model (clipped), or the cleaned prompt when nothing was sent.
  message: string
  // The summary lines sent as `conversation`, one per line; null when nothing was sent.
  memory: string | null
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
