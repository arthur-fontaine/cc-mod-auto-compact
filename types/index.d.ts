export type AutoCompactLabel = string | null

declare module 'claude-code' {
  interface PluginState {
    'auto-compact': { label: AutoCompactLabel }
  }
}
