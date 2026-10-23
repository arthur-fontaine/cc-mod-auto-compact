export type SpisevLabel = string | null

declare module 'claude-code' {
  interface PluginState {
    spisev: { label: SpisevLabel }
  }
}
