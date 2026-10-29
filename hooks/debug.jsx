// The debug pane: the last prompt the mod looked at, the memory it sent with it, and the decision.

import { CATEGORIES } from './task.js'
import { MEMORY_LIMIT } from './tree.js'

export const PANE = 'auto-compact-debug'
export const TITLE = 'auto-compact debug'

const ACTION_COLOR = { keep: 'success', compact_view: 'warning', compact_branch: 'warning', clear: 'error' }

const pct = (x) => Math.round(x * 100) + '%'

// `raw  37 B · 475 left`: where the line came from, its size, and the room left before the
// summarizer would have to compress it. A summary line is compressed already, and its inputs are
// drawn dimmed under it.
function size(line) {
  if (!line) return ''
  const bytes = line.source.padEnd(7) + String(line.bytes).padStart(4) + ' B'
  return line.source === 'summary' ? bytes : bytes + ' · ' + String(MEMORY_LIMIT - line.bytes).padStart(3) + ' left'
}

function clock(at) {
  const d = new Date(at)
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((n) => String(n).padStart(2, '0')).join(':')
}

function Heading({ Text, children }) {
  return <Text bold color="claude">{children}</Text>
}

export function drawDebug($, e, trace) {
  const { Box, Text } = $.ui.resolve(e)
  if (!trace) {
    return (
      <Box flexDirection="column">
        <Text dimColor>No prompt looked at yet. Send one and it shows here.</Text>
      </Box>
    )
  }
  const columns = e.props.bodyColumns ?? 60
  const memory = trace.memory === null ? [] : trace.memory.split('\n').filter(Boolean).map((line) => {
    const bar = line.indexOf('|')
    return bar < 0 ? ['', line] : [line.slice(0, bar), line.slice(bar + 1)]
  })
  const nameWidth = Math.max(0, ...memory.map(([name]) => name.length))
  const lines = new Map((trace.lines ?? []).map((line) => [line.name, line]))
  const sizeWidth = Math.max(0, ...memory.map(([name]) => size(lines.get(name)).length))
  const d = trace.decision
  const barWidth = Math.max(8, Math.min(30, columns - 26))

  return (
    <Box flexDirection="column" gap={1}>
      <Box flexDirection="column">
        <Box flexDirection="row" justifyContent="space-between">
          <Heading Text={Text}>Message</Heading>
          <Text dimColor>
            {clock(trace.at) + (trace.latencyMs === null ? '' : ' · ' + trace.latencyMs + ' ms')}
          </Text>
        </Box>
        <Text wrap="wrap">{trace.message}</Text>
      </Box>

      <Box flexDirection="column">
        <Heading Text={Text}>
          {'Memory' + (trace.memory === null ? '' : ' · ' + memory.length + ' lines') +
            (trace.messages === null ? '' : ' over ' + trace.messages + ' messages')}
        </Heading>
        {trace.memory === null && <Text dimColor>Not sent.</Text>}
        {memory.map(([name, text]) => (
          <Box flexDirection="column">
            <Box flexDirection="row" gap={1}>
              <Box width={nameWidth} flexShrink={0}>
                <Text dimColor>{name}</Text>
              </Box>
              <Box width={sizeWidth} flexShrink={0}>
                <Text color={lines.get(name)?.source === 'summary' ? 'claude' : 'subtle'}>{size(lines.get(name))}</Text>
              </Box>
              <Box flexGrow={1} flexShrink={1}>
                <Text wrap="wrap">{text}</Text>
              </Box>
            </Box>
            {(lines.get(name)?.from ?? []).map((input) => (
              <Box flexDirection="row" gap={1}>
                <Box width={nameWidth} flexShrink={0} />
                <Box width={sizeWidth} flexShrink={0}>
                  <Text dimColor>{'↳ ' + (input.name ?? 'message')}</Text>
                </Box>
                <Box flexGrow={1} flexShrink={1}>
                  <Text dimColor wrap="wrap">{input.text}</Text>
                </Box>
              </Box>
            ))}
          </Box>
        ))}
      </Box>

      <Box flexDirection="column">
        <Heading Text={Text}>Decision</Heading>
        {d === null && <Text dimColor>None taken.</Text>}
        {d !== null && d.category === null && <Text>No answer from the model.</Text>}
        {d !== null && d.category !== null && (
          <Box flexDirection="row" gap={1}>
            <Text bold>{d.category + ' · ' + pct(d.confidence)}</Text>
            <Text>→</Text>
            <Text color={ACTION_COLOR[d.action] ?? 'text'}>{d.action}</Text>
          </Box>
        )}
        {d !== null && d.category !== null && CATEGORIES.map((category) => {
          const p = Number(d.probabilities?.[category] ?? 0)
          const filled = Math.round(p * barWidth)
          const chosen = category === d.category
          return (
            <Box flexDirection="row" gap={1}>
              <Box width={12} flexShrink={0}>
                <Text bold={chosen} dimColor={!chosen}>{category}</Text>
              </Box>
              <Text color={chosen ? 'claude' : 'subtle'}>{'█'.repeat(filled) + '░'.repeat(barWidth - filled)}</Text>
              <Text dimColor={!chosen}>{pct(p).padStart(4)}</Text>
            </Box>
          )
        })}
        {d !== null && <Text dimColor>{'reason: ' + d.reason}</Text>}
        <Text>{'outcome: ' + trace.outcome}</Text>
      </Box>
    </Box>
  )
}
