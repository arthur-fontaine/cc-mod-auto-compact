import { expect, mock, test } from 'claude-code/testing'

import { FIXTURES } from './fixtures/parity.js'

type Probabilities = Record<string, number>

function answer(choice: string, p: number) {
  const probabilities: Probabilities = { continuation: 0, follow_up: 0, related: 0, independent: 0, return: 0 }
  for (const k of Object.keys(probabilities)) probabilities[k] = (1 - p) / 4
  probabilities[choice] = p
  const body = { answers: { boundary: { type: 'choice', choice, probabilities, confidence: p } }, usage: { input_tokens: 900 } }
  return { status: 200, ok: true, headers: {}, text: JSON.stringify(body) }
}

const ROWS = [
  { role: 'user', text: 'Add a --verbose flag to the CLI', toolUses: [] },
  { role: 'assistant', text: 'Added it in src/cli.py and a test.', toolUses: [{ tool: 'Edit', input: { file_path: 'src/cli.py' } }] },
  { role: 'user', text: 'thanks, now the README section', toolUses: [] },
  { role: 'assistant', text: 'Done.', toolUses: [] },
]

const MAMMOTH = {
  AUTO_COMPACT_ENDPOINT: 'https://inference.baseten.co/v1/systemone',
  AUTO_COMPACT_MODEL: 'decision-models/baseten-mammoth-27b-latest',
  AUTO_COMPACT_AUTH: 'Api-Key',
  AUTO_COMPACT_API_KEY: 'test-key',
  AUTO_COMPACT_DATA_DIR: '/data',
  HOME: '/home/u',
}

type Setup = {
  env?: Record<string, string>
  rows?: unknown[]
  fetch?: (e: any) => unknown
  complete?: (e: any) => unknown
}

function stub(on: any, { env = MAMMOTH, rows = ROWS, fetch, complete }: Setup = {}) {
  const seen = {
    requests: [] as any[], compactions: [] as any[], logs: [] as string[], statuses: [] as unknown[],
    labels: [] as unknown[], completions: [] as any[], writes: new Map<string, string>(), prompts: [] as string[],
    clock: null as any, commands: [] as string[], origins: [] as any[], box: '',
  }
  seen.clock = mock.clock(on)
  mock.env(on, env)
  mock.store(on, {})
  on('session.id', () => ({ value: 'session-1' }))
  on('session.messages', () => ({ value: rows }))
  on('fs.read', ($: any, e: any) => {
    if (String(e.path).endsWith('prompts/compact.md')) return { value: FIXTURES.compactPrompt }
    if (seen.writes.has(e.path)) return { value: seen.writes.get(e.path) }
    return { deny: 'ENOENT' }
  })
  on('fs.write', ($: any, e: any) => {
    seen.writes.set(e.path, e.text)
    return { value: undefined }
  })
  on('model.complete', ($: any, e: any) => {
    seen.completions.push(e)
    return complete ? complete(e) : { value: { isAnswered: true, text: 'task: verbose flag [done]', usage: {} } }
  })
  on('http.fetch', ($: any, e: any) => {
    seen.requests.push({ url: e.url, headers: e.init?.headers, body: JSON.parse(e.init?.body ?? '{}') })
    return fetch ? fetch(e) : { value: answer('follow_up', 0.92) }
  })
  on('session.compact', ($: any, e: any) => {
    seen.compactions.push(e)
    return { messages: [{ role: 'user', text: 'summary', toolUses: [] }] }
  })
  on('ui.log', ($: any, e: any) => {
    seen.logs.push(e.text)
    return { value: undefined }
  })
  on('ui.status', ($: any, e: any) => {
    seen.statuses.push(e.text)
    return { value: undefined }
  })
  on('command.run', ($: any, e: any) => {
    seen.commands.push(e.command)
    return { value: { text: '' } }
  })
  on('tool.call', { tool: 'AskUserQuestion' }, () => ({ deny: 'no one to ask (-p)' }))
  const state = new Map<string, { value: unknown; version: number }>()
  on('state.get', ($: any, e: any) => ({ value: state.get(e.plugin + '.' + e.key) ?? { value: undefined, version: 0 } }))
  on('state.set', ($: any, e: any) => {
    const version = (state.get(e.plugin + '.' + e.key)?.version ?? 0) + 1
    state.set(e.plugin + '.' + e.key, { value: e.value, version })
    if (e.key === 'label') seen.labels.push(e.value)
    return { value: { isSet: true, version } }
  })
  on('prompt.submit', ($: any, e: any) => {
    seen.prompts.push(e.text)
    seen.origins.push(e.origin)
    return { text: e.text }
  })
  on('prompt.read', () => ({ value: { text: seen.box, cursor: seen.box.length } }))
  on('prompt.fill', ($: any, e: any) => {
    seen.box = e.text
    return { isFilled: true }
  })
  on('turn.complete', () => ({ text: '' }))
  return seen
}

const submit = ($: any, text: string, origin = 'composer') => $.prompt.submit({ text, wait: false, origin: { kind: origin } })
const decisions = (seen: { logs: string[] }) => seen.logs.filter((l) => l.startsWith('[auto-compact] ') && l.includes(' → '))

test('a follow-up keeps the context and logs the decision', async ($, on) => {
  const seen = stub(on)
  const entered = await submit($, 'and add a changelog entry for it')
  expect(entered.text).toBe('and add a changelog entry for it')
  expect(seen.compactions.length).toBe(0)
  expect(decisions(seen)).toEqual(['[auto-compact] follow_up · 0.92 → keep'])
  expect(seen.labels).toContain('context kept · 92%')
})

test('the request is the state the model was trained on, sent with the configured auth', async ($, on) => {
  const seen = stub(on)
  await submit($, 'what is a monad?')
  const [req] = seen.requests
  expect(req.url).toBe(MAMMOTH.AUTO_COMPACT_ENDPOINT)
  expect(req.headers.Authorization).toBe('Api-Key test-key')
  expect(req.body.model).toBe(MAMMOTH.AUTO_COMPACT_MODEL)
  expect(Object.keys(req.body)).toEqual(['model', 'state', 'questions'])
  expect(req.body.state.latest_user_message).toBe('what is a monad?')
  expect(req.body.state.conversation.split('\n')[0]).toBe('0+1|user: Add a --verbose flag to the CLI')
  expect(req.body.questions.boundary.criteria).toEqual(FIXTURES.request.questions.boundary.criteria)
})

test('in suggest mode nothing happens without the press', async ($, on) => {
  const seen = stub(on, { fetch: () => ({ value: answer('independent', 0.91) }) })
  const entered = await submit($, 'unrelated: write me a haiku about rain')
  expect(entered.text).toBe('unrelated: write me a haiku about rain')
  expect(seen.compactions.length).toBe(0)
  expect(decisions(seen)).toEqual(['[auto-compact] independent · 0.91 → suggested'])
  expect(seen.labels).toContain('clear? independent · 91%')
})

test('auto compacts a related task, passing the view lines as what to keep', async ($, on) => {
  const seen = stub(on, { env: { ...MAMMOTH, AUTO_COMPACT_MODE: 'auto' }, fetch: () => ({ value: answer('related', 0.9) }) })
  const held = await submit($, 'now set up CI for this repo')
  expect(held.drop).toMatch(/compacting the context, then sending this prompt/)
  expect(seen.prompts).toEqual([])
  await seen.clock.advance(1)
  expect(seen.compactions.length).toBe(1)
  expect(seen.compactions[0].instructions).toContain('0+1|user: Add a --verbose flag to the CLI')
  expect(seen.labels).toContain('compacted · related · 90%')
  expect(seen.prompts).toEqual(['now set up CI for this repo'])
})

test('auto clears an independent task with /clear, and the prompt still goes', async ($, on) => {
  const seen = stub(on, { env: { ...MAMMOTH, AUTO_COMPACT_MODE: 'auto' }, fetch: () => ({ value: answer('independent', 0.95) }) })
  const held = await submit($, 'write a poem')
  expect(held.drop).toMatch(/clearing the context/)
  await seen.clock.advance(1)
  expect(seen.commands).toEqual(['clear'])
  expect(seen.compactions.length).toBe(0)
  expect(seen.labels).toContain('cleared · independent · 95%')
  expect(seen.prompts).toEqual(['write a poem'])
})

test('the held prompt goes as the person typed it, and leaves the box', async ($, on) => {
  const seen = stub(on, { env: { ...MAMMOTH, AUTO_COMPACT_MODE: 'auto' }, fetch: () => ({ value: answer('independent', 0.95) }) })
  await submit($, 'write a poem')
  seen.box = 'write a poem' // what the engine puts back on a drop
  await seen.clock.advance(1)
  expect(seen.box).toBe('')
  expect(seen.origins.at(-1)).toMatchObject({ kind: 'plugin', asUser: true })
})

test('a draft typed since the drop is left alone', async ($, on) => {
  const seen = stub(on, { env: { ...MAMMOTH, AUTO_COMPACT_MODE: 'auto' }, fetch: () => ({ value: answer('independent', 0.95) }) })
  await submit($, 'write a poem')
  seen.box = 'something else'
  await seen.clock.advance(1)
  expect(seen.box).toBe('something else')
  expect(seen.prompts).toEqual(['write a poem'])
})

test('auto does not act below the confidence threshold, nor on continuation or follow_up', async ($, on) => {
  const answers = [answer('related', 0.6), answer('continuation', 0.99), answer('follow_up', 0.99)]
  const seen = stub(on, { env: { ...MAMMOTH, AUTO_COMPACT_MODE: 'auto' }, fetch: () => ({ value: answers.shift() }) })
  for (const text of ['a', 'b', 'c']) await submit($, text)
  expect(seen.compactions.length).toBe(0)
  expect(decisions(seen)).toEqual(['[auto-compact] related · 0.60 → keep (low confidence)',
    '[auto-compact] continuation · 0.99 → keep', '[auto-compact] follow_up · 0.99 → keep'])
})

test('with no provider nothing is asked', async ($, on) => {
  const seen = stub(on, { env: { HOME: '/home/u' } })
  await submit($, 'hello there')
  expect(seen.requests.length).toBe(0)
  expect(seen.statuses).toContain('auto-compact: run /auto-compact setup to choose a provider')
})

test('a provider error or timeout keeps the context', async ($, on) => {
  const seen = stub(on, { env: { ...MAMMOTH, AUTO_COMPACT_MODE: 'auto' }, fetch: () => ({ value: { status: 429, ok: false, headers: {}, text: 'slow down' } }) })
  const entered = await submit($, 'next')
  expect(entered.text).toBe('next')
  expect(seen.compactions.length).toBe(0)
  expect(seen.logs.some((l) => l.includes('HTTP 429'))).toBe(true)
})

test("a delivery into a running turn and a subagent's turn are left alone", async ($, on) => {
  const seen = stub(on)
  await $.prompt.submit({ text: 'peer message', wait: false, turnId: 't1', origin: { kind: 'peer' } } as any)
  await $.turn.complete({ turnId: 't2', agentId: 'agent-1', answer: 'x', durationMs: 1, isAborted: false } as any)
  expect(seen.requests.length).toBe(0)
  expect(seen.completions.length).toBe(0)
})

test('turn.complete builds the summaries with the pinned Haiku and saves them; then the prompt is decided', async ($, on) => {
  const rows = [...ROWS.slice(0, 3), { role: 'assistant', text: 'Long reply. '.repeat(100), toolUses: [] }]
  const seen = stub(on, { rows })
  await $.turn.complete({ turnId: 't1', answer: 'x', durationMs: 1, isAborted: false } as any)
  expect(seen.completions.length).toBe(1)
  const call = seen.completions[0]
  expect(call.model).toBe('claude-haiku-5-5')
  expect(call.effort).toBe('low')
  expect(call.system).toBe(FIXTURES.compactPrompt)
  expect(call.prompt).toContain('Compress message 3 into one line.')
  const saved = [...seen.writes.keys()]
  expect(saved).toEqual(['/data/trees/session-1.json'])
  await submit($, 'one more thing')
  expect(decisions(seen).length).toBe(1)
  expect(seen.requests[0].body.state.conversation).toContain('3+1|task: verbose flag [done]')
})

test('with Spisev, turn.complete prefills the state without a message', async ($, on) => {
  const env = { AUTO_COMPACT_PROVIDER: 'spisev', AUTO_COMPACT_DATA_DIR: '/data', HOME: '/home/u' }
  const models = { status: 200, ok: true, headers: {}, text: JSON.stringify({ data: [{ id: 'spisev-alpha' }] }) }
  const seen = stub(on, { env, fetch: (e) => ({ value: String(e.url).endsWith('/v1/models') ? models : answer('follow_up', 0.92) }) })
  await $.turn.complete({ turnId: 't1', answer: 'x', durationMs: 1, isAborted: false } as any)
  await seen.clock.advance(0)
  const asks = () => seen.requests.filter((r) => r.url.endsWith('/v1/systemone'))
  expect(asks().length).toBe(1)
  expect(asks()[0].body.state.latest_user_message).toBe('')
  await submit($, 'one more thing')
  // The decision's state starts with exactly what was prefilled.
  expect(asks()[1].body.state.conversation).toBe(asks()[0].body.state.conversation)
  expect(asks()[1].body.state.latest_user_message).toBe('one more thing')
})

test('a cloud provider is never sent a prefill', async ($, on) => {
  const seen = stub(on)
  await $.turn.complete({ turnId: 't1', answer: 'x', durationMs: 1, isAborted: false } as any)
  await seen.clock.advance(0)
  expect(seen.requests.length).toBe(0)
})

test('a missing summary keeps the context and says summaries are catching up', async ($, on) => {
  const rows = [...ROWS.slice(0, 3), { role: 'assistant', text: 'Long reply. '.repeat(100), toolUses: [] }]
  const seen = stub(on, { rows })
  await submit($, 'next')
  expect(seen.requests.length).toBe(0)
  expect(seen.statuses).toContain('auto-compact: summaries catching up')
})

test('a usage limit while summarizing pauses the summaries', async ($, on) => {
  const rows = [...ROWS.slice(0, 3), { role: 'assistant', text: 'Long reply. '.repeat(100), toolUses: [] }]
  const seen = stub(on, { rows, complete: () => ({ value: { isAnswered: false, reason: 'api-error', status: 429, error: 'rate_limit', usage: {} } }) })
  await $.turn.complete({ turnId: 't1', answer: 'x', durationMs: 1, isAborted: false } as any)
  expect(seen.statuses).toContain('auto-compact: Claude usage limit, summaries paused')
})

const DEBUG_PANE = {
  plugin: 'auto-compact', component: 'Pane', requestId: 'auto-compact-debug',
  props: { title: 'auto-compact debug', isFocused: false, bodyColumns: 60, placement: 'dock', scroll: { offset: 0, bodyRows: 40 }, view: {} },
} as const

test('the debug pane shows the message, the memory sent and the decision, on every surface', async ($, on) => {
  stub(on)
  await submit($, 'and add a changelog entry for it')
  for (const surface of ['terminal', 'desktop', 'vscode', 'mobile'] as const) {
    const ui = await $.ui.mount({ ...DEBUG_PANE, surface })
    expect((await ui.find({ type: 'Text', text: /changelog entry/ }))?.text).toBe('and add a changelog entry for it')
    expect((await ui.find({ type: 'Text', text: /^Memory/ }))?.text).toBe('Memory · 4 lines over 4 messages')
    expect((await ui.find({ type: 'Text', text: /verbose flag to the CLI/ }))?.text).toBe('user: Add a --verbose flag to the CLI')
    expect((await ui.find({ type: 'Text', text: /^raw/ }))?.text).toMatch(/^raw +37 B · 475 left$/)
    expect((await ui.find({ type: 'Text', text: /follow_up ·/ }))?.text).toBe('follow_up · 92%')
    expect((await ui.find({ type: 'Text', text: /^outcome/ }))?.text).toBe('outcome: keep')
    await ui.unmount()
  }
})

test('the debug pane marks a summarized line, with its size and the message it was made from', async ($, on) => {
  const rows = [...ROWS.slice(0, 3), { role: 'assistant', text: 'Long reply. '.repeat(100), toolUses: [] }]
  stub(on, { rows })
  await $.turn.complete({ turnId: 't1', answer: 'x', durationMs: 1, isAborted: false } as any)
  await submit($, 'one more thing')
  const ui = await $.ui.mount({ ...DEBUG_PANE, surface: 'terminal' })
  expect((await ui.find({ type: 'Text', text: /^summary/ }))?.text).toMatch(/^summary +25 B$/)
  expect((await ui.find({ type: 'Text', text: /^↳/ }))?.text).toBe('↳ message')
  // 1206 bytes, clipped in the middle to 1024.
  expect((await ui.find({ type: 'Text', text: /^agent: Long reply/ }))?.text).toContain('\n[… truncated …]\n')
  await ui.unmount()
})

test('the debug pane marks two short messages merged without a call as joined', async ($, on) => {
  stub(on, { rows: Array.from({ length: 9 }, () => ROWS).flat() })
  await submit($, 'one more thing')
  const ui = await $.ui.mount({ ...DEBUG_PANE, surface: 'terminal' })
  expect((await ui.find({ type: 'Text', text: /^joined/ }))?.text).toMatch(/^joined +\d+ B · +\d+ left$/)
  await ui.unmount()
})

test('the debug pane follows a held prompt to its outcome', async ($, on) => {
  const seen = stub(on, { env: { ...MAMMOTH, AUTO_COMPACT_MODE: 'auto' }, fetch: () => ({ value: answer('related', 0.9) }) })
  await submit($, 'now set up CI for this repo')
  const before = await $.ui.mount({ ...DEBUG_PANE, surface: 'terminal' })
  expect((await before.find({ type: 'Text', text: /^outcome/ }))?.text).toBe('outcome: compacting')
  await before.unmount()
  await seen.clock.advance(1)
  const after = await $.ui.mount({ ...DEBUG_PANE, surface: 'terminal' })
  expect((await after.find({ type: 'Text', text: /^outcome/ }))?.text).toBe('outcome: compacted')
  await after.unmount()
})

test('the debug pane says when nothing was sent', async ($, on) => {
  const rows = [...ROWS.slice(0, 3), { role: 'assistant', text: 'Long reply. '.repeat(100), toolUses: [] }]
  const other = stub(on, { rows })
  await submit($, 'next')
  expect(other.requests.length).toBe(0)
  const skipped = await $.ui.mount({ ...DEBUG_PANE, surface: 'terminal' })
  expect((await skipped.find({ type: 'Text', text: /^Not sent/ }))?.text).toBe('Not sent.')
  expect((await skipped.find({ type: 'Text', text: /^outcome/ }))?.text).toMatch(/skipped \(summaries catching up/)
  await skipped.unmount()
})

test('the debug pane says so before any prompt', async ($, on) => {
  stub(on)
  const ui = await $.ui.mount({ ...DEBUG_PANE, surface: 'terminal' })
  expect((await ui.find({ type: 'Text' }))?.text).toMatch(/No prompt looked at yet/)
  await ui.unmount()
})
