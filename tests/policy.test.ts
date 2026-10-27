import { expect, test } from 'claude-code/testing'

import { action, decide, describe, headers, logLine, NEVER_ACT, resolveConfig, SPISEV } from '../hooks/policy.js'

const response = (choice: string, p: number) => ({ answers: { boundary: { type: 'choice', choice, probabilities: { [choice]: p } } } })

test('there is no default provider', () => {
  const config = resolveConfig({})
  expect(config.provider).toBeUndefined()
  expect(config.mode).toBe('suggest')
  expect(config.minConfidence).toBe(0.85)
  expect(config.suggestConfidence).toBe(0.6)
})

test('a cloud endpoint needs its model and key, and the key is never read from the store', () => {
  expect(resolveConfig({ endpoint: 'https://x/v1/systemone' }).missing).toEqual(['AUTO_COMPACT_MODEL', 'AUTO_COMPACT_API_KEY'])
  const config = resolveConfig({}, { provider: 'systemone', endpoint: 'https://x', model: 'm', apiKey: 'stored' })
  expect(config.apiKey).toBeUndefined()
  expect(config.missing).toEqual(['AUTO_COMPACT_API_KEY'])
})

test('Spisev runs locally with no key, on a local port', () => {
  const config = resolveConfig({ provider: 'spisev' })
  expect(config.endpoint).toBe(SPISEV.endpoint)
  expect(config.port).toBe(8766)
  expect(config.missing).toEqual([])
  expect(resolveConfig({ provider: 'spisev', endpoint: 'https://example.com/v1/systemone' }).problems.length).toBe(1)
})

test('the saved mode applies, the environment wins', () => {
  expect(resolveConfig({}, { mode: 'auto' }).mode).toBe('auto')
  expect(resolveConfig({ mode: 'off' }, { mode: 'auto' }).mode).toBe('off')
  expect(resolveConfig({ mode: 'nonsense' }).mode).toBe('suggest')
})

test('Baseten keys go in an Api-Key header, others as a bearer token', () => {
  expect(headers({ apiKey: 'k', auth: 'Api-Key' } as any).Authorization).toBe('Api-Key k')
  expect(headers({ apiKey: 'k', auth: 'Bearer' } as any).Authorization).toBe('Bearer k')
  expect(headers({} as any).Authorization).toBeUndefined()
})

test('continuation and follow_up never act; auto needs the threshold; suggest offers', () => {
  for (const c of NEVER_ACT) expect(action(c)).toBe('keep')
  const auto = { mode: 'auto', minConfidence: 0.85 } as any
  expect(decide(response('follow_up', 0.99), auto).act).toBe(false)
  expect(decide(response('independent', 0.9), auto)).toMatchObject({ act: true, action: 'clear' })
  expect(decide(response('related', 0.8), auto)).toMatchObject({ act: false, reason: 'low confidence' })
  expect(decide(response('return', 0.6), { ...auto, mode: 'suggest' })).toMatchObject({ suggest: true, action: 'compact_branch' })
  expect(decide({}, auto)).toMatchObject({ action: 'keep', reason: 'no answer' })
})

test('suggest mode does not ask below its own threshold', () => {
  const suggest = resolveConfig({ mode: 'suggest' })
  expect(decide(response('independent', 0.4), suggest)).toMatchObject({ suggest: false, act: false, reason: 'low confidence' })
  expect(decide(response('independent', 0.6), suggest)).toMatchObject({ suggest: true, action: 'clear' })
  expect(decide(response('related', 0.5), resolveConfig({ mode: 'suggest', suggestConfidence: '0.3' })).suggest).toBe(true)
})

test('footer and debug lines', () => {
  const kept = decide(response('follow_up', 0.92), { mode: 'suggest' } as any)
  expect(describe(kept)).toBe('context kept · 92%')
  const related = decide(response('related', 0.71), { mode: 'suggest' } as any)
  expect(describe(related)).toBe('compact? related · 71%')
  const clear = decide(response('independent', 0.91), { mode: 'auto', minConfidence: 0.85 } as any)
  expect(describe(clear, true)).toBe('cleared · independent · 91%')
  expect(logLine(clear, 'cleared')).toBe('[auto-compact] independent · 0.91 → cleared')
})
