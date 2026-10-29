import { expect, test } from 'claude-code/testing'

import { FIXTURES } from './fixtures/parity.js'
import { CATEGORIES, NEVER_ACT, POLICY } from '../hooks/policy.js'
import { summaryRequest, SUMMARIZER, systemPromptV2 } from '../hooks/summarizer.js'
import { agentReply, clipState, clipStateV2, request, requestV2 } from '../hooks/task.js'
import { clip, clipEnd, cleanPrompt, estimateTokens, messages, render } from '../hooks/text.js'
import * as tree from '../hooks/tree.js'

// Each value below was computed by the Python pipeline (training/scripts/make_fixtures.py):
// the mod must clip, group, summarize and ask exactly as training did.

test('clip keeps both ends within the byte and token budgets, as Python does', () => {
  for (const c of FIXTURES.clip) expect(clip(c.text, c.maxBytes, c.maxTokens)).toBe(c.out)
})

test('the token estimate matches', () => {
  for (const c of FIXTURES.tokens) expect(estimateTokens(c.text)).toBe(c.tokens)
})

test('the prompt filter keeps what the user typed and drops the rest', () => {
  for (const c of FIXTURES.prompts) expect(cleanPrompt(c.text)).toBe(c.out)
})

test('transcript rows group into the same messages and render the same', () => {
  const msgs = messages(FIXTURES.rows)
  expect(msgs).toEqual(FIXTURES.messages)
  expect(msgs.map(render)).toEqual(FIXTURES.rendered)
})

test('views and the nodes they need are the same', () => {
  for (const v of FIXTURES.views) expect(tree.view(v.t).map(tree.name)).toEqual(v.lines)
  for (const v of FIXTURES.contextViews) {
    expect(tree.view(v.t, tree.CONTEXT.leaves, tree.CONTEXT.coarse).map(tree.name)).toEqual(v.lines)
  }
  for (const n of FIXTURES.needed) expect(tree.needed([tree.view(n.t)]).map(tree.name)).toEqual(n.order)
})

test('summary requests are built like the Python summarizer builds them', () => {
  const msgs = messages(FIXTURES.rows)
  const context = [[[0, 0], 'user: Add a --verbose flag'], [[0, 1], 'task: verbose flag [in-progress] reads src/cli.py']]
  const leaf = tree.leafPrompt(2, render(msgs[0]), context)
  expect(leaf).toBe(FIXTURES.summaryPrompts.leaf)
  expect(tree.mergePrompt([1, 2], [1, 3], 'task: a [done]', 'task: b [started]', context)).toBe(FIXTURES.summaryPrompts.merge)
  expect(tree.retryPrompt(leaf, 'é'.repeat(300) + 'z'.repeat(30))).toBe(FIXTURES.summaryPrompts.retry)
  expect(SUMMARIZER).toEqual(FIXTURES.summarizer)
  expect(summaryRequest(leaf, FIXTURES.compactPrompt)).toEqual(FIXTURES.summaryRequest)
})

test('states are clipped to the same budgets and the request is the same', () => {
  for (const s of FIXTURES.states) expect(clipState(s.conversation, s.message)).toEqual(s.state)
  const conversation = FIXTURES.states[0].conversation
  expect(request(clipState(conversation, 'now the README'), 'spisev-alpha')).toEqual(FIXTURES.request)
})

test('the policy table is policy.py', () => {
  expect(POLICY).toEqual(FIXTURES.policy)
  expect([...NEVER_ACT].sort()).toEqual(FIXTURES.neverAct)
  expect(CATEGORIES).toEqual(FIXTURES.categories)
})

test('v2: the batched memory views and the nodes they need are the same', () => {
  for (const v of FIXTURES.v2.memoryViews) expect(tree.memoryView(v.t).map(tree.name)).toEqual(v.lines)
  for (const n of FIXTURES.v2.neededMemory) expect(tree.neededMemory([tree.memoryView(n.t)]).map(tree.name)).toEqual(n.order)
})

test('v2: the agent\'s last reply is filtered and clipped the same, and the state is the same', () => {
  for (const c of FIXTURES.v2.clipEnd) expect(clipEnd(c.text, c.maxBytes)).toBe(c.out)
  for (const r of FIXTURES.v2.replies) expect(agentReply(r.message)).toBe(r.out)
  for (const s of FIXTURES.v2.states) expect(clipStateV2(s.memory, s.last, s.message)).toEqual(s.state)
  const conversation = FIXTURES.states[0].conversation
  expect(requestV2(clipStateV2(conversation, 'Done.', 'now the README'), 'spisev-alpha')).toEqual(FIXTURES.v2.request)
})

test('v2: the summary prompt and its retry are the same', () => {
  expect(systemPromptV2(FIXTURES.v2.promptTemplate, 160)).toBe(FIXTURES.v2.prompt160)
  expect(tree.retryPrompt(FIXTURES.v2.leaf, 'é'.repeat(100) + 'z'.repeat(30), 160)).toBe(FIXTURES.v2.retry)
  for (const c of FIXTURES.v2.cleanLines) expect(tree.cleanLineV2(c.text)).toBe(c.out)
})
