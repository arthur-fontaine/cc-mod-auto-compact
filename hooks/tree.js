// The OptChat summary tree (training/pipeline/tree.py, same code): node(l, i) covers the 2^l
// messages from i·2^l on, in at most 512 bytes. The view at time T keeps the last `leaves`
// messages one line each and merges older pairs, most due first, down to `leaves + coarse` lines.

import { utf8Len } from './text.js'

export const LIMIT = 512
export const RETRIES = 5
export const STATE = { leaves: 16, coarse: 16 }
export const CONTEXT = { leaves: 8, coarse: 8 }
export const RULER_NOTE = '| ← LIMIT'

export const key = ([level, index]) => level + ':' + index

export function name([level, index]) {
  return (index * 2 ** level) + '+' + 2 ** level
}

export function span([level, index]) {
  return [index * 2 ** level, (index + 1) * 2 ** level]
}

function moreDue([la, lastA], [lb, lastB], t) {
  return (t - lastA) * 2 ** lb > (t - lastB) * 2 ** la
}

export function view(t, leaves = STATE.leaves, coarse = STATE.coarse) {
  const lines = Array.from({ length: t }, (_, i) => [0, i])
  const tail = Math.max(0, t - leaves)
  while (lines.length > leaves + coarse) {
    let best = null
    for (let k = 0; k < lines.length - 1; k++) {
      const [l1, i1] = lines[k]
      const [l2, i2] = lines[k + 1]
      if (l1 !== l2 || i1 % 2 || i2 !== i1 + 1) continue
      const last = span(lines[k + 1])[1] - 1
      if (last >= tail) continue
      if (best === null || moreDue([l1, last], best.due, t)) best = { k, due: [l1, last] }
    }
    if (best === null) break
    const [level, index] = lines[best.k]
    lines.splice(best.k, 2, [level + 1, Math.floor(index / 2)])
  }
  return lines
}

export function children([level, index]) {
  return level ? [[level - 1, 2 * index], [level - 1, 2 * index + 1]] : []
}

export function contextView(node) {
  return view(span(node)[0], CONTEXT.leaves, CONTEXT.coarse)
}

// Every node the views need, each after what it depends on.
export function needed(views) {
  const seen = new Set()
  const order = []
  const visit = (node) => {
    if (seen.has(key(node))) return
    seen.add(key(node))
    for (const dep of [...children(node), ...contextView(node)]) visit(dep)
    order.push(node)
  }
  for (const v of views) for (const node of v) visit(node)
  return order
}

export function viewText(lines) {
  return lines.map(([node, text]) => name(node) + '|' + text.split(/\s+/).filter(Boolean).join(' ')).join('\n')
}

export function leafPrompt(index, rendered, context) {
  return '<chat>\n' + viewText(context) + '\n</chat>\n' +
    'Compress message ' + index + ' into one line.\n' +
    '<input>\n' + rendered + '\n</input>'
}

export function mergePrompt(a, b, textA, textB, context) {
  return '<chat>\n' + viewText(context) + '\n</chat>\n' +
    'Merge lines ' + name(a) + ' and ' + name(b) + ', adjacent, into one line.\n' +
    '<input>\n' + textA + '\n' + textB + '\n</input>'
}

export function retryPrompt(prompt, line, limit = LIMIT) {
  const bytes = new TextEncoder().encode(line).slice(0, limit)
  const cut = new TextDecoder('utf-8', { fatal: false }).decode(bytes).replace(/�+$/, '')
  return prompt + '\n\nToo long: your line is ' + utf8Len(line) + ' bytes, over the ' + limit + '-byte limit. ' +
    'Write the whole line again for the same <input>, cutting just enough of the least valuable ' +
    'items to fit before this cut:\n' + cut + RULER_NOTE
}

export function noCallText(node, sources, limit = LIMIT) {
  if (node[0] === 0) return utf8Len(sources) <= limit ? sources : null
  const joined = sources[0] + '\n' + sources[1]
  return utf8Len(joined) <= limit ? joined : null
}

export function shortest(lines) {
  return lines.reduce((a, b) => (utf8Len(b) < utf8Len(a) ? b : a))
}

// The v2 memory (memory_view in tree.py): shorter lines, and UniiChat's batched merges so that
// between two batches the memory only grows at its end and stays in the prompt cache.
export const MEMORY_LIMIT = 160
export const MEMORY = { low: 10, high: 16 }
const memories = new Map()

function mergeMostDue(lines, t) {
  let best = null
  for (let k = 0; k < lines.length - 1; k++) {
    const [l1, i1] = lines[k]
    const [l2, i2] = lines[k + 1]
    if (l1 !== l2 || i1 % 2 || i2 !== i1 + 1) continue
    const last = span(lines[k + 1])[1] - 1
    if (best === null || moreDue([l1, last], best.due, t)) best = { k, due: [l1, last] }
  }
  if (best === null) return false
  const [level, index] = lines[best.k]
  lines.splice(best.k, 2, [level + 1, Math.floor(index / 2)])
  return true
}

export function memoryView(t, low = MEMORY.low, high = MEMORY.high) {
  const k = low + ':' + high
  if (!memories.has(k)) memories.set(k, [[]])
  const done = memories.get(k)
  while (done.length <= t) {
    const lines = [...done[done.length - 1], [0, done.length - 1]]
    if (lines.length > high) while (lines.length > low && mergeMostDue(lines, done.length));
    done.push(lines)
  }
  return done[t].map((n) => [...n])
}

export function memoryContext(node, low = MEMORY.low, high = MEMORY.high) {
  return memoryView(span(node)[0], low, high)
}

export function neededMemory(views, low = MEMORY.low, high = MEMORY.high) {
  const seen = new Set()
  const order = []
  const visit = (node) => {
    if (seen.has(key(node))) return
    seen.add(key(node))
    for (const dep of [...children(node), ...memoryContext(node, low, high)]) visit(dep)
    order.push(node)
  }
  for (const v of views) for (const node of v) visit(node)
  return order
}
