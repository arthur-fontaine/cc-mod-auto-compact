import { atom, read, update } from 'claude-code'

import { PANE, TITLE, drawDebug } from './debug.jsx'
import { parseBuild, parseServerOutput, serverArgs, serverCommand, servesOurModel } from './llama.js'
import {
  SPISEV, compactInstructions, decide, describe, headers, logLine, MODES, PRESETS, resolveConfig,
} from './policy.js'
import { PROMPT_PATH, SUMMARIZER, summarize } from './summarizer.js'
import { clipState, request } from './task.js'
import { clip, cleanPrompt, messages, render, utf8Len } from './text.js'
import * as tree from './tree.js'

const USER_ORIGINS = ['composer', 'bridge', 'sdk']
const SUMMARY_CONCURRENCY = 4
const BUILD_WAIT_MS = 120_000
// What the debug pane keeps of a summary's input: a message can run to MAX_AGENT_BYTES.
const INPUT_BYTES = 1024

// The footer label, kept in $.state so a reload keeps it.
const label = atom({ plugin: 'auto-compact', key: 'label' }, null)
// What the debug pane draws. Bump the shape when AutoCompactTrace changes, so a reload drops the old one.
const trace = atom({ plugin: 'auto-compact', key: 'trace' }, null, { shape: 'trace-2' })

let systemPrompt = null
let nodes = null // summary line by hash of its request, for this session
let nodesPath = null
let building = null
let lastDecision = null
let lastError = null
let clearProblem = null
let warned = false
let server = null // { state, stream }
const resending = new Set() // prompts held back while compacting, sent again as they were

function show($, text) {
  return update($, label, () => text).catch(() => {})
}

function record($, fn) {
  return update($, trace, fn).catch(() => {})
}

function debug($, text) {
  $.ui.log(text, { to: 'debug' })
}

// FNV-1a over the request: a stable key for a node, whatever position it has in the transcript.
function hash(text) {
  let a = 0x811c9dc5
  let b = 0x01000193
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i)
    a = Math.imul(a ^ c, 0x01000193) >>> 0
    b = Math.imul(b ^ c, 0x5bd1e995) >>> 0
  }
  return a.toString(16).padStart(8, '0') + b.toString(16).padStart(8, '0')
}

async function storedConfig($) {
  try {
    const saved = await $.store.get('config')
    return saved && typeof saved === 'object' ? saved : {}
  } catch {
    return {}
  }
}

async function loadConfig($) {
  const env = {
    provider: await $.env.get('AUTO_COMPACT_PROVIDER'),
    endpoint: await $.env.get('AUTO_COMPACT_ENDPOINT'),
    model: await $.env.get('AUTO_COMPACT_MODEL'),
    auth: await $.env.get('AUTO_COMPACT_AUTH'),
    apiKey: await $.env.get('AUTO_COMPACT_API_KEY'),
    mode: await $.env.get('AUTO_COMPACT_MODE'),
    minConfidence: await $.env.get('AUTO_COMPACT_MIN_CONFIDENCE'),
    suggestConfidence: await $.env.get('AUTO_COMPACT_SUGGEST_CONFIDENCE'),
    timeoutMs: await $.env.get('AUTO_COMPACT_TIMEOUT_MS'),
    llamaServer: await $.env.get('AUTO_COMPACT_LLAMA_SERVER'),
  }
  return resolveConfig(env, await storedConfig($))
}

async function saveConfig($, patch) {
  try {
    await $.store.set('config', { ...(await storedConfig($)), ...patch })
    return true
  } catch {
    return false
  }
}

async function dataDir($) {
  const own = await $.env.get('AUTO_COMPACT_DATA_DIR')
  if (own) return own
  const plugin = await $.env.get('CLAUDE_PLUGIN_DATA')
  if (plugin) return plugin
  return (await $.env.get('HOME')) + '/.claude/plugins/data/auto-compact'
}

async function system($) {
  if (systemPrompt === null) systemPrompt = await $.fs.read($.plugin.root + '/' + PROMPT_PATH)
  return systemPrompt
}

async function summarizerSettings($) {
  return { ...SUMMARIZER, model: (await $.env.get('AUTO_COMPACT_SUMMARIZER_MODEL')) || SUMMARIZER.model }
}

// The session's summary lines, saved per session id with $.fs (not $.store: 4 MiB, shared).
async function loadNodes($) {
  const path = (await dataDir($)) + '/trees/' + (await $.session.id()) + '.json'
  if (nodes && nodesPath === path) return nodes
  nodesPath = path
  try {
    nodes = JSON.parse(await $.fs.read(path)).nodes ?? {}
  } catch {
    nodes = {}
  }
  return nodes
}

async function saveNodes($) {
  if (!nodesPath) return
  try {
    await $.fs.write(nodesPath, JSON.stringify({ version: 1, summarizer: SUMMARIZER.model, nodes }))
  } catch (error) {
    debug($, '[auto-compact] could not save the tree: ' + error.message)
  }
}

class Missing extends Error {}

// The text of every node the view at msgs.length needs, with where it came from: `raw` (the message,
// under tree.LIMIT), `joined` (both halves, under tree.LIMIT together) or `summary` (the summarizer,
// from the inputs given: the message, or both halves by name).
// With `allowCalls`, missing summaries are built with $.model.complete, up to SUMMARY_CONCURRENCY at
// once; without, a missing one throws.
async function treeTexts($, msgs, { allowCalls }) {
  const store = await loadNodes($)
  const sys = allowCalls ? await system($) : null
  const settings = await summarizerSettings($)
  const memo = new Map()
  const sources = new Map()
  const inputs = new Map()
  let active = 0
  const waiting = []
  const slot = async () => {
    if (active < SUMMARY_CONCURRENCY) return void active++
    await new Promise((resolve) => waiting.push(resolve))
    active++
  }
  const release = () => {
    active--
    waiting.shift()?.()
  }
  const text = (node) => {
    const k = tree.key(node)
    if (!memo.has(k)) memo.set(k, build(node))
    return memo.get(k)
  }
  const build = async (node) => {
    let source
    if (node[0] === 0) source = render(msgs[node[1]])
    else source = await Promise.all(tree.children(node).map(text))
    const free = tree.noCallText(node, source)
    sources.set(tree.key(node), free === null ? 'summary' : node[0] === 0 ? 'raw' : 'joined')
    if (free !== null) return free
    inputs.set(tree.key(node), node[0] === 0
      ? [{ name: null, text: source }]
      : tree.children(node).map((child, i) => ({ name: tree.name(child), text: source[i] })))
    const context = await Promise.all(tree.contextView(node).map(async (n) => [n, await text(n)]))
    const prompt = node[0] === 0
      ? tree.leafPrompt(node[1], source, context)
      : tree.mergePrompt(...tree.children(node), source[0], source[1], context)
    const id = hash(settings.model + '\n' + prompt)
    if (store[id]) return store[id]
    if (!allowCalls) throw new Missing(tree.name(node))
    await slot()
    try {
      const { line } = await summarize((r) => $.model.complete(r), prompt, sys, settings)
      store[id] = line
      return line
    } finally {
      release()
    }
  }
  const lines = tree.view(msgs.length)
  const order = tree.needed([lines])
  const results = await Promise.allSettled(order.map(text))
  const failed = results.find((r) => r.status === 'rejected')
  if (failed) throw failed.reason
  return Promise.all(lines.map(async (n) => [n, await text(n), sources.get(tree.key(n)), inputs.get(tree.key(n))]))
}

async function sessionMessages($) {
  return messages(await $.session.messages())
}

async function buildTree($) {
  const msgs = await sessionMessages($)
  try {
    await treeTexts($, msgs, { allowCalls: true })
    lastError = null
  } catch (error) {
    lastError = error.message
    debug($, '[auto-compact] summaries: ' + error.message)
    if (error.isLimit) $.ui.status('auto-compact: Claude usage limit, summaries paused')
  } finally {
    await saveNodes($)
  }
}

async function llamaBuild($, binary) {
  try {
    const { stdout, stderr } = await $.process.run([binary, '--version'], { timeoutMs: 10_000 })
    return parseBuild(stdout + stderr)
  } catch {
    return null
  }
}

async function findServer($, config) {
  for (const binary of config.llamaServer ? [config.llamaServer] : SPISEV.servers) {
    const build = await llamaBuild($, binary)
    if (build !== null) return { binary, build }
  }
  return { binary: config.llamaServer ?? SPISEV.servers[0], build: null }
}

async function isServing($, config) {
  try {
    const r = await $.http.fetch('http://127.0.0.1:' + config.port + '/v1/models')
    return r.ok && servesOurModel(r.text)
  } catch {
    return false
  }
}

// Starts llama-server for the rest of the session unless our model is already served; never waits.
async function startSpisev($, config) {
  if (server && server.state !== 'stopped') return server.state
  if (await isServing($, config)) {
    server = { state: 'ready', stream: null }
    return 'ready'
  }
  const { binary } = await findServer($, config)
  const current = { state: 'starting', stream: null }
  server = current
  void show($, 'Spisev starting · the first start downloads ' + SPISEV.sizeLabel)
  void (async () => {
    try {
      current.stream = $.process.spawn({ argv: serverArgs(config, binary) })
      for await (const { text } of current.stream) {
        const seen = parseServerOutput(text ?? '')
        if (seen.ready && current.state !== 'ready') {
          current.state = 'ready'
          void show($, 'Spisev ready')
        } else if (seen.percent !== undefined && current.state !== 'ready') {
          current.state = 'downloading'
          void show($, 'Spisev downloading · ' + seen.percent + '%')
        }
      }
    } catch (error) {
      debug($, '[auto-compact] llama-server failed to start: ' + error.message)
      $.ui.status('auto-compact: llama-server failed to start, see the debug log')
    }
    current.state = 'stopped'
  })()
  return current.state
}

// The first prompt of a session usually comes while llama-server loads the model (a few seconds):
// wait for it a little rather than skip the decision. A download is never waited for.
const START_WAIT_MS = 10_000

async function spisevReady($, config) {
  let state = await startSpisev($, config)
  for (let waited = 0; state === 'starting' && waited < START_WAIT_MS; waited += 250) {
    await $.clock.sleep(250)
    state = server?.state ?? 'stopped'
    if (state === 'starting' && (await isServing($, config))) state = server.state = 'ready'
  }
  return state
}

async function askProvider($, config, state) {
  const model = config.provider === 'spisev' ? SPISEV.alias : config.model
  const call = $.http.fetch(config.endpoint, {
    method: 'POST', headers: headers(config), body: JSON.stringify(request(state, model)),
  })
  const timeout = $.clock.sleep(config.timeoutMs).then(() => null)
  const response = await Promise.race([call, timeout])
  if (response === null) throw new Error('timed out after ' + config.timeoutMs + 'ms')
  if (!response.ok) throw new Error('HTTP ' + response.status + ' ' + response.text.slice(0, 200))
  return JSON.parse(response.text)
}

async function perform($, decision, view, prompt) {
  if (decision.action === 'clear') {
    // The engine's own /clear, queued for once the session is idle; the prompt is sent after it.
    await $.command.run({ command: 'clear' })
    return 'cleared'
  }
  const result = await $.session.compact({ instructions: compactInstructions(decision, view, prompt) })
  if (result?.skip) throw new Error(result.skip)
  return 'compacted'
}

async function offer($, decision) {
  const verb = decision.action === 'clear' ? 'Clear the context first' : 'Compact the context first'
  const question = 'auto-compact: this prompt looks ' + decision.category.replace('_', ' ') + ' (' +
    Math.round(decision.confidence * 100) + '%). ' + verb + '?'
  try {
    const choice = await $.ui.ask(question, { header: 'auto-compact', options: [verb, 'Send as is'] })
    return choice === verb
  } catch {
    return false // dismissed, or no one to ask (a -p run): nothing happens without the press
  }
}

async function onPrompt($, e) {
  const config = await loadConfig($)
  if (config.mode === 'off') return
  if (!config.provider || config.missing.length || config.problems.length) {
    if (!warned) {
      warned = true
      $.ui.status(config.provider
        ? 'auto-compact: ' + [...config.missing.map((m) => 'set ' + m), ...config.problems].join('; ')
        : 'auto-compact: run /auto-compact setup to choose a provider')
    }
    return
  }
  if (building) await Promise.race([building, $.clock.sleep(2000)])
  const msgs = await sessionMessages($)
  const prompt = cleanPrompt(e.text) ?? e.text
  const seen = { at: await $.clock.now(), message: prompt, memory: null, lines: null, messages: msgs.length,
    decision: null, latencyMs: null }
  const skip = (outcome) => record($, () => ({ ...seen, outcome }))
  if (msgs.length < 2) return void (await skip('skipped (fewer than 2 messages)'))
  let view
  try {
    view = await treeTexts($, msgs, { allowCalls: false })
  } catch (error) {
    if (!(error instanceof Missing)) throw error
    $.ui.status('auto-compact: summaries catching up')
    debug($, '[auto-compact] summaries catching up (' + error.message + ' not built) → keep')
    return void (await skip('skipped (summaries catching up: ' + error.message + ' not built)'))
  }
  if (config.provider === 'spisev' && (await spisevReady($, config)) !== 'ready') {
    debug($, '[auto-compact] Spisev ' + server.state + ' → keep')
    return void (await skip('skipped (Spisev ' + server.state + ')'))
  }
  const viewText = tree.viewText(view)
  const state = clipState(viewText, prompt)
  seen.message = state.latest_user_message
  seen.memory = state.conversation
  seen.lines = view.map(([node, text, source, from]) => ({
    name: tree.name(node), source, bytes: utf8Len(text),
    from: from ? from.map((input) => ({ ...input, text: clip(input.text, INPUT_BYTES) })) : null,
  }))
  let decision
  const asked = await $.clock.now()
  try {
    decision = decide(await askProvider($, config, state), config)
  } catch (error) {
    lastError = error.message
    debug($, '[auto-compact] provider: ' + error.message + ' → keep')
    seen.latencyMs = (await $.clock.now()) - asked
    return void (await skip('keep (provider: ' + error.message + ')'))
  }
  seen.latencyMs = (await $.clock.now()) - asked
  seen.decision = {
    category: decision.category, confidence: decision.confidence, action: decision.action,
    reason: decision.reason, probabilities: decision.probabilities ?? {},
  }
  let outcome = decision.action === 'keep' ? 'keep' : decision.reason === 'suggest' ? 'suggested' : 'keep (' + decision.reason + ')'
  const done = false
  await skip(decision.suggest ? 'waiting for your answer' : decision.act ? 'acting' : outcome)
  const wants = decision.act || (decision.suggest && (await offer($, decision)))
  lastDecision = { ...decision, outcome }
  if (wants && e.attachments?.length) {
    outcome = 'keep (a prompt with attachments is never held back: compact by hand)'
  } else if (wants) {
    // $.session.compact can't run under the prompt.submit hook, which holds the turn: the prompt is
    // held back, the context compacted or cleared once the hook returns, and the same prompt sent again.
    debug($, logLine(decision, decision.action === 'clear' ? 'clear' : 'compact'))
    await skip(decision.action === 'clear' ? 'clearing' : 'compacting')
    await show($, describe(decision))
    return { decision, viewText, prompt }
  }
  lastDecision = { ...decision, outcome }
  await skip(outcome)
  debug($, logLine(decision, outcome))
  $.ui.status(undefined)
  await show($, describe(decision, done))
  return null
}

// The engine puts a dropped prompt back in the box. The mod sends it itself, so the box is emptied,
// but only while it still holds that prompt: what the person typed since stays.
async function clearDraft($, text) {
  try {
    const box = await $.prompt.read()
    if (box.text.trim() === text.trim()) await $.prompt.fill({ text: '', mode: 'replace' })
  } catch {
    // No box to clear (headless).
  }
}

function actThenResend($, plan, text) {
  $.clock.after(0, async () => {
    const { decision, viewText, prompt } = plan
    await clearDraft($, text)
    let outcome
    try {
      outcome = await perform($, decision, viewText, prompt)
      await show($, describe(decision, true))
    } catch (error) {
      if (decision.action === 'clear') clearProblem = error.message
      outcome = 'keep (could not ' + (decision.action === 'clear' ? 'clear' : 'compact') + ': ' + error.message + ')'
      await show($, describe(decision))
    }
    lastDecision = { ...decision, outcome }
    await record($, (t) => t && { ...t, outcome })
    debug($, '[auto-compact] ' + outcome + ', sending the prompt')
    await clearDraft($, text)
    resending.add(text)
    // asUser: the model reads the person's prompt as they typed it, not as a message from the plugin.
    await $.prompt.submit({ text, asUser: true })
  })
}

async function setup($) {
  let choice
  try {
    choice = await $.ui.ask('Which decision model should spot task boundaries?', {
      header: 'Provider', options: ['Spisev (local)', ...Object.keys(PRESETS)],
    })
  } catch {
    return { text: 'Setup cancelled; nothing changed.' }
  }
  if (choice === 'Spisev (local)') {
    const config = resolveConfig({ provider: 'spisev', llamaServer: await $.env.get('AUTO_COMPACT_LLAMA_SERVER'),
      model: await $.env.get('AUTO_COMPACT_MODEL') }, {})
    const { binary, build } = await findServer($, config)
    if (build === null || build < SPISEV.minBuild) {
      return { text: 'llama.cpp build ' + SPISEV.minBuild + ' or later is needed (llama-server or llama on your PATH, ' +
        'or AUTO_COMPACT_LLAMA_SERVER). Install it from https://github.com/ggml-org/llama.cpp/releases and run /auto-compact setup again.' }
    }
    if (!(await isServing($, config)) && !/\.gguf$/i.test(config.model)) {
      let go
      try {
        go = await $.ui.ask('Download ' + config.model + ' (' + SPISEV.sizeLabel + ') and run it with llama.cpp?', {
          header: 'Download', options: ['Download and start', 'Cancel'],
        })
      } catch {
        go = null
      }
      if (go !== 'Download and start') return { text: 'Setup cancelled; nothing changed.' }
    }
    await saveConfig($, { provider: 'spisev', endpoint: config.endpoint, model: config.model, llamaServer: binary })
    await startSpisev($, { ...config, llamaServer: binary })
    return { text: 'Saved: Spisev (spisev-alpha), served by ' + serverCommand(binary).join(' ') + ' (build ' + build + ') at ' +
      config.endpoint + '. llama-server runs while this session does. Nothing leaves this machine.' }
  }
  const preset = PRESETS[choice]
  if (!preset) return { text: 'Under Other, use AUTO_COMPACT_ENDPOINT and AUTO_COMPACT_MODEL for any System One endpoint.' }
  await saveConfig($, { provider: 'systemone', ...preset })
  const key = await $.env.get('AUTO_COMPACT_API_KEY')
  return { text: 'Saved: ' + choice + ' at ' + preset.endpoint + ', model ' + preset.model + '. ' +
    (key ? 'AUTO_COMPACT_API_KEY is set.' : 'Now set AUTO_COMPACT_API_KEY in your environment; the mod never stores it.') +
    ' Only the clipped state (summary lines and the new message) is sent.' }
}

async function status($) {
  const config = await loadConfig($)
  const lines = ['auto-compact mode: ' + config.mode + (config.mode === 'auto' ? ' (acts at ≥ ' + config.minConfidence + ')' : config.mode === 'suggest' ? ' (asks at ≥ ' + config.suggestConfidence + ')' : '')]
  if (config.provider === 'spisev') lines.push('Provider: Spisev (spisev-alpha) ' + config.model + ' at ' + config.endpoint + ' (' + (server?.state ?? 'stopped') + ')')
  else if (config.provider) lines.push('Provider: ' + config.endpoint + ' · model ' + config.model + ' · key ' + (config.apiKey ? 'set' : 'unset'))
  else lines.push('Provider: none, run /auto-compact setup')
  for (const p of [...config.missing.map((m) => 'set ' + m), ...config.problems]) lines.push('Problem: ' + p)
  let tree_ = 'not built yet'
  try {
    const msgs = await sessionMessages($)
    await treeTexts($, msgs, { allowCalls: false })
    tree_ = msgs.length + ' messages, ' + Object.keys(await loadNodes($)).length + ' summaries, up to date'
  } catch (error) {
    tree_ = 'catching up (' + error.message + ')'
  }
  lines.push('Tree: ' + tree_ + ' · summarizer ' + (await summarizerSettings($)).model)
  lines.push('Last decision: ' + (lastDecision ? describe(lastDecision, /ed$/.test(lastDecision.outcome)) + ' → ' + lastDecision.outcome : 'none yet'))
  if (clearProblem) lines.push('Clear falls back to suggest: ' + clearProblem)
  if (lastError) lines.push('Last error: ' + lastError)
  return { text: lines.join('\n') }
}

export function register(on) {
  on('session.start', async ($, e, next) => {
    try {
      await $.command.register({
        name: 'auto-compact', description: 'Show or set up the task-boundary classifier that offers to compact',
        argumentHint: '[setup|debug|mode off|suggest|auto]', immediate: true,
      })
    } catch (error) {
      debug($, '[auto-compact] could not register /auto-compact: ' + error.message)
    }
    const result = await next(e)
    const config = await loadConfig($)
    if (config.mode !== 'off' && config.provider === 'spisev' && !config.problems.length) await startSpisev($, config)
    return result
  })

  on('prompt.submit', async ($, e, next) => {
    // A delivery into a running turn is no new request; slash commands and hidden prompts aren't either.
    const intoRunningTurn = e.turnId && !USER_ORIGINS.includes(e.origin?.kind)
    if (intoRunningTurn || e.isMeta || e.command || !e.text.trim()) return next(e)
    if (resending.delete(e.text)) return next(e)
    try {
      const plan = await onPrompt($, e)
      if (plan) {
        actThenResend($, plan, e.text)
        return { drop: 'auto-compact: ' + (plan.decision.action === 'clear' ? 'clearing' : 'compacting') +
          ' the context, then sending this prompt' }
      }
    } catch (error) {
      debug($, '[auto-compact] ' + error.message + ' → keep')
    }
    return next(e)
  })

  on('turn.complete', async ($, e, next) => {
    const result = await next(e)
    if (e.agentId) return result
    const config = await loadConfig($)
    if (config.mode === 'off' || !config.provider) return result
    // Awaited (bounded) so a -p run saves the tree before it exits.
    building = buildTree($).finally(() => {
      building = null
    })
    await Promise.race([building, $.clock.sleep(BUILD_WAIT_MS)])
    return result
  })

  on('session.end', async ($, e, next) => {
    await saveNodes($)
    nodes = null // a /clear starts another session id, with a tree of its own
    // A /clear ends the session but not the process: llama-server keeps serving the next one.
    if (e.reason !== 'clear') {
      try {
        await server?.stream?.return?.()
      } catch {
        // Already gone.
      }
    }
    return next(e)
  })

  on('ui.render', { component: 'SessionMode' }, async ($, e, next) => {
    const text = await read($, label)
    return text ? next({ ...e, props: { ...e.props, modes: [...e.props.modes, text] } }) : next(e)
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => drawDebug($, e, await read($, trace)))

  on('command.run', { command: 'auto-compact' }, async ($, e) => {
    const [arg, value] = e.args.trim().toLowerCase().split(/\s+/)
    if (arg === 'setup') return setup($)
    if (arg === 'debug') {
      const opened = await $.ui.open({ id: PANE, title: TITLE })
      return { text: opened.isPlaced ? 'Debug pane opened.' : 'Debug pane waiting: ' + opened.reason }
    }
    if (arg === 'mode') {
      if (!MODES.includes(value)) return { text: 'Usage: /auto-compact mode off|suggest|auto' }
      const saved = await saveConfig($, { mode: value })
      await show($, value === 'off' ? 'auto-compact off' : null)
      return { text: 'auto-compact mode: ' + value + (saved ? ' (saved for later sessions).' : ' (this session only).') }
    }
    if (arg) return { text: 'Usage: /auto-compact [setup|debug|mode off|suggest|auto]' }
    return status($)
  })
}
