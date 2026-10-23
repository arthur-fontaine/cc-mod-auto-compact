// The decision task, as training/pipeline/task.py builds it: the view and the new message,
// clipped to the same token and byte budgets, and the question spisev-alpha was trained on.

import { clip, estimateTokens, pyJson } from './text.js'

export const QUESTION_ID = 'boundary'
export const CATEGORIES = ['continuation', 'follow_up', 'related', 'independent', 'return']

export const INSTRUCTIONS =
  'conversation is a coding session between a developer and an AI coding agent, as one-line summaries ' +
  '(id+n|summary: the n messages from id on), oldest first; recent lines cover one message, older lines ' +
  "cover more. latest_user_message is the developer's new message. How does it relate to the work in the " +
  'conversation? The answer decides how much of the conversation the agent keeps in its context.'

export const CRITERIA = {
  continuation: 'Carries on the current task directly: answers the agent, approves, corrects, or asks for ' +
    'the next step of the same work.',
  follow_up: 'Builds on the task just done or in progress: a fix, refinement, test or extension that needs ' +
    'its details.',
  related: 'A new task in the same project or area: it needs the general picture (repo, files, decisions), ' +
    'not the details of the recent work.',
  independent: 'Unrelated to the conversation: a new topic or task that needs none of it.',
  return: 'Goes back to an earlier task of the conversation, set aside since, rather than the recent work.',
}

export const MAX_TOKENS = 12_000
export const MAX_MESSAGE_BYTES = 24_000
export const MAX_CONVERSATION_BYTES = 20_000

export function question() {
  return { type: 'choice', instructions: INSTRUCTIONS, criteria: { ...CRITERIA } }
}

const fixedTokens = () => estimateTokens(pyJson({ questions: { [QUESTION_ID]: question() } }))

export function sentTokens(state) {
  return estimateTokens(pyJson({ state, questions: { [QUESTION_ID]: question() } }))
}

// Decision servers refuse a state holding chat or media tokens (`<|im_start|>`): break them.
export function neutralize(text) {
  return text.replace(/<\|([A-Za-z0-9_]+)\|>/g, '< |$1| >')
}

export function clipState(conversation, message) {
  conversation = clip(neutralize(conversation), MAX_CONVERSATION_BYTES)
  message = neutralize(message)
  let room = MAX_TOKENS - fixedTokens() - estimateTokens(pyJson(conversation)) - 16
  const state = { conversation, latest_user_message: clip(message, MAX_MESSAGE_BYTES, Math.max(room, 64)) }
  for (let i = 0; i < 4; i++) {
    const over = sentTokens(state) - MAX_TOKENS
    if (over <= 0) break
    room -= over
    state.latest_user_message = clip(message, MAX_MESSAGE_BYTES, Math.max(room, 64))
  }
  return state
}

// State first, then the fixed question, so a server reuses the cached prefix of earlier requests.
export function request(state, model) {
  return { model, state, questions: { [QUESTION_ID]: question() } }
}
