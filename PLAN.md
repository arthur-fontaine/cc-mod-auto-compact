# spisev v2 — faster and more accurate decisions

Plan of record for the next version. Claude follows it step by step: one phase at a time, each
gate checked and its numbers written in the [log](#log) before the next phase starts. A failed
gate stops the plan: report, do not loosen the gate, do not skip ahead. Changing this plan takes
Arthur's go.

## Why

The idea (Arthur): the UniiChat memory tells which task is under way; the agent's last message
and the user's new message show whether the user changes task. Latency counts from the moment
the user sends: everything that does not depend on the new message is done while they read
and type.

What v1 does instead (measured 2026-10-29 on the 345 test states, Spisev tokenizer):

- The agent's last message is a Haiku summary in 305 of 345 states (88%), raw in 14. The
  part that shows what the user answers (the agent's question or proposal) is mostly lost.
- The view is 67% of the tokens; lines average 103 tokens (the 512-byte ruler gets filled);
  65% of view bytes are one-message summaries of agent turns, full of details the decision
  does not need.
- Spisev matches Mammoth under 1,000 tokens (right category 87.2% vs 85.2% at 500–1,000)
  and falls ~20 points behind above (68.8% vs 84.4% at 1,000–2,000; 42.9% vs 71.4% over 4,000).
- Latency is prefill: 74 of 345 states leave more than ~700 uncached tokens, cache reuse is
  18% on the slowest tenth (one merge per message rewrites the view), and nothing is
  prefilled before the user sends.
- In the live tree of a session: the task header repeats on every line, statuses never leave
  `in-progress`, the agent's corrected guesses stay as facts, languages mix, tool calls fill
  the 32 lines.

## Targets

On the benchmark's 418 held-out decisions (334 prompts + 84 splices, hindsight answer key),
Spisev v2 against Spisev v1:

- Right action ≥ v1 (93.3%), and not worse by McNemar (p ≥ 0.05 if lower).
- Acted when it should > v1 (79.4%).
- Right category > v1 (81.5%).
- Latency after send, with the prefix prefilled at `turn.complete`: p50 ≤ 100 ms, p95 ≤ 400 ms.
- Cold latency (no prefill, the fallback): p95 under the mod's 4 s timeout on every point.

## 0. Baseline and measurement

- Tokenize the splices too, and keep the per-point token breakdown script in
  `training/pipeline/` (not the scratchpad), with a test.
- `benchmark.py --warm`: send the state minus `latest_user_message` first, then the full
  request on the same slot, and time only the second. Report cold and warm latency.
- Record v1 baselines (Spisev and Mammoth): the README table, warm latency, and accuracy by
  token bin.
- Gate: v1 numbers reproduce the README within 0.5 point; warm latency measured.

## 1. Prefill in the mod (no retraining)

- At `turn.complete`, once the tree is built, send llama-server the request up to the new
  message (`cache_prompt`, the session's slot, no generation), so only the user's message is
  left at send time. If the user sends before it finishes, behave as today.
- Clip `latest_user_message` to its head and tail (start at 512 tokens; tune on dev).
- Tests for both; parity with Python for the clip.
- Gate: on the benchmark, accuracy not worse than v1 (McNemar), warm p50 ≤ 100 ms. This
  phase ships on its own if it passes (version bump, README latency updated).

## 2. State format v2

State, in this order (everything before the last field stable and prefillable):

1. `memory`: the UniiChat view, for the task structure only.
2. `last_agent_message`: the agent's last reply, raw, clipped keeping more of its end
   (start at 1.5 KB; tune on dev).
3. `latest_user_message`.

The question text and options describe the new fields; the categories and policy stay.

Memory (prompts/compact.md v2, `tree`, `view`):

- One language for every line: English.
- Lines short: target ~160 bytes (try 128 / 192 / 256 on dev), the ruler at that size.
- The task header once per task change, not on every line; statuses add `left` (the
  conversation moved on, nothing confirmed) next to `done` and `abandoned`.
- By value: the user's words and decisions, the outcome of each exchange (proposed /
  accepted / refused / corrected), task changes; drop numbers, tool steps and the agent's
  guesses, keep corrections over what they correct.
- Tool calls and results fold into the agent turn they belong to; they are not lines.
- Merge in batches on a byte budget, as UniiChat does (grow to a high mark, merge back to a
  low mark), so the memory changes only at its end between batches. Marks set so state
  stays under ~1,000 tokens (memory ≤ ~600).
- Python pipeline and JS mod keep one implementation each, checked by the parity tests.
- Gate: unit and parity tests pass; on 20 real sessions, memory p95 ≤ 600 tokens and cache
  reuse between consecutive states ≥ 80%.

## 3. Teacher check before relabeling

Pick the format on dev, never on test:

- Build a hindsight answer key for a dev sample (`hindsight.py`, the same two teachers,
  ~300 points, the 96% agreement rule).
- Run Mammoth on that sample with v1 states and with each v2 variant (line size, clip sizes).
- Estimate cost first, `--dry-run`, stop above MAX_SPEND_USD, 6b limit rules apply.
- Gate: the chosen variant is at least as accurate as v1 for Mammoth on the dev key (right
  action and category). If none is, stop and report: v2 is not worth retraining.

## 4. Rebuild, relabel, retrain

- Rebuild the tree with prompt v2 on every session (Haiku on the subscription; estimate,
  cache, usage-limit rules), relabel with Mammoth, rebuild splits with the same session split.
- Retrain Qwen3-1.7B with the v1 recipe, best epoch on dev, calibrate, `import_adapter.py`
  dev check, export GGUF Q8_0.
- Check `df -h` before; tell Arthur the size if over ~1 GB; delete intermediate checkpoints
  and merged weights once the GGUF is verified.
- Gate: dev agreement with Mammoth ≥ v1's.

## 5. Evaluate and ship

- Benchmark v2 on the 418 held-out points, used here for the first time: the targets above,
  cold and warm latency, accuracy by token bin, McNemar against v1.
- Gate: every target met. If not, report which and stop.
- Ship: mod updated (prompt, tree, task format, prefill), `claude plugin validate --strict .`
  and `claude plugin test` pass, version bump, README benchmark and SPEC updated.
- Ask before anything outward: pushing, uploading the GGUF to Hugging Face, publishing.

## Rules

- Held-out sessions never used to build, tune or choose anything before phase 5.
- No API billing, no faked or substituted teacher answers or summaries; keys never printed.
- One Conventional Commit per logical change, on the `v2` branch.
- Clean up as each phase closes: scratch files, intermediate data, stopped servers.

## Log

| Date | Phase | Result |
| :- | :- | :- |
| 2026-10-29 | analysis | Numbers in [Why](#why), from the 345 test states and the live tree of one session. |
