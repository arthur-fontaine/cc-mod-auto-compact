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
  `in-progress`, the agent's corrected guesses stay as facts, languages mix. (Tool calls do
  not fill lines: `text.messages` already folds them into the agent's turn.)

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
- Tool calls stay folded into the agent turn they belong to, as v1 already does.
- Merge in batches on a byte budget, as UniiChat does (grow to a high mark, merge back to a
  low mark), so the memory changes only at its end between batches. Marks set so state
  stays under ~1,000 tokens (memory ≤ ~600).
- Python pipeline and JS mod keep one implementation each, checked by the parity tests.
- Gate: unit and parity tests pass; on 20 real sessions, memory p95 ≤ 600 tokens and, with
  the state prefilled, tokens left to read after send p95 ≤ 600. (Was "cache reuse between
  consecutive states ≥ 80%", which v2 cannot reach by design: the agent's last reply changes
  every turn. Replaced with what it stood for, Arthur's go on 2026-10-29.) A line size that
  fails this gate is not a phase 3 candidate.

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
| 2026-10-29 | 0 · passed | Spisev v1 rerun on the 418 points: right action 93.3%, acted 79.4%, category 81.5%, auto 87.1%, the README's to 0.0 point. Latency p50 / p95: cold 153 / 2,487 ms (4 timeouts), prefilled 54 / 284 ms (1 timeout), same answers. Tokens left after prefill: p50 23, p95 351, max 9,577 (one long message). By state size (test + splices), right category Spisev / Mammoth: < 500 88.2 / 94.1%, 500–1k 88.7 / 87.2%, 1–2k 70.9 / 86.1%, 2–4k 71.7 / 83.0%, 4k+ 50.0 / 80.0%. Dev key (`hindsight.py --split dev`): 249 of 266 points where both teachers agree on the action, only 14 that act (no splices on dev), always keep 94.4%: on dev, right category (216 points) is what separates formats. |
| 2026-10-29 | 1 · passed | Message clip chosen on dev: 1,024, 512 and 256 tokens change none of Spisev's 266 picks; 512 kept for margin. On the 418 points, prefilled and clipped: right action 93.5% (v1 93.3%, McNemar 1 point to 0, p = 1.0), acted 80.4%, category 82.1%, auto 87.3%, no timeout; latency p50 60 ms (gate ≤ 100), p95 445 ms, max 2.1 s. The slowest decisions had 21–77 tokens left to read, so the p95 here is machine noise, not prefill. Released locally as 0.1.3 (not pushed). |
| 2026-10-29 | 2 · passed | v2 built in Python and JS (parity tests, 94 Python and 41 mod tests pass). First dev build audited and fixed before any Mammoth call: Haiku copied `id+n|` names and backticks into lines, kept quotes untranslated, and almost never used `left` (1 of 483 statuses); prompt and cleaning fixed, dev rebuilt (Haiku, subscription). At 128 bytes over the 69 dev sessions: memory p50 150 / p95 490 / max 608 tokens, whole state p50 876 / p95 1,380 (v1 dev-like states reach 10k), tokens left after prefill p50 21 / p95 322; statuses now include 22 left and 97 done. The reuse gate as first written failed (whole prompt 53%, memory lines kept 78%, a rewrite at 13% of decisions) and was replaced, see the gate. |
| 2026-10-29 | 2 · recheck | Final v2 (memory through the agent's last reply), 128 bytes: memory p50 175 / p95 518 / max 712 tokens, state p95 1,411, left after prefill p50 21 / p95 322: passes. 512 bytes: memory p95 1,201, fails, so not a candidate. |
| 2026-10-29 | 3 · **not met** | Mammoth changed version at Baseten (q46n67k → qvxd569) and the labeler stopped; with Arthur's go every phase 3 run, v1 included, was asked again to qvxd569 in a cache of its own (`data/teacher-cache-eval`), training labels untouched. Dev keys: the first (`hindsight-dev`, teachers saw v1 lines, 249 points) and a neutral one (`--neutral`, raw messages only, 254 points); only 14–15 points act, so "acted when it should" decides nothing. Right action / right category, first key then neutral: v1 95.2 / 85.2, 94.9 / 85.0. Final v2-128 92.4 / 82.4, 91.7 / 82.3. Best v2 on action, v2a-128 (memory without the last reply's line): 94.0 / 82.9. v2-512 93.6 / 84.7, 92.1 / 84.1 (fails phase 2). Eight v2 variants (line size 128–512, batch marks 10/16 and 14/22, task granularity, with and without the raw reply) all fall 1–4 points of action below v1 on both keys; no single gap is significant at 0.05 but for v2b-192 (p = 0.013), and the direction never changes. Two findings. 1) The raw `last_agent_message` costs Mammoth about 2 points of action, with v1 lines too (v1 + raw reply 92.8 / 85.2 against v1 with the v2 question and no reply 94.4 / 85.6): when the reply ends offering a menu ("want me to add X, Y or Z?") an off-menu request reads as `related`, and "working tree is clean" as `independent`. 2) 128-byte lines cost 2–3 points of category against 512. Not measured here: Spisev's own gain on short states (phase 0: 11 to 30 points of category behind Mammoth above 1,000 tokens, level with it below), which only training would show. Spend: 2,868 answers, 2.9 M input tokens, about $0.29 at $0.10 / M, plus a few calls before the version stop. Phase 4 not started. |
| 2026-10-29 | 3 · Haiku pre-test | Arthur asked whether Haiku could replace Mammoth as the teacher (hard labels, as in Nisev and tev1; Spisev already has `--hard-targets`). Haiku 5.5 through `claude -p` on the subscription (`claude_teacher.py`, `evaluate.py --claude`), the student's own prompt, one letter back. Right action / right category, first key then neutral: Mammoth v1 95.2 / 85.2, 94.9 / 85.0; Haiku v1 87.6 / 76.4, 88.2 / 75.5; Haiku v2-128 87.6 / 72.7, 86.6 / 70.9; always keep 94.4, 94.1. Haiku v1 against Mammoth v1: action 6 points to 25 (p = 0.001), category 18 to 37 (p = 0.014), the same on the neutral key (p = 0.003, 0.005). Haiku drops context it should keep: kept 207 of 235 keep points (Mammoth 227), picks `related` 33 times where the key has 13, mostly for follow-ups. It agrees with itself on 37 of 50 points asked twice, so its v2-against-v1 gaps (p ≥ 0.24) say nothing about the format. Haiku is not a candidate teacher; Mammoth stays. 532 calls, billed $0. |
| 2026-10-29 | 3 · tuning key | With Arthur's go, a tuning key on train so formats are chosen without wearing out dev: 154 train sessions drawn at random (`tune_split.py`, seed 0), 816 real points and 102 splices, labeled by Sonnet 5.5 alone on the neutral view (`hindsight.py --split tune --neutral`, $14.53 list-price equivalent, subscription); 917 points, 177 that act. Mammoth (qvxd569) right action / acted / right category: v1 89.2 / 61.6 / 71.0; v2-128 85.9 / 65.0 / 69.6 (action 22 to 52, p = 0.001: follow-ups read as `related`, 37 against 12); `rules` (edge cases in the criteria, judge the message not how the reply ended) 87.5 / 65.0 / 71.8, better than v2 (action p = 0.024, category p = 0.029), action still below v1 (p = 0.052); `rules2` (follow_up defined as the same app, program or module, related as a different part of the project) 88.3 / 54.8 / 71.4, level with v1 (action p = 0.35, category p = 0.77) but acting less. No candidate yet at or above v1 on action. Mammoth spend so far about $0.89 of the $1 credit. |
| 2026-10-29 | 3 · operating point | `rules3` (follow_up and related split on whether the request needs the recent details): 85.9 / 71.2 / 70.3, acting more and keeping less (action p = 0.002 against v1). The four question variants only move one keep-or-act cursor. Compared at the same cursor (act when P(related + independent + return) passes a threshold), Mammoth tells keep from act as well with v2 as with v1: AUC on the tuning key v1 0.936, v2-128 0.938 (difference +0.002, paired bootstrap 95% CI −0.011 to +0.016), `rules` 0.932, `rules2` 0.936, `rules3` 0.935; best right action over thresholds 89.5% (v1) and 89.1% (v2-128). Dev, from the runs already made: v1 0.971 and v2-128 0.957 (CI −0.043 to +0.016) on the first key, 0.967 and 0.958 on the neutral one, with 14–15 acting points. So the v2 gap at the argmax is a shift of operating point, not lost information, and rewording the criteria buys none. Mammoth spend about $0.81 of the $1 credit. |
| 2026-10-29 | 3 · Jev pre-test | With Arthur's go and his OpenCode key, Jev on OpenCode Zen (`evaluate.py --jev`, `teacher.provider`): `jev-1.13-free` answered 233 points then only 429s, the rest went to the paid `jev-1.13` ($0.042 / M input), no version id served, probabilities rounded to 0.01. Tuning key, right action / acted / right category: Mammoth v1 89.2 / 61.6 / 71.0; Jev v1 89.2 / 59.9 / 72.6 (action 22 to 22); Jev v2-128 88.4 / 67.2 / 74.2 (against Mammoth v1: action 33 to 40, p = 0.48; category 94 to 65, p = 0.026). AUC keep against act: Jev v1 0.943, Jev v2-128 0.935, Mammoth v1 0.936 (Jev v2-128 − Mammoth v1 −0.001, CI −0.020 to +0.020); best right action over thresholds Jev v2-128 89.9%, Mammoth v1 89.5%. Dev, first key then neutral: Mammoth v1 95.2 / 85.2, 94.9 / 85.0; Jev v1 94.4 / 88.9, 94.9 / 88.2; Jev v2-128 93.6 / 89.4, 93.7 / 88.2 (no gap significant). Unlike Mammoth, Jev keeps its category on v2 and acts more when it should. Jev spend about $0.13. |
