# Auto Compact

A Claude Code mod that notices when a prompt starts a new task and offers to compact or clear
the context before it is sent. A decision model reads a summary of the session and the new
prompt, and puts the prompt in one of five categories: it continues the current task, follows
up on it, starts a related task, starts an unrelated one, or returns to an earlier one. A
follow-up keeps everything; a related task is worth a compaction that keeps the general
picture; an unrelated one is worth a `/clear`.

The local model is **Spisev** (spisev-alpha), a small model distilled from Baseten's Mammoth
that runs on your machine with llama.cpp; any System One endpoint (Mammoth, Jev) works too.

## Install

At the prompt of a terminal session:

```
/plugin install auto-compact --marketplace arthur-fontaine/cc-mod-auto-compact
```

From a shell: `claude plugin install auto-compact --marketplace arthur-fontaine/cc-mod-auto-compact`.

Then pick a provider with `/auto-compact setup`. Until then the mod does nothing.

## Providers

- **Spisev (local)**: `/auto-compact setup` asks before downloading the GGUF (about 1.9 GB)
  and starts `llama-server` on 127.0.0.1 with `-hf`, for the rest of the session. It needs
  llama.cpp build 11361 or later (`llama-server` or `llama` on your PATH, or
  `AUTO_COMPACT_LLAMA_SERVER`). A `.gguf` path in `AUTO_COMPACT_MODEL` is served with `-m`.
- **Mammoth (Baseten)** or **Jev (OpenCode Zen)**: setup saves the endpoint and model; the
  key is read from `AUTO_COMPACT_API_KEY` in the environment and never stored. Any other
  System One endpoint works with `AUTO_COMPACT_ENDPOINT` and `AUTO_COMPACT_MODEL`.

## Privacy

With Spisev nothing leaves the machine. With a cloud provider, only the clipped state goes
out: the session's summary lines and the new prompt. The summaries themselves are written by
Claude Haiku through your own Claude Code session (`$.model.complete`), like any other request
of the session.

## Commands

- `/auto-compact`: the provider, the mode, the state of the summary tree and the last decision.
- `/auto-compact setup`: choose a provider.
- `/auto-compact mode off|suggest|auto`: saved across sessions.

In **suggest** mode (the default), a prompt that looks like a related, returning or unrelated
task opens a question before it is sent: compact (or clear) first, or send it as is. Nothing
happens without your press. In **auto** mode the mod acts only when the decision's confidence
is at least `AUTO_COMPACT_MIN_CONFIDENCE` (0.85). `continuation` and `follow_up` never act. The
mod never acts on subagents, on a timeout or an error, without a provider, or while the
previous turn's summaries are still being built (the status line says
`auto-compact: summaries catching up`).

Acting holds the prompt back (Claude Code does not compact under a prompt being submitted),
compacts with the current summary lines as what to keep, or runs `/clear`, then sends the same
prompt. A prompt with attachments is never held back. The footer shows each decision:
`context kept · 92%`, `compact? related · 71%`, `cleared · independent · 91%`; the debug log has
`[auto-compact] <category> · <confidence> → <action>`.

## Environment variables

| Variable | Default | |
| :- | :- | :- |
| `AUTO_COMPACT_PROVIDER` | none | `spisev`, or `systemone` (implied by an endpoint) |
| `AUTO_COMPACT_ENDPOINT` | Spisev: `http://127.0.0.1:8766/v1/systemone` | a System One endpoint |
| `AUTO_COMPACT_MODEL` | Spisev: `arthur-fontaine/spisev-alpha-GGUF:Q8_0` | the endpoint's model, or a GGUF for llama-server |
| `AUTO_COMPACT_AUTH` | `Bearer` | `Api-Key` for Baseten |
| `AUTO_COMPACT_API_KEY` | | the cloud provider's key, never stored |
| `AUTO_COMPACT_MODE` | `suggest` | overrides the saved mode |
| `AUTO_COMPACT_MIN_CONFIDENCE` | `0.85` | auto mode's threshold |
| `AUTO_COMPACT_TIMEOUT_MS` | `4000` | a slower answer keeps the context |
| `AUTO_COMPACT_SUMMARIZER_MODEL` | `claude-haiku-5-5` | a full model id |
| `AUTO_COMPACT_DATA_DIR` | `~/.claude/plugins/data/auto-compact` | where each session's summary tree is saved |
| `AUTO_COMPACT_LLAMA_SERVER` | `llama-server`, then `llama` | |

## How it decides

After each turn the mod summarizes the new messages with Haiku into an
[OptChat](https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449) tree: each message
becomes a line of at most 512 bytes, and adjacent lines merge in pairs. On each prompt it sends
the view (the last 16 messages one line each, older ones as coarser lines) and the prompt to
the decision model, clipped to 12,000 tokens, and maps the category it picks through an
editable table. The same code builds the training data
([training/README.md](training/README.md)); parity tests check that the mod and the pipeline
clip, summarize and ask alike.

## Benchmark

See [training/README.md](training/README.md#results).

## Development

```sh
pnpm test        # claude plugin test
pnpm validate    # claude plugin validate . --strict
pnpm eval        # the configured provider on sample decisions (reads .env)
claude --plugin-dir .
```
