"""The OptChat summary tree over one session's messages (gist VictorTaelin/91837951a5ce5b38f341ec1ba1df6449).

    node(0, i) = message i, in at most 512 bytes
    node(l, i) = node(l-1, 2i) and node(l-1, 2i+1) merged, in at most 512 bytes

`node(l, i)` covers the 2^l messages from i·2^l on and is named `id+n`. A source that
fits in 512 bytes is its own node, with no model call; two short lines are joined by a
newline. The rest is one Haiku call (pipeline/summarizer.py), with "too long" retries.

The view at time T covers messages [0, T): the last `leaves` messages one line each,
everything older merged pair by pair, most due first (due = (T - last) / 2^l, last
being the pair's last message; ties go to the oldest), until at most `leaves + coarse`
lines remain. Budgets are in lines, so which nodes a view needs is known before any is
built. hooks/tree.js is the same code.
"""
from text import utf8_len

LIMIT = 512
RETRIES = 5
STATE_LEAVES, STATE_COARSE = 16, 16  # the decision state
CONTEXT_LEAVES, CONTEXT_COARSE = 8, 8  # a summary call's <chat> context
RULER_NOTE = "| ← LIMIT"


def name(node):
    level, index = node
    return f"{index << level}+{1 << level}"


def span(node):
    level, index = node
    return index << level, (index + 1) << level


def _more_due(a, b, t):
    """True when pair a=(level, last) is strictly more due than pair b at time t."""
    (la, last_a), (lb, last_b) = a, b
    return (t - last_a) * (1 << lb) > (t - last_b) * (1 << la)


def view(t, leaves=STATE_LEAVES, coarse=STATE_COARSE):
    lines = [(0, i) for i in range(t)]
    tail = max(0, t - leaves)
    while len(lines) > leaves + coarse:
        best = None
        for k in range(len(lines) - 1):
            (l1, i1), (l2, i2) = lines[k], lines[k + 1]
            if l1 != l2 or i1 % 2 or i2 != i1 + 1:
                continue
            last = span(lines[k + 1])[1] - 1
            if last >= tail:
                continue
            if best is None or _more_due((l1, last), best[1], t):
                best = (k, (l1, last))
        if best is None:
            break
        k = best[0]
        level, index = lines[k]
        lines[k:k + 2] = [(level + 1, index // 2)]
    return lines


def children(node):
    level, index = node
    return [(level - 1, 2 * index), (level - 1, 2 * index + 1)] if level else []


def context_view(node):
    return view(span(node)[0], CONTEXT_LEAVES, CONTEXT_COARSE)


def needed(views):
    """Every node the given views need, children and summary contexts included, in an
    order where each node comes after everything it depends on."""
    seen, order = set(), []

    def visit(node):
        if node in seen:
            return
        seen.add(node)
        for dep in children(node) + context_view(node):
            visit(dep)
        order.append(node)

    for v in views:
        for node in v:
            visit(node)
    return order


def view_text(lines):
    """Lines as the state and the <chat> context show them: `id+n|text`, one per line."""
    return "\n".join(f"{name(node)}|{' '.join(text.split())}" for node, text in lines)


def leaf_prompt(index, rendered, context):
    return (f"<chat>\n{view_text(context)}\n</chat>\n"
            f"Compress message {index} into one line.\n"
            f"<input>\n{rendered}\n</input>")


def merge_prompt(a, b, text_a, text_b, context):
    return (f"<chat>\n{view_text(context)}\n</chat>\n"
            f"Merge lines {name(a)} and {name(b)}, adjacent, into one line.\n"
            f"<input>\n{text_a}\n{text_b}\n</input>")


def retry_prompt(prompt, line, limit=LIMIT):
    cut = line.encode("utf-8")[:limit].decode("utf-8", "ignore")
    return (f"{prompt}\n\nToo long: your line is {utf8_len(line)} bytes, over the {limit}-byte limit. "
            f"Write the whole line again for the same <input>, cutting just enough of the least valuable "
            f"items to fit before this cut:\n{cut}{RULER_NOTE}")


def no_call_text(node, sources, limit=LIMIT):
    """The node's text when it needs no model call, else None. `sources` is the rendered
    message for a leaf, or the two child lines for a merge."""
    if node[0] == 0:
        return sources if utf8_len(sources) <= limit else None
    joined = sources[0] + "\n" + sources[1]
    return joined if utf8_len(joined) <= limit else None


def shortest(lines):
    return min(lines, key=utf8_len)


def clean_line(text):
    return (text or "").strip()


# The v2 memory (PLAN.md section 2): shorter lines, and UniiChat's batched merges so that
# between two batches the memory only grows at its end and stays in the prompt cache.
# Budgets are in lines, so the view is still known from t alone.
MEMORY_LIMIT = 160
MEMORY_LOW, MEMORY_HIGH = 10, 16
_memory = {}


def _merge_most_due(lines, t):
    best = None
    for k in range(len(lines) - 1):
        (l1, i1), (l2, i2) = lines[k], lines[k + 1]
        if l1 != l2 or i1 % 2 or i2 != i1 + 1:
            continue
        last = span(lines[k + 1])[1] - 1
        if best is None or _more_due((l1, last), best[1], t):
            best = (k, (l1, last))
    if best is None:
        return False
    level, index = lines[best[0]]
    lines[best[0]:best[0] + 2] = [(level + 1, index // 2)]
    return True


def memory_view(t, low=MEMORY_LOW, high=MEMORY_HIGH):
    """Messages [0, t) as the memory shows them: one line per new message; once past `high`
    lines, one batch merges the most due pairs back to `low` lines."""
    key = (low, high)
    done = _memory.setdefault(key, [[]])
    while len(done) <= t:
        lines = done[-1] + [(0, len(done) - 1)]
        if len(lines) > high:
            while len(lines) > low and _merge_most_due(lines, len(done)):
                pass
        done.append(lines)
    return list(done[t])


def memory_context(node, low=MEMORY_LOW, high=MEMORY_HIGH):
    """A v2 summary call's <chat>: the memory as it stood before the node's first message."""
    return memory_view(span(node)[0], low, high)


def needed_memory(views, low=MEMORY_LOW, high=MEMORY_HIGH):
    """`needed` for v2 memories: children and memory contexts first."""
    seen, order = set(), []

    def visit(node):
        if node in seen:
            return
        seen.add(node)
        for dep in children(node) + memory_context(node, low, high):
            visit(dep)
        order.append(node)

    for v in views:
        for node in v:
            visit(node)
    return order
