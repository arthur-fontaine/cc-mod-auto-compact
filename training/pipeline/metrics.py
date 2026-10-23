"""Scores shared by checkpoint selection, calibration and the section 11 evaluation.

The reference is Mammoth: agreement is with its pick, KL is to its distribution, and the
calibration error compares a model's confidence with how often its pick is Mammoth's.
"""
import math
from math import comb

from policy import action
from task import CATEGORIES


def top1(probs):
    return max(CATEGORIES, key=lambda c: probs.get(c, 0.0))


def kl(p, q, eps=1e-9):
    """KL(p || q) over the categories, p the reference."""
    return sum(p[c] * math.log((p[c] + eps) / (q.get(c, 0.0) + eps)) for c in CATEGORIES if p.get(c, 0) > 0)


def ece(confidences, correct, bins=10):
    n = len(confidences)
    if not n:
        return None
    groups = {}
    for c, ok in zip(confidences, correct):
        groups.setdefault(min(int(c * bins), bins - 1), []).append((c, ok))
    return sum(len(g) / n * abs(sum(ok for _, ok in g) / len(g) - sum(c for c, _ in g) / len(g))
               for g in groups.values())


def independent_precision(rows):
    """rows: (weak label or None, predicted category). Precision of `independent` against
    the weak labels, over weakly labeled points."""
    picked = [w for w, c in rows if w and c == "independent"]
    return (sum(w == "new_task" for w in picked) / len(picked) if picked else None), len(picked)


def mcnemar_exact(a_correct, b_correct):
    """Exact (binomial) McNemar test on paired outcomes: two-sided p-value and the
    discordant counts (a right b wrong, a wrong b right)."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return 1.0, b, c
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail), b, c


def score(rows):
    """rows: {probs, reference, weak}. Agreement, KL, ECE, independent precision, actions."""
    n = len(rows)
    picks = [top1(r["probs"]) for r in rows]
    refs = [top1(r["reference"]) for r in rows]
    correct = [p == q for p, q in zip(picks, refs)]
    prec, n_ind = independent_precision([(r.get("weak"), p) for r, p in zip(rows, picks)])
    return {
        "n": n,
        "agreement": sum(correct) / n,
        "kl": sum(kl(r["reference"], r["probs"]) for r in rows) / n,
        "ece": ece([r["probs"][p] for r, p in zip(rows, picks)], correct),
        "independent_precision": prec, "independent_n": n_ind,
        "action_agreement": sum(action(p) == action(q) for p, q in zip(picks, refs)) / n,
        "correct": correct,
    }
