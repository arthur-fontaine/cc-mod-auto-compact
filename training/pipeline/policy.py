"""What the mod does with each category (Nisev's lesson: the model picks a category, an
editable table picks the action). hooks/policy.js holds the same table; tests check both.

- keep: nothing happens;
- compact_view: compact, keeping the current view lines as what to keep;
- clear: clear the context, keeping the prompt being submitted;
- compact_branch: compact, keeping the earlier branch the prompt returns to.
"""
ACTIONS = ["keep", "compact_view", "clear", "compact_branch"]

POLICY = {
    "continuation": "keep",
    "follow_up": "keep",
    "related": "compact_view",
    "independent": "clear",
    "return": "compact_branch",
}

# False positives destroy context: these never act, whatever the confidence.
NEVER_ACT = {"continuation", "follow_up"}


def action(category, policy=POLICY):
    return policy.get(category, "keep")


def action_probabilities(probabilities, policy=POLICY):
    out = {a: 0.0 for a in ACTIONS}
    for category, p in probabilities.items():
        out[action(category, policy)] += p
    return out
