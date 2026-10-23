import json
import shutil
import subprocess
import unittest

from helpers import REPO

import policy
from task import CATEGORIES


class PolicyTest(unittest.TestCase):
    def test_default_table(self):
        self.assertEqual(policy.action("continuation"), "keep")
        self.assertEqual(policy.action("follow_up"), "keep")
        self.assertEqual(policy.action("related"), "compact_view")
        self.assertEqual(policy.action("independent"), "clear")
        self.assertEqual(policy.action("return"), "compact_branch")
        self.assertEqual(policy.action("unknown"), "keep")
        self.assertEqual(set(policy.POLICY), set(CATEGORIES))
        self.assertEqual(policy.NEVER_ACT, {"continuation", "follow_up"})
        for c in policy.NEVER_ACT:
            self.assertEqual(policy.action(c), "keep")

    def test_action_probabilities_sum_categories(self):
        probs = {"continuation": 0.5, "follow_up": 0.2, "related": 0.1, "independent": 0.15, "return": 0.05}
        out = policy.action_probabilities(probs)
        self.assertAlmostEqual(out["keep"], 0.7)
        self.assertAlmostEqual(sum(out.values()), 1.0)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_hooks_policy_js_has_the_same_table(self):
        code = ("import('./hooks/policy.js').then(m => console.log(JSON.stringify({policy: m.POLICY, "
                "never: [...m.NEVER_ACT], categories: m.CATEGORIES})))")
        out = json.loads(subprocess.run(["node", "-e", code], cwd=REPO, capture_output=True, text=True,
                                        check=True).stdout)
        self.assertEqual(out["policy"], policy.POLICY)
        self.assertEqual(set(out["never"]), policy.NEVER_ACT)
        self.assertEqual(out["categories"], CATEGORIES)


if __name__ == "__main__":
    unittest.main()
