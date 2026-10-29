"""Score the decision task with a fine-tuned model: one forward pass, no sampling.

The answer is read from the logits of the five option letters at the first generated
position, so every call returns a full distribution over categories. The tokens every
request shares (system prompt, question, options) are run once and their cache reused.
From Nisev's infer.py.
"""
import copy
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task import CATEGORIES, LETTERS, prompt  # noqa: E402


def encode_prompt(tokenizer, msgs):
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    return list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)


class Classifier:
    def __init__(self, model_path, adapter_path=None):
        mx.set_cache_limit(512 * 1024**2)
        self.model, self.tokenizer = load(str(model_path), adapter_path=str(adapter_path) if adapter_path else None)
        self.model.eval()
        self.letter_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]
        calibration = Path(model_path) / "calibration.json"
        self.temperature = json.loads(calibration.read_text())["temperature"] if calibration.exists() else 1.0
        self._prefix = self._prefix_cache = None

    def tokens(self, state):
        return encode_prompt(self.tokenizer, prompt(state))

    def _cached_prefix(self, ids):
        if self._prefix is None:
            a = self.tokens({"conversation": "a", "latest_user_message": "a"})
            b = self.tokens({"conversation": "b", "latest_user_message": "b"})
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            self._prefix = a[: max(0, n - 4)]
            cache = make_prompt_cache(self.model)
            self.model(mx.array(self._prefix)[None], cache=cache)
            mx.eval([c.state for c in cache])
            self._prefix_cache = cache
        if ids[: len(self._prefix)] != self._prefix:
            return make_prompt_cache(self.model), 0
        return copy.deepcopy(self._prefix_cache), len(self._prefix)

    def letter_logits(self, state, use_cache=True):
        ids = self.tokens(state)
        cache, skip = self._cached_prefix(ids) if use_cache else (make_prompt_cache(self.model), 0)
        logits = self.model(mx.array(ids[skip:])[None], cache=cache)[0, -1]
        return logits[mx.array(self.letter_ids)].astype(mx.float32), len(ids)

    def predict(self, state, use_cache=True):
        started = time.perf_counter()
        logits, n = self.letter_logits(state, use_cache)
        probs = mx.softmax(logits / self.temperature).tolist()
        return {"probabilities": dict(zip(CATEGORIES, probs)),
                "choice": CATEGORIES[max(range(len(probs)), key=probs.__getitem__)],
                "tokens": n, "ms": (time.perf_counter() - started) * 1000}
