import json
import unittest

import helpers  # noqa: F401
import export_gguf
import task


def render(template, state, q=None):
    import jinja2

    env = jinja2.Environment(keep_trailing_newline=True)  # as llama.cpp renders it
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
    q = q or task.question()
    options = [{"key": k, "description": d} for k, d in q["criteria"].items()]
    return env.from_string(template).render(instructions=q["instructions"], options=options, state=state)


def qwen_chat(messages):
    """Qwen's chat format in non-thinking mode, as apply_chat_template renders it."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + export_gguf.ASSISTANT_PREFIX


class TemplateTest(unittest.TestCase):
    STATES = [
        {"conversation": "0+4|task: parser [done]\n4+1|user: thanks", "latest_user_message": "now the README"},
        {"conversation": "0+1|user: \"quotes\" and \\ backslash\ttab", "latest_user_message": "é 日本語 😀\nline"},
    ]

    def test_systemone_template_renders_the_training_prompt(self):
        for state in self.STATES:
            self.assertEqual(render(export_gguf.systemone_template(), state), qwen_chat(task.messages(state)))

    def test_v2_template_renders_the_v2_training_prompt(self):
        state = task.clip_state_v2("0+1|user: a \"b\"\n1+1|= [done] é", "Done. Commit?", "yes 日本語")
        self.assertEqual(render(export_gguf.systemone_template("v2"), state, task.question_v2()),
                         qwen_chat(task.messages_v2(state)))

    def test_against_the_real_qwen_template_when_cached(self):
        try:
            from huggingface_hub import try_to_load_from_cache
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers not installed")
        for repo in ("Qwen/Qwen3-1.7B", "Qwen/Qwen3.5-2B"):
            path = try_to_load_from_cache(repo, "tokenizer_config.json", cache_dir=helpers.ROOT / ".hf" / "hub")
            if not isinstance(path, str):
                continue
            tok = AutoTokenizer.from_pretrained(repo, cache_dir=helpers.ROOT / ".hf" / "hub", local_files_only=True)
            for state in self.STATES:
                ours = render(export_gguf.systemone_template(), state)
                theirs = tok.apply_chat_template(task.messages(state), tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
                self.assertEqual(ours, theirs, repo)


if __name__ == "__main__":
    unittest.main()
