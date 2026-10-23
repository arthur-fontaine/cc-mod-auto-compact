import unittest

import helpers  # noqa: F401


class MlxSoftLossTest(unittest.TestCase):
    def test_one_hot_soft_loss_is_cross_entropy(self):
        import mlx.core as mx
        import mlx.nn as nn
        from train import soft_cross_entropy

        mx.random.seed(0)
        logits = mx.random.normal((4, 50))
        letters = mx.array([3, 7, 11, 19, 23])
        gold = [0, 2, 4, 1]
        targets = mx.zeros((4, 5))
        for i, g in enumerate(gold):
            targets[i, g] = 1.0
        ours = soft_cross_entropy(logits, letters, targets)
        reference = nn.losses.cross_entropy(logits, letters[mx.array(gold)], reduction="none")
        self.assertTrue(mx.allclose(ours, reference, atol=1e-5).item())

    def test_soft_targets_mix_the_letters(self):
        import mlx.core as mx
        from train import soft_cross_entropy

        logits = mx.zeros((1, 10))
        loss = soft_cross_entropy(logits, mx.array([0, 1]), mx.array([[0.5, 0.5]]))
        self.assertAlmostEqual(loss.item(), float(mx.log(mx.array(10.0)).item()), places=5)


class TorchSoftLossTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed (uv sync --group merge)")

    def test_one_hot_soft_loss_is_cross_entropy(self):
        import torch
        import torch.nn.functional as F
        from soft_loss import soft_cross_entropy

        torch.manual_seed(0)
        logits = torch.randn(4, 50)
        letters = torch.tensor([3, 7, 11, 19, 23])
        gold = torch.tensor([0, 2, 4, 1])
        targets = F.one_hot(gold, 5).float()
        ours = soft_cross_entropy(logits, letters, targets)
        reference = F.cross_entropy(logits, letters[gold], reduction="none")
        self.assertTrue(torch.allclose(ours, reference, atol=1e-5))


class Qwen35TargetsTest(unittest.TestCase):
    def test_the_target_list_covers_every_linear_layer_of_the_text_model(self):
        try:
            import torch
            from transformers import AutoConfig, AutoModelForImageTextToText
        except ImportError:
            self.skipTest("torch/transformers not installed (uv sync --group merge)")
        from soft_loss import QWEN35_TARGETS

        config = AutoConfig.from_pretrained(helpers.FIXTURES / "qwen3.5-2b-config.json")
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(config)
        text = {name.split(".")[-1] for name, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and ".visual." not in name and not name.startswith("model.visual")
                and not name.endswith("lm_head")}
        self.assertEqual(text, set(QWEN35_TARGETS))
        self.assertTrue({"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"} <= set(QWEN35_TARGETS))

    def test_the_trainer_refuses_a_partial_target_list(self):
        source = (helpers.ROOT / "cloud" / "train_unsloth.py").read_text()
        self.assertIn("targets = args.targets.split(\",\") if args.targets else sorted(every)", source)
        self.assertIn("--targets must list every linear layer", source)

    def test_the_default_targets_are_every_linear_layer_of_qwen35(self):
        try:
            import torch
            from transformers import AutoConfig, AutoModelForImageTextToText
        except ImportError:
            self.skipTest("torch/transformers not installed")
        from soft_loss import QWEN35_TARGETS, linear_layer_names

        config = AutoConfig.from_pretrained(helpers.FIXTURES / "qwen3.5-2b-config.json")
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(config)
        self.assertEqual(linear_layer_names(model), set(QWEN35_TARGETS))

    def test_answer_logits_on_a_tiny_qwen35_with_lora_on_every_target(self):
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import AutoConfig, AutoModelForImageTextToText
        except ImportError:
            self.skipTest("torch/transformers/peft not installed")
        import json
        from soft_loss import QWEN35_TARGETS, answer_logits

        cfg = json.loads((helpers.FIXTURES / "qwen3.5-2b-config.json").read_text())
        t = cfg["text_config"]
        t.update(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=2,
                 num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                 linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=300, full_attention_interval=4,
                 layer_types=["linear_attention"] * 3 + ["full_attention"], mtp_num_hidden_layers=0)
        cfg["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
        config = AutoConfig.for_model(**cfg)
        torch.manual_seed(0)
        model = AutoModelForImageTextToText.from_config(config)
        peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=QWEN35_TARGETS,
                                                exclude_modules=r".*visual.*"))
        adapted = {n.split(".lora_A")[0].split(".")[-1] for n, _ in peft.named_parameters() if "lora_A" in n}
        self.assertEqual(adapted, set(QWEN35_TARGETS))
        ids = torch.randint(0, 300, (2, 12))
        logits = answer_logits(peft, ids, torch.ones_like(ids), torch.tensor([11, 5]))
        self.assertEqual(tuple(logits.shape), (2, 300))
        full = peft(input_ids=ids).logits
        self.assertTrue(torch.allclose(logits[0], full[0, 11], atol=1e-4))


if __name__ == "__main__":
    unittest.main()
