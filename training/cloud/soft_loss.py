"""The soft-target loss in PyTorch, shipped to the VM with train_unsloth.py.

The same as pipeline/train.py's MLX one: cross-entropy at the answer position against
Mammoth's distribution over the option letters, over the whole vocabulary, so a one-hot
target gives the standard cross-entropy. tests/test_losses.py checks both.
"""
import torch

# Every linear layer of Qwen3.5's text model: full attention, MLP, and the five
# projections of the Gated DeltaNet (linear attention) layers. Adapting only attention
# and MLP leaves most of the token mixing untrained (Nisev's README).
QWEN35_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                  "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]


def soft_cross_entropy(logits, letter_ids, targets):
    """logits (B, V), letter_ids (C,), targets (B, C) summing to 1 -> loss per example (B,)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(targets * logp[:, letter_ids]).sum(dim=-1)


def answer_logits(model, input_ids, attention_mask, position):
    """Logits at each example's answer position. The LM head runs on that position alone:
    projecting all 8k positions onto a 248k-token vocabulary would not fit on a T4."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    body = getattr(base, "model", base)  # Qwen3_5Model under Qwen3_5ForConditionalGeneration, LoRA inside
    hidden = body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h = hidden[torch.arange(hidden.shape[0], device=hidden.device), position.to(hidden.device)]
    return base.get_output_embeddings()(h)


def linear_layer_names(model):
    """The names of every linear layer of the text model: no vision tower, no LM head."""
    return {name.split(".")[-1] for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and ".visual." not in name and not name.startswith("model.visual")
            and not name.endswith("lm_head")}
