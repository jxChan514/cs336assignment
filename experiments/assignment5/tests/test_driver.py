"""Exercise all training routes with synthetic CPU models and mocked generation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import train


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 7)

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.lm_head = nn.Linear(7, 17)
        self.config = SimpleNamespace(model_type="olmo2", vocab_size=17)

    def gradient_checkpointing_enable(self, **kwargs):
        pass


class FakeSampler:
    def __init__(self, *args):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def init_weight_sync(self, device):
        pass

    def sync_policy_weights(self, policy):
        pass

    def generate(self, examples, tokenizer, template, n, seed):
        return [{**r, "sample_index": j, "prompt_token_ids": [1, 2],
                 "response_token_ids": [3 + j, 6, 7], "response": "synthetic",
                 "finish_reason": "stop", "rewards": {"reward": float(j % 2 == 0), "format_reward": 1.}}
                for r in examples for j in range(n)]


class DriverTests(unittest.TestCase):
    def test_all_methods_have_correct_update_counts_and_resumable_state(self):
        root = Path(__file__).resolve().parent.parent
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0,
                                     get_vocab=lambda: {str(i): i for i in range(17)},
                                     encode=lambda s, **kwargs: [1, 2])
        for method in (*train.RL_METHODS, "dapo", "opd"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                p = Path(directory)
                for split, count in (("train", 8), ("val", 4)):
                    (p / (split + ".jsonl")).write_text("".join(json.dumps(
                        {"question": f"{split} question {i}", "answer": "reasoning #### 1"}) + "\n"
                        for i in range(count)))
                profile = method if method in ("opd", "dapo") else "rl"
                cfg = json.loads((root / "configs" / (profile + ".json")).read_text())
                cfg.update(rollout_steps=2, train_examples=8, val_examples=4, group_size=2,
                           rollout_batch_size=8, train_batch_size=8, microbatch_size=2,
                           offpolicy_train_batch_size=2, max_candidate_groups=8,
                           generation_prompt_batch=4, head_chunk_tokens=2,
                           checkpoint_every=1, eval_every=1)
                if method == "dapo":
                    cfg["train_batch_size"] = 4
                if method == "opd":
                    cfg.update(group_size=1, rollout_batch_size=4, train_batch_size=4)
                (p / "config.json").write_text(json.dumps(cfg))
                argv = ["train.py", "--method", method, "--train-file", str(p / "train.jsonl"),
                        "--val-file", str(p / "val.jsonl"), "--config", str(p / "config.json"),
                        "--output", str(p / "run"), "--trainer-device", "cpu"]
                with patch.object(sys, "argv", argv), patch("sampler.Sampler", FakeSampler), \
                     patch("torch.cuda.set_device"), patch("torch.cuda.current_device", return_value=0), \
                     patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                     patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **k: Model()), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                state = torch.load(p / "run/checkpoint.pt", map_location="cpu", weights_only=False)
                expected = 8 if method.startswith("offpolicy_") else 4 if method == "dapo" else 2
                self.assertEqual(state["progress"]["optimizer_updates"], expected)
                self.assertEqual(state["progress"]["step"], 2)
                self.assertEqual(json.loads((p / "run/result.json").read_text())["optimizer_updates"], expected)
                self.assertTrue(state["optimizer"]["state"])


if __name__ == "__main__":
    unittest.main()
