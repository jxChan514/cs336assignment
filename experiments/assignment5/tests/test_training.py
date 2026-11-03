import copy
import random
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence
from algorithms import advantages_for_method
from checkpoints import restore_checkpoint, save_checkpoint
from distillation import categorical_reverse_kl, response_objective
from model_utils import pack_prompt_and_response_ids, response_log_probs
from train import collect_rollout, optimize_batch


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 7)

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.lm_head = nn.Linear(7, 17)
        self.config = SimpleNamespace(model_type="olmo2")


class TrainingTests(unittest.TestCase):
    def test_kl_independent_oracle_and_frozen_teacher(self):
        torch.manual_seed(4)
        student = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        actual = categorical_reverse_kl(student, teacher)[0].sum()
        oracle = kl_divergence(Categorical(logits=student), Categorical(logits=teacher.detach())).sum()
        a = torch.autograd.grad(actual, (student, teacher), allow_unused=True, retain_graph=True)
        expected = torch.autograd.grad(oracle, student)[0]
        torch.testing.assert_close(actual, oracle)
        torch.testing.assert_close(a[0], expected)
        self.assertIsNone(a[1])
        self.assertTrue(torch.autograd.gradcheck(lambda x: categorical_reverse_kl(x, teacher)[0].sum(), (student,)))

    def test_eos_padding_and_shift(self):
        batch = pack_prompt_and_response_ids([[1, 2], [3]], [[4, 0], [5]], 0)
        self.assertEqual(batch["labels"].tolist(), [[2, 4, 0], [5, 0, 0]])
        self.assertEqual(batch["response_mask"].tolist(), [[False, True, True], [True, False, False]])

    def test_uneven_microbatches_match_one_update(self):
        torch.manual_seed(6)
        batch = pack_prompt_and_response_ids([[1, 2], [3], [4, 5]], [[6], [7, 8, 9], [10, 11]], 0)
        rewards = torch.tensor([1., 0., 1.])
        # All RL reductions and full-vocabulary OPD must preserve accumulation.
        cases = [("grpo", "mean", "std", "sequence"),
                 ("dr_grpo", "mean", "none", "constant"),
                 ("rft", "none", "none", "constant"),
                 ("offpolicy_noclip", "mean", "std", "sequence"),
                 ("dapo", None, None, None), ("opd", None, None, None)]
        for method, baseline, normalizer, reduction in cases:
            first = Toy()
            second = copy.deepcopy(first)
            teacher = Toy().requires_grad_(False)
            settings = {"name": method, "baseline": baseline, "advantage_normalizer": normalizer,
                        "loss_normalization": reduction}
            advantages = (advantages_for_method(rewards, 3, settings)
                          if method not in ("dapo", "opd") else torch.tensor([1., -1., .5]))
            with torch.no_grad():
                old = response_log_probs(first, **batch)["log_probs"]
            cfg = {"microbatch_size": 3, "head_chunk_tokens": 2, "max_new_tokens": 8,
                   "max_grad_norm": 1., "clip_low": .2, "clip_high": .28}
            one = torch.optim.SGD(first.parameters(), lr=.01)
            micro = torch.optim.SGD(second.parameters(), lr=.01)
            a = optimize_batch(first, one, batch, [0, 1, 2], cfg, method, settings,
                               advantages, old, teacher if method == "opd" else None)
            b = optimize_batch(second, micro, batch, [0, 1, 2], {**cfg, "microbatch_size": 2},
                               method, settings, advantages, old, teacher if method == "opd" else None)
            self.assertAlmostEqual(a["loss"], b["loss"], places=6)
            for x, y in zip(first.parameters(), second.parameters()):
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_checkpoint_restores_rng_adam_and_next_update(self):
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        rng = random.Random(19)
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        x = torch.randn(4, 3)
        def update():
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
        update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, rng, {"optimizer_updates": 1}, {"method": "toy"})
            expected_rng = (random.random(), np.random.random(), torch.rand(1), rng.random())
            update()
            expected = copy.deepcopy(model.state_dict())
            state = restore_checkpoint(path, model, optimizer, rng)
            self.assertEqual(state["progress"]["optimizer_updates"], 1)
            actual_rng = (random.random(), np.random.random(), torch.rand(1), rng.random())
            self.assertEqual(expected_rng[:2], actual_rng[:2])
            torch.testing.assert_close(expected_rng[2], actual_rng[2])
            self.assertEqual(expected_rng[3], actual_rng[3])
            update()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_dapo_bounded_sampling_does_not_accept_uniform_groups(self):
        sampler = Mock()
        sampler.generate.return_value = [
            {"id": i, "sample_index": j, "rewards": {"reward": 0.}}
            for i in range(2) for j in range(2)]
        cfg = {"max_candidate_groups": 2, "generation_prompt_batch": 2,
               "group_size": 2, "rollout_batch_size": 4, "seed": 0}
        with self.assertRaisesRegex(RuntimeError, "candidate budget"):
            collect_rollout(sampler, [{"id": 0}, {"id": 1}], None, "", cfg, "dapo", random.Random(0), 1)
        self.assertEqual(sampler.generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
