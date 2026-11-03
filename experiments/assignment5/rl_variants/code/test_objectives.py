"""Compare independent implementations, derivatives, and pruning normalization."""
from pathlib import Path
import copy
import json
import math
import hashlib
import sys
import unittest
import torch

from objectives import advantages_for_method, loss_for_method
from reference.grpo import (compute_group_normalized_rewards, compute_policy_gradient_loss,
                            aggregate_loss_across_microbatch)


class Objectives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent
        protocol = root / "protocol.json"
        if not protocol.exists():
            protocol = root.parent / "protocol.json"
        cls.methods = json.loads(protocol.read_text())["methods_in_order"]

    def test_all_standard_advantages_match_reference(self):
        rewards = torch.tensor([0., 1., 1., 1., 0., 0., 1., 0.])
        for settings in self.methods:
            if settings["name"] == "soft_maxrl":
                continue
            with self.subTest(method=settings["name"]):
                ours = advantages_for_method(rewards, 2, settings)
                reference, _ = compute_group_normalized_rewards(
                    rewards, 2, baseline=settings["baseline"],
                    advantage_normalizer=settings["advantage_normalizer"])
                torch.testing.assert_close(ours, reference)

    def test_soft_maxrl_endpoints(self):
        rewards = torch.tensor([0., 1., 1., 1., 0., 0., 1., 0.])
        soft = copy.deepcopy(self.methods[-1])
        for alpha, name in [(0., "dr_grpo"), (1., "maxrl")]:
            soft["advantage_alpha"] = alpha
            reference = next(s for s in self.methods if s["name"] == name)
            torch.testing.assert_close(advantages_for_method(rewards, 2, soft),
                                       advantages_for_method(rewards, 2, reference))

    def test_all_losses_and_gradients_match_reference(self):
        torch.manual_seed(42)
        base = torch.randn(4, 5) - 2
        old = base + torch.randn_like(base) * 0.1
        mask = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 1, 1, 1],
                             [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=torch.bool)
        rewards = torch.tensor([0., 1., 1., 1.])
        for settings in self.methods:
            with self.subTest(method=settings["name"]):
                a = advantages_for_method(rewards, 2, settings)
                ours = base.clone().requires_grad_()
                reference = base.clone().requires_grad_()
                loss, _ = loss_for_method(ours, old, a, mask, settings, 4, 8)
                if settings["name"] in {"offpolicy_noclip", "offpolicy_clip", "offpolicy_gspo"}:
                    # The saved course helper predates off-policy support. Use a
                    # scalar formula plus numerical derivatives as an independent oracle.
                    values = base.double().tolist()
                    def objective(xs):
                        rows = []
                        for i, row in enumerate(xs):
                            positions = torch.where(mask[i])[0].tolist()
                            ratios = [math.exp(row[t] - float(old[i, t])) for t in positions]
                            advantage = float(a[i])
                            eps = settings.get("cliprange")
                            if settings["name"] == "offpolicy_gspo":
                                ratio = math.exp(sum(row[t] - float(old[i, t]) for t in positions) / len(positions))
                                result = -min(advantage * ratio, advantage * max(1-eps, min(1+eps, ratio)))
                            elif settings["name"] == "offpolicy_clip":
                                result = -sum(min(advantage * ratio, advantage * max(1-eps, min(1+eps, ratio)))
                                              for ratio in ratios) / len(positions)
                            else:
                                result = -advantage * sum(ratios) / len(positions)
                            rows.append(result)
                        return sum(rows) / len(rows)
                    numerical = torch.zeros_like(base)
                    for i in range(4):
                        for t in range(5):
                            plus, minus = copy.deepcopy(values), copy.deepcopy(values)
                            plus[i][t] += 1e-5
                            minus[i][t] -= 1e-5
                            numerical[i, t] = (objective(plus) - objective(minus)) / 2e-5
                    loss.backward()
                    self.assertAlmostEqual(float(loss.detach()), objective(values), places=6)
                    torch.testing.assert_close(ours.grad, numerical, atol=2e-6, rtol=1e-4)
                    continue
                per_token, _ = compute_policy_gradient_loss(
                    a, reference, importance_reweighting_method=settings["importance_reweighting_method"],
                    old_log_probs=old, cliprange=settings.get("cliprange"), response_mask=mask)
                expected = aggregate_loss_across_microbatch(
                    per_token, mask, settings["loss_normalization"], normalization_constant=32)
                loss.backward()
                expected.backward()
                torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(ours.grad, reference.grad, atol=1e-6, rtol=1e-5)

    def test_accumulation_and_zero_advantage_pruning(self):
        rewards = torch.tensor([0., 1., 1., 1., 0., 0., 1., 0.])
        mask = torch.ones(8, 5, dtype=torch.bool)
        mask[::2, -2:] = False
        base = torch.linspace(-4, -1, 40).reshape(8, 5)
        for settings in self.methods:
            if settings["name"].startswith("offpolicy_"):
                continue
            for prune in [False, True]:
                with self.subTest(method=settings["name"], prune=prune):
                    a = advantages_for_method(rewards, 2, settings)
                    full = base.clone().requires_grad_()
                    split = base.clone().requires_grad_()
                    expected, _ = loss_for_method(full, None, a, mask, settings, 8, 8)
                    expected.backward()
                    ids = torch.where(a != 0)[0] if prune else torch.arange(8)
                    reported = 0.
                    for selected in ids.tensor_split(3):
                        if selected.numel() == 0:
                            continue
                        loss, _ = loss_for_method(split[selected], None, a[selected], mask[selected], settings, 8, 8)
                        if settings["loss_normalization"] == "sequence":
                            loss = loss * (len(selected) / 8)
                        loss.backward()
                        reported += float(loss.detach())
                    torch.testing.assert_close(split.grad, full.grad)
                    self.assertAlmostEqual(reported, float(expected.detach()), places=6)

    def test_first_ratio_one_still_has_gradient(self):
        lp = torch.tensor([[-2., -3.]], requires_grad=True)
        old = lp.detach().clone().requires_grad_()
        settings = next(s for s in self.methods if s["name"] == "offpolicy_noclip")
        loss, _ = loss_for_method(lp, old, torch.tensor([1.]),
                                  torch.ones_like(lp, dtype=torch.bool), settings, 1)
        loss.backward()
        torch.testing.assert_close(lp.grad, torch.tensor([[-0.5, -0.5]]))
        self.assertIsNone(old.grad)


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    if not result.wasSuccessful():
        sys.exit(1)
    code = Path(__file__).resolve().parent
    root = code if (code / "protocol.json").exists() else code.parent
    path = root / "preflight/objectives-pass.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"status": "PASS", "test_methods": result.testsRun,
        "methods": [s["name"] for s in Objectives.methods],
        "sha256": {name: hashlib.sha256((code / name).read_bytes()).hexdigest()
                   for name in ["objectives.py", "losses.py", "test_objectives.py"]}}, indent=2) + "\n")
