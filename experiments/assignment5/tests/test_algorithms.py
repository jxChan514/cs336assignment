import copy
import json
import math
from pathlib import Path
import unittest

import torch
from algorithms import (advantages_for_method, loss_for_method, normalized_advantages,
                        policy_loss, select_mixed_groups, shaped_rewards, token_loss)

ROOT = Path(__file__).resolve().parent.parent


class AlgorithmTests(unittest.TestCase):
    def test_soft_maxrl_endpoints(self):
        methods = json.loads((ROOT / "configs/rl.json").read_text())["methods"]
        rewards = torch.tensor([0., 1., 1., 1., 0., 0., 1., 0.])
        for alpha, target in [(0., "dr_grpo"), (1., "maxrl")]:
            settings = {**methods["soft_maxrl"], "advantage_alpha": alpha}
            torch.testing.assert_close(advantages_for_method(rewards, 2, settings),
                                       advantages_for_method(rewards, 2, methods[target]))

    def test_offpolicy_values_and_gradients_against_scalar_oracle(self):
        methods = json.loads((ROOT / "configs/rl.json").read_text())["methods"]
        base = torch.tensor([[-2., -3., -1.], [-1., -2., -3.]])
        old = base + torch.tensor([[.1, -.3, .2], [-.2, .1, .2]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        advantages = torch.tensor([.7, -.4])
        for method in (k for k in methods if k.startswith("offpolicy_")):
            eps = methods[method].get("cliprange")
            values = base.double().tolist()
            def oracle(xs):
                rows = []
                for i, row in enumerate(xs):
                    ts = torch.where(mask[i])[0].tolist()
                    ratios = [math.exp(row[t] - float(old[i, t])) for t in ts]
                    a = float(advantages[i])
                    if method == "offpolicy_naive":
                        v = -a * sum(row[t] for t in ts) / len(ts)
                    elif method == "offpolicy_noclip":
                        v = -a * sum(ratios) / len(ts)
                    elif method == "offpolicy_clip":
                        v = -sum(min(a*r, a*max(1-eps, min(1+eps, r))) for r in ratios) / len(ts)
                    else:
                        r = math.exp(sum(row[t] - float(old[i, t]) for t in ts) / len(ts))
                        v = -min(a*r, a*max(1-eps, min(1+eps, r)))
                    rows.append(v)
                return sum(rows) / len(rows)
            current = base.clone().requires_grad_()
            loss, _ = policy_loss(current, old, advantages, mask, method, eps)
            loss.backward()
            numerical = torch.zeros_like(base)
            for i in range(len(base)):
                for t in range(base.shape[1]):
                    plus, minus = copy.deepcopy(values), copy.deepcopy(values)
                    plus[i][t] += 1e-5
                    minus[i][t] -= 1e-5
                    numerical[i, t] = (oracle(plus) - oracle(minus)) / 2e-5
            self.assertAlmostEqual(float(loss.detach()), oracle(values), places=6)
            torch.testing.assert_close(current.grad, numerical, atol=2e-6, rtol=1e-4)

    def test_first_ratio_one_has_gradient_and_old_is_frozen(self):
        current = torch.tensor([[-2., -3.]], requires_grad=True)
        old = current.detach().clone().requires_grad_()
        a = torch.tensor([1.], requires_grad=True)
        loss, _ = policy_loss(current, old, a, torch.ones_like(current, dtype=torch.bool),
                              "offpolicy_noclip")
        loss.backward()
        torch.testing.assert_close(current.grad, torch.tensor([[-.5, -.5]]))
        self.assertIsNone(old.grad)
        self.assertIsNone(a.grad)

    def test_dapo_clip_gradient(self):
        current = torch.tensor([1.4, .7, 1.4, .7, 1.25]).log().reshape(5, 1).requires_grad_()
        old = torch.zeros_like(current, requires_grad=True)
        a = torch.tensor([1., 1., -1., -1., 1.], requires_grad=True)
        loss, meta = token_loss(current, old, a, torch.ones_like(current, dtype=torch.bool), 5)
        loss.backward()
        torch.testing.assert_close(current.grad.flatten(), torch.tensor([0., -.7/5, 1.4/5, 0., -1.25/5]))
        self.assertEqual(meta["clipped_tokens"], 2)
        self.assertIsNone(old.grad)
        self.assertIsNone(a.grad)

    def test_dapo_shaping_and_raw_group_filter(self):
        reward, penalty = shaped_rewards(torch.ones(4), [128, 384, 448, 512])
        torch.testing.assert_close(reward, torch.tensor([1., 1., .5, 0.]))
        torch.testing.assert_close(penalty, torch.tensor([0., 0., -.5, -1.]))
        records = []
        for qid, rs in enumerate([[0]*8, [1]*8, [1, 0]*4]):
            records.extend({"id": qid, "sample_index": j, "rewards": {"reward": r}}
                           for j, r in enumerate(rs))
        kept, counts = select_mixed_groups(records)
        self.assertEqual(counts, {"all_wrong": 1, "all_correct": 1, "mixed": 1})
        self.assertEqual(len(kept), 8)
        a = normalized_advantages(torch.tensor([1., 1., -1., -1., -1., -1., -1., -1.]))
        self.assertAlmostEqual(float(a.sum()), 0., places=5)


if __name__ == "__main__":
    unittest.main()
