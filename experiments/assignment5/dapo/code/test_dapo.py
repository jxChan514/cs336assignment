import unittest
from unittest.mock import patch
import io
import random
import tempfile
import json
from pathlib import Path
from history import canonical_rows, lifetime_work
from common import write_json, append_json
import numpy as np
import torch
from dapo import shaped_rewards, normalized_advantages, select_mixed_groups, token_loss, warmup_lr
from checkpoints import capture_state, restore_state, save_recoverable


class DAPOTests(unittest.TestCase):
    def test_asymmetric_clip_gradient_and_negative_tail(self):
        current = torch.tensor([1.4, .7, 1.4, .7, 1.25]).log().reshape(5, 1).requires_grad_()
        old = torch.zeros_like(current, requires_grad=True)
        adv = torch.tensor([1., 1., -1., -1., 1.], requires_grad=True)
        loss, meta = token_loss(current, old, adv, torch.ones_like(current, dtype=torch.bool), 5)
        loss.backward()
        torch.testing.assert_close(current.grad.flatten(), torch.tensor([0., -.7/5, 1.4/5, 0., -1.25/5]))
        self.assertIsNone(old.grad)
        self.assertIsNone(adv.grad)
        self.assertEqual(meta['clipped_tokens'], 2)

    def test_global_token_denominator_and_microbatch_gradients(self):
        torch.manual_seed(23)
        current = (torch.randn(7, 6)*.1).requires_grad_()
        mask = torch.arange(6)[None, :] < torch.tensor([1, 6, 2, 5, 3, 1, 4])[:, None]
        old = torch.zeros_like(current)
        adv = torch.tensor([1., -.5, .7, -1., .3, -2., 1.])
        denominator = int(mask.sum())
        full, _ = token_loss(current, old, adv, mask, denominator)
        full.backward()
        expected = current.grad.clone()
        parts = current.detach().clone().requires_grad_()
        for start in [0, 3, 6]:
            loss, _ = token_loss(parts[start:start+3], old[start:start+3], adv[start:start+3], mask[start:start+3], denominator)
            loss.backward()
        torch.testing.assert_close(parts.grad, expected)
        self.assertTrue((parts.grad[~mask] == 0).all())

    def test_reward_boundaries_and_baseline(self):
        reward, penalty = shaped_rewards(torch.ones(4), [128, 384, 448, 512])
        torch.testing.assert_close(penalty, torch.tensor([0., 0., -.5, -1.]))
        torch.testing.assert_close(reward, torch.tensor([1., 1., .5, 0.]))
        reward, _ = shaped_rewards(torch.tensor([1., 1., 0., 0., 0., 0., 0., 0.]), [128]*8)
        adv = normalized_advantages(reward)
        self.assertAlmostEqual(float(adv.sum()), 0., places=5)
        self.assertTrue((adv[:2] > 0).all() and (adv[2:] < 0).all())

    def test_group_filter_uses_task_accuracy(self):
        records = []
        for qid, rewards in enumerate([[0]*8, [1]*8, [1, 0]*4]):
            records.extend({'id': qid, 'sample_index': j, 'rewards': {'reward': r}}
                           for j, r in enumerate(rewards))
        kept, counts = select_mixed_groups(records)
        self.assertEqual(counts, {'all_wrong': 1, 'all_correct': 1, 'mixed': 1})
        self.assertEqual(len(kept), 8)
        self.assertTrue(all(r['id'] == 2 for r in kept))

    def test_first_update_and_warmup(self):
        current = torch.zeros(2, 2, requires_grad=True)
        old = current.detach().clone()
        loss, meta = token_loss(current, old, torch.tensor([1., -1.]), torch.ones_like(current, dtype=torch.bool), 4)
        loss.backward()
        torch.testing.assert_close(current.grad, torch.tensor([[-.25, -.25], [.25, .25]]))
        self.assertEqual(meta['clipped_tokens'], 0)
        self.assertEqual(warmup_lr(80), 1e-6)
        self.assertEqual(warmup_lr(81), 1e-6)
        self.assertEqual(warmup_lr(1), 1e-6/80)

    def test_resume_preserves_tail_and_counts_canonical_and_lifetime_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Four completed updates at rollout20, then four rolled-back at21.
            for step in [20, 21]:
                for j in range(4):append_json(root/'updates.jsonl',
                    {'rollout_step': step, 'optimizer_update': (step-1)*4+j+1})
                append_json(root/'sampling-events.jsonl',
                    {'rollout_step': step, 'generated_responses': 256, 'generated_tokens': 512})
            original = (root/'updates.jsonl').read_bytes()
            attempt = root/'attempts/attempt-0001'; attempt.mkdir(parents=True)
            write_json(attempt/'resume.json', {'parent_attempt': '.',
                'checkpoint_rollout_step': 20, 'checkpoint_optimizer_updates': 80})
            for j in range(4):append_json(attempt/'updates.jsonl',
                {'rollout_step': 21, 'optimizer_update': 81+j})
            append_json(attempt/'sampling-events.jsonl',
                {'rollout_step': 21, 'generated_responses': 512, 'generated_tokens': 1024})
            canonical = canonical_rows(root, 'attempts/attempt-0001', 'updates.jsonl')
            self.assertEqual(len(canonical), 8)
            self.assertEqual(len([r for r in canonical if r['rollout_step'] == 21]), 4)
            self.assertTrue(all(r['_attempt'] == 'attempts/attempt-0001' for r in canonical[-4:]))
            self.assertEqual(lifetime_work(root), {'lifetime_optimizer_updates': 12,
                'lifetime_generated_responses': 1024, 'lifetime_generated_tokens': 2048})
            self.assertEqual((root/'updates.jsonl').read_bytes(), original)

    def test_local_checkpoint_rotation_without_remote_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters())
            model(torch.ones(1,2)).sum().backward();optimizer.step();optimizer.zero_grad()
            progress={'rollout_step':20,'optimizer_updates':80,'wandb_run_id':'test'}
            with patch('checkpoints.checkpoint_policy',return_value={'remote_backup_required':False,'source_keep':1}):
                first=save_recoverable(directory,model,optimizer,random.Random(0),progress,{},None)
                self.assertTrue(Path(first['path']).exists())
                progress={'rollout_step':40,'optimizer_updates':160,'wandb_run_id':'test'}
                second=save_recoverable(directory,model,optimizer,random.Random(0),progress,{},None)
            self.assertFalse(Path(first['path']).exists())
            self.assertTrue(Path(second['path']).exists())
            self.assertFalse(second['effective_retention_policy']['remote_backup_required'])
            self.assertFalse((Path(directory)/'checkpoint-backups').exists())
            state=torch.load(second['path'],weights_only=False)
            self.assertEqual(state['progress']['optimizer_updates'],160)
            self.assertEqual(len(state['optimizer']['state']),2)

    def test_checkpoint_restores_optimizer_rng_and_next_update(self):
        random.seed(9); np.random.seed(9); torch.manual_seed(9)
        data_rng = random.Random(57)
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(3, 4)).square().mean().backward(); opt.step(); opt.zero_grad()
        stream = io.BytesIO()
        torch.save(capture_state(model, opt, data_rng, {'optimizer_updates': 1}, {'seed': 0}), stream)
        expected_rng = (random.random(), np.random.rand(), torch.rand(2), data_rng.random())
        x = torch.randn(3, 4); model(x).square().mean().backward(); opt.step(); opt.zero_grad()
        expected = {k: v.clone() for k, v in model.state_dict().items()}
        restored = torch.nn.Linear(4, 2); restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        stream.seek(0)
        state = torch.load(stream, weights_only=False)
        progress = restore_state(state, restored, restored_opt, data_rng)
        self.assertEqual(progress['optimizer_updates'], 1)
        self.assertEqual(random.random(), expected_rng[0]); self.assertEqual(np.random.rand(), expected_rng[1])
        torch.testing.assert_close(torch.rand(2), expected_rng[2]); self.assertEqual(data_rng.random(), expected_rng[3])
        replay_x = torch.randn(3, 4); torch.testing.assert_close(replay_x, x)
        restored(replay_x).square().mean().backward(); restored_opt.step()
        for k, v in restored.state_dict().items():torch.testing.assert_close(v, expected[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
