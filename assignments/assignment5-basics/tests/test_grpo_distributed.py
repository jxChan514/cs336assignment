"""Ensure DDP accumulation + sharded AdamW matches a single full-batch update."""
import copy
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel

from cs336_alignment.grpo import grpo_train_step
from .test_grpo import _train_step_inputs, _train_step_reward_fn


def _worker(rank, model, tokenizer, rendezvous, result_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        model = copy.deepcopy(model)
        ddp = DistributedDataParallel(model, gradient_as_bucket_view=True)
        calls = {"count": 0}
        def counting_hook(state, bucket):
            state["count"] += 1
            return default_hooks.allreduce_hook(None, bucket)
        ddp.register_comm_hook(calls, counting_hook)
        optimizer = ZeroRedundancyOptimizer(ddp.parameters(), optimizer_class=torch.optim.AdamW,
                                            lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
        prompts, responses, truths, _ = _train_step_inputs(tokenizer)
        part = slice(rank * 2, (rank + 1) * 2)
        for _ in range(2):
            loss, stats = grpo_train_step(ddp, tokenizer, optimizer, 2, 1.0, _train_step_reward_fn,
                                          prompts[part], responses[part], truths[part], group_size=2)
        dist.all_reduce(loss)
        assert calls["count"] == 2, "Only the last backward of each update should synchronize gradients"
        assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in model.parameters())
        assert len(optimizer.optim.state) < len(list(model.parameters())), "Optimizer states must be sharded"
        if rank == 0:
            torch.save({"loss": loss / 2, "state": model.state_dict(), "grad_norm": stats["grad_norm"]}, result_path)
    finally:
        dist.destroy_process_group()


def test_ddp_accumulation_and_sharded_adamw_match_full_batch(tiny_train_model, tokenizer, tmp_path):
    expected = copy.deepcopy(tiny_train_model)
    prompts, responses, truths, _ = _train_step_inputs(tokenizer)
    optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
    for _ in range(2):
        loss, stats = grpo_train_step(expected, tokenizer, optimizer, 4, 1.0, _train_step_reward_fn,
                                      prompts, responses, truths, group_size=2)
    result_path = str(tmp_path / "result.pt")
    torch.multiprocessing.spawn(_worker, args=(tiny_train_model, tokenizer, str(tmp_path / "rendezvous"), result_path),
                                 nprocs=2, join=True)
    actual = torch.load(result_path, weights_only=True)
    torch.testing.assert_close(actual["loss"], loss, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual["grad_norm"], stats["grad_norm"], atol=1e-6, rtol=1e-5)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual["state"][name], value, atol=2e-6, rtol=1e-4)
