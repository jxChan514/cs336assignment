"""Separate inference GPU; vLLM 0.22 weight-update lifecycle."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import socket
import subprocess
import sys

import torch

from reference.vllm_utils import VLLMServer, wait_for_server, _http_json


class Sampler(VLLMServer):
    def __init__(self, model_id, output, seed=0, gpu=1, port=19073):
        super().__init__(model_id=model_id, gpu=gpu, seed=seed, port=port,
                         gpu_memory_utilization=0.35, startup_timeout=900)
        self.output = Path(output)

    def start(self):
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind((self.host, self.port))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", self.model_id,
                   "--host", "127.0.0.1", "--port", str(self.port), "--dtype", "bfloat16",
                   "--seed", str(self.seed), "--tensor-parallel-size", "1",
                   "--max-model-len", "2048", "--gpu-memory-utilization", "0.35",
                   "--max-num-seqs", "256", "--max-num-batched-tokens", "8192",
                   "--enable-prefix-caching", "--enforce-eager", "--generation-config", "vllm",
                   "--weight-transfer-config", '{"backend":"nccl"}']
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "vllm.log").open("a") as log:
            self.process = subprocess.Popen(command, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        (self.output / "server.json").write_text(json.dumps(
            {"pid": self.process.pid, "command": command, "base_url": self.base_url}, indent=2) + "\n")
        wait_for_server(self.base_url, self.process, self.startup_timeout)

    def sync_policy_weights(self, policy):
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine,
        )
        if self.weight_sync_group is None:
            raise RuntimeError("Initialize NCCL weight synchronization first")
        torch.cuda.set_device(next(policy.parameters()).device)
        with torch.no_grad():
            weights = [(name, param.detach().to(torch.bfloat16).contiguous())
                       for name, param in policy.named_parameters()]
        info = {"names": [name for name, _ in weights],
                "dtype_names": ["bfloat16"] * len(weights),
                "shapes": [list(param.shape) for _, param in weights], "packed": False}
        _http_json("POST", self.base_url + "/pause", timeout=60)
        _http_json("POST", self.base_url + "/start_weight_update",
                   {"is_checkpoint_format": True}, timeout=60)
        with ThreadPoolExecutor(max_workers=1) as executor:
            receiver = executor.submit(_http_json, "POST", self.base_url + "/update_weights",
                                       {"update_info": info}, 180)
            NCCLWeightTransferEngine.trainer_send_weights(
                iter(weights), NCCLTrainerSendWeightsArgs(group=self.weight_sync_group, packed=False))
            receiver.result()
        _http_json("POST", self.base_url + "/finish_weight_update", {}, timeout=60)
        _http_json("POST", self.base_url + "/reset_prefix_cache", timeout=60)
        _http_json("POST", self.base_url + "/resume", timeout=60)
