"""Local vLLM generation and trainer-to-sampler weight synchronization (v0.22)."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def http_json(method, url, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else {}


class Sampler:
    def __init__(self, model, output, cfg, gpu=1, port=0):
        self.model, self.output, self.cfg, self.gpu = model, Path(output), cfg, gpu
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
            self.port = probe.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.process = None
        self.group = None

    def start(self):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu = visible.split(",")[self.gpu] if visible else str(self.gpu)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
               "VLLM_SERVER_DEV_MODE": "1", "VLLM_NO_USAGE_STATS": "1"}
        command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", self.model,
                   "--host", "127.0.0.1", "--port", str(self.port), "--dtype", "bfloat16",
                   "--seed", str(self.cfg["seed"]), "--tensor-parallel-size", "1",
                   "--max-model-len", str(self.cfg["context_length"]),
                   "--gpu-memory-utilization", str(self.cfg["sampler_memory_fraction"]),
                   "--max-num-seqs", "256", "--max-num-batched-tokens", "8192",
                   "--enable-prefix-caching", "--enforce-eager", "--generation-config", "vllm",
                   "--weight-transfer-config", '{"backend":"nccl"}']
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "vllm.log").open("a") as log:
            self.process = subprocess.Popen(command, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("vLLM exited; inspect the output directory's vllm.log")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5):
                    return
            except OSError:
                time.sleep(1)
        raise TimeoutError("vLLM startup timed out")

    def stop(self):
        # Only stop the process group created by this Sampler.
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()

    def init_weight_sync(self, device):
        import torch
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
        from vllm.utils.network_utils import get_ip, get_open_port
        info = {"master_address": get_ip(), "master_port": get_open_port(),
                "world_size": http_json("GET", self.url + "/get_world_size")["world_size"] + 1}
        torch.cuda.set_device(device)
        with ThreadPoolExecutor(max_workers=1) as executor:
            receiver = executor.submit(http_json, "POST", self.url + "/init_weight_transfer_engine",
                                       {"init_info": {**info, "rank_offset": 1}})
            self.group = NCCLWeightTransferEngine.trainer_init(info)
            receiver.result()

    def sync_policy_weights(self, policy):
        import torch
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine,
        )
        if self.group is None:
            raise RuntimeError("Initialize weight synchronization first")
        torch.cuda.set_device(next(policy.parameters()).device)
        with torch.no_grad():
            weights = [(name, param.detach().to(torch.bfloat16).contiguous())
                       for name, param in policy.named_parameters()]
        info = {"names": [name for name, _ in weights],
                "dtype_names": ["bfloat16"] * len(weights),
                "shapes": [list(param.shape) for _, param in weights], "packed": False}
        http_json("POST", self.url + "/pause")
        http_json("POST", self.url + "/start_weight_update", {"is_checkpoint_format": True})
        with ThreadPoolExecutor(max_workers=1) as executor:
            receiver = executor.submit(http_json, "POST", self.url + "/update_weights",
                                       {"update_info": info}, 180)
            NCCLWeightTransferEngine.trainer_send_weights(
                iter(weights), NCCLTrainerSendWeightsArgs(group=self.group, packed=False))
            receiver.result()
        http_json("POST", self.url + "/finish_weight_update", {})
        http_json("POST", self.url + "/reset_prefix_cache", {})
        http_json("POST", self.url + "/resume", {})

    def generate(self, examples, tokenizer, template, n, seed, log_probs=False):
        from reward import r1_zero_reward_fn
        prompts = [tokenizer.encode(template.format(question=r["question"]),
                                    add_special_tokens=False) for r in examples]
        if max(map(len, prompts)) + self.cfg["max_new_tokens"] > self.cfg["context_length"]:
            raise ValueError("Prompt plus answer exceeds context_length")
        records = []
        for begin in range(0, len(examples), 128):
            payload = {"model": self.model, "prompt": prompts[begin:begin + 128], "n": n,
                       "temperature": self.cfg["temperature"], "top_p": self.cfg["top_p"],
                       "top_k": -1, "min_p": 0., "max_tokens": self.cfg["max_new_tokens"],
                       "seed": seed, "stop": ["</answer>"], "include_stop_str_in_output": True,
                       "return_token_ids": True}
            if log_probs:
                payload["logprobs"] = 1
            result = http_json("POST", self.url + "/v1/completions", payload, 900)
            choices = sorted(result["choices"], key=lambda x: x["index"])
            if len(choices) != len(prompts[begin:begin + 128]) * n:
                raise RuntimeError("Incomplete sampler response")
            for j, example in enumerate(examples[begin:begin + 128]):
                for sample_index in range(n):
                    choice = choices[j * n + sample_index]
                    ids = choice["token_ids"]
                    if not ids or len(ids) > self.cfg["max_new_tokens"]:
                        raise RuntimeError("Invalid sampled token trajectory")
                    row = {**example, "sample_index": sample_index,
                           "prompt_token_ids": prompts[begin + j], "response_token_ids": ids,
                           "response": choice["text"], "finish_reason": choice.get("finish_reason"),
                           "rewards": r1_zero_reward_fn(choice["text"], example["ground_truth"])}
                    if log_probs:
                        row["sampler_log_probs"] = choice["logprobs"]["token_logprobs"]
                        if len(row["sampler_log_probs"]) != len(ids) or None in row["sampler_log_probs"]:
                            raise RuntimeError("Missing sampled-token probabilities")
                    records.append(row)
        return records
