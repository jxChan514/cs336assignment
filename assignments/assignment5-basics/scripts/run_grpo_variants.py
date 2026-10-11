"""Run three independent seed-0 on-policy variants against a completed baseline."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from train_grpo import METHODS, load_gsm8k, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline_dir.resolve()
    config = json.loads((baseline / "config.json").read_text())
    baseline_done = json.loads((baseline / "completed.json").read_text())
    if config["seed"] != 0 or baseline_done["steps"] != 200:
        raise ValueError("Expected the completed 200-step seed-0 baseline")
    if (config["baseline"], config["advantage_normalizer"], config["loss_normalization"]) != METHODS["grpo"]:
        raise ValueError("Expected standard GRPO baseline")
    if config["training_world_size"] != 1 or config["sampling_gpus"] != [1]:
        raise ValueError("This queue uses GPU 0 for training and GPU 1 for sampling")
    for label in ("train", "val"):
        path = Path(config[f"{label}_file"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != config[f"{label}_sha256"]:
            raise ValueError(f"{label} dataset differs from the baseline")
    ids = [r["id"] for r in load_gsm8k(Path(config["val_file"]), config["val_examples"])]
    if ids != config["validation_ids"]:
        raise ValueError("Validation subset differs from the baseline")
    root = args.output_dir.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ".a5-grpo-variants.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(exist_ok=False)
        methods = ["maxrl", "rft", "dr_grpo"]
        state = {"status": "starting", "pid": os.getpid(), "methods": methods,
                 "seed": 0, "steps_per_method": 200, "baseline_dir": str(baseline),
                 "baseline_result": baseline_done, "runs": [], "started_at_unix": time.time(),
                 "budget_scope": "one seed each; GRPO_constant is not scheduled"}
        tracked = [PROJECT / "scripts/train_grpo.py", PROJECT / "cs336_alignment/grpo.py",
                   PROJECT / "scripts/run_grpo_two_gpu.sh", Path(__file__).resolve(),
                   PROJECT / "cs336_alignment/prompts/r1_zero.prompt"]
        hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
        state["source_sha256"] = hashes
        provenance = root / "provenance"
        provenance.mkdir()
        for path in tracked:
            shutil.copy2(path, provenance / path.name)
        state_path = root / "sweep.json"
        child = None

        def interrupted(signum, frame):
            if child is not None and child.poll() is None:
                child.send_signal(signum)
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise KeyboardInterrupt(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        write_json(state_path, state)
        copied_args = ["model", "train_file", "val_file", "prompt", "seed", "steps",
                       "train_examples", "val_examples", "rollout_batch_size", "group_size",
                       "microbatch_size", "learning_rate", "attention_implementation", "temperature",
                       "max_tokens", "max_grad_norm", "eval_every", "samples_every", "sample_count",
                       "vllm_memory_utilization", "vllm_max_model_len", "request_batch_size",
                       "wandb_project", "wandb_entity", "wandb_mode"]
        try:
            for index, method in enumerate(methods):
                if shutil.disk_usage(root).free < ((len(methods) - index) * 3.2 + 2) * 2**30:
                    raise RuntimeError("Insufficient data-disk space for remaining checkpoints")
                for path in tracked:
                    if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(PROJECT))]:
                        raise RuntimeError(f"Source changed during queue: {path}")
                output = root / f"{method}-seed0"
                command = ["bash", str(PROJECT / "scripts/run_grpo_two_gpu.sh"), "standard",
                           "--method", method, "--port", str(18140 + index * 20),
                           "--wandb-group", root.name, "--output-dir", str(output)]
                for key in copied_args:
                    if config.get(key) is not None:
                        command.extend(["--" + key.replace("_", "-"), str(config[key])])
                entry = {"method": method, "seed": 0, "status": "running", "output_dir": str(output),
                         "command": command, "started_at_unix": time.time()}
                state["runs"].append(entry)
                state.update(status="running", current_method=method)
                write_json(state_path, state)
                print(f"Starting {method}, seed 0, 200 steps: {output}", flush=True)
                with (root / f"{method}.log").open("w") as log:
                    child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True)
                    entry["pid"] = child.pid
                    write_json(state_path, state)
                    while child.poll() is None:
                        wandb_path = output / "wandb-run.json"
                        if "wandb_url" not in entry and wandb_path.exists():
                            info = json.loads(wandb_path.read_text())
                            entry.update(wandb_url=info["url"], wandb_id=info["id"])
                            write_json(state_path, state)
                        time.sleep(5)
                    if child.returncode:
                        entry.update(status="failed", exit_code=child.returncode)
                        raise RuntimeError(f"{method} failed; see {root / (method + '.log')}")
                done = json.loads((output / "completed.json").read_text())
                if done["steps"] != 200 or not (output / "final-model/model.safetensors").exists():
                    raise RuntimeError(f"{method} did not finish its complete run")
                entry.update(status="completed", finished_at_unix=time.time(), **done)
                write_json(state_path, state)
                print(f"Completed {method}: final accuracy={done['final_val_reward']:.4f}", flush=True)
            state.update(status="completed", current_method=None, finished_at_unix=time.time())
            write_json(state_path, state)
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error=f"{type(error).__name__}: {error}")
            write_json(state_path, state)
            raise


if __name__ == "__main__":
    main()
