"""Run independent, sequential A5 baseline seeds and aggregate completed runs."""
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


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--wandb-entity")
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("CS336_A5_MODEL_DIR", str(PROJECT / "models/OLMo-2-0425-1B"))))
    data = Path(os.environ.get("CS336_A5_DATA_DIR", str(PROJECT / "data/gsm8k")))
    parser.add_argument("--train-file", type=Path, default=data / "train.jsonl")
    parser.add_argument("--val-file", type=Path, default=data / "test.jsonl")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    root = args.output_dir.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ".a5-grpo-seed-sweep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(exist_ok=False)
        state = {"status": "starting", "pid": os.getpid(), "seeds": args.seeds,
                 "steps_per_seed": 200, "prompt": "r1_zero", "runs": [],
                 "started_at_unix": time.time()}
        manifest = root / "sweep.json"
        tracked = [PROJECT / "scripts/train_grpo.py", PROJECT / "cs336_alignment/grpo.py",
                   PROJECT / "cs336_alignment/prompts/r1_zero.prompt"]
        hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
        state["source_sha256"] = hashes
        provenance = root / "provenance"
        provenance.mkdir()
        for path in tracked:
            shutil.copy2(path, provenance / path.name)
        model = args.model.expanduser().resolve()
        group = root.name
        state.update(model=str(model), wandb_group=group)
        child = None

        def interrupted(signum, frame):
            if child is not None and child.poll() is None:
                child.send_signal(signum)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise KeyboardInterrupt(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        write_json(manifest, state)
        try:
            for index, seed in enumerate(args.seeds):
                if shutil.disk_usage(root).free < ((len(args.seeds) - index) * 3.2 + 2) * 2**30:
                    raise RuntimeError("Insufficient data-disk space for remaining checkpoints")
                for path in tracked:
                    if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(PROJECT))]:
                        raise RuntimeError(f"Training source changed between seeds: {path}")
                output = root / f"seed-{seed}"
                entry = {"seed": seed, "status": "running", "output_dir": str(output)}
                state["runs"].append(entry)
                state.update(status="running", current_seed=seed)
                write_json(manifest, state)
                command = ["bash", str(PROJECT / "scripts/run_grpo_two_gpu.sh"), "standard",
                           "--seed", str(seed), "--prompt", "r1_zero", "--model", str(model),
                           "--microbatch-size", "8", "--attention-implementation", "sdpa",
                           "--vllm-memory-utilization", "0.35", "--request-batch-size", "32",
                           "--train-file", str(args.train_file.expanduser().resolve()),
                           "--val-file", str(args.val_file.expanduser().resolve()),
                           "--wandb-group", group,
                           "--output-dir", str(output)]
                if args.wandb_entity:
                    command.extend(["--wandb-entity", args.wandb_entity])
                entry["command"] = command
                print(f"Starting seed {seed}: {output}", flush=True)
                with (root / f"seed-{seed}.log").open("w") as log:
                    child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL)
                    entry["pid"] = child.pid
                    write_json(manifest, state)
                    while child.poll() is None:
                        wandb_path = output / "wandb-run.json"
                        if "wandb_url" not in entry and wandb_path.exists():
                            info = json.loads(wandb_path.read_text())
                            entry.update(wandb_url=info["url"], wandb_id=info["id"])
                            write_json(manifest, state)
                        time.sleep(5)
                    if child.returncode:
                        entry.update(status="failed", exit_code=child.returncode)
                        raise RuntimeError(f"Seed {seed} failed; see {root / f'seed-{seed}.log'}")
                done = json.loads((output / "completed.json").read_text())
                if done["steps"] != 200 or not (output / "final-model/model.safetensors").exists():
                    raise RuntimeError(f"Seed {seed} did not finish its full baseline")
                entry.update(status="completed", **done)
                state.update(status="aggregating", current_seed=None)
                write_json(manifest, state)
                aggregate = [sys.executable, str(PROJECT / "scripts/plot_grpo_seed_sweep.py"),
                             "--sweep-dir", str(root)]
                if index == len(args.seeds) - 1:
                    aggregate.append("--upload-wandb")
                    if args.wandb_entity:
                        aggregate.extend(["--wandb-entity", args.wandb_entity])
                subprocess.run(aggregate, cwd=PROJECT, check=True)
                print(f"Completed seed {seed}: final accuracy={done['final_val_reward']:.4f}", flush=True)
            summary = json.loads((root / "summary.json").read_text())
            state.update(status="completed", finished_at_unix=time.time(),
                         mean_final_val_reward=summary["mean_final_val_reward"],
                         meets_assignment_accuracy_target=summary["meets_assignment_accuracy_target"])
            write_json(manifest, state)
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error=f"{type(error).__name__}: {error}")
            write_json(manifest, state)
            raise


if __name__ == "__main__":
    main()
