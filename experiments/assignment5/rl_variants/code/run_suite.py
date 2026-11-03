"""Serial experimental queue, stopping on failed verification or failed training."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time


def main():
    code = Path(__file__).resolve().parent
    root = code.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["validation", "formal"], default="validation")
    parser.add_argument("--dry-run", action="store_true", help="Print formal jobs without launching or writing files")
    parser.add_argument("--methods", nargs="+", help="Run only these formal methods, preserving protocol order")
    args = parser.parse_args()
    protocol = json.loads((root / "protocol.json").read_text())
    performance = json.loads((root / "performance-policy.json").read_text())
    def performance_args(method):
        return ["--microbatch-size", str(8 if method.startswith("offpolicy_") else performance["on_policy_microbatch_size"]),
                "--old-log-probs-batch-size", str(performance["old_log_probs_batch_size"]),
                "--scorer", performance["scorer"], "--head-chunk-size", str(performance["head_chunk_size"]),
                *(["--no-gradient-checkpointing"] if not performance["gradient_checkpointing"] else [])]
    jobs = [(method, seed) for method in protocol["methods_in_order"]
            for seed in protocol["common"]["seeds"]]
    assert len(jobs) == protocol["formal_training_run_count"]
    if args.methods:
        if args.stage != "formal":
            parser.error("--methods applies only to formal runs")
        unknown = set(args.methods) - {m["name"] for m in protocol["methods_in_order"]}
        if unknown:
            parser.error(f"Unknown methods: {sorted(unknown)}")
        jobs = [(method, seed) for method, seed in jobs if method["name"] in args.methods]
    if args.dry_run:
        if args.stage != "formal":
            parser.error("--dry-run currently describes the formal job queue")
        print(json.dumps({"formal_runs": len(jobs), "jobs": [
            {"method": method["name"], "seed": seed,
             "output": str(root / "outputs/formal" / method["name"] / f"seed{seed}"),
             "rollout_steps": protocol["common"]["rollout_steps"],
             "performance_args": performance_args(method["name"])}
            for method, seed in jobs]}, indent=2))
        return
    status = root / "outputs/queue-status.json"
    def save(**values):
        status.parent.mkdir(exist_ok=True)
        status.write_text(json.dumps(values, indent=2) + "\n")
    def run(name, argv, output):
        output.mkdir(parents=True, exist_ok=True)
        if (output / "completed.json").exists():
            result = json.loads((output / "completed.json").read_text())
            assert result["status"] == "PASS"
            return result
        if (output / "config.json").exists():
            raise RuntimeError(f"Incomplete run requires inspection, preserved: {output}")
        save(status="running", stage=args.stage, current=name, output=str(output))
        with (output / "driver.log").open("a") as log:
            completed = subprocess.run([sys.executable, str(code / "train.py"), *argv,
                                        "--output", str(output)], stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Run failed with exit {completed.returncode}: {output}")
        result = json.loads((output / "completed.json").read_text())
        assert result["status"] == "PASS" and result["signal_rollouts"] > 0
        return result
    try:
        assert json.loads((root / "preflight/objectives-pass.json").read_text())["status"] == "PASS"
        assert json.loads((root / "outputs/baseline-3shot/completed.json").read_text())["status"] == "PASS"
        assert json.loads((root / "preflight/performance-validation-pass.json").read_text())["status"] == "PASS"
        if performance.get("memory_safety_required"):
            memory_receipt = json.loads((root / "preflight/memory-safety-pass.json").read_text())
            assert memory_receipt["status"] == "PASS"
            assert memory_receipt["performance_policy_sha256"] == hashlib.sha256((root / "performance-policy.json").read_bytes()).hexdigest()
        if args.stage == "validation":
            for method in protocol["methods_in_order"]:
                name = method["name"]
                run(name, ["--method", name, "--profile", "smoke", "--seed", "0"],
                    root / "outputs/smoke" / f"{name}-seed0")
            pilot = run("grpo-pilot50", ["--method", "grpo", "--profile", "pilot", "--seed", "0", *performance_args("grpo")],
                        root / "outputs/pilot/grpo-seed0")
            passed = pilot["best_val_reward"] > pilot["initial_validation"]["reward"]
            save(status="validation_passed" if passed else "needs_pilot_review", stage=args.stage,
                 reason="Pilot must show validation reward improvement before formal runs", pilot=pilot)
            if not passed:
                sys.exit(2)
            (root / "preflight/training-validation-pass.json").write_text(json.dumps(
                {"status": "PASS", "smoke_methods": 10, "pilot_rollouts": 50,
                 "pilot_initial_reward": pilot["initial_validation"]["reward"],
                 "pilot_best_reward": pilot["best_val_reward"]}, indent=2) + "\n")
        else:
            assert json.loads((root / "preflight/training-validation-pass.json").read_text())["status"] == "PASS"
            for method, seed in jobs:
                save_model = seed == 0
                needed = 4 if save_model else 2
                if shutil.disk_usage(root).free < needed * 1024**3:
                    # User explicitly authorized removing unrelated data-disk contents.
                    previous = root.parent / "ocr_multihop_rl/outputs"
                    if previous.exists():
                        shutil.rmtree(previous)
                if shutil.disk_usage(root).free < needed * 1024**3:
                    raise RuntimeError("Export completed A5 outputs before continuing; A5 results were preserved")
                argv = ["--method", method["name"], "--profile", "formal", "--seed", str(seed), *performance_args(method["name"])]
                if save_model:
                    argv.append("--save-model")
                run(f'{method["name"]}-seed{seed}', argv,
                    root / "outputs/formal" / method["name"] / f"seed{seed}")
            subprocess.run([sys.executable, str(code / "aggregate.py")], check=True)
            complete_count = sum((root / "outputs/formal" / method["name"] / f"seed{seed}" / "completed.json").exists()
                                 for method in protocol["methods_in_order"] for seed in protocol["common"]["seeds"])
            interrupted = [method["name"] for method in protocol["methods_in_order"]
                           if (root / "outputs/formal" / method["name"] / "seed0/interrupted.json").exists()
                           and not (root / "outputs/formal" / method["name"] / "seed0/completed.json").exists()]
            save(status="complete_with_interruption" if interrupted else "complete", stage=args.stage,
                 selected_runs_completed=len(jobs), completed_formal_runs=complete_count,
                 interrupted_methods=interrupted, seeds=protocol["common"]["seeds"])
    except BaseException as error:
        save(status="stopped", stage=args.stage, error=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    main()
