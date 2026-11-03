"""Summarize configured runs, retaining final and peak validation scores."""
from pathlib import Path
import csv
import json


def read_jsonl(path):
    rows = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def load_run(path, method, seed, target_steps):
    history = read_jsonl(path / "metrics.jsonl")
    if (path / "completed.json").exists():
        summary = json.loads((path / "completed.json").read_text())
        assert summary["status"] == "PASS" and summary["completed_rollout_steps"] == target_steps
        assert summary["seed"] == seed
        assert summary["optimizer_updates"] == target_steps * method["optimizer_updates_per_rollout"]
        return summary, history
    interruption = json.loads((path / "interrupted.json").read_text())
    assert interruption["status"] == "INTERRUPTED"
    assert interruption.get("user_authorized_skip_remaining") or interruption.get("monitoring_policy_stop")
    train = [r for r in history if "train/reward" in r]
    assert train and train[-1]["rollout_step"] < target_steps
    updates = read_jsonl(path / "updates.jsonl")
    summary = {"status": "INTERRUPTED", "seed": seed,
               "completed_rollout_steps": train[-1]["rollout_step"],
               "optimizer_updates": updates[-1]["optimizer_update"],
               "final_validation": None,
               "best_val_reward": max(r["val/reward"] for r in history if "val/reward" in r),
               "elapsed_seconds": history[-1]["system/cumulative_wall_seconds"],
               "wandb_url": json.loads((path / "wandb-run.json").read_text())["url"]}
    return summary, history


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    root = Path(__file__).resolve().parent.parent
    protocol = json.loads((root / "protocol.json").read_text())
    out = root / "outputs/report"
    out.mkdir(exist_ok=True)
    table = []
    histories = {}
    for method in protocol["methods_in_order"]:
        runs = []
        hs = []
        for seed in protocol["common"]["seeds"]:
            path = root / "outputs/formal" / method["name"] / f"seed{seed}"
            s, history = load_run(path, method, seed, protocol["common"]["rollout_steps"])
            runs.append(s)
            hs.append(history)
        rewards = [s["final_validation"]["reward"] for s in runs if s["status"] == "PASS"]
        table.append({"method": method["name"], "seeds": len(runs),
                      "status": "complete" if len(rewards) == len(runs) else "interrupted",
                      "completed_rollout_steps": min(s["completed_rollout_steps"] for s in runs),
                      "optimizer_updates_recorded": sum(s["optimizer_updates"] for s in runs),
                      "final_reward_mean": float(np.mean(rewards)) if len(rewards) == len(runs) else None,
                      "final_reward_std": float(np.std(rewards, ddof=1)) if len(rewards) == len(runs) and len(runs) > 1 else None,
                      "last_observed_reward": float(np.mean([next(r["val/reward"] for r in reversed(h) if "val/reward" in r) for h in hs])),
                      "peak_reward_mean": float(np.mean([s["best_val_reward"] for s in runs])),
                      "seconds_mean": float(np.mean([s["elapsed_seconds"] for s in runs])),
                      "wandb_urls": [s["wandb_url"] for s in runs]})
        histories[method["name"]] = hs
    (out / "summary.json").write_text(json.dumps(table, indent=2, allow_nan=False) + "\n")
    with (out / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in table[0] if k != "wandb_urls"])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "wandb_urls"} for row in table)
    metrics = ["val/reward", "val/format_reward", "val/response_length", "train/loss",
               "train/grad_norm", "train/token_entropy", "train/reward", "train/format_reward",
               "train/clip_fraction"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, hs in histories.items():
            curves = [{r["rollout_step"]: r[metric] for r in h
                       if r.get(metric) is not None} for h in hs]
            common = sorted(set.intersection(*(set(c) for c in curves)))
            if not common:
                continue
            ys = np.asarray([[c[x] for x in common] for c in curves])
            mean = ys.mean(0)
            interrupted = next(r["status"] for r in table if r["method"] == name) == "interrupted"
            line, = ax.plot(common, mean, label=name + (" (interrupted)" if interrupted else ""),
                            linestyle="--" if interrupted else "-")
            if len(curves) > 1:
                std = ys.std(0, ddof=1)
                ax.fill_between(common, mean-std, mean+std, color=line.get_color(), alpha=.12)
        ax.set(xlabel="Rollout batches", ylabel=metric,
               title=(f"Fixed 3-shot; single run, seed={protocol['common']['seeds'][0]}"
                      if len(protocol["common"]["seeds"]) == 1
                      else f"Fixed 3-shot; mean ± sample SD over {len(protocol['common']['seeds'])} seeds"))
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=.2)
        fig.tight_layout()
        for suffix in ["png", "svg"]:
            fig.savefig(out / f'{metric.replace("/", "_")}.{suffix}', dpi=180)
        plt.close(fig)
    (out / "plot-notes.txt").write_text(
        "Train entropy is over nonzero-advantage responses for pruned on-policy variants; "
        "cross-method entropy differences therefore also reflect response selection. "
        "Off-policy runs perform 32 optimizer updates per rollout; on-policy runs perform one. "
        "Final rewards are reported separately from best observed validation rewards; only final model weights are saved. "
        "Interrupted runs retain observed curves only; their final reward is null and their curves are dashed. "
        + ("Single run per method; between-seed variability is not estimated.\n"
           if len(protocol["common"]["seeds"]) == 1 else "\n"))
    print("A5_ALL_EXPERIMENTS_AGGREGATED", out, flush=True)


if __name__ == "__main__":
    main()
