"""四组学习率实验结束后，汇总结果并绘制步数/耗时两种对照曲线。"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    runs = []
    results = []
    for folder in sorted(args.directory.glob("gpu*-lr*")):
        if not (folder / "metrics.csv").exists():
            results.append({"directory": str(folder), "status": "failed_before_logging"})
            continue
        with (folder / "metrics.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        configuration = json.loads((folder / "config.json").read_text())["arguments"]
        summary = json.loads((folder / "summary.json").read_text()) if (folder / "summary.json").exists() else {"status": "failed"}
        identity = json.loads((folder / "wandb_run.json").read_text()) if (folder / "wandb_run.json").exists() else None
        result = {"max_learning_rate": configuration["max_learning_rate"], "directory": str(folder), "wandb": identity, **summary}
        results.append(result)
        runs.append((configuration["max_learning_rate"], rows, summary["status"]))
    completed = [item for item in results if item["status"] == "completed" and item.get("final_valid_loss") is not None]
    winner = min(completed, key=lambda item: item["final_valid_loss"]) if completed else None
    report = {"runs": results, "best_completed_run": winner, "target_valid_loss": 1.45, "target_reached": bool(winner and winner["final_valid_loss"] <= 1.45), "divergence_observed": any(item["status"] == "diverged" for item in results)}
    (args.directory / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for x_name, label, suffix in (("step", "Gradient steps", "steps"), ("elapsed_seconds", "Elapsed training time (seconds)", "time")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for lr, rows, status in runs:
            training = [(float(row[x_name]), float(row["train_loss"])) for row in rows if math.isfinite(float(row["train_loss"]))]
            validation = [(float(row[x_name]), float(row["valid_loss"])) for row in rows if row["valid_loss"] and math.isfinite(float(row["valid_loss"]))]
            name = f"LR={lr:g} ({status})"
            if training:
                axes[0].plot(*zip(*training), label=name, linewidth=1, alpha=0.8)
            if validation:
                axes[1].plot(*zip(*validation), label=name, linewidth=1.4)
        axes[0].set_yscale("log")
        axes[0].set_title("Training batch loss (log scale)")
        axes[1].set_title("Mean validation loss")
        axes[1].axhline(1.45, linestyle="--", color="gray", label="Assignment target: 1.45")
        for ax in axes:
            ax.set_xlabel(label)
            ax.set_ylabel("Loss per token")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.directory / f"lr_comparison_{suffix}.png", dpi=170)
        plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
