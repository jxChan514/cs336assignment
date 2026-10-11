"""Plot completed A5 seeds without inventing missing runs or smoothing metrics."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = {
    "train/loss": "Policy gradient loss",
    "train/grad_norm": "Gradient norm (before clipping)",
    "train/token_entropy": "Response token entropy",
    "train/reward": "Training reward",
    "train/format_reward": "Training format reward",
    "val/reward": "Validation reward / accuracy",
    "val/format_reward": "Validation format reward",
    "val/response_length": "Validation response length (tokens)",
}


def summarize(sweep_dir: Path, upload_wandb=False, wandb_entity=None):
    manifest = json.loads((sweep_dir / "sweep.json").read_text())
    entries = [entry for entry in manifest["runs"] if entry["status"] == "completed"]
    if not entries:
        raise ValueError("No completed seeds to aggregate")
    runs = []
    reference = None
    comparable = ["model", "prompt", "steps", "train_sha256", "val_sha256", "validation_ids",
                  "rollout_batch_size", "group_size", "microbatch_size", "learning_rate", "max_tokens",
                  "temperature", "script_sha256", "grpo_sha256", "prompt_sha256"]
    for entry in entries:
        # Resolve by seed under this sweep so copied results remain portable.
        output = sweep_dir / f"seed-{entry['seed']}"
        config = json.loads((output / "config.json").read_text())
        if config["seed"] != entry["seed"]:
            raise ValueError("Seed manifest and run config disagree")
        signature = {key: config[key] for key in comparable}
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("Runs differ in more than random seed")
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
        training_steps = [r["rollout_step"] for r in records if "train/loss" in r]
        if training_steps != list(range(1, config["steps"] + 1)):
            raise ValueError("A run is missing training steps")
        final = json.loads((output / "completed.json").read_text())
        if final["steps"] != config["steps"]:
            raise ValueError("Completion count disagrees with config")
        final_val = [r for r in records if "val/reward" in r][-1]
        if final_val["rollout_step"] != config["steps"] or final_val["val/reward"] != final["final_val_reward"]:
            raise ValueError("Final validation is missing or inconsistent")
        runs.append({"seed": entry["seed"], "records": records, "done": final})

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
                         "font.size": 15, "axes.linewidth": 2, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    figures = sweep_dir / "figures"
    figures.mkdir(exist_ok=True)
    palette = ["#3775BA", "#B64342", "#42949E", "#9A4D8E"]
    statistics = {}
    for key, label in METRICS.items():
        series = [{r["rollout_step"]: r[key] for r in run["records"] if key in r} for run in runs]
        steps = sorted(series[0])
        if any(sorted(values) != steps for values in series):
            raise ValueError(f"Metric {key} is missing steps in a seed")
        values = np.asarray([[mapping[step] for step in steps] for mapping in series], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Metric {key} contains NaN or infinity")
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if len(runs) > 1 else None
        statistics[key] = {"steps": steps, "mean": mean.tolist(),
                           "sample_std": std.tolist() if std is not None else None,
                           "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}
        fig, ax = plt.subplots(figsize=(9, 5.6))
        for index, run in enumerate(runs):
            ax.plot(steps, values[index], color=palette[index % len(palette)], linewidth=1.6,
                    alpha=0.65, linestyle="--", label=f"seed {run['seed']}")
        ax.plot(steps, mean, color="#0F4D92", linewidth=2.6, label=f"Mean ({len(runs)} seeds)")
        if std is not None:
            ax.fill_between(steps, mean - std, mean + std, color="#3775BA", alpha=0.18,
                            label="Mean ± sample SD")
        ax.set(xlabel="Rollout step", ylabel=label, title=label)
        if "reward" in key:
            ax.set_ylim(0, 1)
        ax.legend(fontsize=11, ncol=2)
        fig.tight_layout(pad=1.5)
        name = key.replace("/", "_")
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(figures / f"{name}.{extension}", dpi=300, facecolor="white")
        plt.close(fig)

    finals = np.asarray([run["done"]["final_val_reward"] for run in runs])
    summary = {"completed_seeds": [run["seed"] for run in runs], "requested_seeds": manifest["seeds"],
               "mean_final_val_reward": float(finals.mean()),
               "sample_std_final_val_reward": float(finals.std(ddof=1)) if len(runs) > 1 else None,
               "meets_assignment_accuracy_target": bool(finals.mean() >= 0.25),
               "meets_four_seed_requirement": len(runs) >= 4,
               "all_requested_seeds_completed": len(runs) == len(manifest["seeds"]),
               "final_per_seed": [{"seed": run["seed"], **run["done"]} for run in runs],
               "metric_statistics": statistics}
    (sweep_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    table = "".join(f'<tr><td>{run["seed"]}</td><td>{run["done"]["final_val_reward"]:.2%}</td>'
                    f'<td>{run["done"]["best_val_reward"]:.2%}</td>'
                    f'<td><a href="{html.escape(run["done"]["wandb_url"], quote=True)}">W&B</a> · '
                    f'<a href="seed-{run["seed"]}/samples/latest-comparison.html">回答对比</a></td></tr>' for run in runs)
    images = "".join(f'<section><h2>{html.escape(label)}</h2><img src="figures/{key.replace("/", "_")}.png"></section>'
                     for key, label in METRICS.items())
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>A5 基线汇总</title>'
            '<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:16px;'
            'color:#172033}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;'
            'text-align:left}img{width:100%;height:auto}section{margin:32px 0}</style><h1>A5 零样本 GRPO 基线</h1>'
            f'<p>已完成 {len(runs)}/{len(manifest["seeds"])} 个实验；最终验证奖励平均 {finals.mean():.2%}。'
            f'作业原要求 4 个 seeds，本次选择 {len(manifest["seeds"])} 个。阴影为样本标准差，不是置信区间；单个实验不画方差带。</p>'
            '<table><tr><th>Seed</th><th>最终奖励</th><th>最高奖励</th><th>记录</th></tr>' + table + '</table>' + images)
    (sweep_dir / "report.html").write_text(page)
    if upload_wandb:
        import wandb
        with wandb.init(project="cs336-a5-grpo", entity=wandb_entity, group=manifest["wandb_group"],
                        name=f"{manifest['wandb_group']}-summary", job_type="aggregate", dir=str(sweep_dir),
                        mode="online", config={"seeds": summary["completed_seeds"], "prompt": "r1_zero",
                                               "steps_per_seed": manifest["steps_per_seed"], "budget_reduced": True}) as run:
            run.log({f"aggregate/{key.replace('/', '_')}": wandb.Image(str(figures / f"{key.replace('/', '_')}.png"))
                     for key in METRICS})
            run.log({"final_validation": wandb.Table(columns=["seed", "final_reward", "best_reward"],
                data=[[r["seed"], r["done"]["final_val_reward"], r["done"]["best_val_reward"]] for r in runs])})
            run.summary.update({key: value for key, value in summary.items() if key != "metric_statistics"})
            summary["aggregate_wandb_url"] = run.url
        (sweep_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "metric_statistics"}, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--upload-wandb", action="store_true")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    summarize(args.sweep_dir.resolve(), args.upload_wandb, args.wandb_entity)


if __name__ == "__main__":
    main()
