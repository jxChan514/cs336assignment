"""将 metrics.csv 画成以训练步数、累计训练时间为横轴的两张 loss 曲线。"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

# 使用文件绘图后端，服务器没有图形界面时也能保存 PNG。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metrics(metrics_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """读取 CSV 并保存图片；也接受直接传入实验目录。"""
    metrics_path = Path(metrics_path)
    if metrics_path.is_dir():
        metrics_path = metrics_path / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"step", "elapsed_seconds", "train_loss", "valid_loss"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"缺少绘图需要的 CSV 列：{metrics_path}")
        rows = list(reader)
    if not rows:
        return []

    destination = Path(output_dir) if output_dir is not None else metrics_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    train_losses = [float(row["train_loss"]) for row in rows]
    valid_rows = [row for row in rows if row["valid_loss"] != ""]
    valid_losses = [float(row["valid_loss"]) for row in valid_rows]
    nonfinite_count = sum(not math.isfinite(value) for value in train_losses + valid_losses)
    # 发散实验的 NaN/Inf 保留在 CSV 中；画图时断开曲线并标注数量。
    train_plot = [value if math.isfinite(value) else math.nan for value in train_losses]
    valid_plot = [value if math.isfinite(value) else math.nan for value in valid_losses]

    images = []
    for column, xlabel, filename in (
        ("step", "Training steps", "loss_vs_steps.png"),
        ("elapsed_seconds", "Elapsed training time (seconds)", "loss_vs_time.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        ax.plot([float(row[column]) for row in rows], train_plot,
                label="Training batch loss", color="#2563eb", linewidth=1.6, marker="o", markersize=3)
        if valid_rows:
            ax.plot([float(row[column]) for row in valid_rows], valid_plot,
                    label="Mean validation loss", color="#ea580c", linewidth=1.8, marker="o", markersize=4)
        ax.set(xlabel=xlabel, ylabel="Cross-entropy loss (nats/token)", title="Training and validation loss")
        ax.grid(alpha=0.25)
        ax.legend()
        if nonfinite_count:
            ax.text(0.02, 0.98, f"Non-finite loss records: {nonfinite_count}",
                    transform=ax.transAxes, va="top", color="#b91c1c")
        path = destination / filename
        fig.savefig(path, dpi=170)
        plt.close(fig)
        images.append(path)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="metrics.csv 的路径，或包含它的实验目录")
    parser.add_argument("--output-dir", type=Path, help="图片保存目录，默认与 CSV 放在一起")
    args = parser.parse_args()
    images = plot_metrics(args.metrics, args.output_dir)
    for path in images:
        print(f"已保存曲线：{path}")
    if not images:
        print("CSV 尚无指标记录，暂不生成曲线。")


if __name__ == "__main__":
    main()
