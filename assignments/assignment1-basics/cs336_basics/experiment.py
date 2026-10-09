"""Assignment 1 §7.1：保存实验配置、指标和实验笔记。"""

import csv
import json
from datetime import datetime
from pathlib import Path
import shutil
import time


class ExperimentLogger:
    """一行 CSV 对应一个训练步；未执行验证时，valid_loss 留空。

    同一实验续训时，累计已记录的训练耗时，不计中途停机的时间。
    若恢复到较早的 checkpoint，先备份完整日志，再保留该步之前的记录。
    """

    fieldnames = ["step", "elapsed_seconds", "train_loss", "valid_loss", "lr"]

    def __init__(self, directory: Path, configuration: dict, start_step: int = 0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.directory / "metrics.csv"
        self.config_path = self.directory / "config.json"
        notes_path = self.directory / "notes.txt"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # ---------- 1. 读取旧指标；从头训练时不混入上次实验的数据 ----------
        previous_rows = []
        if self.metrics_path.exists():
            with self.metrics_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames != self.fieldnames:
                    raise ValueError(f"CSV 表头不匹配：{self.metrics_path}")
                previous_rows = list(reader)
        retained_rows = [row for row in previous_rows if int(row["step"]) <= start_step] if start_step else []

        # 保存旧日志和配置，随后再重建本次要继续使用的 CSV。
        if previous_rows != retained_rows or (not start_step and self.config_path.exists()):
            archive = self.directory / "log_history" / stamp
            archive.mkdir(parents=True, exist_ok=True)
            for path in (self.metrics_path, self.config_path, notes_path):
                if path.exists():
                    shutil.copy2(path, archive / path.name)

        with self.metrics_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(retained_rows)

        # ---------- 2. 保存参数；续训配置单独存放，保留最初的实验配置 ----------
        config_target = self.config_path
        if start_step and config_target.exists():
            config_target = self.directory / f"resume_config_{stamp}.json"
        with config_target.open("w", encoding="utf-8") as file:
            # Path、torch.dtype 等值转成字符串；数字、列表仍保留原类型。
            json.dump(configuration, file, ensure_ascii=False, indent=2, default=str)

        if not start_step or not notes_path.exists():
            purpose = configuration.get("arguments", {}).get("experiment_note", "")
            notes_path.write_text(
                f"实验目的或变更：{purpose}\n\n"
                "观察：\n\n结论：\n\n"
                "配置见 config.json；指标见 metrics.csv。\n",
                encoding="utf-8",
            )

        # ---------- 3. 从训练循环开始计时；续训时接上旧日志的累计耗时 ----------
        self.elapsed_offset = float(retained_rows[-1]["elapsed_seconds"]) if retained_rows else 0.0
        self.last_step = start_step
        self.has_records = bool(retained_rows)
        self.start_time = time.perf_counter()

    def log(self, step: int, train_loss: float, valid_loss: float | None, lr: float) -> None:
        """写入当前步的指标；train_loss 是本步 batch loss，valid_loss 是验证均值。"""
        if step <= self.last_step:
            raise ValueError(f"日志步数必须递增：{step} <= {self.last_step}")
        record = {
            "step": step,
            "elapsed_seconds": self.elapsed_offset + time.perf_counter() - self.start_time,
            "train_loss": train_loss,
            "valid_loss": valid_loss,  # csv 将 None 写成空单元格，不复用上一轮验证值。
            "lr": lr,
        }
        # 每次记录都关闭文件，程序中断时已写入的记录仍保留。
        with self.metrics_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self.fieldnames).writerow(record)
        self.last_step = step
        self.has_records = True
