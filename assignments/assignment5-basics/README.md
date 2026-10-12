# CS336 Spring 2026 Assignment 5: Alignment

代码基于 Stanford 的 [Assignment 5 起始仓库](https://github.com/stanford-cs336/assignment5-alignment)，保留 `cs336_alignment/`、`tests/adapters.py`、官方测试、测试夹具、依赖文件及提交脚本的结构。本仓库沿用 `assignments/assignment5-basics/` 作为目录名；对应作业为 Alignment / Reasoning RL。

作业说明：[主作业 PDF](cs336_spring2026_assignment5_alignment.pdf)。[Safety / RLHF 补充作业](cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)为可选内容，当前未实现。

## 当前完成范围

| 内容 | 状态 | 入口 |
| --- | --- | --- |
| Prompting：question-only、zero-shot R1、3-shot R1 | 已实现评测与结果保存 | `scripts/evaluate_a5_prompting.py` |
| 提示与回答分词、下一词元标签、回答掩码 | 已实现 | `cs336_alignment/grpo.py` |
| 回答 token log probability 与 entropy | 已实现 | `cs336_alignment/grpo.py` |
| 官方评分器调用与奖励统计 | 已实现 | `cs336_alignment/grpo.py` |
| 组内 mean / none baseline；std / none / mean 归一化 | 已实现，支持 GRPO、Dr.GRPO、RFT、MaxRL 的回答权重 | `cs336_alignment/grpo.py` |
| On-policy token loss、sequence loss 汇总、梯度累积与裁剪 | 已实现 | `cs336_alignment/grpo.py` |
| 标准 on-policy GRPO 训练循环 | 已实现，包含验证、vLLM 权重同步、W&B、回答样例和最终 checkpoint | `scripts/train_grpo.py` |
| 顺序运行多个种子及八项指标汇总图 | 已实现 | `scripts/run_grpo_seed_sweep.py`、`scripts/plot_grpo_seed_sweep.py` |
| 固定 loss 归一化、跳过零 advantage 回答、变体训练步 | 已实现，单训练 GPU 剪枝；DDP 保留同步 batch | `cs336_alignment/grpo.py` |
| MaxRL、RFT、Dr.GRPO 顺序实验队列 | 已实现，从完成的 seed0 基线复制实验配置 | `scripts/run_grpo_variants.py` |
| 实际采样 token 轨迹训练、标签一致性检查 | 已实现，保留采样 IDs；字符串接口用于课程 adapter | `cs336_alignment/grpo.py`、`scripts/train_grpo.py` |
| Off-policy、importance weighting、PPO clipping、GSPO | 待实现 | 第六节 |

训练循环支持 `--method grpo/grpo_constant/dr_grpo/rft/maxrl`。完整作业测试中仍包含待实现的 off-policy 部分，因此不能将本分支视为全部作业完成。

## 环境与测试

基础测试使用 Python 3.12 与 uv：

```bash
cd assignments/assignment5-basics
uv sync --locked
uv run pytest tests/test_grpo.py -k compute_group_normalized_rewards -q
bash scripts/test_standard_grpo.sh
```

`test_standard_grpo.sh` 仅选择已完成的组件与补充集成测试，强制使用 CPU。官方快照中的小模型权重由 Transformers 5.7.0 环境生成；历史 GPU 训练环境使用 Transformers 4.57.6，直接混用可能导致随机初始化模型的训练步快照不匹配。需要复用现有训练环境时，可以将独立的快照依赖目录设置为 `CS336_A5_TEST_DEPS`：

```bash
# 在已经激活的训练环境中，仅安装到独立目录，不覆盖训练环境。
uv pip install --python "$(command -v python)" --target .test-deps \
  --no-deps transformers==5.7.0 huggingface-hub==1.5.0
export CS336_A5_TEST_DEPS="$PWD/.test-deps"
bash scripts/test_standard_grpo.sh
```

2026-10-11：脱敏后的独立目录中运行上述 CPU 测试入口，结果为 **15 passed、10 deselected**，包括九个已完成官方组件测试、两个真实评分器检查、三个训练步集成检查及一个 DDP / ZeRO 检查。未完成部分不计入通过范围。Python / shell 语法与离线依赖锁定检查通过；本次发布未重新启动 GPU 实验。

同日补上固定归一化和完整 on-policy 变体后，CPU 检查为 **20 passed、5 deselected**，包括固定 loss 汇总及四种变体训练步的官方快照；off-policy 测试仍不计入通过范围。

2026-10-12：补上实际采样轨迹路径后，CPU 检查为 **24 passed、5 deselected**。新增检查覆盖重复 token 保留、EOS 与 padding 掩码、GRPO / RFT 实际动作目标的参数更新一致性，以及不完整轨迹输入的拒绝。

GPU 训练需要 Linux、CUDA、PyTorch、vLLM 与 W&B。`pyproject.toml` 中的 `gpu` extra 描述 GPU 依赖，`plots` extra 包含 W&B；汇总绘图的 matplotlib 单独列在 `requirements-plots.txt`，可运行 `uv pip install --python "$(command -v python)" -r requirements-plots.txt` 安装，保留现有训练依赖锁定文件。GPU / attention backend 兼容性需按设备配置；两卡入口支持 `--attention-implementation sdpa`。单卡交替生成与训练入口保留，但不以两卡验证代替其 GPU 实测。

## 配置数据与模型

模型为 OLMo-2-0425-1B；数据文件使用 GSM8K JSONL，每行包含 `question` 与 `answer`，答案文本应包含 `####`。默认资源位置为作业目录内的 `models/OLMo-2-0425-1B/` 与 `data/gsm8k/{train,test}.jsonl`，这些目录不纳入 Git。

可通过环境变量或命令行指定资源位置；下面路径为示例，需要替换为自己的目录：

```bash
export CS336_A5_ENV="/path/to/venv"               # 可省略，默认当前环境或 .venv
export CS336_A5_MODEL_DIR="/path/to/OLMo-2-0425-1B"
export CS336_A5_DATA_DIR="/path/to/gsm8k"
source ./activate.sh
```

`activate.sh` 不依赖仓库外的私人工作区脚本。`--model`、`--train-file`、`--val-file` 可以覆盖上述默认值。W&B 使用当前登录账号，或通过 `--wandb-entity` 指定；API key 仅保存在用户本地配置或环境变量中。

## Prompting

```bash
python scripts/evaluate_a5_prompting.py \
  --gpu 0 --output-dir outputs/prompting-seed0 --seed 0
```

默认分别评测三个模板，采样参数为 temperature 1、top-p 1、max tokens 512、每题一条回答。R1 模板使用 `</answer>` 停止字符串并保留该字符串。评分器保持课程提供的规则，不放宽标签解析。

脚本保存配置哈希、逐题回答、三类评分结果、正确率、格式率、输出长度、结束原因及供人工审查的样例。`--limit` 只用于短流程检查；完整评测要求 1319 道 GSM8K test 题。`--test-file` 支持指定数据位置。

## 标准 GRPO 训练

```bash
# 同一机器：GPU 0 训练，GPU 1 生成。
bash scripts/run_grpo_two_gpu.sh smoke --rollout-batch-size 8 \
  --val-examples 8 --attention-implementation sdpa --no-save-model

bash scripts/run_grpo_two_gpu.sh pilot --seed 0 --microbatch-size 8 \
  --attention-implementation sdpa

# 顺序运行两个种子，每个 200 个 rollout 更新。
python scripts/run_grpo_seed_sweep.py --seeds 0 1 \
  --output-dir outputs/grpo-baseline-two-seeds
```

`smoke` 为 2 步，`pilot` 为 50 步，`standard` 为 200 步。正式配置使用 6400 道训练题、1024 道固定验证题，每轮 32 道题 × 每题 8 条回答，共 256 条回答；学习率 1e-5、AdamW betas (0.9, 0.95)、weight decay 0、梯度裁剪 1、temperature 1、max new tokens 512。每 10 步验证，每 40 步保存训练回答样例。

两卡配置中，microbatch 8 对应 32 次梯度累积，最后只调用一次 optimizer.step。训练使用 FP32 主参数及 AdamW 状态、BF16 autocast 与 gradient checkpointing；生成使用 BF16 vLLM。四卡 DDP / ZeRO 入口为 `scripts/run_grpo_3090.sh`。自定义 NCCL 动态库仅在显式设置 `VLLM_NCCL_SO_PATH` 时加载。

真实训练入口将 prompt 以 token IDs 发送给 vLLM，保存返回的完整回答 IDs，并直接据此构建训练 labels 和 response_mask；每次更新前逐条校验回答标签。文字仍用于奖励判分及样例展示。配置记录 `trajectory_token_source`，W&B 记录 `debug/sampled_token_ids_used` 和完整轨迹的回答 token 数。

作业完整基线要求四个随机种子，可显式指定 `--seeds 0 1 2 3`；本分支默认队列为两个种子，不代表已满足四种子的要求。每个种子都从原始预训练模型开始，脚本验证跨种子源代码一致。输出目录已有配置时拒绝覆盖。

W&B 记录 loss、裁剪前梯度范数、token entropy、训练及验证奖励/格式率、验证回答长度，还包括截断率、运行时间与显存。真正的更新计数为 `rollout_step`；W&B 默认 `_step` 包含多次日志写入，可能超过训练步数。

样例包括训练回答、固定验证题、按评分结果选取的回答和训练前后对比 HTML。最后保存 BF16 `final-model`，不保存优化器状态，因此不支持精确续训。

汇总入口：

```bash
python scripts/plot_grpo_seed_sweep.py --sweep-dir outputs/grpo-baseline-two-seeds
```

生成八项指标的 PNG / PDF / SVG、各种子曲线、均值及样本标准差、HTML 报告与汇总 JSON。单个已完成种子不绘制标准差带，未完成种子不补造结果。`--upload-wandb` 可将汇总上传至当前登录账号。

## 两卡 seed0 配置

同一机器的 GPU 0 训练、GPU 1 生成；模型和数据路径沿用上面的环境变量。配置入口显式保存本次 200 次更新的训练与采样参数：

```bash
bash configs/grpo_rft_seed0.sh grpo
bash configs/grpo_rft_seed0.sh rft
```

分别从原始预训练模型开始。可追加 `--output-dir`、`--port`、`--wandb-mode` 等参数覆盖默认值。运行前准备资源及登录 W&B；配置脚本不包含服务器连接或账号信息。

## W&B 训练曲线

[GRPO / RFT seed0 曲线](figures/grpo_rft_seed0_curves.png) · [SVG](figures/grpo_rft_seed0_curves.svg)。曲线使用 W&B 导出的未平滑数值，横轴为 `rollout_step`。对应数值保存在 `figures/grpo_seed0_metrics.csv` 和 `figures/rft_seed0_metrics.csv`；空值表示该步未记录该指标，未补值或插值。

loss 的定义随方法变化；GRPO 的训练 entropy 统计全部回答，RFT 统计非零 advantage 回答。两组均为 seed0，验证每 10 次更新一次。

## On-policy 变体实验

```bash
python scripts/run_grpo_variants.py \
  --baseline-dir outputs/grpo-baseline/seed-0 \
  --output-dir outputs/onpolicy-variants-seed0
```

顺序运行 MaxRL、RFT、Dr.GRPO，各 seed0、200 步，复制基线的模型、数据、采样及优化器设置。每组从原始预训练模型独立开始；不续训基线权重，也不运行第二个随机种子。队列使用不同端口，保存配置哈希、日志、状态及 W&B 链接。

可用 `--methods grpo rft --port-base 18340` 只运行 GRPO 与 RFT，仍从原始预训练模型各自开始。指定新的输出目录，避免覆盖已有实验。

三种方法使用固定全局分母 `rollout_batch_size * max_tokens`，单卡训练时为 256 × 512。传入 `grpo_train_step` 的 `normalization_constant` 是完整本地 batch 的分母；各 microbatch 的 token loss 除以该分母后直接相加，不再额外除以梯度累积次数。剪掉零 advantage 回答后分母保持不变。

单训练 GPU 下，变体的 `train/token_entropy` 仅统计实际送入模型的非零 advantage 回答，配置中以 `entropy_scope` 标明；它与基线的全回答训练 entropy 不能直接作为同分布比较。奖励、格式率、长度及验证仍统计完整采样结果。所有方法保留训练和验证回答表、固定样例及训练前后 HTML。

此预算配置只运行三种方法各一个随机种子，尚未运行作业要求的 GRPO_constant 实验或四种子对照。

## 课程提交形式与排除范围

保留课程 `test_and_make_submission.sh`，从作业目录运行会执行 GRPO 测试并生成 `code.zip`。当前仍有待实现测试，脚本按课程原版设计在测试失败后继续打包；生成 zip 不表示完整测试通过。

GitHub 中保留官方测试所需的两份小型 GPT-2 模型夹具；它们来自课程起始仓库，属于测试资源。仅公开训练代码、可移植配置和脱敏后的 W&B 指标曲线。个人训练权重、GSM8K 数据、生成回答、原始 W&B 文件、日志、环境、编辑器私人配置、认证文件与学习笔记均不上传。提交 zip 另按课程规则排除模型夹具和数据，并增加私人配置与运行输出的排除项。

## 来源与修改范围

上游起始版本：[c2734a26308710949fe13226960a1e8cece94b7e](https://github.com/stanford-cs336/assignment5-alignment/commit/c2734a26308710949fe13226960a1e8cece94b7e)。官方测试与快照保留，adapter 接入个人实现；新增 GRPO 实现、prompting / 训练 / 汇总脚本、集成检查与使用说明。辅助代码、提示模板、作业 PDF、官方更新日志及 AI 指导文件保留上游来源。
