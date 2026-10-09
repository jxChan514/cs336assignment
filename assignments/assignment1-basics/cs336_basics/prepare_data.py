"""准备调试数据：读取 TinyStories 小样本，按完整故事划分训练和验证文本。"""

import sys  # 直接运行文件时，将作业目录加入 Python 的模块搜索路径。
from pathlib import Path  # 定位样本文件，并以 UTF-8 读取文本。

# 支持直接运行本文件，也支持通过 python -m 按包运行。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
import numpy as np


# ---------- 1. 路径与划分配置 ----------
# 当前文件位于 cs336_basics 中，往上两级就是 assignment1-basics。
project_root = Path(__file__).resolve().parent.parent
sample_path = project_root / "tests" / "fixtures" / "tinystories_sample.txt"
special_token = "<|endoftext|>"  # 样本中每篇故事结束时的标记。
train_ratio = 0.9  # 训练故事数量占比；取整数后，小样本的实际比例可能不同。


# ---------- 2. 读取原始文本，再拆成故事列表 ----------
text = sample_path.read_text(encoding="utf-8")  # str：整个文件的文本。
stories = text.split(special_token)  # list[str]：每一项是一篇故事，结束标记被移除。

# 最后一个结束标记后可能只有换行；过滤这种空白片段。
# story.strip() 只用于判断，列表中仍保留原始 story，不删除故事内的空格和换行。
stories = [story for story in stories if story.strip()]


# ---------- 3. 按故事数量划分，保留完整故事 ----------
# 当前样本有 5 篇故事：int(5 * 0.9) = 4，因此前 4 篇训练，最后 1 篇验证。
split_index = int(len(stories) * train_ratio)
train_stories = stories[:split_index]  # list[str]：训练故事列表。
valid_stories = stories[split_index:]  # list[str]：验证故事列表。


# ---------- 4. 分别拼回文本，恢复故事结束标记 ----------
# join 在故事之间插入标记；最后再补一个标记，表示最后一篇故事也已结束。
train_text = special_token.join(train_stories) + special_token  # str：训练文本。
valid_text = special_token.join(valid_stories) + special_token  # str：验证文本。

print(f"故事总数：{len(stories)}")
print(f"训练集：{len(train_stories)} 篇故事，{len(train_text)} 个字符")
print(f"验证集：{len(valid_stories)} 篇故事，{len(valid_text)} 个字符")

# 下一步：把 train_text 保存到文件，供 train_bpe 学习词表和合并规则。
# 然后用同一个 Tokenizer 分别编码 train_text 和 valid_text。

# train_bpe 接受文件路径，因此先把训练文本保存成文件。
train_text_path = project_root / "data" / "train.txt"
train_text_path.parent.mkdir(parents=True, exist_ok=True)
train_text_path.write_text(train_text, encoding="utf-8")

# 用关键字参数传入三个值。
vocab_train, merges = train_bpe(
    input_path=train_text_path,
    vocab_size=10_000,
    special_tokens=[special_token],
)
tokenizer=Tokenizer(vocab=vocab_train,merges=merges,special_tokens=[special_token])
train_ids = tokenizer.encode(train_text)
valid_ids = tokenizer.encode(valid_text)


train_arr = np.asarray(train_ids, dtype=np.uint16)
valid_arr=np.asarray(valid_ids,dtype=np.uint16)
tokenizer.save("data/vocab.json","data/merges.json")


# train.py 在 cs336_basics 下，project_root 定位到 assignment1-basics
project_root = Path(__file__).parent.parent

# 拼接路径，data文件夹在作业根目录
train_bin_path = project_root / "data" / "train.bin"
valid_bin_path = project_root / "data" / "valid.bin"


train_arr.tofile(train_bin_path)
valid_arr.tofile(valid_bin_path)

