"""字节级 BPE Tokenizer。

模块顺序：初始化 → 文件保存/加载 → 查表/解码 → BPE 合并 → 编码/流式编码。
训练在 bpe.py 中完成；此处只应用已有词表和合并规则，不学习新规则。
"""

import json
from collections.abc import Iterable, Iterator, Sequence
from os import PathLike

import regex as re
from cs336_basics.bpe import PAT, merge_token_sequence


class Tokenizer:
    """保存训练好的词表与合并规则，供后续编码、解码使用。"""

    # ---------- 1. 初始化：准备正向词表、反向词表、合并优先级 ----------
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """接收 ID 到字节串的词表、按学习顺序排列的合并规则及特殊 token。"""

        # 解码用的正向词表：例如 257 → b"ab"。
        # 复制输入，避免补充特殊 token 时修改调用方的原始词表。
        self.vocab = dict(vocab)
        # 保存已有合并规则；初始化时不重新训练，也不执行合并。
        self.merges = list(merges)
        # 未提供特殊 token 时统一使用空列表，后续可以安全遍历和判断成员。
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        if "" in self.special_tokens:
            raise ValueError("特殊 token 不能为空字符串")
        # 交换词表的 key/value，方便编码时由字节串查 ID：b"ab" → 257。
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        # 已存在的特殊 token 保留 ID；缺失的从最大 ID + 1 开始添加。
        next_token_id = max(self.vocab, default=-1) + 1
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                self.vocab[next_token_id] = token_bytes
                self.token_to_id[token_bytes] = next_token_id
                next_token_id += 1

        self.merge_ranks = {}
        for rank, pair in enumerate(self.merges):
            # rank 越小，训练时越早出现，编码时越优先合并。
            self.merge_ranks.setdefault(pair, rank)

        self._special_token_set = set(self.special_tokens)
        self._special_token_pattern = None
        if self.special_tokens:
            # 正则从左到右尝试分支：长 token 优先，避免被短 token 抢先切开。
            ordered_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped_specials = [re.escape(token) for token in ordered_specials]
            self._special_token_pattern = re.compile("(" + "|".join(escaped_specials) + ")")
        self._pre_token_pattern = re.compile(PAT)

    # ---------- 2. 文件接口：用 JSON 无损保存任意字节 ----------
    def save(self, vocab_filepath: str | PathLike, merges_filepath: str | PathLike) -> None:
        """保存训练结果，供 from_files 加载。

        vocab JSON：{"0": [0], "257": [97, 98], ...}，即 ID → 字节整数列表。
        merges JSON：[[[97], [98]], ...]，即按 rank 排序的字节串对。
        任意 token 都可能包含不完整 UTF-8，因此不能直接把 bytes 解码成文本保存。
        训练后可调用 Tokenizer(vocab, merges, special_tokens).save(...)。
        """
        with open(vocab_filepath, "w", encoding="utf-8") as file:
            json.dump({token_id: list(token) for token_id, token in self.vocab.items()}, file)
        with open(merges_filepath, "w", encoding="utf-8") as file:
            json.dump([[list(left), list(right)] for left, right in self.merges], file)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike,
        merges_filepath: str | PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """加载 save 写出的两个 JSON 文件；特殊 token 列表由调用方另行传入。

        这是本实现的字节整数格式，不是 GPT-2 的 Unicode 映射词表格式。
        """
        with open(vocab_filepath, encoding="utf-8") as file:
            serialized_vocab = json.load(file)
        with open(merges_filepath, encoding="utf-8") as file:
            serialized_merges = json.load(file)
        vocab = {int(token_id): bytes(token) for token_id, token in serialized_vocab.items()}
        merges = [(bytes(left), bytes(right)) for left, right in serialized_merges]
        return cls(vocab, merges, special_tokens)

    # ---------- 3. 查表与解码：bytes token ↔ 整数 ID ----------
    def tokens_to_ids(self, tokens: Sequence[bytes]) -> list[int]:
        """将已经分好的 bytes token 按原顺序转换为整数 ID 列表。

        例如 [b"abab", b"ab"] → [258, 257]。
        此方法只负责查表，输入不是原始文本，不在这里执行 BPE 分词。
        """

        token_ids = []
        for token_bytes in tokens:
            token_id = self.token_to_id[token_bytes]
            token_ids.append(token_id)
        return token_ids

    def decode(self, ids: list[int]) -> str:
        """按 ID 查出字节串，拼接后以 UTF-8 解码为文本。"""

        token_bytes_list = []
        for token_id in ids:
            # ID 是整数，因此查询 self.vocab，而不是 self.token_to_id。
            token_bytes = self.vocab[token_id]
            token_bytes_list.append(token_bytes)
        # 一个字符的 UTF-8 字节可能分散在多个 token 中，必须先拼接再解码。
        # 单个 ID 可能只对应一个字符的部分字节；非法 UTF-8 按要求用 � 替代。
        decoded_text = b"".join(token_bytes_list).decode("utf-8", errors="replace")
        return decoded_text

    # ---------- 4. BPE：只在当前预分词片段内部，按训练 rank 反复合并 ----------
    def find_best_merge_pair(self, split_tokens: Sequence[bytes]) -> tuple[bytes, bytes] | None:
        """返回当前相邻 pair 中 rank 最小的一对；没有可合并项则返回 None。"""
        best_pair = None
        best_rank = float("inf")
        for position in range(len(split_tokens) - 1):
            pair = (split_tokens[position], split_tokens[position + 1])
            if pair in self.merge_ranks:
                rank = self.merge_ranks[pair]
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
        return best_pair

    def apply_bpe(self, tokens: Sequence[bytes]) -> tuple[bytes, ...]:
        """如 [b'a', b'b', b'a', b'b'] → (b'abab',)，结果始终为元组。"""
        tokens = tuple(tokens)
        while True:
            best_pair = self.find_best_merge_pair(tokens)
            if best_pair is None:
                break
            tokens = merge_token_sequence(tokens, best_pair)
        return tokens

    # ---------- 5. 编码：文本 → 特殊/普通片段 → 字节 → BPE → ID ----------
    def encode(self, text: str) -> list[int]:
        """将原始文本转换为按原文顺序排列的一维 ID 列表。

        以下用 text = "abab<|endoftext|>ab" 举例。
        特殊 token 在同一位置存在多个匹配时，优先匹配最长的。
        """
        # list[int]：最终收集结果的容器，例如 [258, 256, 257]。
        # 一个 ID 用 append；一组 ID 用 extend，避免得到 [[258], [256], [257]]。
        token_ids = []

        if self._special_token_pattern is not None:
            # 捕获组让切分结果保留特殊 token，以便直接查表。
            # list[str]：例如 ["abab", "<|endoftext|>", "ab"]，尚未转成字节或 ID。
            text_segments = self._special_token_pattern.split(text)
        else:
            # 没有特殊 token 时保留整段文本，随后仍进行 GPT-2 正则预分词。
            text_segments = [text]

        for segment_text in text_segments:
            # segment_text 是 str：一次取一个片段，例如 "abab" 或 "<|endoftext|>"。
            if segment_text == "":
                continue
            if segment_text in self._special_token_set:
                # special_token_bytes 是 bytes，例如 b"<|endoftext|>"；segment_text 保持 str。
                special_token_bytes = segment_text.encode("utf-8")
                # int：查到单个特殊 token 的 ID，例如 256。
                special_token_id = self.token_to_id[special_token_bytes]
                token_ids.append(special_token_id)
            else:
                # 匹配对象的迭代器，不是字符串列表；需要逐个取出并调用 .group()。
                # 例如 segment_text="abab ab!"，依次匹配 "abab"、" ab"、"!"。
                pre_token_matches = self._pre_token_pattern.finditer(segment_text)
                for pre_token_match in pre_token_matches:
                    # list[bytes]：仅收集当前一个预分词片段的单字节 token。
                    # 每个匹配都重新创建，不能把不同匹配的字节拼到一起做 BPE。
                    byte_tokens = []
                    # str：一个匹配片段，例如 "abab"；包含的前导空格也要保留。
                    pre_token_text = pre_token_match.group()
                    # bytes：整个片段编码后的字节串，例如 b"abab"，还没有拆成列表。
                    pre_token_bytes = pre_token_text.encode("utf-8")
                    for byte_value in pre_token_bytes:
                        # byte_value 是 int，例如 97；遍历 bytes 得到整数。
                        # byte_token 是长度为 1 的 bytes，例如 b"a"。
                        byte_token = bytes([byte_value])
                        byte_tokens.append(byte_token)
                    # 字节循环结束：byte_tokens == [b"a", b"b", b"a", b"b"]。
                    # 合并后的字节 token 序列，例如 (b"abab",)，还不是 ID。
                    # apply_bpe 始终返回 tuple[bytes, ...]。
                    merged_tokens = self.apply_bpe(byte_tokens)
                    # list[int]：当前一个预分词片段的 ID，例如 [258]。
                    pre_token_ids = self.tokens_to_ids(merged_tokens)
                    # 把当前片段的所有 ID 展开加入整段文本的结果，保留顺序。
                    token_ids.extend(pre_token_ids)
        return token_ids

    # ---------- 6. 流式编码：逐段读取，逐个产出 ID ----------
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """例如输入文件对象，按行编码，惰性产出 int，而不是 list[int]。

        只保留当前输入段及其 ID，不把整个文件读入或累计所有输出。
        每个元素独立编码；调用方应按行/文档提供文本，不要在词或特殊 token
        内部任意切块，否则分词 ID 可能与整段 encode 不同。
        """
        for text in iterable:
            yield from self.encode(text)


if __name__ == "__main__":
    # 仅用于查表和解码的小词表；完整 BPE 词表还包括 256 个基础字节 token。
    vocab = {
        256: b"<|endoftext|>",
        257: b"ab",
        258: b"abab",
    }
    merges = [(b"a", b"b"), (b"ab", b"ab")]
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    # 从已分好的 token 出发，验证 bytes → ID → 文本的转换。
    tokens = [b"abab", b"<|endoftext|>", b"ab"]
    token_ids = tokenizer.tokens_to_ids(tokens)

    decoded_text = tokenizer.decode(token_ids)

    print(tokenizer.merge_ranks)
    print(token_ids)
    print(decoded_text)

    print("查表测试通过")
