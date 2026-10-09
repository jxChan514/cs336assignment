"""正式 TinyStories 实验的数据适配：复用自己的 BPE，按特殊 token 边界并行处理。

原来的 prepare_data.py 继续用于学习和小样本调试。本脚本单独保存正式数据，
不会覆盖 data/train.bin、data/valid.bin 或原来的 765 词词表。
"""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import heapq
import json
from pathlib import Path
import pickle
import time

import numpy as np

from cs336_basics.bpe import (
    init_vocab_and_merges, initialize_pair_statistics, merge_token_sequence,
    pre_token_split, regex_pre_tokenize, str_to_bytes_pre_tokens,
)
from cs336_basics.tokenizer import Tokenizer

SPECIAL = "<|endoftext|>"


def document_ranges(path: Path, chunk_size: int = 4 * 1024**2):
    """每块结束在完整的 EOS 后；不会把一个单词或 UTF-8 字符截断。"""
    marker = SPECIAL.encode()
    size = path.stat().st_size
    start = 0
    with path.open("rb") as file:
        while start < size:
            file.seek(start)
            block = file.read(chunk_size)
            while start + len(block) < size and marker not in block:
                block += file.read(chunk_size)
            if start + len(block) == size:
                end = size
            else:
                end = start + block.rfind(marker) + len(marker)
            yield str(path), start, end
            start = end


def read_range(item):
    path, start, end = item
    with open(path, "rb") as file:
        file.seek(start)
        return file.read(end - start).decode("utf-8")


def count_range(item):
    return regex_pre_tokenize(pre_token_split(read_range(item), [SPECIAL]))


def ordered_results(pool, function, items, workers):
    """只预取有限的任务；按原文顺序写出结果，避免把整个语料堆进内存。"""
    items = iter(items)
    pending = deque()
    for _ in range(2 * workers):
        item = next(items, None)
        if item is not None:
            pending.append((item, pool.submit(function, item)))
    while pending:
        item, future = pending.popleft()
        yield item, future.result()
        following = next(items, None)
        if following is not None:
            pending.append((following, pool.submit(function, following)))


class PairPriority:
    """堆顶为最高频 pair；同频时保持作业要求的最大字典序。"""
    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair


def train_from_counts(counts, vocab_size):
    """复用 bpe.py 的统计/合并函数，只用堆加速最佳 pair 的查询。"""
    vocab, merges = init_vocab_and_merges([SPECIAL])
    byte_counts = str_to_bytes_pre_tokens(counts)
    sequences, frequencies = list(byte_counts), list(byte_counts.values())
    pair_counts, index = initialize_pair_statistics(sequences, frequencies)
    heap = [(-count, PairPriority(pair)) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    while len(vocab) < vocab_size:
        while heap:
            negative_count, candidate = heapq.heappop(heap)
            best = candidate.pair
            if -negative_count == pair_counts.get(best, 0):
                break
        else:
            break
        touched = set()
        for sequence_id in tuple(index.pop(best, ())):
            old = sequences[sequence_id]
            frequency = frequencies[sequence_id]
            old_pairs = list(zip(old, old[1:]))
            for pair in old_pairs:
                pair_counts[pair] -= frequency
                touched.add(pair)
            for pair in set(old_pairs):
                holders = index.get(pair)
                if holders is not None:
                    holders.discard(sequence_id)
                    if not holders:
                        index.pop(pair, None)
            new = merge_token_sequence(old, best)
            sequences[sequence_id] = new
            new_pairs = list(zip(new, new[1:]))
            for pair in new_pairs:
                pair_counts[pair] += frequency
                touched.add(pair)
            for pair in set(new_pairs):
                index.setdefault(pair, set()).add(sequence_id)
        for pair in touched:
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
            else:
                heapq.heappush(heap, (-pair_counts[pair], PairPriority(pair)))
        vocab[len(vocab)] = best[0] + best[1]
        merges.append(best)
        if len(merges) % 500 == 0:
            print(f"BPE merges={len(merges)} vocab={len(vocab)}", flush=True)
    return vocab, merges


class CachedTokenizer(Tokenizer):
    """重复单词直接复用分词结果，规则仍由自己的 Tokenizer.apply_bpe 决定。"""
    @lru_cache(maxsize=200_000)
    def _cached_bpe(self, tokens):
        return super().apply_bpe(tokens)

    def apply_bpe(self, tokens):
        return self._cached_bpe(tuple(tokens))


_tokenizer = None


def init_encoder(directory):
    global _tokenizer
    directory = Path(directory)
    _tokenizer = CachedTokenizer.from_files(directory / "vocab.json", directory / "merges.json", [SPECIAL])


def encode_range(item):
    ids = _tokenizer.encode(read_range(item))
    if ids and max(ids) >= 65536:
        raise ValueError("词表超出 uint16 的范围")
    return np.asarray(ids, dtype="<u2").tobytes()


def encode_file(path, target, directory, workers):
    digest = hashlib.sha256()
    count = 0
    part = target.with_suffix(".bin.part")
    with ProcessPoolExecutor(max_workers=workers, initializer=init_encoder, initargs=(directory,)) as pool:
        with part.open("wb") as file:
            for item, encoded in ordered_results(pool, encode_range, document_ranges(path), workers):
                file.write(encoded)
                digest.update(encoded)
                count += len(encoded) // 2
                print(f"encode {path.name}: {item[2] / path.stat().st_size:.1%}, tokens={count:,}", flush=True)
    part.replace(target)
    return {"tokens": count, "bytes": target.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--valid-text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or not 257 <= args.vocab_size <= 65536:
        parser.error("workers 必须为正数，vocab-size 需要在 [257,65536] 内")
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # 缓存由本脚本创建，仅供可信的本地续跑使用。
    cache = directory / "pre_token_counts.pkl"
    if not (directory / "vocab.json").exists():
        if cache.exists():
            with cache.open("rb") as file:
                counts = pickle.load(file)
        else:
            counts = Counter()
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for item, result in ordered_results(pool, count_range, document_ranges(args.train_text), args.workers):
                    counts.update(result)
                    print(f"count {item[2] / args.train_text.stat().st_size:.1%}: {len(counts):,} unique pre-tokens", flush=True)
            with cache.open("wb") as file:
                pickle.dump(counts, file, protocol=5)
        vocab, merges = train_from_counts(counts, args.vocab_size)
        if len(vocab) != args.vocab_size:
            raise ValueError(f"实际词表只有 {len(vocab)}，不足 {args.vocab_size}")
        Tokenizer(vocab, merges, [SPECIAL]).save(directory / "vocab.json", directory / "merges.json")
    metadata = {"dtype": "uint16", "vocab_size": args.vocab_size, "special_tokens": [SPECIAL], "sources": {}}
    for split, path in (("train", args.train_text), ("valid", args.valid_text)):
        target = directory / f"{split}.bin"
        # 写出新二进制前保留旧结果；本脚本不会碰小样本的 data/train.bin。
        if target.exists():
            target.replace(target.with_suffix(f".bin.previous-{time.time_ns()}"))
        metadata[split] = encode_file(path, target, directory, args.workers)
        metadata["sources"][split] = {"path": str(path), "bytes": path.stat().st_size}
    metadata["preparation_seconds"] = time.perf_counter() - started
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
