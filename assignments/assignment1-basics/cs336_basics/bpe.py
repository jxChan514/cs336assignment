
"""字节级 BPE 训练的基础步骤。

数据依次经过：特殊 token 切分、正则预分词、UTF-8 字节化、
相邻 token 对计数、最佳 pair 选择，以及 pair 合并。
"""

from heapq import merge
import token

from numpy import byte
import regex as re
from collections import Counter

def init_vocab_and_merges(special_token_list:list):
    """初始化 256 个单字节 token、特殊 token，以及空的合并记录。"""

    # 每个可能的字节值 0～255 都对应一个基础 token，token ID 等于字节值。
    token_nums=256 
    vocab={}
    for byte_val in range(token_nums):
        vocab[byte_val]=bytes([byte_val])
    
    # 特殊 token 作为完整的 UTF-8 字节串加入词表，不参与普通文本的 BPE 合并。
    for special_token in special_token_list:
        special_token_encode=special_token.encode("utf-8")
        
        vocab[token_nums]=special_token_encode
        token_nums=len(vocab)
    
    # merges 按学习顺序记录每轮被合并的 pair。
    merge=[]
    return vocab,merge

def pre_token_split(text:str,special_token_list:list):
    """按特殊 token 切开文本，防止 BPE 跨越特殊 token 边界合并。"""

    if not special_token_list:
        return [text]
    
    # 转义特殊 token 中可能具有正则含义的字符，再组合成“或”模式。
    escaped_specials= [re.escape(token) for token in special_token_list]
    pattern="|".join(escaped_specials)
    parts=re.split(pattern=pattern,string=text)
    return parts


# 作业给定的 GPT-2 预分词正则；匹配结果中的前导空格必须保留。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def regex_pre_tokenize(parts):
    """对普通文本片段做正则预分词，并统计每种匹配文本的出现次数。"""

    # Counter 的 key 是匹配文本，value 是该文本在语料中的频次。
    data=Counter()
    for part in parts:
        # 特殊 token 位于开头、结尾或连续出现时，切分结果中可能含有空字符串。
        if not part:
            continue

        tokens=re.finditer(pattern=PAT,string=part)
        for token in tokens:
            data[token.group()]+=1
    return data


"""
举个例子：
`pair = "in"`
1. `"in".encode("utf-8")` → `b'in'`
2. 遍历`b'in'`，拿到整数 `105`、`110`
3. 包装：`bytes([105])=b'i'`，`bytes([110])=b'n'`
4. 列表 `[b'i',b'n']` → 转元组 `(b'i', b'n')`
5. `byte_token_counts[(b'i', b'n')] = count`

"""
def str_to_bytes_pre_tokens(data:Counter):
    """将预分词字符串转换为单字节 token 元组，并保留其出现次数。"""

    byte_token_counts={}
    
    for pair,count in data.items():
        byte_tokens = []
        pairs_encode=pair.encode("utf-8")
        # 遍历 bytes 得到的是整数，需要重新包装成长度为 1 的 bytes token。
        for pair_encode in pairs_encode:
            pre_token=bytes([pair_encode])
            byte_tokens.append(pre_token)
        # 元组表示一个预分词片段当前的 token 序列，可作为字典的 key。
        token=tuple(byte_tokens)
        byte_token_counts[token]=count
    return byte_token_counts

def count_pairs(byte_token_counts:dict):
    """统计所有 token 序列中相邻 pair 的加权出现频次。"""

    pair_counts=Counter()
    for token_seq,count in byte_token_counts.items():
        for i in range(len(token_seq)-1):
            pair=(token_seq[i],token_seq[i+1])
            # 一个预分词片段出现 count 次，其中的相邻 pair 也贡献 count 次。
            pair_counts[pair]+=count
    return pair_counts

def select_best_pair(
    pair_counts: Counter[tuple[bytes, bytes]],
) -> tuple[bytes, bytes] | None:
    """选择频率最高的 pair；频率相同时选择字典序更大的 pair。"""
    
    best_pair = None
    best_count = -1
    for pair,count in pair_counts.items():
        if count>best_count:
            best_count=count
            best_pair=pair
        elif count==best_count:
            if best_pair==None:
                pair=best_pair
            else:
                if best_pair<pair:
                    best_pair=pair
    
    return best_pair

def merge_pair(byte_token_counts:dict,best_pair):
    """从左到右、非重叠地合并所有预分词序列中的指定 pair。"""

    co_byte_token_counts={}
    for token_seq,count in byte_token_counts.items():
        # 不直接修改旧序列，而是逐个构造合并后的新序列。
        new_token_seq=[]
        i=0
        seq_len=len(token_seq)
        while i<seq_len:
            if i+1<seq_len and (token_seq[i],token_seq[i+1])==best_pair:
                merged=token_seq[i]+token_seq[i+1]
                new_token_seq.append(merged)
                # 两个旧 token 已被消耗，跳过两个位置以避免重叠合并。
                i+=2
            else:
                new_token_seq.append(token_seq[i])
                i+=1    
        new_tuple = tuple(new_token_seq)    
        co_byte_token_counts[new_tuple]=count    
    return co_byte_token_counts

def merge_token_sequence(token_seq, best_pair):
    """只合并一个 token 序列，供增量训练循环处理受影响的预分词片段。"""

    new_token_seq = []
    i = 0
    while i < len(token_seq):
        if i + 1 < len(token_seq) and (token_seq[i], token_seq[i + 1]) == best_pair:
            new_token_seq.append(token_seq[i] + token_seq[i + 1])
            i += 2
        else:
            new_token_seq.append(token_seq[i])
            i += 1
    return tuple(new_token_seq)

def initialize_pair_statistics(token_sequences, frequencies):
    """一次性建立全局 pair 频次，以及每个 pair 出现在哪些预分词序列中。"""

    pair_counts = Counter()
    pair_to_sequence_ids = {}

    for sequence_id, (token_seq, frequency) in enumerate(zip(token_sequences, frequencies)):
        pairs_in_sequence = set()
        for pair in zip(token_seq, token_seq[1:]):
            pair_counts[pair] += frequency
            pairs_in_sequence.add(pair)

        for pair in pairs_in_sequence:
            pair_to_sequence_ids.setdefault(pair, set()).add(sequence_id)

    return pair_counts, pair_to_sequence_ids

def train_bpe(input_path, vocab_size, special_tokens):
    # 打开文件、读取文本、调用预分词函数
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    vocab, merges = init_vocab_and_merges(
        special_token_list=special_tokens
    )

    parts = pre_token_split(
        text=text,
        special_token_list=special_tokens,
    )

    data = regex_pre_tokenize(parts=parts)
    byte_token_counts = str_to_bytes_pre_tokens(data=data)

    # 为每一种预分词片段分配稳定 ID。后续只处理包含本轮 best_pair 的片段，
    # 避免每一轮都重新扫描整个语料。
    token_sequences = list(byte_token_counts.keys())
    frequencies = list(byte_token_counts.values())
    pair_counts, pair_to_sequence_ids = initialize_pair_statistics(
        token_sequences,
        frequencies,
    )

    while len(vocab)<vocab_size:
        if not pair_counts:
            break

        best_pair=select_best_pair(pair_counts=pair_counts)
        # 空统计可能返回 None；确认存在 pair 后才能索引或执行合并。
        if best_pair is None:
            break
        affected_sequence_ids = tuple(pair_to_sequence_ids.pop(best_pair, ()))
        touched_pairs = set()

        for sequence_id in affected_sequence_ids:
            old_sequence = token_sequences[sequence_id]
            frequency = frequencies[sequence_id]
            old_pairs = list(zip(old_sequence, old_sequence[1:]))

            # 先移除旧序列对全局频次和倒排索引的贡献。
            for pair in old_pairs:
                pair_counts[pair] -= frequency
                touched_pairs.add(pair)
            for pair in set(old_pairs):
                sequence_ids = pair_to_sequence_ids.get(pair)
                if sequence_ids is not None:
                    sequence_ids.discard(sequence_id)
                    if not sequence_ids:
                        pair_to_sequence_ids.pop(pair, None)

            new_sequence = merge_token_sequence(old_sequence, best_pair)
            token_sequences[sequence_id] = new_sequence
            new_pairs = list(zip(new_sequence, new_sequence[1:]))

            # 再加入新序列产生的相邻 pair。
            for pair in new_pairs:
                pair_counts[pair] += frequency
                touched_pairs.add(pair)
            for pair in set(new_pairs):
                pair_to_sequence_ids.setdefault(pair, set()).add(sequence_id)

        # 频率为 0 的 pair 不能继续参与下一轮最佳 pair 选择。
        for pair in touched_pairs:
            if pair_counts[pair] <= 0:
                del pair_counts[pair]

        merges.append(best_pair)
        merged_token=best_pair[0]+best_pair[1]
        new_token_id=len(vocab)
        vocab[new_token_id]=merged_token
    return vocab, merges

if __name__ == "__main__":
    # 手工样例：验证特殊 token 切分、正则预分词和字节化。
    str1 = (
        "Hello world! Hello, world. I'm learning BPE in 2026. "
        "BPE is fun, and BPE is useful."
        "<|endoftext|>"
        "Hello again! I'm testing numbers: 123 123 456. "
        "Cats, cats, and dogs; cats and dogs!"
    )

    special_token_list = ["<|endoftext|>"]

    parts=pre_token_split(str1,special_token_list)
    print(parts)
    data=regex_pre_tokenize(parts)
    print(data)
    byte_token_counts=str_to_bytes_pre_tokens(data=data)
    print(byte_token_counts)
    pair_counts=count_pairs(byte_token_counts=byte_token_counts)
    print(pair_counts)
    best_pair=select_best_pair(pair_counts=pair_counts)
    print(best_pair)
    co_byte_token_counts=merge_pair(byte_token_counts=byte_token_counts,best_pair=best_pair)
    print(co_byte_token_counts)
