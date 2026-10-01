"""补充官方测试未直接覆盖的文件接口与边界条件。"""

from itertools import islice

from cs336_basics.tokenizer import Tokenizer


def test_missing_special_tokens_preserve_existing_ids_and_input():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[300] = b"<existing>"
    original_vocab = dict(vocab)
    tokenizer = Tokenizer(vocab, [], ["<existing>", "<new>", "<new>"])
    assert vocab == original_vocab
    assert tokenizer.encode("<existing><new>") == [300, 301]
    assert len(tokenizer.vocab) == len(vocab) + 1


def test_overlapping_specials_and_regex_metacharacters():
    tokenizer = Tokenizer({i: bytes([i]) for i in range(256)}, [], ["[x]", "[x][x]", ".*"])
    assert tokenizer.encode("[x][x]a.*[x]") == [257, 97, 258, 256]


def test_decode_invalid_utf8_and_split_character():
    tokenizer = Tokenizer({i: bytes([i]) for i in range(256)}, [])
    assert tokenizer.decode([0xFF]) == "\ufffd"
    assert tokenizer.decode([0xE4, 0xB8, 0xAD]) == "中"
    assert tokenizer.decode([0xE4]) == "\ufffd"


def test_file_roundtrip_preserves_bytes_merges_and_encoding(tmp_path):
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256: b"ab", 257: b"abab"})
    merges = [(b"a", b"b"), (b"ab", b"ab")]
    tokenizer = Tokenizer(vocab, merges, ["<end>"])
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.json"
    tokenizer.save(vocab_path, merges_path)
    restored = Tokenizer.from_files(vocab_path, merges_path, ["<end>"])
    assert restored.vocab == tokenizer.vocab
    assert restored.merges == merges
    assert restored.encode("abab<end>中") == [257, 258, 228, 184, 173]
    assert restored.decode(restored.encode("abab<end>中")) == "abab<end>中"


def test_encode_iterable_is_lazy_and_yields_individual_ids():
    def source():
        yield "ab"
        raise AssertionError("不应提前读取下一个输入段")

    tokenizer = Tokenizer({i: bytes([i]) for i in range(256)}, [])
    ids = tokenizer.encode_iterable(source())
    assert iter(ids) is ids
    assert list(islice(ids, 2)) == [97, 98]
    assert list(tokenizer.encode_iterable([])) == []
