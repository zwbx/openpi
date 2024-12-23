from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize(["Hello, world!", "This is a test"])

    assert tokens.shape == (2, 10)
    assert masks.shape == (2, 10)
