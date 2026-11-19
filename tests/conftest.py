import pytest
from transformers import BertTokenizer


@pytest.fixture
def tokenizer():
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "word", "##piece", "##s", "."]
    vocab = {token: index for index, token in enumerate(tokens)}
    return BertTokenizer(vocab=vocab, model_max_length=512)
