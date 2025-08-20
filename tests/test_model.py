import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import MiniLLM, ModelConfig


def test_forward_shape():
    config = ModelConfig(vocab_size=10, hidden_size=8)
    model = MiniLLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 4))
    out = model(ids)
    assert out.shape == (2, 4, config.vocab_size)


def test_parameter_tying():
    config = ModelConfig(vocab_size=10, hidden_size=8, tie_weights=True)
    model = MiniLLM(config)
    assert model.linear.weight.data_ptr() == model.embedding.weight.data_ptr()
    ids = torch.randint(0, config.vocab_size, (1, 3))
    out = model(ids)
    assert out.shape == (1, 3, config.vocab_size)


def test_positional_encoding_options():
    torch.manual_seed(0)
    config_none = ModelConfig(vocab_size=10, hidden_size=8, max_seq_len=4, positional_encoding=None)
    model_none = MiniLLM(config_none)
    ids = torch.randint(0, config_none.vocab_size, (2, 4))
    out_none = model_none(ids)

    torch.manual_seed(0)
    config_sin = ModelConfig(vocab_size=10, hidden_size=8, max_seq_len=4, positional_encoding="sinusoidal")
    model_sin = MiniLLM(config_sin)
    out_sin = model_sin(ids)

    assert out_sin.shape == out_none.shape == (2, 4, config_none.vocab_size)
    assert model_none.positional_encoding is None
    assert model_sin.positional_encoding is not None
    assert not torch.allclose(out_sin, out_none)
