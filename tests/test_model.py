import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import MiniLLM, ModelConfig


def test_forward_shape():
    config = ModelConfig(vocab_size=10, emb_dim=8)
    model = MiniLLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 4))
    out = model(ids)
    assert out.shape == (2, 4, config.vocab_size)


def test_parameter_tying():
    config = ModelConfig(vocab_size=10, emb_dim=8, tie_weights=True)
    model = MiniLLM(config)
    assert model.lm_head.weight is model.embedding.embedding.weight
    ids = torch.randint(0, config.vocab_size, (1, 3))
    out = model(ids)
    assert out.shape == (1, 3, config.vocab_size)


def test_positional_encoding_options():
    torch.manual_seed(0)
    config_sin = ModelConfig(vocab_size=10, emb_dim=8, max_seq_len=4, learnable_pos=False)
    model_sin = MiniLLM(config_sin)
    ids = torch.randint(0, config_sin.vocab_size, (2, 4))
    out_sin = model_sin(ids)

    torch.manual_seed(0)
    config_learn = ModelConfig(vocab_size=10, emb_dim=8, max_seq_len=4, learnable_pos=True)
    model_learn = MiniLLM(config_learn)
    out_learn = model_learn(ids)

    assert out_sin.shape == out_learn.shape == (2, 4, config_sin.vocab_size)
    assert model_sin.pos_encoding.learnable is False
    assert model_learn.pos_encoding.learnable is True
    assert not torch.allclose(out_sin, out_learn)
