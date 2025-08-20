"""Training script for MiniLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from .model import MiniTransformer, ModelConfig
from .tokenizer import Tokenizer

VOCAB_PATH = Path("data/vocab.json")
TRAIN_PATH = Path("data/splits/train.json")
VAL_PATH = Path("data/splits/val.json")


class QADataset(Dataset):
    """Dataset of tokenised question–answer pairs."""

    def __init__(self, path: Path, tokenizer: Tokenizer, limit: int | None = None) -> None:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if limit is not None:
            data = data[:limit]
        self.pairs = data
        self.tokenizer = tokenizer
        self.encoded: List[List[int]] = []
        for item in self.pairs:
            ids = tokenizer.encode(item["question"], add_bos=True)
            ids += tokenizer.encode(item["answer"], add_eos=True)
            self.encoded.append(ids)
        self.max_len = max((len(seq) for seq in self.encoded), default=0)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.encoded[idx], dtype=torch.long)


def make_collate_fn(pad_id: int):
    """Create a collate function that pads sequences and builds targets."""

    def collate(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(seq.size(0) for seq in batch)
        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(batch):
            padded[i, : seq.size(0)] = seq
        inputs = padded[:, :-1]
        targets = padded[:, 1:]
        return inputs, targets

    return collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MiniLLM model")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--run-dir", type=str, default="experiments/run", help="output directory")
    parser.add_argument("--vocab-size", type=int, default=8000, help="maximum vocabulary size")
    parser.add_argument("--num-layers", type=int, default=2, help="number of transformer layers")
    parser.add_argument("--limit", type=int, default=None, help="limit number of training examples")
    return parser.parse_args()


def generate_answer(model: MiniTransformer, tokenizer: Tokenizer, question: str, max_len: int) -> str:
    """Generate an answer for ``question`` using greedy decoding."""

    model.eval()
    ids = tokenizer.encode(question, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_len - len(ids)):
        with torch.no_grad():
            logits = model(input_ids)
        next_id = int(torch.argmax(logits[0, -1]))
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
        input_ids = torch.tensor([ids], dtype=torch.long)
    return tokenizer.decode(ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = Tokenizer()

    # Build vocabulary from train and validation texts
    texts: List[str] = []
    for path in [TRAIN_PATH, VAL_PATH]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            texts.extend(f"{d['question']} {d['answer']}".strip() for d in data)
    tokenizer.fit(texts, vocab_size=args.vocab_size)
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_vocab(str(VOCAB_PATH))
    tokenizer.save_vocab(str(run_dir / "vocab.json"))

    train_dataset = QADataset(TRAIN_PATH, tokenizer, limit=args.limit)
    val_dataset = QADataset(VAL_PATH, tokenizer)

    collate_fn = make_collate_fn(tokenizer.pad_id)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    max_seq_len = max(train_dataset.max_len, val_dataset.max_len)
    config = ModelConfig(
        vocab_size=len(tokenizer.token_to_id),
        max_seq_len=max_seq_len,
        emb_dim=32,
        num_layers=args.num_layers,
        num_heads=2,
        ffn_dim=64,
    )
    model = MiniTransformer(config)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_path = run_dir / "log.jsonl"
    sample_question = val_dataset.pairs[0]["question"] if val_dataset.pairs else ""

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)
                )
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        generated = generate_answer(model, tokenizer, sample_question, max_seq_len)
        # Save model parameters as JSON to avoid binary .pt files
        state_dict = {k: v.tolist() for k, v in model.state_dict().items()}
        with (run_dir / f"model_epoch{epoch}.json").open("w", encoding="utf-8") as f:
            json.dump(state_dict, f)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "sample_question": sample_question,
            "generated_answer": generated,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry))

    print("Training complete")


if __name__ == "__main__":
    main()
