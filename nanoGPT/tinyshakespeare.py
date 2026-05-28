import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def prepare_tinyshakespeare(data_dir, source_url=TINY_SHAKESPEARE_URL, train_ratio=0.9, download=True):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    input_path = data_dir / "input.txt"
    if not input_path.exists():
        if not download:
            raise FileNotFoundError(f"TinyShakespeare source text is missing: {input_path}")
        urllib.request.urlretrieve(source_url, input_path)

    text = input_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"TinyShakespeare source text is empty: {input_path}")
    chars = sorted(set(text))
    stoi = {char: idx for idx, char in enumerate(chars)}
    encoded = np.asarray([stoi[char] for char in text], dtype=np.uint16)
    split_idx = int(len(encoded) * float(train_ratio))
    if split_idx <= 0 or split_idx >= len(encoded):
        raise ValueError("train_ratio must leave non-empty train and validation splits.")

    encoded[:split_idx].tofile(data_dir / "train.bin")
    encoded[split_idx:].tofile(data_dir / "val.bin")
    metadata = {
        "dataset": "tinyshakespeare",
        "vocab_size": len(chars),
        "itos": chars,
        "stoi": stoi,
        "train_tokens": split_idx,
        "val_tokens": len(encoded) - split_idx,
        "source_url": source_url,
    }
    (data_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class TinyShakespeareBatches:
    def __init__(self, data_dir, block_size, device):
        data_dir = Path(data_dir)
        meta_path = data_dir / "meta.json"
        required = [data_dir / "train.bin", data_dir / "val.bin", meta_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Prepared TinyShakespeare files are missing: " + ", ".join(missing))

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
        self.block_size = int(block_size)
        self.device = device
        for name, data in (("train", self.train), ("val", self.val)):
            if len(data) <= self.block_size:
                raise ValueError(f"{name} split requires more than block_size={self.block_size} tokens.")

    def get_batch(self, split, batch_size):
        data = self.train if split == "train" else self.val
        indices = torch.randint(len(data) - self.block_size, (int(batch_size),))
        x = torch.stack(
            [torch.from_numpy(np.asarray(data[i : i + self.block_size], dtype=np.int64)) for i in indices]
        )
        y = torch.stack(
            [torch.from_numpy(np.asarray(data[i + 1 : i + 1 + self.block_size], dtype=np.int64)) for i in indices]
        )
        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def encode(self, text):
        unknown = sorted(set(text) - set(self.meta["stoi"]))
        if unknown:
            raise ValueError(f"Prompt includes characters outside the TinyShakespeare vocabulary: {unknown}")
        return [self.meta["stoi"][char] for char in text]

    def decode(self, tokens):
        return "".join(self.meta["itos"][int(token)] for token in tokens)
