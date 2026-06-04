import json
import os
import random
from collections import Counter

import torch


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
DEMO_PAIRS = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("thank you", "merci"),
    ("yes", "oui"),
    ("no", "non"),
    ("i love apples", "j aime les pommes"),
    ("i love books", "j aime les livres"),
    ("we study math", "nous etudions les maths"),
    ("we study language", "nous etudions la langue"),
    ("this is a small model", "ceci est un petit modele"),
    ("the cat is black", "le chat est noir"),
    ("the dog is white", "le chien est blanc"),
    ("open the door", "ouvre la porte"),
    ("close the window", "ferme la fenetre"),
    ("where is the station", "ou est la gare"),
    ("i want water", "je veux de l eau"),
    ("i want coffee", "je veux du cafe"),
    ("see you tomorrow", "a demain"),
    ("have a nice day", "bonne journee"),
]


def tokenize(text):
    text = str(text).strip()
    if not text:
        return []
    if any(char.isspace() for char in text):
        return text.split()
    if all(ord(char) < 128 for char in text):
        return [text]
    return list(text)


class TranslationVocab:
    def __init__(self, tokens=None):
        tokens = tokens or SPECIAL_TOKENS
        self.itos = list(tokens)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    @property
    def pad_id(self):
        return self.stoi["<pad>"]

    @property
    def bos_id(self):
        return self.stoi["<bos>"]

    @property
    def eos_id(self):
        return self.stoi["<eos>"]

    @property
    def unk_id(self):
        return self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len, add_bos=False, add_eos=True):
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(token, self.unk_id) for token in tokenize(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids[:max_len]

    def decode(self, ids, skip_special=True):
        tokens = []
        for idx in ids:
            token = self.itos[int(idx)] if int(idx) < len(self.itos) else "<unk>"
            if token == "<eos>":
                break
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def to_json(self):
        return {"tokens": self.itos}

    @classmethod
    def from_json(cls, data):
        return cls(data["tokens"])


def _read_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no} must contain exactly one tab: source<TAB>target")
            pairs.append((parts[0].strip(), parts[1].strip()))
    if not pairs:
        raise ValueError(f"No translation pairs found in {path}.")
    return pairs


def _write_pairs(path, pairs):
    with open(path, "w", encoding="utf-8") as handle:
        for source, target in pairs:
            handle.write(f"{source}\t{target}\n")


def _write_jsonl(path, pairs):
    with open(path, "w", encoding="utf-8") as handle:
        for source, target in pairs:
            handle.write(json.dumps({"src": source, "tgt": target}, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append((row["src"], row["tgt"]))
    return rows


def prepare_translation_data(data_dir, input_file=None, train_ratio=0.9, min_freq=1, overwrite=False):
    os.makedirs(data_dir, exist_ok=True)
    pairs_path = os.path.join(data_dir, "pairs.tsv")
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    meta_path = os.path.join(data_dir, "meta.json")

    if not overwrite and all(os.path.exists(path) for path in (train_path, val_path, meta_path)):
        with open(meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    if input_file:
        pairs = _read_pairs(input_file)
        if os.path.abspath(input_file) != os.path.abspath(pairs_path):
            _write_pairs(pairs_path, pairs)
    elif os.path.exists(pairs_path):
        pairs = _read_pairs(pairs_path)
    else:
        pairs = list(DEMO_PAIRS)
        _write_pairs(pairs_path, pairs)

    rng = random.Random(0)
    rng.shuffle(pairs)
    split = max(1, int(len(pairs) * float(train_ratio)))
    split = min(split, len(pairs) - 1) if len(pairs) > 1 else len(pairs)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:] or pairs[:1]

    counter = Counter()
    for source, target in train_pairs:
        counter.update(tokenize(source))
        counter.update(tokenize(target))
    tokens = list(SPECIAL_TOKENS)
    tokens.extend(token for token, count in sorted(counter.items()) if count >= min_freq and token not in SPECIAL_TOKENS)
    vocab = TranslationVocab(tokens)
    meta = {
        "vocab_size": len(vocab),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "special_tokens": {token: idx for idx, token in enumerate(SPECIAL_TOKENS)},
        "vocab": vocab.to_json(),
    }
    _write_jsonl(train_path, train_pairs)
    _write_jsonl(val_path, val_pairs)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    return meta


class TranslationBatches:
    def __init__(self, data_dir, max_src_len, max_tgt_len, device):
        self.data_dir = data_dir
        self.max_src_len = int(max_src_len)
        self.max_tgt_len = int(max_tgt_len)
        self.device = device
        with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.vocab = TranslationVocab.from_json(self.meta["vocab"])
        self.splits = {
            "train": _read_jsonl(os.path.join(data_dir, "train.jsonl")),
            "val": _read_jsonl(os.path.join(data_dir, "val.jsonl")),
        }

    def _pad(self, ids, max_len):
        ids = list(ids[:max_len])
        ids.extend([self.vocab.pad_id] * (max_len - len(ids)))
        return ids

    def _encode_pair(self, source, target):
        src = self._pad(self.vocab.encode(source, self.max_src_len, add_bos=False, add_eos=True), self.max_src_len)
        tgt_full = self.vocab.encode(target, self.max_tgt_len + 1, add_bos=True, add_eos=True)
        tgt_input = self._pad(tgt_full[:-1], self.max_tgt_len)
        tgt_labels = self._pad(tgt_full[1:], self.max_tgt_len)
        return src, tgt_input, tgt_labels

    def get_batch(self, split, batch_size):
        pairs = self.splits[split]
        indices = torch.randint(len(pairs), (int(batch_size),))
        src_rows, input_rows, label_rows = [], [], []
        for idx in indices.tolist():
            src, tgt_input, tgt_labels = self._encode_pair(*pairs[idx])
            src_rows.append(src)
            input_rows.append(tgt_input)
            label_rows.append(tgt_labels)
        return (
            torch.tensor(src_rows, dtype=torch.long, device=self.device),
            torch.tensor(input_rows, dtype=torch.long, device=self.device),
            torch.tensor(label_rows, dtype=torch.long, device=self.device),
        )

    def encode_source(self, text):
        ids = self._pad(self.vocab.encode(text, self.max_src_len, add_bos=False, add_eos=True), self.max_src_len)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def decode_target(self, ids):
        return self.vocab.decode(ids)
