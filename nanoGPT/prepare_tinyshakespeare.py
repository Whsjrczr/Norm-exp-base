#!/usr/bin/env python3
import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from nanoGPT.tinyshakespeare import TINY_SHAKESPEARE_URL, prepare_tinyshakespeare


def main():
    parser = argparse.ArgumentParser("Prepare TinyShakespeare character-level data")
    parser.add_argument("--data-dir", default="./dataset/tinyshakespeare")
    parser.add_argument("--url", default=TINY_SHAKESPEARE_URL)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()

    meta = prepare_tinyshakespeare(args.data_dir, source_url=args.url, train_ratio=args.train_ratio)
    print(
        "Prepared TinyShakespeare: "
        f"vocab={meta['vocab_size']}, train_tokens={meta['train_tokens']}, "
        f"val_tokens={meta['val_tokens']} at {args.data_dir}"
    )


if __name__ == "__main__":
    main()
