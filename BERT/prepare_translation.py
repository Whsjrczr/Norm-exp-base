#!/usr/bin/env python3
import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from BERT.translation_data import prepare_translation_data


def main():
    parser = argparse.ArgumentParser("Prepare text translation data")
    parser.add_argument("--data-dir", default="./dataset/text_translation")
    parser.add_argument("--input-file", default=None, help="TSV file with source<TAB>target per line")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    meta = prepare_translation_data(
        args.data_dir,
        input_file=args.input_file,
        train_ratio=args.train_ratio,
        min_freq=args.min_freq,
        overwrite=args.overwrite,
    )
    print(f"prepared {args.data_dir}: vocab={meta['vocab_size']} train={meta['train_pairs']} val={meta['val_pairs']}")


if __name__ == "__main__":
    main()
