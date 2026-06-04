#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request
import zipfile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from BERT.translation_data import tokenize


DEFAULT_URL = "https://object.pouta.csc.fi/OPUS-Books/v1/moses/en-fr.txt.zip"


def _download(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        return
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, output_path)


def _find_member(members, suffix):
    matches = [name for name in members if name.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"Could not find zip member ending with {suffix}. Members: {members[:10]}")
    return matches[0]


def _clean(text):
    return " ".join(str(text).strip().split())


def convert_opus_zip(
    zip_path,
    output_tsv,
    source_lang="en",
    target_lang="fr",
    pair_name="en-fr",
    max_pairs=50000,
    max_src_tokens=64,
    max_tgt_tokens=64,
):
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.namelist()
        src_member = _find_member(members, f".{pair_name}.{source_lang}")
        tgt_member = _find_member(members, f".{pair_name}.{target_lang}")
        with archive.open(src_member) as src_file, archive.open(tgt_member) as tgt_file:
            pairs = []
            for src_raw, tgt_raw in zip(src_file, tgt_file):
                source = _clean(src_raw.decode("utf-8", errors="replace"))
                target = _clean(tgt_raw.decode("utf-8", errors="replace"))
                if not source or not target:
                    continue
                if "\t" in source or "\t" in target:
                    continue
                if max_src_tokens > 0 and len(tokenize(source)) > max_src_tokens:
                    continue
                if max_tgt_tokens > 0 and len(tokenize(target)) > max_tgt_tokens:
                    continue
                pairs.append((source, target))
                if max_pairs > 0 and len(pairs) >= max_pairs:
                    break

    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    with open(output_tsv, "w", encoding="utf-8") as handle:
        for source, target in pairs:
            handle.write(f"{source}\t{target}\n")
    return len(pairs), src_member, tgt_member


def main():
    parser = argparse.ArgumentParser("Download and convert OPUS Books translation data")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--data-dir", default="./dataset/opus_books_en_fr")
    parser.add_argument("--output-tsv", default=None)
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="fr")
    parser.add_argument("--pair-name", default="en-fr")
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--max-src-tokens", type=int, default=64)
    parser.add_argument("--max-tgt-tokens", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_tsv = args.output_tsv or os.path.join(args.data_dir, "pairs.tsv")
    zip_path = os.path.join(args.data_dir, os.path.basename(args.url))
    if os.path.exists(output_tsv) and not args.overwrite:
        print(f"pairs file already exists: {output_tsv}")
        return

    _download(args.url, zip_path)
    count, src_member, tgt_member = convert_opus_zip(
        zip_path=zip_path,
        output_tsv=output_tsv,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        pair_name=args.pair_name,
        max_pairs=args.max_pairs,
        max_src_tokens=args.max_src_tokens,
        max_tgt_tokens=args.max_tgt_tokens,
    )
    print(f"wrote {count} pairs to {output_tsv}")
    print(f"source member: {src_member}; target member: {tgt_member}")


if __name__ == "__main__":
    main()
