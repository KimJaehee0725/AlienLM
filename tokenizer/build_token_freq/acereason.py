import argparse
import json
import os
from itertools import islice

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def _default_cache_dir() -> str:
    return (
        os.environ.get("HF_DATASETS_CACHE")
        or os.environ.get("HF_HOME")
        or os.path.join(os.getcwd(), ".cache", "hf_datasets")
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build token frequency dictionary from AceReason SFT.")
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Tokenizer name or path.",
    )
    parser.add_argument(
        "--dataset",
        default="nvidia/AceReason-1.1-SFT",
        help="HF dataset name.",
    )
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument(
        "--cache_dir",
        default=_default_cache_dir(),
        help="HF datasets cache dir.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=25,
        help="Number of arrow shards to include (starts from 1).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500000,
        help="Limit samples (0 = no limit).",
    )
    parser.add_argument(
        "--output",
        default="acereason_tok_dict.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def get_token_freq_dict(tokenizer, data, max_samples: int = 0):
    token_freq_dict = {tok: 0 for tok, tok_id in tokenizer.vocab.items()}
    iterator = enumerate(data)
    if max_samples and max_samples > 0:
        iterator = islice(iterator, max_samples + 1)
    for i, sample in tqdm(iterator, total=max_samples or None):
        if max_samples and i > max_samples:
            break
        tokenized = tokenizer.tokenize(sample['input'], add_special_tokens=False)
        for tok in tokenized:
            token_freq_dict[tok] += 1
        tokenized = tokenizer.tokenize(sample['output'], add_special_tokens=False)
        for tok in tokenized:
            token_freq_dict[tok] += 1
    return token_freq_dict


def main() -> None:
    args = _parse_args()
    file_list = [
        f"sft_data.parquet/data-00{str(i).zfill(3)}-of-00187.arrow"
        for i in range(1, args.max_files + 1)
    ]
    dataset = load_dataset(args.dataset, data_files=file_list, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    acereason_tok_dict = get_token_freq_dict(tokenizer, dataset[args.split], args.max_samples)
    with open(args.output, "w") as f:
        json.dump(acereason_tok_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
