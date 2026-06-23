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
    parser = argparse.ArgumentParser(description="Build token frequency dictionary from Tulu-3 dataset.")
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Tokenizer name or path.",
    )
    parser.add_argument(
        "--dataset",
        default="allenai/tulu-3-sft-olmo-2-mixture",
        help="HF dataset name.",
    )
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument(
        "--cache_dir",
        default=_default_cache_dir(),
        help="HF datasets cache dir.",
    )
    parser.add_argument(
        "--output",
        default="tulu3_tok_dict.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Limit samples (0 = no limit).",
    )
    return parser.parse_args()

def get_token_freq_dict(tokenizer, data, max_samples: int = 0):
    token_freq_dict = {tok: 0 for tok, tok_id in tokenizer.vocab.items()}
    iterator = data
    if max_samples and max_samples > 0:
        iterator = islice(data, max_samples)
    for sample in tqdm(iterator):
        for msg in sample['messages']:
            tokenized = tokenizer.tokenize(msg['content'], add_special_tokens=False)
            for tok in tokenized:
                token_freq_dict[tok] += 1
    return token_freq_dict


def main() -> None:
    args = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)
    tulu3_tok_dict = get_token_freq_dict(tokenizer, dataset[args.split], args.max_samples)
    with open(args.output, "w") as f:
        json.dump(tulu3_tok_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
