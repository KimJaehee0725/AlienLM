import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from Efficient_Token_Matcher import OptimizedTokenMatcher

os.environ.setdefault("OMP_NUM_THREADS", "32")


def _default_cache_dir() -> str:
    return (
        os.environ.get("TRANSFORMERS_CACHE")
        or os.environ.get("HF_HOME")
        or os.path.join(os.getcwd(), ".cache", "hf_models")
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match tokens using proxy embeddings.")
    parser.add_argument(
        "--base_model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model/tokenizer.",
    )
    parser.add_argument(
        "--proxy_model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Proxy model/tokenizer for embeddings.",
    )
    parser.add_argument(
        "--token_freq_json",
        required=True,
        help="Token frequency JSON path.",
    )
    parser.add_argument(
        "--cache_dir",
        default=_default_cache_dir(),
        help="HF cache dir for models/tokenizers.",
    )
    parser.add_argument(
        "--output",
        default="matches-sim-and-diff.txt",
        help="Output match file path.",
    )
    return parser.parse_args()

def _load_token_freq(path: str) -> dict:
    token_path = Path(path)
    if not token_path.exists():
        raise FileNotFoundError(f"Token frequency file not found: {token_path}")
    with token_path.open("r") as f:
        return json.load(f)

def main() -> None:
    args = _parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.proxy_model, cache_dir=args.cache_dir)
    proxy_lm_head = model.lm_head.weight.detach()
    proxy_tokenizer = AutoTokenizer.from_pretrained(args.proxy_model, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)

    lm_head = []
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])

    non_matched = 0
    non_matched_len = []
    for tok, _tok_id in tqdm(vocab):
        proxy_tok_id = proxy_tokenizer.encode(tok)
        proxy_tok_embedding = proxy_lm_head[proxy_tok_id]
        if len(proxy_tok_id) > 1:
            non_matched += 1
            non_matched_len.append(len(proxy_tok_id))

            proxy_tok = proxy_tokenizer.convert_ids_to_tokens(proxy_tok_id)
            proxy_tok_len = [len(tok) for tok in proxy_tok]
            proxy_tok_prop = [tok_len / sum(proxy_tok_len) for tok_len in proxy_tok_len]
            proxy_tok_embedding = [proxy_tok_embedding[i] * proxy_tok_prop[i] for i in range(len(proxy_tok_id))]
            proxy_tok_embedding = torch.stack(proxy_tok_embedding).sum(dim=0).unsqueeze(0)
        lm_head.append(proxy_tok_embedding)
    lm_head = torch.stack(lm_head).squeeze(1)

    if non_matched_len:
        print(f">>> {non_matched} tokens are not matched")
        print(f">>> max token length: {max(non_matched_len)}")
        print(f">>> avg token length: {sum(non_matched_len) / len(non_matched_len)}")

    token_freq = _load_token_freq(args.token_freq_json)

    tokens = tokenizer.get_vocab()
    tokens = [(tokenizer.convert_ids_to_tokens(idx), idx) for _tok, idx in tokens.items()]
    # tokenizer.get_added_vocab()

    # sort token by its frequency
    tokens = sorted(tokens, key=lambda x: token_freq.get(x[0], 0), reverse=True)

    # additional_special_tokens = list(tokenizer.vocab.keys())
    # additional_special_tokens = [tok for tok in tqdm(additional_special_tokens) if tok.startswith('<|') and tok.endswith('|>')]
    added_vocab = tokenizer.get_added_vocab()
    total_special_tokens = {tok for tok, _id in added_vocab.items()}
    # delete special tokens
    tokens = [token for token in tokens if token[0] not in total_special_tokens]

    embeddings = lm_head[[token[1] for token in tokens]]
    id_to_idx = {token[1]: idx for idx, token in enumerate(tokens)}
    idx_to_id = {idx: token[1] for idx, token in enumerate(tokens)}
    tokens = [(token[0], id_to_idx[token[1]]) for token in tokens]

    import faiss

    faiss.omp_set_num_threads(64)

    english_matcher = OptimizedTokenMatcher(
        embeddings=embeddings,
        tokens=tokens,
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        batch_size=100,
        n_neighbors=50,
    )

    matches = english_matcher.find_matches(lev_weight=1, sim_weight=0.01)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for match in matches:
            f.write(f"{match[0]}\t{match[1]}\t{match[2]}\n")
if __name__ == "__main__":
    main()
