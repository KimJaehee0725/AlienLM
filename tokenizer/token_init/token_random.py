import random
import numpy as np
from typing import List, Dict, Tuple


def probabilistic_bucket_reordering(
    tokens: List[Tuple],
    token_freq: Dict[str, int],
    n_buckets: int = 10,
    intra_bucket_swap_prob: float = 0.1,
    inter_bucket_swap_prob: float = 0.05,
    random_seed: int = 42,
) -> List[Tuple]:
    """
    Probabilistic bucket-based token reordering.

    Args:
        tokens: list of tokens
        token_freq: token frequency dict
        n_buckets: number of buckets
        intra_bucket_swap_prob: swap probability within a bucket
        inter_bucket_swap_prob: swap probability across buckets
        random_seed: random seed
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    tokens_with_freq = [(token, token_freq.get(token[0], 1)) for token in tokens]
    tokens_sorted = sorted(tokens_with_freq, key=lambda x: x[1], reverse=True)

    bucket_size = len(tokens_sorted) // n_buckets
    buckets = []

    for i in range(n_buckets):
        start_idx = i * bucket_size
        if i == n_buckets - 1:
            end_idx = len(tokens_sorted)
        else:
            end_idx = (i + 1) * bucket_size

        bucket = [token for token, freq in tokens_sorted[start_idx:end_idx]]
        buckets.append(bucket)

    print(f"Tokens per bucket: {[len(bucket) for bucket in buckets]}")

    for bucket_idx, bucket in enumerate(buckets):
        n_swaps = int(len(bucket) * intra_bucket_swap_prob)
        print(f"Bucket {bucket_idx}: {n_swaps} swaps")

        for _ in range(n_swaps):
            if len(bucket) >= 2:
                i, j = random.sample(range(len(bucket)), 2)
                bucket[i], bucket[j] = bucket[j], bucket[i]

    total_tokens = sum(len(bucket) for bucket in buckets)
    n_inter_swaps = int(total_tokens * inter_bucket_swap_prob)
    print(f"Inter-bucket swaps: {n_inter_swaps}")

    for _ in range(n_inter_swaps):
        bucket1_idx = random.randint(0, n_buckets - 2)
        bucket2_idx = bucket1_idx + 1

        if len(buckets[bucket1_idx]) > 0 and len(buckets[bucket2_idx]) > 0:
            token1_idx = random.randint(0, len(buckets[bucket1_idx]) - 1)
            token2_idx = random.randint(0, len(buckets[bucket2_idx]) - 1)

            token1 = buckets[bucket1_idx].pop(token1_idx)
            token2 = buckets[bucket2_idx].pop(token2_idx)

            buckets[bucket1_idx].append(token2)
            buckets[bucket2_idx].append(token1)

    result_tokens = []
    for bucket in buckets:
        result_tokens.extend(bucket)

    return result_tokens


def advanced_probabilistic_reordering(
    tokens: List[Tuple],
    token_freq: Dict[str, int],
    n_buckets: int = 10,
    intra_bucket_swap_prob: float = 0.1,
    inter_bucket_swap_prob: float = 0.05,
    distance_decay: float = 0.5,
    random_seed: int = 42,
) -> List[Tuple]:
    """Probabilistic reordering with distance-based swap decay."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    tokens_with_freq = [(token, token_freq.get(token[0], 1)) for token in tokens]
    tokens_sorted = sorted(tokens_with_freq, key=lambda x: x[1], reverse=True)

    bucket_size = len(tokens_sorted) // n_buckets
    buckets = []

    for i in range(n_buckets):
        start_idx = i * bucket_size
        end_idx = (i + 1) * bucket_size if i < n_buckets - 1 else len(tokens_sorted)
        bucket = [token for token, freq in tokens_sorted[start_idx:end_idx]]
        buckets.append(bucket)

    for bucket_idx, bucket in enumerate(buckets):
        n_swaps = int(len(bucket) * intra_bucket_swap_prob)
        for _ in range(n_swaps):
            if len(bucket) >= 2:
                i, j = random.sample(range(len(bucket)), 2)
                bucket[i], bucket[j] = bucket[j], bucket[i]

    total_tokens = sum(len(bucket) for bucket in buckets)
    n_inter_swaps = int(total_tokens * inter_bucket_swap_prob)

    for _ in range(n_inter_swaps):
        bucket1_idx = random.randint(0, n_buckets - 1)

        distances = [abs(i - bucket1_idx) for i in range(n_buckets)]
        probs = [distance_decay ** d for d in distances]
        probs[bucket1_idx] = 0

        if sum(probs) > 0:
            probs = np.array(probs) / sum(probs)
            bucket2_idx = np.random.choice(n_buckets, p=probs)

            if len(buckets[bucket1_idx]) > 0 and len(buckets[bucket2_idx]) > 0:
                token1_idx = random.randint(0, len(buckets[bucket1_idx]) - 1)
                token2_idx = random.randint(0, len(buckets[bucket2_idx]) - 1)

                token1 = buckets[bucket1_idx].pop(token1_idx)
                token2 = buckets[bucket2_idx].pop(token2_idx)

                buckets[bucket1_idx].append(token2)
                buckets[bucket2_idx].append(token1)

    result_tokens = []
    for bucket in buckets:
        result_tokens.extend(bucket)

    return result_tokens


# Example helper
def get_probabilistic_bucket_ordering(tokens, token_freq, strategy="basic", random_seed=42):
    """Bucket reordering strategies."""

    if strategy == "basic":
        return probabilistic_bucket_reordering(
            tokens,
            token_freq,
            n_buckets=10,
            intra_bucket_swap_prob=0.1,
            inter_bucket_swap_prob=0.05,
            random_seed=random_seed,
        )

    if strategy == "high_noise":
        return probabilistic_bucket_reordering(
            tokens,
            token_freq,
            n_buckets=8,
            intra_bucket_swap_prob=0.2,
            inter_bucket_swap_prob=0.1,
            random_seed=random_seed,
        )

    if strategy == "conservative":
        return probabilistic_bucket_reordering(
            tokens,
            token_freq,
            n_buckets=15,
            intra_bucket_swap_prob=0.05,
            inter_bucket_swap_prob=0.02,
            random_seed=random_seed,
        )

    if strategy == "distance_aware":
        return advanced_probabilistic_reordering(
            tokens,
            token_freq,
            n_buckets=10,
            intra_bucket_swap_prob=0.1,
            inter_bucket_swap_prob=0.05,
            distance_decay=0.5,
            random_seed=random_seed,
        )

    return tokens
