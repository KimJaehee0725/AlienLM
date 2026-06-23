import numpy as np
import torch
from scipy.spatial.distance import cosine
import Levenshtein
from typing import Dict, List, Tuple
from tqdm import tqdm
import faiss
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import random

faiss.omp_set_num_threads(64)

import re


def _strip_space_token(token: str, space_token="Ġ"):
    """
    Normalize a token:
    1) Remove leading space token.
    2) Strip non-alpha at the start/end when adjacent chars are alpha.
    3) Lowercase.
    """
    if len(token) == 1:
        return token.lower()
    if token.startswith(space_token):
        if len(token) > 1:
            return token[1:].lower()

    if not re.match(r"[a-zA-Z]", token[0]) and re.match(r"[a-zA-Z]", token[1]):
        return token[1:].lower()

    if not re.match(r"[a-zA-Z]", token[-1]) and re.match(r"[a-zA-Z]", token[-2]):
        return token[:-1].lower()

    return token.lower()


def _calculate_edit_distances(args) -> List[Tuple[int, int, float]]:
    """Helper for edit-distance computation (parallelizable)."""
    token1, idx1, tokens, indices = args
    distances = []
    for idx2, token2 in zip(indices, tokens):
        token1, token2 = _strip_space_token(token1[0]), _strip_space_token(token2[0])
        if idx1 != idx2:
            if token1 in token2 or token2 in token1:
                dist = Levenshtein.ratio(token1, token2) / max(len(token1), len(token2))
                distances.append((idx1, idx2, dist))
            else:
                dist = Levenshtein.ratio(token1, token2) / max(len(token1), len(token2))
                distances.append((idx1, idx2, dist))
    return distances


class OptimizedTokenMatcher:
    def __init__(
        self,
        embeddings: Dict[str, torch.Tensor],
        tokens: List[str],
        id_to_idx: Dict[int, int],
        idx_to_id: Dict[int, int],
        batch_size: int = 1000,
        n_neighbors: int = 100,
    ):
        """
        Optimized token matcher.

        Args:
            embeddings: token -> embedding dict
            tokens: list of tokens
            batch_size: batch size
            n_neighbors: neighbors per token
        """
        self.embeddings = embeddings
        self.tokens = tokens
        self.id_to_idx = id_to_idx
        self.idx_to_id = idx_to_id
        self.n_tokens = len(tokens)
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors
        self.device = "cpu"
        faiss.omp_set_num_threads(64)

        self._init_faiss_index()
        self.NO_MATCH_CNT = 0

    def _init_faiss_index(self):
        """Initialize FAISS index."""
        print("Initializing FAISS index...")
        embedding_dim = next(iter(self.embeddings)).shape[0]

        if self.device == "cuda":
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatIP(res, embedding_dim)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)

        embeddings_matrix = np.stack([self.embeddings[token[1]].cpu().numpy() for token in self.tokens])
        faiss.normalize_L2(embeddings_matrix)
        self.index.add(embeddings_matrix)

    def _reinit_faiss_index(self, tokens: List[str]):
        """Reinitialize FAISS index."""
        print("Reinitializing FAISS index...")
        self.tokens = tokens
        self.n_tokens = len(tokens)
        self._init_faiss_index()

    def find_matches(self, lev_weight: float = 0.5, sim_weight: float = 0.5) -> List[Tuple[str, str, float]]:
        """Find token matches."""
        matches = []
        available_indices = set(range(self.n_tokens))
        n_workers = multiprocessing.cpu_count()

        pbar = tqdm(range(0, self.n_tokens, self.batch_size))

        for batch_idx, batch_start in enumerate(pbar):
            desc = f"No match count: {self.NO_MATCH_CNT} | Remaining tokens: {len(available_indices)}"
            pbar.set_description_str(desc)
            pbar.refresh()
            if not available_indices:
                break

            batch_end = min(batch_start + self.batch_size, self.n_tokens)
            batch_tokens = self.tokens[batch_start:batch_end]

            # 1) Nearest neighbors via FAISS
            batch_embeddings = np.stack([self.embeddings[token[1]].cpu().numpy() for token in batch_tokens])
            similarities, neighbor_indices = self.index.search(batch_embeddings, self.n_neighbors)

            # 2) Edit distance
            edit_distance_args = []
            for i, (token, indices) in enumerate(zip(batch_tokens, neighbor_indices)):
                neighbor_tokens = [self.tokens[idx] for idx in indices if idx in available_indices]
                indices = [idx for idx in indices if idx in available_indices]
                edit_distance_args.append((token, batch_start + i, neighbor_tokens, indices))

            all_distances = [_calculate_edit_distances(args) for args in edit_distance_args]

            # 3) Build matching matrix per batch
            for i, (token1, distances, sims, indices) in enumerate(
                zip(batch_tokens, all_distances, similarities, neighbor_indices)
            ):
                if batch_start + i not in available_indices:
                    continue

                scores = []
                for (idx1, idx2, dist), sim in zip(distances, sims):
                    if idx1 not in available_indices:
                        continue
                    if idx2 in available_indices:
                        score = -dist * lev_weight + sim * sim_weight
                        scores.append((score, idx2))

                if scores:
                    best_score, best_idx = max(scores)
                    token2 = self.tokens[best_idx]

                    token1 = (token1[0], self.idx_to_id[token1[1]])
                    token2 = (token2[0], self.idx_to_id[token2[1]])
                    matches.append((token1, token2, best_score))
                    available_indices.remove(batch_start + i)
                    available_indices.remove(best_idx)
                else:
                    self.NO_MATCH_CNT += 1

        # Randomly pair remaining tokens
        available_indices = list(available_indices)
        random.shuffle(available_indices)
        for i in range(0, len(available_indices), 2):
            if i + 1 < len(available_indices):
                token1 = self.tokens[available_indices[i]]
                token2 = self.tokens[available_indices[i + 1]]
                token1 = (token1[0], self.idx_to_id[token1[1]])
                token2 = (token2[0], self.idx_to_id[token2[1]])
                matches.append((token1, token2, -100))

        return sorted(matches, key=lambda x: x[2], reverse=True)


def optimize_memory_usage(func):
    """Decorator to reduce GPU memory usage."""
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        result = func(*args, **kwargs)
        torch.cuda.empty_cache()
        return result

    return wrapper


@optimize_memory_usage
def main():
    vocab_size = 300000
    embedding_dim = 768
    batch_size = 1000
    n_neighbors = 100

    embeddings = {f"token_{i}": torch.randn(embedding_dim) for i in range(vocab_size)}
    tokens = list(embeddings.keys())

    matcher = OptimizedTokenMatcher(
        embeddings=embeddings,
        tokens=tokens,
        batch_size=batch_size,
        n_neighbors=n_neighbors,
    )

    matches = matcher.find_matches(w1=0.6, w2=0.4)

    print("\nTop 5 matches:")
    for token1, token2, score in matches[:5]:
        print(f"{token1} -> {token2} (score: {score:.3f})")


if __name__ == "__main__":
    main()
