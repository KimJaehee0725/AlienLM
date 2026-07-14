"""Recover token IDs from text returned by an unmodified model server.

Tokenizer decoding can erase token boundaries. If a black-box API returns only
decoded text, re-encoding that text is therefore not guaranteed to reproduce
the generated IDs. This module builds the inverse decoder lattice and ranks
minimum-token candidates using tokenizer-internal signals only.

The current implementation supports the decoder layouts used by Llama 3
(``ByteLevel``) and Gemma 2 (``Replace``/``ByteFallback``/``Fuse``).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import unicodedata


_BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>")
_ALLOWED_FORMAT_CHARS = frozenset({"\u200d", "\ufe0e", "\ufe0f"})


def _gpt2_byte_decoder() -> dict[str, int]:
    """Return the inverse alphabet used by the ByteLevel pre-tokenizer."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(byte_values)
    extra = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): byte_value for byte_value, codepoint in zip(byte_values, codepoints)}


@dataclass(frozen=True)
class RecoveryResult:
    text: str
    token_ids: tuple[int, ...]
    minimum_tokens: int
    candidate_count: int


class TextResponseRecoverer:
    """Invert supported tokenizer decoders for short text-only API responses."""

    def __init__(
        self,
        base_tokenizer,
        alien_tokenizer,
        *,
        max_candidates: int = 1_024,
        max_expansions: int = 50_000,
    ) -> None:
        self.base_tokenizer = base_tokenizer
        self.alien_tokenizer = alien_tokenizer
        self.max_candidates = max_candidates
        self.max_expansions = max_expansions
        self._terminal = object()
        self._trie: dict = {}
        self._byte_id_by_value: dict[int, int] = {}
        self._single_byte_decodings: dict[str, list[tuple[int, ...]]] = {}
        self._mode = self._detect_decoder_mode()
        self._build_decoder_index()

    def _detect_decoder_mode(self) -> str:
        backend = json.loads(self.base_tokenizer.backend_tokenizer.to_str())
        decoder = backend.get("decoder") or {}
        if decoder.get("type") == "ByteLevel":
            return "byte_level"

        decoders = decoder.get("decoders", ())
        decoder_types = [item.get("type") for item in decoders]
        if decoder.get("type") == "Sequence" and decoder_types == [
            "Replace",
            "ByteFallback",
            "Fuse",
        ]:
            replace = decoders[0]
            pattern = replace.get("pattern", {})
            if pattern.get("String") == "▁" and replace.get("content") == " ":
                return "metaspace_fallback"

        raise NotImplementedError(
            "Text-only response recovery currently supports Llama 3 ByteLevel "
            "and Gemma 2 Replace/ByteFallback/Fuse decoders."
        )

    def _insert_piece(self, piece, token_id: int) -> None:
        if not piece:
            return
        node = self._trie
        for unit in piece:
            node = node.setdefault(unit, {})
        node.setdefault(self._terminal, []).append(token_id)

    def _build_decoder_index(self) -> None:
        special_ids = set(self.base_tokenizer.all_special_ids)
        vocab = sorted(self.base_tokenizer.get_vocab().items(), key=lambda item: item[1])

        if self._mode == "byte_level":
            byte_decoder = _gpt2_byte_decoder()
            for token, token_id in vocab:
                if token_id in special_ids:
                    continue
                try:
                    piece = bytes(byte_decoder[character] for character in token)
                except KeyError:
                    continue
                self._insert_piece(piece, token_id)
            return

        for token, token_id in vocab:
            byte_match = _BYTE_TOKEN.fullmatch(token)
            if byte_match:
                self._byte_id_by_value[int(byte_match.group(1), 16)] = token_id
                continue
            if token_id in special_ids:
                continue
            self._insert_piece(token.replace("▁", " "), token_id)

        for byte_value, token_id in sorted(self._byte_id_by_value.items()):
            visible = bytes([byte_value]).decode("utf-8", errors="replace")
            self._single_byte_decodings.setdefault(visible, []).append((token_id,))

    def _byte_arcs(self, character: str) -> list[tuple[int, ...]]:
        arcs = list(self._single_byte_decodings.get(character, ()))
        utf8 = character.encode("utf-8")
        if all(byte in self._byte_id_by_value for byte in utf8):
            arcs.append(tuple(self._byte_id_by_value[byte] for byte in utf8))
        return list(dict.fromkeys(arcs))

    def _edges(self, server_text: str) -> tuple[list[list[tuple[int, tuple[int, ...]]]], int]:
        source = server_text.encode("utf-8") if self._mode == "byte_level" else server_text
        edges: list[list[tuple[int, tuple[int, ...]]]] = [
            [] for _ in range(len(source) + 1)
        ]

        for start in range(len(source)):
            node = self._trie
            end = start
            while end < len(source) and source[end] in node:
                node = node[source[end]]
                end += 1
                for token_id in node.get(self._terminal, ()):
                    edges[start].append((end, (token_id,)))

            if self._mode == "metaspace_fallback":
                for byte_ids in self._byte_arcs(source[start]):
                    edges[start].append((start + 1, byte_ids))
            edges[start] = list(dict.fromkeys(edges[start]))
        return edges, len(source)

    @staticmethod
    def _minimum_distances(
        edges: list[list[tuple[int, tuple[int, ...]]]],
    ) -> tuple[list[int], list[int]]:
        length = len(edges) - 1
        infinity = 10**12
        forward = [infinity] * (length + 1)
        backward = [infinity] * (length + 1)
        forward[0] = 0
        backward[length] = 0

        for position in range(length):
            if forward[position] == infinity:
                continue
            for next_position, token_ids in edges[position]:
                forward[next_position] = min(
                    forward[next_position],
                    forward[position] + len(token_ids),
                )

        for position in range(length - 1, -1, -1):
            for next_position, token_ids in edges[position]:
                if backward[next_position] != infinity:
                    backward[position] = min(
                        backward[position],
                        len(token_ids) + backward[next_position],
                    )
        return forward, backward

    def _minimum_candidates(
        self,
        server_text: str,
    ) -> tuple[list[tuple[tuple[int, ...], str, bool]], int]:
        edges, source_length = self._edges(server_text)
        forward, backward = self._minimum_distances(edges)
        minimum_tokens = forward[-1]
        if minimum_tokens >= 10**12:
            return [], 0

        candidates_by_text: dict[str, tuple[tuple[int, ...], str, bool]] = {}
        stack: list[tuple[int, tuple[int, ...]]] = [(0, ())]
        expansions = 0

        while stack and len(candidates_by_text) < self.max_candidates:
            position, token_ids = stack.pop()
            expansions += 1
            if expansions > self.max_expansions:
                break

            if position == source_length:
                recovered = self.alien_tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                canonical = tuple(
                    self.alien_tokenizer.encode(recovered, add_special_tokens=False)
                ) == token_ids
                existing = candidates_by_text.get(recovered)
                if existing is None or (canonical and not existing[2]):
                    candidates_by_text[recovered] = (token_ids, recovered, canonical)
                continue

            shortest_edges = []
            for next_position, arc_ids in edges[position]:
                if (
                    len(token_ids)
                    + len(arc_ids)
                    + backward[next_position]
                    == minimum_tokens
                ):
                    shortest_edges.append((next_position, arc_ids))
            for next_position, arc_ids in reversed(shortest_edges):
                stack.append((next_position, token_ids + arc_ids))

        return list(candidates_by_text.values()), int(minimum_tokens)

    @staticmethod
    def _structural_penalty(text: str) -> float:
        penalty = 0.0
        for character in text:
            category = unicodedata.category(character)
            if (
                category in {"Cc", "Cf", "Cs", "Co", "Cn"}
                and character not in {"\n", "\r", "\t"}
                and character not in _ALLOWED_FORMAT_CHARS
            ):
                penalty += 8.0
            if character == "\ufffd":
                penalty += 16.0
        for opening, closing in (("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")):
            penalty += 0.5 * abs(text.count(opening) - text.count(closing))
        return penalty

    def _bpe_rank(self, text: str) -> tuple[int, float]:
        token_ids = self.base_tokenizer.encode(text, add_special_tokens=False)
        return len(token_ids), sum(math.log1p(token_id) for token_id in token_ids)

    def recover(self, server_text: str) -> RecoveryResult:
        if not server_text:
            return RecoveryResult("", (), 0, 1)

        candidates, minimum_tokens = self._minimum_candidates(server_text)
        if not candidates:
            raise ValueError("No token segmentation can reproduce the server response.")

        def penalty(candidate: tuple[tuple[int, ...], str, bool]) -> tuple:
            _token_ids, recovered, canonical = candidate
            return (
                not canonical,
                self._structural_penalty(recovered),
                *self._bpe_rank(recovered),
            )

        best = min(candidates, key=penalty)
        return RecoveryResult(
            text=best[1],
            token_ids=best[0],
            minimum_tokens=minimum_tokens,
            candidate_count=len(candidates),
        )
