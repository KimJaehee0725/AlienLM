"""
Single-alien tokenizer wrapper for Axolotl.

This wrapper translates plain text -> alien text using one alien tokenizer,
then encodes with the base tokenizer. It can also decode back.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase

TextOrList = Union[str, Sequence[str]]


def _load_tokenizer(name_or_path: str, trust_remote_code: bool = False):
    return AutoTokenizer.from_pretrained(
        name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )


def _vocab_id_range(tokenizer) -> tuple[int, int]:
    vocab = tokenizer.get_vocab()
    if not vocab:
        raise ValueError("Tokenizer vocab is empty.")
    ids = vocab.values()
    return min(ids), max(ids)


def _resolve_path(path: str) -> Path:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path(os.getcwd(), path_obj).resolve()
    return path_obj


class SingleAlienTokenizer(PreTrainedTokenizerBase):
    """
    A tokenizer wrapper that translates text with a single alien tokenizer.

    - encode: plain -> alien -> base encode
    - decode: base decode -> alien -> plain
    """

    def __init__(
        self,
        base_tokenizer_path: str,
        alien_tokenizer_path: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> None:
        if not alien_tokenizer_path:
            alien_tokenizer_path = os.environ.get("ALIEN_TOKENIZER_PATH")
        if not alien_tokenizer_path:
            raise ValueError(
                "alien_tokenizer_path must be provided. "
                "Set tokenizer_kwargs.alien_tokenizer_path or ALIEN_TOKENIZER_PATH."
            )

        self.base_tokenizer = _load_tokenizer(base_tokenizer_path, trust_remote_code=trust_remote_code)

        alien_path = _resolve_path(alien_tokenizer_path)
        if not alien_path.exists():
            raise FileNotFoundError(f"Alien tokenizer path not found: {alien_path}")
        self.alien_tokenizer = _load_tokenizer(str(alien_path), trust_remote_code=trust_remote_code)

        self._assert_compatibility()

        super().__init__(**kwargs)
        self.chat_template = self.base_tokenizer.chat_template

    def _assert_compatibility(self) -> None:
        base = self.base_tokenizer
        alien = self.alien_tokenizer

        if alien.__class__.__name__ != base.__class__.__name__:
            raise AssertionError(
                "Tokenizer class names differ: "
                f"{alien.__class__.__name__} != {base.__class__.__name__}"
            )

        alien_min, alien_max = _vocab_id_range(alien)
        base_min, base_max = _vocab_id_range(base)
        if (alien_min, alien_max) != (base_min, base_max):
            raise AssertionError(
                "Tokenizer id ranges differ: "
                f"alien [{alien_min}, {alien_max}] vs base [{base_min}, {base_max}]"
            )

        if len(alien) != len(base):
            raise AssertionError(
                "Tokenizer vocab sizes differ: "
                f"alien {len(alien)} vs base {len(base)}"
            )

    def alien2plain(self, text: TextOrList) -> TextOrList:
        return self._translate(text, source=self.alien_tokenizer, target=self.base_tokenizer)

    def plain2alien(self, text: TextOrList) -> TextOrList:
        return self._translate(text, source=self.base_tokenizer, target=self.alien_tokenizer)

    def _translate(self, text: TextOrList, source, target) -> TextOrList:
        if isinstance(text, str):
            token_ids = source.encode(text, add_special_tokens=False)
            return target.decode(token_ids, skip_special_tokens=False)
        if isinstance(text, (list, tuple)):
            return [self._translate(t, source, target) for t in text]
        raise TypeError(f"Unsupported text type: {type(text)}")

    def _convert_token_to_id(self, token):
        return self.base_tokenizer._convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        return self.base_tokenizer._convert_id_to_token(index)

    def get_vocab(self):
        return self.base_tokenizer.get_vocab()

    def encode(self, text, **kwargs):
        if kwargs.get("add_special_tokens", True):
            kwargs["add_special_tokens"] = False
        alien_text = self.plain2alien(text)
        return self.base_tokenizer.encode(alien_text, **kwargs)

    def __call__(self, text, **kwargs):
        if kwargs.get("add_special_tokens", True):
            kwargs["add_special_tokens"] = False
        alien_text = self.plain2alien(text)
        return self.base_tokenizer(alien_text, **kwargs)

    def decode(self, token_ids, **kwargs):
        base_text = self.base_tokenizer.decode(token_ids, **kwargs)
        return self.alien2plain(base_text)

    @property
    def pad_token_id(self):
        return self.base_tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.base_tokenizer.eos_token_id

    @property
    def bos_token_id(self):
        return self.base_tokenizer.bos_token_id
