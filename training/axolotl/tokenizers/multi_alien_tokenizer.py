"""
Multi-alien tokenizer wrapper for axolotl training.

This wrapper translates plain text -> alien text using one of K alien tokenizers,
then encodes with the base (original) tokenizer. It can switch the active alien
tokenizer per update step via set_step().
"""
from __future__ import annotations

import os
from pathlib import Path
import random
from typing import List, Sequence, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase
try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - optional dependency in some runtimes
    TrainerCallback = object


TextOrList = Union[str, Sequence[str]]

DEFAULT_ALIEN_TOKENIZER_PATHS: List[str] = []


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


def _normalize_paths(paths: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(paths, str):
        split_paths = [p.strip() for p in paths.split(",") if p.strip()]
        return split_paths
    return [str(p) for p in paths]


def _resolve_path(path: str) -> Path:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path(os.getcwd(), path_obj).resolve()
    return path_obj


class MultiAlienTokenizer(PreTrainedTokenizerBase):
    """
    A tokenizer wrapper that translates text using multiple alien tokenizers.

    - encode: plain -> alien (active) -> base encode
    - decode: generated IDs -> alien decode (active; no lossy text round trip)
    """

    def __init__(
        self,
        base_tokenizer_path: str,
        alien_tokenizer_paths: Union[str, Sequence[str]] | None = None,
        random_per_call: bool = True,
        random_seed: int | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> None:
        if not alien_tokenizer_paths:
            env_paths = os.environ.get("ALIEN_TOKENIZER_PATHS", "")
            alien_tokenizer_paths = env_paths if env_paths else DEFAULT_ALIEN_TOKENIZER_PATHS
        if not alien_tokenizer_paths:
            raise ValueError(
                "alien_tokenizer_paths must be provided. "
                "Set tokenizer_kwargs.alien_tokenizer_paths or ALIEN_TOKENIZER_PATHS."
            )

        self.base_tokenizer = _load_tokenizer(base_tokenizer_path, trust_remote_code=trust_remote_code)

        alien_paths = _normalize_paths(alien_tokenizer_paths)
        if not alien_paths:
            raise ValueError("alien_tokenizer_paths resolved to empty.")

        self.alien_tokenizers = []
        for path in alien_paths:
            alien_path = _resolve_path(path)
            if not alien_path.exists():
                raise FileNotFoundError(f"Alien tokenizer path not found: {alien_path}")
            self.alien_tokenizers.append(_load_tokenizer(str(alien_path), trust_remote_code=trust_remote_code))

        self._assert_compatibility()

        self._step = 0
        self._active_index = 0
        self._last_used_index = 0
        self._random_per_call = random_per_call
        if random_seed is not None:
            random.seed(random_seed)

        super().__init__(**kwargs)
        self.chat_template = self.base_tokenizer.chat_template

    def _assert_compatibility(self) -> None:
        base = self.base_tokenizer
        base_min, base_max = _vocab_id_range(base)
        base_size = len(base)
        base_class = base.__class__.__name__

        for idx, alien in enumerate(self.alien_tokenizers):
            print(f"the number of alien tokenizers: {len(self.alien_tokenizers)}")
            if alien.__class__.__name__ != base_class:
                raise AssertionError(
                    "Tokenizer class names differ: "
                    f"{alien.__class__.__name__} != {base_class} (alien index {idx})"
                )
            alien_min, alien_max = _vocab_id_range(alien)
            if (alien_min, alien_max) != (base_min, base_max):
                raise AssertionError(
                    "Tokenizer id ranges differ: "
                    f"alien[{idx}] [{alien_min}, {alien_max}] vs base [{base_min}, {base_max}]"
                )
            if len(alien) != base_size:
                raise AssertionError(
                    "Tokenizer vocab sizes differ: "
                    f"alien[{idx}] {len(alien)} vs base {base_size}"
                )

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("step must be >= 0")
        self._step = step
        self._active_index = step % len(self.alien_tokenizers)

    def advance_step(self) -> None:
        self.set_step(self._step + 1)

    @property
    def active_alien_index(self) -> int:
        return self._active_index

    def _active_alien(self):
        return self.alien_tokenizers[self._active_index]

    def _choose_alien(self):
        if self._random_per_call:
            idx = random.randrange(len(self.alien_tokenizers))
            self._last_used_index = idx
            return self.alien_tokenizers[idx]
        self._last_used_index = self._active_index
        return self.alien_tokenizers[self._active_index]

    def _plain2alien(self, text: TextOrList, alien) -> TextOrList:
        if isinstance(text, str):
            token_ids = self.base_tokenizer.encode(text, add_special_tokens=False)
            return alien.decode(token_ids, skip_special_tokens=False)
        if isinstance(text, (list, tuple)):
            return [self._plain2alien(t, alien) for t in text]
        raise TypeError(f"Unsupported text type: {type(text)}")

    def _alien2plain(self, text: TextOrList, alien) -> TextOrList:
        if isinstance(text, str):
            token_ids = alien.encode(text, add_special_tokens=False)
            return self.base_tokenizer.decode(token_ids, skip_special_tokens=False)
        if isinstance(text, (list, tuple)):
            return [self._alien2plain(t, alien) for t in text]
        raise TypeError(f"Unsupported text type: {type(text)}")

    def encode(self, text, add_special_tokens=True, **kwargs):
        """Encode text with plain->alien translation applied."""
        alien = self._choose_alien()
        alien_text = self._plain2alien(text, alien) if isinstance(text, (str, list, tuple)) else text
        return self.base_tokenizer.encode(alien_text, add_special_tokens=add_special_tokens, **kwargs)

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """Decode IDs directly with the selected alien tokenizer."""
        alien = self._choose_alien()
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        return alien.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    def __call__(self, text, **kwargs):
        """Call tokenizer with plain->alien translation applied."""
        alien = self._choose_alien()
        alien_text = self._plain2alien(text, alien) if isinstance(text, (str, list, tuple)) else text
        return self.base_tokenizer(alien_text, **kwargs)

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False, **kwargs):
        """Apply chat template to conversation with plain->alien translation."""
        alien = self._choose_alien()
        if isinstance(conversation, list):
            translated = []
            for msg in conversation:
                if isinstance(msg, dict) and "content" in msg:
                    new_msg = msg.copy()
                    new_msg["content"] = self._plain2alien(msg["content"], alien)
                    translated.append(new_msg)
                else:
                    translated.append(msg)
            return self.base_tokenizer.apply_chat_template(
                translated,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        return self.base_tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def _add_tokens(self, new_tokens, special_tokens=False):
        return self.base_tokenizer._add_tokens(new_tokens, special_tokens=special_tokens)

    def add_tokens(self, new_tokens, special_tokens=False):
        return self.base_tokenizer.add_tokens(new_tokens, special_tokens=special_tokens)

    def add_special_tokens(self, special_tokens_dict, replace_additional_special_tokens=True):
        return self.base_tokenizer.add_special_tokens(
            special_tokens_dict,
            replace_additional_special_tokens=replace_additional_special_tokens,
        )

    def save_pretrained(self, save_directory, **kwargs):
        return self.base_tokenizer.save_pretrained(save_directory, **kwargs)

    def __getattr__(self, name):
        try:
            base_tokenizer = object.__getattribute__(self, "base_tokenizer")
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return object.__getattribute__(base_tokenizer, name)
        except AttributeError:
            try:
                return getattr(base_tokenizer, name)
            except RecursionError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __len__(self):
        return len(self.base_tokenizer)

    @property
    def last_used_alien_index(self) -> int:
        return self._last_used_index

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        alien_tokenizer_paths: Union[str, Sequence[str]] = DEFAULT_ALIEN_TOKENIZER_PATHS,
        **kwargs,
    ):
        return cls(
            pretrained_model_name_or_path,
            alien_tokenizer_paths=alien_tokenizer_paths,
            **kwargs,
        )


class TokenizerStepCallback(TrainerCallback):
    """
    Trainer callback that updates MultiAlienTokenizer step each optimizer step.
    """

    def on_step_begin(self, args, state, control, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is not None and hasattr(tokenizer, "set_step"):
            tokenizer.set_step(state.global_step)
            if getattr(state, "is_local_process_zero", True):
                active_idx = getattr(tokenizer, "active_alien_index", None)
                print(
                    f"[TokenizerStepCallback] global_step={state.global_step} "
                    f"active_alien_index={active_idx}"
                )
