import argparse
import sys
from pathlib import Path
from typing import Sequence, Union

from transformers import AutoTokenizer

try:
    from .inverse_segmentation import TextResponseRecoverer
except ImportError:  # Support `python translator/translator.py ...`.
    from inverse_segmentation import TextResponseRecoverer


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


class TokenizerTranslator:
    """
    Translate between alien-tokenized and open-source-tokenized text by
    encoding with one tokenizer and decoding with the other.

    alien2plain: alien encode -> open-source decode
    plain2alien: open-source encode -> alien decode
    """

    def __init__(
        self,
        alien_tokenizer_path: str,
        opensource_tokenizer: str = "meta-llama/Meta-Llama-3-8B",
        trust_remote_code: bool = False,
    ) -> None:
        alien_path = Path(alien_tokenizer_path)
        if not alien_path.is_absolute():
            raise ValueError("alien_tokenizer_path must be an absolute path.")
        if not alien_path.exists():
            raise FileNotFoundError(f"Alien tokenizer path not found: {alien_path}")

        self.alien_tokenizer = _load_tokenizer(str(alien_path), trust_remote_code=trust_remote_code)
        self.opensource_tokenizer = _load_tokenizer(opensource_tokenizer, trust_remote_code=trust_remote_code)

        self._assert_compatibility()
        self._response_recoverer = None

    def _assert_compatibility(self) -> None:
        # Class name check
        if self.alien_tokenizer.__class__.__name__ != self.opensource_tokenizer.__class__.__name__:
            raise AssertionError(
                "Tokenizer class names differ: "
                f"{self.alien_tokenizer.__class__.__name__} != {self.opensource_tokenizer.__class__.__name__}"
            )

        # Vocab size + id range checks
        alien_min, alien_max = _vocab_id_range(self.alien_tokenizer)
        open_min, open_max = _vocab_id_range(self.opensource_tokenizer)

        if (alien_min, alien_max) != (open_min, open_max):
            raise AssertionError(
                "Tokenizer id ranges differ: "
                f"alien [{alien_min}, {alien_max}] vs open [{open_min}, {open_max}]"
            )

        if len(self.alien_tokenizer) != len(self.opensource_tokenizer):
            raise AssertionError(
                "Tokenizer vocab sizes differ: "
                f"alien {len(self.alien_tokenizer)} vs open {len(self.opensource_tokenizer)}"
            )

    def alien2plain(self, text: TextOrList) -> TextOrList:
        return self._translate(text, source=self.alien_tokenizer, target=self.opensource_tokenizer)

    def plain2alien(self, text: TextOrList) -> TextOrList:
        return self._translate(text, source=self.opensource_tokenizer, target=self.alien_tokenizer)

    def recover_server_response(self, text: TextOrList) -> TextOrList:
        """Recover plain text when a fixed server returns decoded text only.

        This differs from ``alien2plain``: the input is assumed to be the
        output of ``opensource_tokenizer.decode(generated_ids)`` and may have
        lost its original token boundaries.
        """
        if isinstance(text, str):
            if self._response_recoverer is None:
                self._response_recoverer = TextResponseRecoverer(
                    self.opensource_tokenizer,
                    self.alien_tokenizer,
                )
            return self._response_recoverer.recover(text).text
        if isinstance(text, (list, tuple)):
            return [self.recover_server_response(item) for item in text]
        raise TypeError(f"Unsupported text type: {type(text)}")

    def decode_token_ids(self, token_ids, **kwargs) -> str:
        """Decode IDs directly when local inference or evaluation exposes them."""
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        return self.alien_tokenizer.decode(token_ids, **kwargs)

    def _translate(self, text: TextOrList, source, target) -> TextOrList:
        if isinstance(text, str):
            token_ids = source.encode(text, add_special_tokens=False)
            return target.decode(token_ids, skip_special_tokens=False)
        if isinstance(text, (list, tuple)):
            return [self._translate(t, source, target) for t in text]
        raise TypeError(f"Unsupported text type: {type(text)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate text between alien and open-source tokenizers.")
    parser.add_argument(
        "--alien-tokenizer-path",
        required=True,
        help="Absolute path to alien tokenizer directory.",
    )
    parser.add_argument(
        "--opensource-tokenizer",
        default="meta-llama/Meta-Llama-3-8B",
        help="Open-source tokenizer name or path (default: Meta-Llama-3-8B).",
    )
    parser.add_argument(
        "--direction",
        choices=["alien2plain", "plain2alien", "recover-server-response"],
        default="alien2plain",
    )
    parser.add_argument("text", nargs="*", help="Text to translate; if empty, read stdin.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    translator = TokenizerTranslator(
        alien_tokenizer_path=args.alien_tokenizer_path,
        opensource_tokenizer=args.opensource_tokenizer,
    )

    if args.text:
        input_text = " ".join(args.text)
    else:
        input_text = sys.stdin.read()

    if args.direction == "alien2plain":
        output_text = translator.alien2plain(input_text)
    elif args.direction == "recover-server-response":
        output_text = translator.recover_server_response(input_text)
    else:
        output_text = translator.plain2alien(input_text)

    print(output_text)


if __name__ == "__main__":
    main()
