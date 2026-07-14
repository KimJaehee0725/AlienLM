from __future__ import annotations

import tempfile
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from translator import TokenizerTranslator


def _write_wordlevel_tokenizer(path: Path, vocab: dict[str, int]) -> None:
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    wrapped.save_pretrained(path)


def main() -> None:
    base_vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "hello": 4,
        "world": 5,
    }
    alien_vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "zul": 4,
        "nara": 5,
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        base_path = tmp_path / "base"
        alien_path = tmp_path / "alien"
        _write_wordlevel_tokenizer(base_path, base_vocab)
        _write_wordlevel_tokenizer(alien_path, alien_vocab)

        translator = TokenizerTranslator(
            alien_tokenizer_path=str(alien_path.resolve()),
            opensource_tokenizer=str(base_path.resolve()),
        )

        plain = "hello world"
        alien = translator.plain2alien(plain)
        round_trip = translator.alien2plain(alien)
        decoded_ids = translator.decode_token_ids(
            [4, 5],
            skip_special_tokens=False,
        )

    assert alien == "zul nara", alien
    assert round_trip == plain, round_trip
    assert decoded_ids == alien, decoded_ids
    print("translator round-trip smoke test passed")


if __name__ == "__main__":
    main()
