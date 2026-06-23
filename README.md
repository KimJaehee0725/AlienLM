# AlienLM

AlienLM is a tokenizer-level "alienization" toolkit. This branch keeps only
the minimal pieces needed to construct an alien token ordering and translate
text between a base tokenizer and its alien counterpart.

For paper training and evaluation code, use the `icml` branch. Attack and
diagnostic experiments live in the separate `AlienLM-attack-evals` repository.

## Layout

- `tokenizer/token_init/`: alien language initialization utilities
- `translator/`: plain-tokenizer to alien-tokenizer text translation
- `scripts/smoke/`: lightweight local smoke checks

Large training outputs, evaluation dumps, caches, checkpoints, and rebuttal-only
analysis files are intentionally excluded from this branch.

## Install

```bash
python -m pip install -r requirements.txt
```

Optional dependencies for token matching and frequency construction:

```bash
python -m pip install ".[init,freq]"
```

## Alien Language Initialization

Build a token-frequency dictionary, then match/reorder tokens against a proxy
model:

```bash
python tokenizer/token_init/token_matching.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --proxy_model Qwen/Qwen2.5-7B-Instruct \
  --token_freq_json /path/to/token_freq.json \
  --output matches-sim-and-diff.txt
```

For randomized bucket reordering, use `tokenizer/token_init/token_random.py`.

## Translator

Translate text by encoding with one compatible tokenizer and decoding with the
other:

```bash
python translator/translator.py \
  --alien-tokenizer-path /abs/path/to/alien_tokenizer \
  --opensource-tokenizer meta-llama/Meta-Llama-3-8B \
  --direction plain2alien \
  "hello world"
```

Run the dependency-backed round-trip smoke test:

```bash
python scripts/smoke/translator_roundtrip.py
```
