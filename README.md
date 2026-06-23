# AlienLM ICML Experiment Snapshot

This branch contains the compact code snapshot for the AlienLM paper
experiments. It keeps tokenizer initialization, translator utilities, tokenizer
assets, Axolotl training entrypoints, and minimal evaluation launchers in one
branch.

The default `main` branch is intentionally smaller: it only contains alien
language initialization and translation utilities. Attack/evaluation rebuttal
experiments are kept in the separate `AlienLM-attack-evals` repository.

## Layout

- `tokenizer/build_token_freq/`: dataset token-frequency builders
- `tokenizer/token_init/`: token matching and randomized reordering utilities
- `translator/`: alien tokenizer to base tokenizer translator
- `training/axolotl/configs/`: paper training configs
- `training/axolotl/tokenizers/`: single/multi alien tokenizer wrappers and tokenizer assets
- `training/axolotl/scripts/`: training launchers
- `eval/`: lm-evaluation-harness and EvalPlus launchers

Checkpoints, raw evaluation dumps, dataset caches, W&B runs, and local logs are
kept outside the tracked tree. By default scripts write caches under `.cache/`
and outputs under `outputs/`, both ignored by git.

## Install

Base utilities:

```bash
python -m pip install -r requirements.txt
```

Training/evaluation environment:

```bash
python -m pip install -r requirements-icml.txt
```

With `uv`, install extras instead:

```bash
uv sync --extra init --extra freq --extra train --extra eval
```

## Tokenizer Construction

Build frequency dictionaries:

```bash
python tokenizer/build_token_freq/tulu3.py \
  --output tulu3_tok_dict.json \
  --max_samples 100000

python tokenizer/build_token_freq/slimorca.py \
  --output slimorca_tok_dict.json

python tokenizer/build_token_freq/acereason.py \
  --output acereason_tok_dict.json \
  --max_files 25 \
  --max_samples 500000
```

Match and reorder tokens:

```bash
python tokenizer/token_init/token_matching.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --proxy_model Qwen/Qwen2.5-7B-Instruct \
  --token_freq_json /path/to/pro_tok_dict.json \
  --output matches-sim-and-diff.txt
```

## Translator

```bash
python translator/translator.py \
  --alien-tokenizer-path /abs/path/to/alien_tokenizer \
  --opensource-tokenizer meta-llama/Meta-Llama-3-8B \
  --direction plain2alien \
  "hello world"
```

Dependency-backed smoke test:

```bash
python scripts/smoke/translator_roundtrip.py
```

## Training

The training configs use relative output paths and local tokenizer assets.

Single alien tokenizer:

```bash
bash training/axolotl/scripts/train_alienlm_single.sh
```

Multi-tenant alien tokenizers:

```bash
bash training/axolotl/scripts/train_alienlm_multi.sh
```

The launcher exports:

```bash
PYTHONPATH="$(pwd)/training/axolotl/tokenizers:$PYTHONPATH"
```

For single-tokenizer runs, either edit
`training/axolotl/configs/alienlm/llama3-8b-instruct-alienlm-single.yaml` or set
`ALIEN_TOKENIZER_PATH=/abs/path/to/alien_tokenizer`.

For multi-tokenizer runs, edit
`training/axolotl/configs/tenant-alienlm/llama3-8b-instruct-multi-tenant.yaml`
or set `ALIEN_TOKENIZER_PATHS=/abs/path1,/abs/path2`.

## Evaluation

LM benchmarks:

```bash
MODEL_PATH=/path/to/model bash eval/scripts/run_lm_eval.sh
```

Code benchmarks:

```bash
MODEL_PATH=/path/to/model bash eval/scripts/run_code_eval.sh
```

Useful environment variables:

- `MODEL_PATH`: model checkpoint or Hugging Face model id
- `TOKENIZER_PATH`: optional tokenizer override
- `OUTPUT_DIR`: evaluation output path
- `DEVICE`: CUDA device list for lm-eval
- `TENSOR_PARALLEL_SIZE`: EvalPlus/vLLM tensor parallelism
