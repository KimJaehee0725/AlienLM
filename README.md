# AlienLM

Official code and research artifacts for **AlienLM: Alienization of Language for
API-Boundary Privacy in Black-Box LLMs**.

AlienLM studies a client-side privacy layer for black-box LLM APIs. Plaintext is
translated into an *Alien Language* through a vocabulary-scale token bijection,
the target model is adapted with Alien Adaptation Training (AAT), and the client
recovers the output back into natural language. The repository contains the
tokenizer construction utilities, translator, tokenizer assets, Axolotl training
configs, and lightweight evaluation entrypoints used for the paper experiments.

## Resources

- 📄 **Paper**: [arXiv:2601.22710](https://arxiv.org/abs/2601.22710)
- 🤗 **Models**: [dsba-lab/AlienLM collection](https://huggingface.co/collections/dsba-lab/alienlm)
- 🤗 **Llama 3 Full Alien checkpoint**: [dsba-lab/llama3-8b-instruct-alienlm-full](https://huggingface.co/dsba-lab/llama3-8b-instruct-alienlm-full)
- 🧪 **Recovery evaluations**: [KimJaehee0725/AlienLM-recovery-evals](https://github.com/KimJaehee0725/AlienLM-recovery-evals)
- 💻 **Official repository**: [KimJaehee0725/AlienLM](https://github.com/KimJaehee0725/AlienLM)

## Repository Guide

This repository is split by intended use:

- `main`: compact tokenizer initialization and translator utilities
- `icml`: paper experiment snapshot with tokenizer assets, Axolotl configs, and
  evaluation launchers
- `AlienLM-recovery-evals`: separate repository for recovery, robustness, and
  post-hoc analysis experiments

The `icml` branch keeps only reproducible code and compact assets. Checkpoints,
dataset caches, raw evaluation dumps, W&B runs, and local logs should stay
outside the tracked tree. The included scripts default to ignored paths such as
`.cache/` and `outputs/`.

## Install

The recommended setup uses [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/KimJaehee0725/AlienLM.git
cd AlienLM
uv sync
```

Install only the extras you need:

```bash
# Token-frequency builders and alien tokenizer initialization
uv sync --extra freq --extra init

# Axolotl training configs and launchers
uv sync --extra train

# lm-evaluation-harness, EvalPlus, and vLLM-backed evaluation utilities
uv sync --extra eval
```

For gated Llama-family tokenizers or checkpoints, log in to Hugging Face first:

```bash
uvx --from huggingface_hub huggingface-cli login
```

If your environment cannot install vLLM, you can still use the tokenizer,
translator, training configs, and non-vLLM lm-evaluation-harness paths by
installing the relevant packages manually in a uv environment.

## Quick Start: Translate Text

`TokenizerTranslator` converts text by preserving token IDs and swapping which
tokenizer decodes those IDs.

- `plain2alien`: encode with the original tokenizer, decode with the alien tokenizer
- `alien2plain`: encode with the alien tokenizer, decode with the original tokenizer

For the Llama 3 8B Instruct Full Alien setup, use the tracked full alien
tokenizer asset with the original Llama 3 tokenizer:

```python
from pathlib import Path

from translator import TokenizerTranslator

translator = TokenizerTranslator(
    alien_tokenizer_path=str(
        Path("training/axolotl/tokenizers/alien/full").resolve()
    ),
    opensource_tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
)

plain = "All happy families are alike; each unhappy family is unhappy in its own way."
alien = translator.plain2alien(plain)
restored = translator.alien2plain(alien)

print("plain:", plain)
print("alien:", alien)
print("restored:", restored)
assert restored == plain
```

The Llama 3 Full Alien model card example is copied below:

<table>
  <tr>
    <th>Natural text</th>
    <th>Alien text</th>
  </tr>
  <tr>
    <td><pre>All happy families are alike; each unhappy family is unhappy in its own way.</pre></td>
    <td><pre>One unhappyamilies                        
 hike..:              
 happy     
                                                                        happy hodin                                                                                                             waypoints,</pre></td>
  </tr>
  <tr>
    <th>Original token IDs</th>
    <th>Alien token IDs</th>
  </tr>
  <tr>
    <td><pre>[2460, 6380, 8689, 527, 27083, 26, 1855, 43251, 3070, 374, 43251, 304, 1202, 1866, 1648, 13]</pre></td>
    <td><pre>[4054, 43251, 60004, 66417, 35331, 114100, 27381, 6380, 39185, 23136, 6380, 109132, 8299, 21649, 82386, 11]</pre></td>
  </tr>
</table>

You can run the translator from the command line as well:

```bash
ALIEN_TOKENIZER="$(pwd)/training/axolotl/tokenizers/alien/full"

uv run python translator/translator.py \
  --alien-tokenizer-path "$ALIEN_TOKENIZER" \
  --opensource-tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
  --direction plain2alien \
  "All happy families are alike; each unhappy family is unhappy in its own way."
```

A dependency-light smoke test is also provided:

```bash
uv run python scripts/smoke/translator_roundtrip.py
```

## Tokenizer Assets

The paper snapshot includes compact alien tokenizer assets under:

- `training/axolotl/tokenizers/alien/full`
- `training/axolotl/tokenizers/alien/qwenv2_bucket_random_5_seed-42`
- `training/axolotl/tokenizers/alien/qwenv2_bucket_random_5_seed-43`
- `training/axolotl/tokenizers/alien/qwenv2_bucket_random_5_seed-44`
- `training/axolotl/tokenizers/alien/qwenv2_bucket_random_5_seed-45`
- `training/axolotl/tokenizers/alien/qwenv2_bucket_random_5_seed-46`

The alien tokenizers are designed to keep the original tokenizer vocabulary size
and ID range while changing the token-string mapping for normal tokens. This
lets the translator move between natural text and alienized text by preserving
token IDs.

## Constructing Alien Tokenizers

Build token-frequency dictionaries:

```bash
uv run python tokenizer/build_token_freq/tulu3.py \
  --output tulu3_tok_dict.json \
  --max_samples 100000

uv run python tokenizer/build_token_freq/slimorca.py \
  --output slimorca_tok_dict.json

uv run python tokenizer/build_token_freq/acereason.py \
  --output acereason_tok_dict.json \
  --max_files 25 \
  --max_samples 500000
```

Match and reorder tokens:

```bash
uv run python tokenizer/token_init/token_matching.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --proxy_model Qwen/Qwen2.5-7B-Instruct \
  --token_freq_json /path/to/pro_tok_dict.json \
  --output matches-sim-and-diff.txt
```

## Training

The Axolotl configs use local tokenizer assets and relative output paths.

Single alien tokenizer:

```bash
uv run bash training/axolotl/scripts/train_alienlm_single.sh
```

Multi-tenant alien tokenizers:

```bash
uv run bash training/axolotl/scripts/train_alienlm_multi.sh
```

The launchers export:

```bash
PYTHONPATH="$(pwd)/training/axolotl/tokenizers:$PYTHONPATH"
```

For single-tokenizer runs, either edit
`training/axolotl/configs/alienlm/llama3-8b-instruct-alienlm-single.yaml` or set:

```bash
export ALIEN_TOKENIZER_PATH=/abs/path/to/alien_tokenizer
```

For multi-tokenizer runs, edit
`training/axolotl/configs/tenant-alienlm/llama3-8b-instruct-multi-tenant.yaml`
or set:

```bash
export ALIEN_TOKENIZER_PATHS=/abs/path1,/abs/path2
```

## Evaluation

LM benchmarks:

```bash
MODEL_PATH=/path/to/model uv run bash eval/scripts/run_lm_eval.sh
```

Code benchmarks:

```bash
MODEL_PATH=/path/to/model uv run bash eval/scripts/run_code_eval.sh
```

Useful environment variables:

- `MODEL_PATH`: model checkpoint path or Hugging Face model ID
- `TOKENIZER_PATH`: optional tokenizer override
- `OUTPUT_DIR`: evaluation output path
- `DEVICE`: CUDA device list for lm-eval
- `TENSOR_PARALLEL_SIZE`: EvalPlus/vLLM tensor parallelism

Recovery and robustness experiments are maintained separately in
[AlienLM-recovery-evals](https://github.com/KimJaehee0725/AlienLM-recovery-evals)
so that the main repository stays focused on tokenizer construction, training,
and standard evaluation entrypoints.

## Layout

- `tokenizer/build_token_freq/`: dataset token-frequency builders
- `tokenizer/token_init/`: token matching and randomized reordering utilities
- `translator/`: alien tokenizer to base tokenizer translator
- `training/axolotl/configs/`: paper training configs
- `training/axolotl/tokenizers/`: single/multi alien tokenizer wrappers and tokenizer assets
- `training/axolotl/scripts/`: training launchers
- `eval/`: lm-evaluation-harness and EvalPlus launchers
- `scripts/smoke/`: small checks that do not require paper-scale compute

## Citation

```bibtex
@inproceedings{kim2026alienlm,
  title={AlienLM: Alienization of Language for API-Boundary Privacy in Black-Box LLMs},
  author={Kim, Jaehee and Kang, Pilsung},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```
