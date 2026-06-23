# Alien Tokenizer Assets

This directory is a placeholder. The single/general AlienLM configs expect the
alien tokenizer assets to be placed at:

```
training/axolotl/tokenizers/alien/full
```

`full` should be a Hugging Face tokenizer folder containing files such as:
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.json` / `merges.txt` (if applicable)

If you store the tokenizer elsewhere, update `tokenizer_config` in the config
or set `ALIEN_TOKENIZER_PATH` for the single-alien wrapper.
