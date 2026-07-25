# Grammora GPT — from-scratch Urdu/English instruction LLM

`grammora_gpt_train.py` trains a **decoder-only GPT from scratch** on **100% of every
JSONL file** in your dataset folder. It replaces the old encoder–decoder paraphrasing
model with a single instruction-following model that handles grammar correction,
paraphrasing, QA, summarization, translation, text generation, and raw Urdu language
modeling — all selected by a chat template.

## Why this uses your *full* dataset (not 20–30%)

- **Streaming loader.** Files are read line-by-line, never loaded into RAM. Your
  84 GB `question_answering.jsonl` (66M records) streams fine.
- **No quality filter that drops pairs.** The old code discarded pairs by word-overlap;
  this one keeps every record.
- **Packing, not padding.** Tokenized examples are packed into full `block_size`
  windows, so almost no compute is wasted on padding — maximum tokens/step.
- **The only cap is on tokenizer training** (`tokenizer_sample_lines_per_file`,
  default 1.5M lines/file). That is just to build the BPE vocab quickly; the *model*
  still trains on every record.

## Zero-UNK tokenizer

SentencePiece BPE with `byte_fallback=True` and `character_coverage=1.0`. Any character
it hasn't merged falls back to raw bytes, so `<unk>` is **never** emitted — the same
guarantee that gave you clean results before. Chat role markers
(`<|system|> <|user|> <|assistant|> <|eot|>`) are registered as single tokens.

## Instruction masking

For chat records the layout is:

```
<s> <|system|> …persona… <|eot|> <|user|> …question… <|eot|> <|assistant|> …answer… <|eot|>
```

Loss is computed **only on the assistant answer + its `<|eot|>`**. The prompt is masked
(`-100`), so the model learns to *answer*, not to echo the prompt. `urdu_corpus.jsonl`
(`text` records) trains as plain language modeling (every token supervised).

## Setup

```bash
pip install torch sentencepiece
# upload your jsonl_datasets folder to the GPU box, then:
export GRAMMORA_DATA_DIR=/workspace/jsonl_datasets
export GRAMMORA_OUT_DIR=/workspace/grammora_out
```

## Run

**Both GPUs (your 2× RTX PRO 6000 — recommended):**
```bash
torchrun --nproc_per_node=2 grammora_gpt_train.py \
  --data_dir /workspace/jsonl_datasets --out_dir /workspace/grammora_out
```

**Single GPU / inside Jupyter:**
```bash
python grammora_gpt_train.py --data_dir /workspace/jsonl_datasets
```
or in a notebook cell: `!python grammora_gpt_train.py --data_dir /workspace/jsonl_datasets`

**Build only the tokenizer first (optional):**
```bash
python grammora_gpt_train.py --mode tokenizer
```

**Chat with a trained checkpoint:**
```bash
python grammora_gpt_train.py --mode chat --prompt "اس کی گرامر درست کریں۔ میں اسکول جاتا ہوں کل۔"
```

Training auto-resumes from the latest checkpoint in `out_dir` (`resume=True`).

## Tuning (edit the `Config` dataclass at the top of the script)

| Goal | Change |
|------|--------|
| Bigger model (you have the VRAM) | `n_embd=1536, n_layer=32` (~1.3B) or `n_embd=2048` |
| Longer context | `block_size=2048` (more VRAM/step) |
| More/less data per step | `micro_batch_size`, `grad_accum_steps` |
| Train longer | raise `max_steps` |
| fp16 instead of bf16 | `dtype="float16"` |
| OOM | lower `micro_batch_size`, raise `grad_accum_steps` |

Effective batch = `micro_batch_size × grad_accum_steps × num_gpus`.
Defaults (`24 × 8 × 2`) ≈ 384 sequences ≈ 393k tokens/optimizer step at `block_size=1024`.

## Scale / cost reality check

One full pass over all files is ~78M examples — the QA file alone dominates. That is a
multi-day job even on 2× RTX PRO 6000. `max_steps` (not epochs) bounds the run; each
step sees a fresh, interleaved, packed batch, so you get broad coverage early and can
stop when the sample outputs look good. Raise `max_steps` for more passes.
