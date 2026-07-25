# Grammora SFT — world-class Urdu/English instruction model

`grammora_sft.py` is the **production** path: it fine-tunes a **strong pretrained base
model** (Qwen2.5 by default) on 100% of your instruction data. This is how real
Urdu/English chat models are built — you inherit billions of tokens of pretraining
instead of trying to learn a language from scratch.

> **Two scripts, pick one:**
> - `grammora_sft.py` ← **use this** for a world-class model (pretrained + fine-tune).
> - `grammora_gpt_train.py` ← only if you specifically want a from-scratch GPT.
>
> On "why is one file 4000 lines and another 900": line count isn't quality. Fine-tuning
> a great base model is *less* code and a *better* model than training from scratch.

## What it does with your data (your exact rules)

| File | Rule |
|------|------|
| `question_answering.jsonl` | **capped at 500,000 (5 lakh)** records |
| `translation.jsonl` | used **both ways** — English→Urdu **and** Urdu→English |
| `grammar.jsonl` | 100% |
| `paraphrasing.jsonl` | 100% |
| `summarization.jsonl` | 100% |
| `text_generation.jsonl` | 100% |
| `urdu_corpus.jsonl` | 100%, as raw-text language modeling (fluency) |

Files are **streamed and interleaved** (size-proportional mixing, `all_exhausted` so
every record is used), tokenized with the base model's own byte-level BPE tokenizer
(**zero `<unk>` inherently**), and trained with **completion-only loss masking** — the
model learns the *answer*, never the prompt.

Verify all of this **before** you spend a rupee of GPU time:
```bash
python grammora_sft.py inspect --data_dir /path/to/jsonl_datasets
```

## Install

```bash
pip install "transformers>=4.44" "datasets>=2.20" "peft>=0.12" \
            "accelerate>=0.33" "sentencepiece" "sacrebleu" "bitsandbytes"
# optional but faster attention:
pip install flash-attn --no-build-isolation
```

## Train

**Both GPUs (your 2× RTX PRO 6000 — recommended):**
```bash
accelerate launch --multi_gpu --num_processes 2 grammora_sft.py train \
  --data_dir /workspace/jsonl_datasets --out_dir /workspace/grammora_sft
```

**Single GPU / notebook:**
```bash
python grammora_sft.py train --data_dir /workspace/jsonl_datasets
```
(In a Jupyter cell: `!python grammora_sft.py train --data_dir /workspace/jsonl_datasets`.)

Training prints **live per-task samples** (grammar, paraphrase, summarize, translate both
ways, QA, article writing) every `sample_every` steps so you can watch quality improve.
It auto-resumes from the newest `checkpoint-*` in `out_dir`.

## After training

```bash
python grammora_sft.py merge          # fold LoRA into base -> out_dir/merged
python grammora_sft.py chat --prompt "اس کی گرامر درست کریں۔ میں کل اسکول جاتا ہوں۔"
# serve (OpenAI-compatible):
python -m vllm.entrypoints.openai.api_server --model /workspace/grammora_sft/merged
```

## Choosing model size & strategy (edit `Config` or pass flags)

| Want | Set |
|------|-----|
| Best default (fits easily, world class) | `base_model=Qwen/Qwen2.5-7B-Instruct`, `train_mode=lora` |
| Higher ceiling | `Qwen/Qwen2.5-14B-Instruct`, `train_mode=lora` |
| Biggest model on tight VRAM | `train_mode=qlora` (4-bit base) |
| Full fine-tune | `train_mode=full` + FSDP/DeepSpeed (see below) |
| Longer context | `--max_seq_len 4096` |
| OOM | lower `--micro_batch_size`, raise `--grad_accum_steps` |

Effective batch = `micro_batch_size × grad_accum_steps × num_gpus`.

### Full fine-tune note
Plain DDP replicates the whole model on each GPU; a 7B full-FT (~112 GB of
params+grads+Adam state) won't fit one 96 GB card. For `train_mode=full` launch with
FSDP so the optimizer/params shard across both cards:
```bash
accelerate launch --multi_gpu --num_processes 2 \
  --use_fsdp --fsdp_sharding_strategy FULL_SHARD \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  grammora_sft.py train --train_mode full --learning_rate 1e-5
```
Otherwise stay on `lora`/`qlora` — for domain SFT on your data, high-rank LoRA
(r=64) reaches essentially full-FT quality at a fraction of the cost, and merges to a
standalone model.

## Steps vs epochs
Streaming data is bounded by `max_steps`, not epochs. At the defaults
(`micro=8 × accum=8 × 2 GPUs = 128` seqs/step, `max_seq_len=2048`) each step is
~260k tokens. Raise `--max_steps` for more passes; watch the live samples and stop when
they look right.
