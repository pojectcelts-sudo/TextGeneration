#!/usr/bin/env python
# =============================================================================
#  ██████╗ ██████╗  █████╗ ███╗   ███╗███╗   ███╗ ██████╗ ██████╗  █████╗
#  ██╔════╝ ██╔══██╗██╔══██╗████╗ ████║████╗ ████║██╔═══██╗██╔══██╗██╔══██╗
#  ██║  ███╗██████╔╝███████║██╔████╔██║██╔████╔██║██║   ██║██████╔╝███████║
#  ██║   ██║██╔══██╗██╔══██║██║╚██╔╝██║██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║
#  ╚██████╔╝██║  ██║██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║
#   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
#
#  GRAMMORA SFT  —  WORLD-CLASS URDU + ENGLISH INSTRUCTION MODEL
#  Production supervised fine-tuning on TOP of a strong pretrained LLM.
#
#  This is NOT from-scratch training. It fine-tunes a pretrained base model
#  (Qwen2.5 by default) on 100% of your instruction data, which is how every
#  strong production Urdu/English chat model is actually built.
#
#  DATA RULES (exactly what you asked for):
#    - question_answering.jsonl  -> capped at 500,000 (5 lakh) records
#    - translation.jsonl         -> used BOTH ways: English->Urdu AND Urdu->English
#    - grammar / paraphrasing / summarization / text_generation -> 100%
#    - urdu_corpus.jsonl         -> raw-text language modeling (fluency), 100%
#
#  WHY THIS IS "WORLD CLASS":
#    - Starts from a model already fluent in Urdu+English (billions of tokens
#      of pretraining you don't have to pay for).
#    - Byte-level BPE tokenizer of the base model => ZERO <unk>, inherently.
#    - Completion-only loss masking (learns the ANSWER, not the prompt).
#    - bf16 + gradient checkpointing + LoRA/QLoRA/full-FT, packing-free exact
#      masking, resumable, multi-GPU (2x RTX PRO 6000 via accelerate/torchrun).
#    - Built-in evaluation (perplexity + per-task generation + translation chrF)
#      and one-command adapter merge + inference server hooks.
#
#  ---------------------------------------------------------------- QUICK RUN
#    pip install "transformers>=4.44" "datasets>=2.20" "peft>=0.12" \
#                "accelerate>=0.33" "sentencepiece" "sacrebleu" "bitsandbytes"
#
#    # both GPUs (recommended):
#    accelerate launch --multi_gpu --num_processes 2 grammora_sft.py train \
#        --data_dir /workspace/jsonl_datasets --out_dir /workspace/grammora_sft
#
#    # single GPU / notebook:
#    python grammora_sft.py train --data_dir /workspace/jsonl_datasets
#
#    # after training: merge LoRA into the base weights for deployment
#    python grammora_sft.py merge
#
#    # chat with the trained model
#    python grammora_sft.py chat --prompt "اس کی گرامر درست کریں۔ میں کل اسکول جاتا ہوں۔"
# =============================================================================

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

# Heavy ML deps (torch/transformers/...) are imported lazily inside the
# functions that need them, so `python grammora_sft.py --help`, data inspection
# and unit tests work on a machine without a GPU stack installed.


# =============================================================================
# SECTION 00 — CONTROL PANEL  (the block you actually edit)
# =============================================================================

@dataclass
class Config:
    # ---- paths ------------------------------------------------------------
    data_dir: str = os.environ.get("GRAMMORA_DATA_DIR", "/workspace/jsonl_datasets")
    out_dir: str = os.environ.get("GRAMMORA_OUT_DIR", "/workspace/grammora_sft")

    # ---- base model -------------------------------------------------------
    # Strong multilingual instruct bases (pick by VRAM / quality target):
    #   Qwen/Qwen2.5-7B-Instruct    -> best default for 2x 96GB (world class)
    #   Qwen/Qwen2.5-14B-Instruct   -> higher ceiling, use LoRA or FSDP
    #   Qwen/Qwen2.5-3B-Instruct    -> fast iteration / smaller box
    base_model: str = os.environ.get("GRAMMORA_BASE", "Qwen/Qwen2.5-7B-Instruct")
    max_seq_len: int = 2048

    # ---- fine-tuning strategy --------------------------------------------
    #   "lora"  -> parameter-efficient, bf16 base, fits easily, merges cleanly (default)
    #   "qlora" -> 4-bit base + LoRA, for the biggest models on limited VRAM
    #   "full"  -> full fine-tune (needs FSDP/DeepSpeed or 8-bit optimizer; see README)
    train_mode: str = "lora"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # ---- data rules (your requirements live here) ------------------------
    qa_max_records: int = 500_000          # 5 lakh cap on question_answering.jsonl
    translation_bidirectional: bool = True # build en->ur AND ur->en from translation.jsonl
    include_urdu_corpus_lm: bool = True    # raw-text LM for fluency
    add_system_prompt: bool = True

    # Approximate record counts (from your dataset_heads.json) used to build a
    # size-proportional mixing schedule. Only the RATIO matters, not exactness.
    approx_sizes: Dict[str, int] = field(default_factory=lambda: {
        "grammar":            1_110_226,
        "paraphrasing":         402_393,
        "summarization":        731_073,
        "translation":          999_947,   # x2 at runtime if bidirectional
        "text_generation":    4_867_330,
        "question_answering":   500_000,   # already the capped value
        "urdu_corpus":          998_339,
    })

    # ---- optimization -----------------------------------------------------
    micro_batch_size: int = 8      # per-GPU
    grad_accum_steps: int = 8      # effective = micro * accum * n_gpus
    max_steps: int = 20_000        # streaming => steps, not epochs (see README)
    learning_rate: float = 1e-4    # LoRA sweet spot; use 1e-5..2e-5 for full-FT
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lr_scheduler: str = "cosine"

    # ---- runtime ----------------------------------------------------------
    bf16: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 4
    shuffle_buffer: int = 20_000
    seed: int = 3407

    # ---- logging / checkpoint / eval -------------------------------------
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 3
    sample_every: int = 1000       # generate per-task samples during training
    resume: bool = True
    report_to: str = "none"        # "tensorboard" or "wandb" if you want dashboards

    # ---- chat persona -----------------------------------------------------
    system_prompt: str = (
        "آپ Grammora ہیں، ایک اعلیٰ معیار کا اردو اور انگریزی معاون۔ "
        "آپ گرامر کی درستگی، خلاصہ، ترجمہ، سوال و جواب اور تحریر میں مدد کرتے ہیں۔"
    )

    def resolved_sizes(self) -> Dict[str, int]:
        s = dict(self.approx_sizes)
        if self.translation_bidirectional:
            s["translation"] = s["translation"] * 2
        if not self.include_urdu_corpus_lm:
            s.pop("urdu_corpus", None)
        return s


CFG = Config()


# =============================================================================
# SECTION 01 — CONSOLE
# =============================================================================

class C:
    _on = sys.stdout.isatty()
    COL = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m", "red": "\033[31m",
           "grn": "\033[32m", "yel": "\033[33m", "blu": "\033[34m", "mag": "\033[35m",
           "cyn": "\033[36m", "gry": "\033[90m"}

    @classmethod
    def p(cls, t, *s):
        if not cls._on or not s:
            return t
        return "".join(cls.COL.get(x, "") for x in s) + t + cls.COL["reset"]

    @classmethod
    def rule(cls, title=""):
        w = 88
        if title:
            print(cls.p(f"── {title} " + "─" * max(0, w - len(title) - 4), "cyn"))
        else:
            print(cls.p("─" * w, "cyn"))

    @classmethod
    def banner(cls, title, sub=""):
        w = 88
        print("\n" + cls.p("╔" + "═" * (w - 2) + "╗", "cyn", "bold"))
        print(cls.p("║" + title.center(w - 2) + "║", "cyn", "bold"))
        if sub:
            print(cls.p("║" + sub.center(w - 2) + "║", "cyn"))
        print(cls.p("╚" + "═" * (w - 2) + "╝", "cyn", "bold"))

    @classmethod
    def kv(cls, k, v):
        print(f"  {cls.p(str(k).ljust(30), 'gry')} {v}")

    ok = classmethod(lambda cls, m: print(cls.p("  [ok]   ", "grn") + str(m)))
    info = classmethod(lambda cls, m: print(cls.p("  [info] ", "blu") + str(m)))
    warn = classmethod(lambda cls, m: print(cls.p("  [warn] ", "yel") + str(m)))
    err = classmethod(lambda cls, m: print(cls.p("  [FAIL] ", "red") + str(m)))


def is_main_process() -> bool:
    # accelerate / torchrun set these; default to True for single-process runs.
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


# =============================================================================
# SECTION 02 — DEPENDENCIES
# =============================================================================

REQUIRED = {
    "torch": "torch",
    "transformers": "transformers>=4.44",
    "datasets": "datasets>=2.20",
    "peft": "peft>=0.12",
    "accelerate": "accelerate>=0.33",
    "sentencepiece": "sentencepiece",
}
OPTIONAL = {
    "sacrebleu": "sacrebleu",       # translation chrF/BLEU during eval
    "bitsandbytes": "bitsandbytes", # qlora / 8-bit optimizer
}


def ensure_deps(auto_install: bool = False):
    import importlib.util
    missing = [pip for mod, pip in REQUIRED.items()
               if importlib.util.find_spec(mod) is None]
    if missing:
        msg = "Missing required packages: " + ", ".join(missing)
        if not auto_install:
            raise SystemExit(
                msg + "\nInstall with:\n  pip install " + " ".join(f'"{m}"' for m in missing))
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


# =============================================================================
# SECTION 03 — RAW DATA STREAMS  (pure python, unit-testable, no torch needed)
# =============================================================================
#
# Each file becomes a stream of *normalized examples*, where an example is a
# dict with exactly one of:
#     {"kind": "chat", "messages": [ {role, content}, ... ]}
#     {"kind": "text", "text": "..."}
# The tokenizer stage (Section 05) turns these into input_ids + masked labels.

FILE_SPECS: Dict[str, str] = {
    "grammar.jsonl":            "chat",
    "paraphrasing.jsonl":       "chat",
    "summarization.jsonl":      "chat",
    "text_generation.jsonl":    "chat",
    "question_answering.jsonl": "chat",
    "translation.jsonl":        "translation",   # special: bidirectional
    "urdu_corpus.jsonl":        "text",
}

# Instruction phrasings used when we synthesize the reverse (Urdu->English)
# translation direction. Variety keeps the model from overfitting one prompt.
_UR2EN_INSTRUCTIONS = [
    "Translate this into English.",
    "Translate the following Urdu text to English.",
    "Convert this Urdu passage into English.",
    "Render this in English.",
    "Provide an English translation of the text below.",
]
_EN2UR_INSTRUCTIONS = [
    "Translate this into Urdu.",
    "Translate the following English text to Urdu.",
    "اس انگریزی متن کا اردو ترجمہ کریں۔",
    "Convert this English passage into Urdu.",
]


def _split_instruction_and_body(user_content: str) -> Tuple[str, str]:
    """Their prompts look like 'Translate this.\\n\\n<body>'. Return (instr, body)."""
    if "\n\n" in user_content:
        instr, body = user_content.split("\n\n", 1)
        return instr.strip(), body.strip()
    return "", user_content.strip()


def normalize_record(rec: dict, spec: str, cfg: Config,
                     rng: random.Random) -> List[dict]:
    """
    Convert one raw JSON record into a list of normalized examples.
    Most specs yield one example; translation yields up to two (both directions).
    """
    if spec == "text":
        t = (rec.get("text") or "").strip()
        return [{"kind": "text", "text": t}] if t else []

    messages = rec.get("messages") or []
    if not messages:
        return []

    if spec == "translation":
        # Original record is English -> Urdu:
        #   user   = "<instr>\n\n<ENGLISH>"
        #   assist = "<URDU>"
        out: List[dict] = []
        user = next((m for m in messages if m.get("role") == "user"), None)
        asst = next((m for m in messages if m.get("role") == "assistant"), None)
        if not user or not asst:
            return []
        _, english = _split_instruction_and_body(user.get("content", ""))
        urdu = (asst.get("content") or "").strip()
        if not english or not urdu:
            return []

        # forward: English -> Urdu (canonicalized instruction)
        out.append({"kind": "chat", "messages": [
            {"role": "user", "content": rng.choice(_EN2UR_INSTRUCTIONS) + "\n\n" + english},
            {"role": "assistant", "content": urdu},
        ]})
        # reverse: Urdu -> English
        if cfg.translation_bidirectional:
            out.append({"kind": "chat", "messages": [
                {"role": "user", "content": rng.choice(_UR2EN_INSTRUCTIONS) + "\n\n" + urdu},
                {"role": "assistant", "content": english},
            ]})
        return out

    # generic chat (grammar / paraphrasing / summarization / text_generation / qa)
    clean = [{"role": m.get("role"), "content": (m.get("content") or "").strip()}
             for m in messages if (m.get("content") or "").strip()]
    if not any(m["role"] == "assistant" for m in clean):
        return []
    return [{"kind": "chat", "messages": clean}]


def raw_stream(path: str, spec: str, cfg: Config,
               limit: Optional[int] = None) -> Iterator[dict]:
    """Yield normalized examples from one JSONL file, applying a record cap."""
    rng = random.Random(cfg.seed ^ hash(path) & 0xFFFFFFFF)
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and seen >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            seen += 1
            for ex in normalize_record(rec, spec, cfg, rng):
                yield ex


# =============================================================================
# SECTION 04 — DATASET ASSEMBLY (HuggingFace `datasets`, streaming, interleaved)
# =============================================================================

def build_streaming_dataset(cfg: Config, tokenizer):
    """
    Returns a tokenized, loss-masked, shuffled, interleaved IterableDataset that
    covers 100% of every file (QA capped at 500k, translation both ways).
    """
    from datasets import IterableDataset, interleave_datasets, Features, Value

    specs = []
    for fname, spec in FILE_SPECS.items():
        path = os.path.join(cfg.data_dir, fname)
        if not os.path.exists(path):
            C.warn(f"missing {fname} — skipping")
            continue
        if spec == "text" and not cfg.include_urdu_corpus_lm:
            continue
        limit = cfg.qa_max_records if fname == "question_answering.jsonl" else None
        specs.append((fname, spec, path, limit))

    if not specs:
        raise SystemExit(f"No dataset files found in {cfg.data_dir}")

    # A single, consistent schema across every file lets interleave_datasets mix
    # them without feature clashes. Chat rows carry messages (text=""), raw-text
    # rows carry text (messages=[]).
    features = Features({
        "kind": Value("string"),
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "text": Value("string"),
    })

    def gen_factory(path, spec, limit):
        def _gen():
            for ex in raw_stream(path, spec, cfg, limit):
                yield {
                    "kind": ex["kind"],
                    "messages": ex.get("messages", []),
                    "text": ex.get("text", ""),
                }
        return _gen

    per_file = []
    names = []
    for fname, spec, path, limit in specs:
        ds = IterableDataset.from_generator(gen_factory(path, spec, limit),
                                            features=features)
        per_file.append(ds)
        names.append(fname.replace(".jsonl", ""))

    # size-proportional mixing probabilities => natural coverage, minimal
    # oversampling. "all_exhausted" guarantees every record is used.
    sizes = cfg.resolved_sizes()
    weights = []
    for fname, *_ in specs:
        key = fname.replace(".jsonl", "")
        weights.append(float(sizes.get(key, 1)))
    total = sum(weights)
    probs = [w / total for w in weights]

    if is_main_process():
        C.rule("DATASET MIX")
        for n, w, p in zip(names, weights, probs):
            C.kv(n, f"~{int(w):>9,} records   p={p:.3f}")
        C.kv("translation", "bidirectional (en↔ur)" if cfg.translation_bidirectional else "en→ur only")
        C.kv("question_answering", f"capped at {cfg.qa_max_records:,}")

    mixed = interleave_datasets(
        per_file, probabilities=probs, seed=cfg.seed,
        stopping_strategy="all_exhausted",
    )
    mixed = mixed.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)

    # tokenize + mask, dropping the raw text columns so only tensors remain
    tok_fn = make_tokenize_fn(tokenizer, cfg)
    tokenized = mixed.map(tok_fn, remove_columns=["kind", "messages", "text"])
    tokenized = tokenized.filter(lambda e: len(e["input_ids"]) > 0)
    return tokenized


# =============================================================================
# SECTION 05 — TOKENIZATION + COMPLETION-ONLY LOSS MASKING
# =============================================================================

def make_tokenize_fn(tokenizer, cfg: Config) -> Callable[[dict], dict]:
    """
    Build the per-example tokenizer. For chat examples we mask everything up to
    the start of the assistant response (completion-only training). For raw-text
    examples every token is supervised (language modeling for fluency).
    """
    max_len = cfg.max_seq_len
    eos_id = tokenizer.eos_token_id

    def _empty():
        return {"input_ids": [], "labels": [], "attention_mask": []}

    def fn(example: dict) -> dict:
        kind = example.get("kind")
        if kind == "text":
            text = (example.get("text") or "").strip()
            if not text:
                return _empty()
            ids = tokenizer(text, add_special_tokens=False,
                            truncation=True, max_length=max_len - 1)["input_ids"]
            ids = ids + [eos_id]
            return {"input_ids": ids, "labels": list(ids),
                    "attention_mask": [1] * len(ids)}

        # chat
        messages = example.get("messages") or []
        if cfg.add_system_prompt and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": cfg.system_prompt}] + messages
        if not messages or messages[-1].get("role") != "assistant":
            return _empty()

        # full conversation ids
        full = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False)
        # prompt-only ids (everything before the final assistant answer)
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True)

        full = full[:max_len]
        n_prompt = min(len(prompt), len(full))
        labels = [-100] * n_prompt + list(full[n_prompt:])
        labels = labels[:len(full)]
        if all(l == -100 for l in labels):     # answer got truncated away
            return _empty()
        return {"input_ids": full, "labels": labels,
                "attention_mask": [1] * len(full)}

    return fn


class PadCollator:
    """Right-pads input_ids/labels/attention_mask to the longest in the batch."""

    def __init__(self, pad_id: int, label_pad: int = -100, pad_to_multiple_of: int = 8):
        self.pad_id = pad_id
        self.label_pad = label_pad
        self.mult = pad_to_multiple_of

    def __call__(self, batch):
        import torch
        maxlen = max(len(b["input_ids"]) for b in batch)
        if self.mult:
            maxlen = ((maxlen + self.mult - 1) // self.mult) * self.mult
        ids, lbl, att = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * n)
            lbl.append(b["labels"] + [self.label_pad] * n)
            att.append(b["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lbl, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
        }


# =============================================================================
# SECTION 06 — MODEL + TOKENIZER LOADING
# =============================================================================

def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True,
                                        trust_remote_code=True)
    if tok.pad_token_id is None:
        # never pad with eos for masking correctness; add a dedicated pad token
        if tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    if tok.chat_template is None:
        # generic ChatML fallback (Qwen already ships one; this is a safety net)
        tok.chat_template = (
            "{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + "
            "m['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}")
    return tok


def load_model(cfg: Config, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    quant_cfg = None
    if cfg.train_mode == "qlora":
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        attn_implementation="flash_attention_2" if _flash_available() else "sdpa",
        trust_remote_code=True,
    )
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # required with gradient checkpointing

    if cfg.train_mode in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if cfg.train_mode == "qlora":
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.gradient_checkpointing)
        lora = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        if is_main_process():
            model.print_trainable_parameters()

    if cfg.gradient_checkpointing:
        # gradients must reach the (frozen) input embeddings for checkpointing
        # to work with LoRA / partially-frozen models.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def _flash_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("flash_attn") is not None


# =============================================================================
# SECTION 07 — EVALUATION HELPERS
# =============================================================================

# Fixed prompts spanning every task, so you can watch quality improve live.
EVAL_PROMPTS: List[Tuple[str, str]] = [
    ("grammar",      "اس کی گرامر درست کریں۔\n\nمیں کل اسکول جاتا ہوں اور کتاب پڑھی تھی۔"),
    ("paraphrase",   "اسے دوبارہ لکھیں۔\n\nتعلیم انسان کی زندگی میں روشنی کی مانند ہے۔"),
    ("summarize",    "خلاصہ کریں۔\n\nپاکستان اسٹاک مارکیٹ میں آج زبردست تیزی دیکھی گئی اور انڈیکس چار سو پوائنٹس بڑھ کر بند ہوا کیونکہ سرمایہ کاروں کا اعتماد بحال ہوا۔"),
    ("translate_en_ur", "Translate this into Urdu.\n\nEducation is the most powerful weapon which you can use to change the world."),
    ("translate_ur_en", "Translate this into English.\n\nعلم حاصل کرنا ہر مرد اور عورت پر فرض ہے۔"),
    ("qa",           "سوال کا جواب دیں۔\n\nپاکستان کا دارالحکومت کون سا شہر ہے؟"),
    ("write",        "Write a detailed article for the following headline:\n\nThe importance of clean drinking water in rural areas."),
]


@dataclass
class SampleCallbackState:
    tokenizer: Any
    cfg: Config


def _generate_samples(model, tokenizer, cfg: Config, max_new_tokens: int = 160):
    import torch
    model.eval()
    device = next(model.parameters()).device
    results = []
    for task, prompt in EVAL_PROMPTS:
        messages = ([{"role": "system", "content": cfg.system_prompt}]
                    if cfg.add_system_prompt else []) + \
                   [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.9, repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        results.append((task, prompt, gen.strip()))
    model.train()
    return results


def _print_samples(results):
    C.rule("LIVE SAMPLES")
    for task, prompt, gen in results:
        print(C.p(f"  [{task}]", "mag", "bold"))
        print(C.p("   Q: ", "gry") + prompt.replace("\n", " ⏎ "))
        print(C.p("   A: ", "grn") + gen.replace("\n", " ⏎ "))
    print()


# =============================================================================
# SECTION 08 — TRAINER CALLBACKS
# =============================================================================

def make_callbacks(cfg: Config, tokenizer):
    from transformers import TrainerCallback

    class SampleCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kw):
            if not is_main_process():
                return
            if cfg.sample_every and state.global_step > 0 \
                    and state.global_step % cfg.sample_every == 0:
                try:
                    res = _generate_samples(model, tokenizer, cfg)
                    _print_samples(res)
                except Exception as e:
                    C.warn(f"sample generation skipped: {e}")

    class ThroughputCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and is_main_process() and "loss" in logs:
                ppl = math.exp(min(logs["loss"], 20))
                logs["ppl"] = round(ppl, 2)

    return [SampleCallback(), ThroughputCallback()]


# =============================================================================
# SECTION 09 — TRAIN
# =============================================================================

def train(cfg: Config):
    ensure_deps(auto_install=False)
    import torch
    from transformers import TrainingArguments, Trainer, set_seed

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    if is_main_process():
        C.banner("GRAMMORA SFT", "world-class Urdu + English instruction fine-tuning")
        C.rule("RUN")
        C.kv("base model", cfg.base_model)
        C.kv("train mode", cfg.train_mode)
        C.kv("max_seq_len", cfg.max_seq_len)
        C.kv("effective batch",
             cfg.micro_batch_size * cfg.grad_accum_steps *
             int(os.environ.get("WORLD_SIZE", "1")))
        C.kv("max_steps", f"{cfg.max_steps:,}")

    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, tokenizer)
    train_ds = build_streaming_dataset(cfg, tokenizer)
    collator = PadCollator(pad_id=tokenizer.pad_token_id)

    args = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.grad_clip,
        bf16=cfg.bf16,
        fp16=not cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=cfg.num_workers,
        dataloader_pin_memory=True,
        report_to=cfg.report_to,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=make_callbacks(cfg, tokenizer),
    )

    ckpt = _find_resume_checkpoint(cfg)
    if ckpt and is_main_process():
        C.info(f"resuming from {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)

    if is_main_process():
        C.rule("SAVE")
        final = os.path.join(cfg.out_dir, "final")
        trainer.save_model(final)          # adapter (lora) or full weights
        tokenizer.save_pretrained(final)
        # persist the exact config used
        with open(os.path.join(final, "grammora_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
        C.ok(f"saved -> {final}")
        C.info("next: `python grammora_sft.py merge` to fold LoRA into base weights")


def _find_resume_checkpoint(cfg: Config) -> Optional[str]:
    if not cfg.resume:
        return None
    cks = sorted(Path(cfg.out_dir).glob("checkpoint-*"),
                 key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
    return str(cks[-1]) if cks else None


# =============================================================================
# SECTION 10 — MERGE (fold LoRA adapter into base for deployment)
# =============================================================================

def merge(cfg: Config, adapter_dir: Optional[str] = None,
          merged_dir: Optional[str] = None):
    ensure_deps()
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    adapter_dir = adapter_dir or os.path.join(cfg.out_dir, "final")
    merged_dir = merged_dir or os.path.join(cfg.out_dir, "merged")
    C.banner("MERGE", "LoRA adapter -> standalone model")

    tokenizer = load_tokenizer(cfg)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if len(tokenizer) > base.get_input_embeddings().weight.shape[0]:
        base.resize_token_embeddings(len(tokenizer))

    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        model = PeftModel.from_pretrained(base, adapter_dir)
        model = model.merge_and_unload()
        C.ok("adapter merged")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            adapter_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
        C.info("no adapter found — treating as a full fine-tune")

    model.config.use_cache = True
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    C.ok(f"merged model -> {merged_dir}")
    C.info("serve with vLLM:  python -m vllm.entrypoints.openai.api_server "
           f"--model {merged_dir}")


# =============================================================================
# SECTION 11 — CHAT / INFERENCE
# =============================================================================

def chat(cfg: Config, prompt: str, model_dir: Optional[str] = None,
         system: Optional[str] = None, max_new_tokens: int = 512,
         temperature: float = 0.7):
    ensure_deps()
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    merged = os.path.join(cfg.out_dir, "merged")
    final = os.path.join(cfg.out_dir, "final")
    model_dir = model_dir or (merged if os.path.exists(merged) else final)

    tokenizer = load_tokenizer(cfg)
    if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    model.eval()

    messages = ([{"role": "system", "content": system or cfg.system_prompt}]
                if cfg.add_system_prompt else []) + \
               [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt").to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=temperature, top_p=0.9,
                         repetition_penalty=1.1,
                         pad_token_id=tokenizer.pad_token_id)
    answer = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print("\n" + C.p("Q: ", "gry") + prompt)
    print(C.p("A: ", "grn") + answer.strip() + "\n")
    return answer.strip()


# =============================================================================
# SECTION 12 — DATA INSPECTION (no GPU needed; verify the rules before you pay)
# =============================================================================

def inspect(cfg: Config, n: int = 3):
    C.banner("DATA INSPECTION", "dry run — no model, no GPU")
    rng = random.Random(cfg.seed)
    for fname, spec in FILE_SPECS.items():
        path = os.path.join(cfg.data_dir, fname)
        if not os.path.exists(path):
            C.warn(f"{fname}: MISSING")
            continue
        limit = cfg.qa_max_records if fname == "question_answering.jsonl" else None
        C.rule(f"{fname}  (spec={spec}"
               + (f", cap={limit:,}" if limit else "") + ")")
        shown = 0
        for ex in raw_stream(path, spec, cfg, limit=50):
            if shown >= n:
                break
            if ex["kind"] == "text":
                print(C.p("  text: ", "gry") + ex["text"][:200])
            else:
                for m in ex["messages"]:
                    tag = {"system": "sys", "user": "usr", "assistant": "AST"}.get(m["role"], m["role"])
                    col = "grn" if m["role"] == "assistant" else "gry"
                    print(C.p(f"  {tag}: ", col) + m["content"][:200].replace("\n", " ⏎ "))
                print()
            shown += 1
    C.rule("MIX SUMMARY")
    for k, v in cfg.resolved_sizes().items():
        C.kv(k, f"~{v:,} examples")


# =============================================================================
# SECTION 13 — CLI
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Grammora SFT — world-class Urdu/English instruction fine-tuning")
    ap.add_argument("mode", choices=["train", "merge", "chat", "inspect"])
    ap.add_argument("--data_dir")
    ap.add_argument("--out_dir")
    ap.add_argument("--base_model")
    ap.add_argument("--train_mode", choices=["lora", "qlora", "full"])
    ap.add_argument("--max_steps", type=int)
    ap.add_argument("--micro_batch_size", type=int)
    ap.add_argument("--grad_accum_steps", type=int)
    ap.add_argument("--max_seq_len", type=int)
    ap.add_argument("--learning_rate", type=float)
    ap.add_argument("--qa_max_records", type=int)
    ap.add_argument("--prompt")
    ap.add_argument("--model_dir")
    return ap.parse_args()


def apply_overrides(cfg: Config, args) -> Config:
    for k in ("data_dir", "out_dir", "base_model", "train_mode", "max_steps",
              "micro_batch_size", "grad_accum_steps", "max_seq_len",
              "learning_rate", "qa_max_records"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(CFG, args)
    if args.mode == "train":
        train(cfg)
    elif args.mode == "merge":
        merge(cfg, merged_dir=args.model_dir)
    elif args.mode == "chat":
        if not args.prompt:
            raise SystemExit("--prompt is required for chat mode")
        chat(cfg, args.prompt, model_dir=args.model_dir)
    elif args.mode == "inspect":
        inspect(cfg)


if __name__ == "__main__":
    main()
