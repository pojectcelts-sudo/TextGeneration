#!/usr/bin/env python
# =============================================================================
#  GRAMMORA AI — ALL-IN-ONE URDU/ENGLISH INSTRUCTION MODEL  (multi-GPU ready)
#
#  ONE file. Runs on BOTH your RTX PRO 6000 GPUs. Before training it prints a
#  full DATA ANALYSIS: tokens per file, total combined tokens, and model
#  parameter counts. Then it fine-tunes Qwen2.5 into a single model that does
#  grammar, paraphrasing, summarization, translation (both ways), Q&A and
#  text generation.
#
#  Whole dataset (no cap) · duplicate removal · Urdu/English-only filter ·
#  completion-only masking · byte-level BPE tokenizer => ZERO <unk> ever.
#
#  ----------------------------------------------------------------- HOW TO RUN
#    # 0) install (once)
#    pip install "transformers>=4.44" "datasets>=2.20" "peft>=0.12" \
#                "accelerate>=0.33" sentencepiece sacrebleu
#
#    # 1) JUST analyze the data (no training, no GPU needed) — see your tokens
#    python train_grammora.py analyze --data_dir /workspace/jsonl_datasets
#
#    # 2) TRAIN ON BOTH GPUs  (this is the important one)
#    accelerate launch --multi_gpu --num_processes 2 train_grammora.py train \
#        --data_dir /workspace/jsonl_datasets --out_dir /workspace/grammora_out
#
#    # 3) merge LoRA into a standalone model, then chat
#    python train_grammora.py merge --out_dir /workspace/grammora_out
#    python train_grammora.py chat  --out_dir /workspace/grammora_out \
#        --prompt "اس کی گرامر درست کریں۔ میں کل اسکول جاتا ہوں۔"
# =============================================================================

from __future__ import annotations
import argparse, gc, hashlib, json, math, os, random, sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    data_dir: str = "/workspace/jsonl_datasets"
    out_dir:  str = "/workspace/grammora_out"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"   # 3B=fast, 14B=higher ceiling
    max_seq_len: int = 2048

    # fine-tuning
    train_mode: str = "lora"                        # "lora" or "qlora"
    lora_r: int = 64; lora_alpha: int = 128; lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj","k_proj","v_proj","o_proj",
                                            "gate_proj","up_proj","down_proj")
    # data rules
    translation_bidirectional: bool = True
    include_urdu_corpus_lm: bool = True
    add_system_prompt: bool = True
    dedup: bool = True
    lang_filter: bool = True
    lang_max_other: float = 0.05
    eval_holdout_per_file: int = 150

    # optimization  (MEMORY-SAFE defaults for 2x 95GB — no OOM)
    micro_batch_size: int = 4         # per GPU
    grad_accum_steps: int = 8
    max_steps: int = 20000
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lr_scheduler: str = "cosine"

    # runtime
    bf16: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 2
    shuffle_buffer: int = 50000
    seed: int = 3407

    # logging / eval  (you see results early)
    logging_steps: int = 10
    save_steps: int = 1000
    save_total_limit: int = 3
    sample_every: int = 250
    eval_every: int = 500
    resume: bool = True

    system_prompt: str = ("آپ Grammora ہیں، ایک اعلیٰ معیار کا اردو اور انگریزی معاون۔ "
        "آپ گرامر کی درستگی، خلاصہ، ترجمہ، سوال و جواب اور تحریر میں مدد کرتے ہیں۔")


# True record counts from your dataset_heads.json (used for token estimation).
TOTAL_RECORDS = {
    "grammar.jsonl":            1_110_226,
    "paraphrasing.jsonl":         402_393,
    "summarization.jsonl":        731_073,
    "text_generation.jsonl":    4_867_330,
    "question_answering.jsonl": 66_220_575,
    "translation.jsonl":          999_947,
    "urdu_corpus.jsonl":          998_339,
}
FILE_SPECS = {"grammar.jsonl":"chat","paraphrasing.jsonl":"chat",
    "summarization.jsonl":"chat","text_generation.jsonl":"chat",
    "question_answering.jsonl":"chat","translation.jsonl":"translation",
    "urdu_corpus.jsonl":"text"}
IGNORE = -100


# =============================================================================
# distributed helpers
# =============================================================================
def rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))
def is_main() -> bool:
    return rank() == 0
def log(*a):
    if is_main(): print(*a, flush=True)


# =============================================================================
# language filter (keep Urdu / English only)
# =============================================================================
def _script_counts(s: str):
    ar = lat = oth = 0
    for ch in s:
        if not ch.isalpha():
            continue
        o = ord(ch)
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or \
           0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
            ar += 1
        elif (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A):
            lat += 1
        else:
            oth += 1
    return ar, lat, oth

def is_ur_or_en(text: str, cfg: Config) -> bool:
    if not cfg.lang_filter:
        return True
    ar, lat, oth = _script_counts(text)
    tot = ar + lat + oth
    if tot < 1:
        return False
    return (oth / tot) <= cfg.lang_max_other


# =============================================================================
# record normalization (bidirectional translation)
# =============================================================================
_UR2EN = ["Translate this into English.","Translate the following Urdu text to English.",
          "Convert this Urdu passage into English.","Render this in English."]
_EN2UR = ["Translate this into Urdu.","Translate the following English text to Urdu.",
          "اس انگریزی متن کا اردو ترجمہ کریں۔","Convert this English passage into Urdu."]

def _split_instr(uc: str):
    if "\n\n" in uc:
        a, b = uc.split("\n\n", 1); return a.strip(), b.strip()
    return "", uc.strip()

def normalize_record(rec: dict, spec: str, cfg: Config, rng: random.Random) -> List[dict]:
    if spec == "text":
        t = (rec.get("text") or "").strip()
        return [{"kind":"text","messages":[],"text":t}] if t else []
    msgs = rec.get("messages") or []
    if not msgs: return []
    if spec == "translation":
        user = next((m for m in msgs if m.get("role")=="user"), None)
        asst = next((m for m in msgs if m.get("role")=="assistant"), None)
        if not user or not asst: return []
        _, english = _split_instr(user.get("content",""))
        urdu = (asst.get("content") or "").strip()
        if not english or not urdu: return []
        out = [{"kind":"chat","text":"","messages":[
            {"role":"user","content":rng.choice(_EN2UR)+"\n\n"+english},
            {"role":"assistant","content":urdu}]}]
        if cfg.translation_bidirectional:
            out.append({"kind":"chat","text":"","messages":[
                {"role":"user","content":rng.choice(_UR2EN)+"\n\n"+urdu},
                {"role":"assistant","content":english}]})
        return out
    clean = [{"role":m.get("role"),"content":(m.get("content") or "").strip()}
             for m in msgs if (m.get("content") or "").strip()]
    if not any(m["role"]=="assistant" for m in clean): return []
    return [{"kind":"chat","text":"","messages":clean}]

def _example_text(ex: dict) -> str:
    if ex["kind"] == "text": return ex["text"]
    return "  ".join(m["role"]+":"+m["content"] for m in ex["messages"])

def _fingerprint(ex: dict) -> bytes:
    return hashlib.blake2b(_example_text(ex).encode("utf-8"), digest_size=8).digest()


# =============================================================================
# combined stream: whole dataset, round-robin, dedup + language filter
# =============================================================================
def combined_stream(cfg: Config):
    seen = set()
    files = [(f,s,os.path.join(cfg.data_dir,f)) for f,s in FILE_SPECS.items()
             if os.path.exists(os.path.join(cfg.data_dir,f))
             and not (s=="text" and not cfg.include_urdu_corpus_lm)]
    handles = {f: open(p,"r",encoding="utf-8") for f,s,p in files}
    specs = {f:s for f,s,p in files}
    rng = random.Random(cfg.seed)
    skip = {f: cfg.eval_holdout_per_file for f in handles}
    alive = set(handles)
    kept = ddup = dlang = 0
    while alive:
        for f in list(handles):
            if f not in alive: continue
            line = handles[f].readline()
            if not line:
                alive.discard(f); handles[f].close(); continue
            if skip[f] > 0:
                skip[f] -= 1; continue
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except Exception: continue
            for ex in normalize_record(rec, specs[f], cfg, rng):
                if not is_ur_or_en(_example_text(ex), cfg):
                    dlang += 1; continue
                if cfg.dedup:
                    fp = _fingerprint(ex)
                    if fp in seen: ddup += 1; continue
                    seen.add(fp)
                kept += 1
                yield ex


# =============================================================================
# tokenizer / model
# =============================================================================
def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    return tok

def make_tokenize_fn(tokenizer, cfg: Config):
    L = cfg.max_seq_len; eos = tokenizer.eos_token_id
    def empty(): return {"input_ids":[], "labels":[], "attention_mask":[]}
    def fn(ex):
        if ex["kind"] == "text":
            t = (ex.get("text") or "").strip()
            if not t: return empty()
            ids = tokenizer(t, add_special_tokens=False, truncation=True,
                            max_length=L-1)["input_ids"] + [eos]
            return {"input_ids":ids, "labels":list(ids), "attention_mask":[1]*len(ids)}
        msgs = ex.get("messages") or []
        if cfg.add_system_prompt and not any(m["role"]=="system" for m in msgs):
            msgs = [{"role":"system","content":cfg.system_prompt}] + msgs
        if not msgs or msgs[-1]["role"] != "assistant": return empty()
        full = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        prm  = tokenizer.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)
        full = full[:L]; n = min(len(prm), len(full))
        labels = ([IGNORE]*n + list(full[n:]))[:len(full)]
        if all(x==IGNORE for x in labels): return empty()
        return {"input_ids":full, "labels":labels, "attention_mask":[1]*len(full)}
    return fn

class PadCollator:
    def __init__(self, pad, mult=8): self.pad=pad; self.mult=mult
    def __call__(self, batch):
        import torch
        m = max(len(b["input_ids"]) for b in batch)
        m = ((m + self.mult - 1)//self.mult)*self.mult
        I=[];Lb=[];A=[]
        for b in batch:
            k = m - len(b["input_ids"])
            I.append(b["input_ids"]+[self.pad]*k)
            Lb.append(b["labels"]+[IGNORE]*k)
            A.append(b["attention_mask"]+[0]*k)
        return {"input_ids":torch.tensor(I),"labels":torch.tensor(Lb),
                "attention_mask":torch.tensor(A)}

def flash_available():
    import importlib.util
    return importlib.util.find_spec("flash_attn") is not None

def load_model(cfg: Config, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    quant = None
    if cfg.train_mode == "qlora":
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=dtype, quantization_config=quant,
        attn_implementation="flash_attention_2" if flash_available() else "sdpa",
        trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if cfg.train_mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules), bias="none", task_type="CAUSAL_LM"))
    if is_main():
        model.print_trainable_parameters()
    if cfg.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


# =============================================================================
# DATA ANALYSIS  (tokens per file, total tokens, parameters) — before training
# =============================================================================
def analyze(cfg: Config, tokenizer=None, sample_per_file: int = 3000, print_params: bool = True):
    if tokenizer is None:
        tokenizer = load_tokenizer(cfg)
    tf = make_tokenize_fn(tokenizer, cfg)

    # ---- confirm the tokenizer is UNK-free (byte-level BPE) ----------------
    probe = "یہ اردو ہے۔ This is English. 123 ؟!"
    ids = tokenizer(probe, add_special_tokens=False)["input_ids"]
    unk = tokenizer.unk_token_id
    n_unk = sum(1 for i in ids if unk is not None and i == unk)
    log("="*84)
    log("TOKENIZER CHECK")
    log(f"  vocab size            : {len(tokenizer):,}")
    log(f"  unk token             : {tokenizer.unk_token}  (byte-level BPE => no UNK)")
    log(f"  UNK in mixed probe    : {n_unk}   -> {'OK, zero UNK' if n_unk==0 else 'WARN'}")

    log("="*84)
    log("DATASET TOKEN ANALYSIS   (sampled estimate; dedup/lang-filter trim slightly)")
    log(f"  {'file':<26}{'records':>13}{'tok/rec':>9}{'examples':>15}{'tokens':>15}")
    log("  " + "-"*78)
    grand_tok = grand_ex = 0.0
    for fname, spec in FILE_SPECS.items():
        path = os.path.join(cfg.data_dir, fname)
        if not os.path.exists(path):
            log(f"  {fname:<26}{'MISSING':>13}"); continue
        if spec == "text" and not cfg.include_urdu_corpus_lm:
            continue
        rng = random.Random(cfg.seed); n = toks = exs = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if n >= sample_per_file: break
                line = line.strip()
                if not line: continue
                try: rec = json.loads(line)
                except Exception: continue
                n += 1
                for ex in normalize_record(rec, spec, cfg, rng):
                    if not is_ur_or_en(_example_text(ex), cfg): continue
                    enc = tf(ex)
                    toks += len(enc["input_ids"]); exs += 1
        if n == 0: continue
        total_rec = TOTAL_RECORDS.get(fname, n)
        tok_per_rec = toks / n
        est_tok = tok_per_rec * total_rec
        est_ex = (exs / n) * total_rec
        grand_tok += est_tok; grand_ex += est_ex
        log(f"  {fname:<26}{total_rec:>13,}{tok_per_rec:>9.0f}"
            f"{est_ex:>15,.0f}{est_tok/1e9:>13.2f}B")
    log("  " + "-"*78)
    log(f"  {'TOTAL':<26}{'':>13}{'':>9}{grand_ex:>15,.0f}{grand_tok/1e9:>13.2f}B")
    log(f"  ~= {grand_tok/1e9:.2f} billion tokens across the whole corpus")

    # ---- one-epoch / time estimate ----------------------------------------
    eff = cfg.micro_batch_size * cfg.grad_accum_steps * max(1, world_size())
    log("="*84)
    log("TRAINING PLAN")
    log(f"  GPUs (processes)      : {world_size()}")
    log(f"  effective batch       : {eff} sequences/step")
    log(f"  max_steps             : {cfg.max_steps:,}")
    tok_per_step = eff * cfg.max_seq_len
    log(f"  ~tokens/step (max)    : {tok_per_step:,}")
    if grand_ex > 0:
        steps_epoch = grand_ex / eff
        log(f"  steps for 1 full epoch: ~{steps_epoch:,.0f}  "
            f"({cfg.max_steps/steps_epoch:.2f} epochs at max_steps)")

    # ---- parameters --------------------------------------------------------
    if print_params:
        try:
            from transformers import AutoConfig
            mc = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
            h = getattr(mc, "hidden_size", 0); l = getattr(mc, "num_hidden_layers", 0)
            v = getattr(mc, "vocab_size", 0)
            log("="*84)
            log("MODEL PARAMETERS")
            log(f"  base model            : {cfg.base_model}")
            log(f"  hidden={h}  layers={l}  vocab={v:,}")
            approx_lora = 2 * cfg.lora_r * h * len(cfg.lora_target_modules) * l
            log(f"  trainable (LoRA r={cfg.lora_r}) ~ {approx_lora/1e6:.0f}M  "
                f"(full base ~7.6B, frozen)")
        except Exception as e:
            log("  (param detail unavailable:", e, ")")
    log("="*84)
    return grand_tok, grand_ex


# =============================================================================
# evaluation during training
# =============================================================================
def build_eval(cfg: Config, per_task=40):
    ev = {}; rng = random.Random(cfg.seed)
    for f, spec in FILE_SPECS.items():
        p = os.path.join(cfg.data_dir, f)
        if not os.path.exists(p): continue
        rows = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                if len(rows) >= per_task: break
                line = line.strip()
                if not line: continue
                try: rec = json.loads(line)
                except Exception: continue
                for ex in normalize_record(rec, spec, cfg, rng):
                    if is_ur_or_en(_example_text(ex), cfg): rows.append(ex)
        ev[f.replace(".jsonl","")] = rows[:per_task]
    return ev

EVAL_PROMPTS = [
 ("grammar","اس کی گرامر درست کریں۔\n\nمیں کل اسکول جاتا ہوں اور کتاب پڑھی تھی۔"),
 ("summarize","خلاصہ کریں۔\n\nپاکستان اسٹاک مارکیٹ میں آج زبردست تیزی دیکھی گئی اور انڈیکس چار سو پوائنٹس بڑھ کر بند ہوا۔"),
 ("translate_en_ur","Translate this into Urdu.\n\nEducation is the most powerful weapon to change the world."),
 ("translate_ur_en","Translate this into English.\n\nعلم حاصل کرنا ہر مرد اور عورت پر فرض ہے۔"),
 ("qa","سوال کا جواب دیں۔\n\nپاکستان کا دارالحکومت کون سا شہر ہے؟"),
 ("write","Write a detailed article for the headline:\n\nThe importance of clean drinking water."),
]

def make_callbacks(cfg: Config, tokenizer, eval_examples):
    import torch, sacrebleu
    from transformers import TrainerCallback

    @torch.no_grad()
    def eval_loss(model):
        model.eval(); tf = make_tokenize_fn(tokenizer, cfg); tl=tt=0
        dev = next(model.parameters()).device
        for rows in eval_examples.values():
            for ex in rows:
                enc = tf(ex)
                if not enc["input_ids"]: continue
                ids = torch.tensor([enc["input_ids"]], device=dev)
                lb  = torch.tensor([enc["labels"]], device=dev)
                out = model(input_ids=ids, labels=lb); k=int((lb!=IGNORE).sum())
                tl += float(out.loss)*k; tt += k
        model.train(); loss=tl/max(tt,1); return loss, math.exp(min(loss,20))

    @torch.no_grad()
    def eval_chrf(model, n=25):
        rows = eval_examples.get("translation", [])
        if not rows: return None
        model.eval(); dev = next(model.parameters()).device; H=[];R=[]
        for ex in rows[:n]:
            m0 = ex["messages"][0]
            pr = ([{"role":"system","content":cfg.system_prompt}]+[m0]) if cfg.add_system_prompt else [m0]
            ids = tokenizer.apply_chat_template(pr, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(dev)
            o = model.generate(ids, max_new_tokens=180, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            H.append(tokenizer.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip())
            R.append(ex["messages"][1]["content"])
        model.train(); return sacrebleu.corpus_chrf(H, [R]).score

    @torch.no_grad()
    def samples(model, mx=140):
        model.eval(); dev = next(model.parameters()).device
        log("── LIVE SAMPLES " + "─"*40)
        for task, p in EVAL_PROMPTS:
            msgs = ([{"role":"system","content":cfg.system_prompt}] if cfg.add_system_prompt else []) + [{"role":"user","content":p}]
            ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(dev)
            o = model.generate(ids, max_new_tokens=mx, do_sample=True, temperature=0.7, top_p=0.9,
                               repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
            log(f"  [{task}] " + tokenizer.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()[:300])
        model.train()

    class CB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs: logs["ppl"] = round(math.exp(min(logs["loss"],20)), 2)
        def on_step_end(self, args, state, control, model=None, **kw):
            if not is_main(): return
            s = state.global_step
            if cfg.sample_every and s>0 and s % cfg.sample_every == 0:
                try: samples(model)
                except Exception as e: log("sample skip:", e)
            if cfg.eval_every and s>0 and s % cfg.eval_every == 0:
                try:
                    l,pp = eval_loss(model); cf = eval_chrf(model)
                    log(f"📊 [eval @ {s}] loss={l:.4f} ppl={pp:.2f}" + (f" chrF={cf:.1f}" if cf else ""))
                except Exception as e: log("eval skip:", e)
                gc.collect(); torch.cuda.empty_cache()
    return [CB()]


# =============================================================================
# TRAIN
# =============================================================================
def train(cfg: Config):
    import torch
    from datasets import IterableDataset, Features, Value
    from transformers import TrainingArguments, Trainer, set_seed
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    tokenizer = load_tokenizer(cfg)
    if is_main():
        analyze(cfg, tokenizer)              # <-- data + params printed BEFORE training

    model = load_model(cfg, tokenizer)

    features = Features({"kind":Value("string"),
        "messages":[{"role":Value("string"),"content":Value("string")}], "text":Value("string")})
    ds = IterableDataset.from_generator(lambda: combined_stream(cfg), features=features)
    ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)
    ds = ds.map(make_tokenize_fn(tokenizer, cfg), remove_columns=["kind","messages","text"])
    ds = ds.filter(lambda e: len(e["input_ids"]) > 0)
    collator = PadCollator(tokenizer.pad_token_id)

    eval_examples = build_eval(cfg) if is_main() else {}

    args = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate, lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio, weight_decay=cfg.weight_decay, max_grad_norm=cfg.grad_clip,
        bf16=cfg.bf16, fp16=not cfg.bf16,
        logging_steps=cfg.logging_steps, save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=cfg.num_workers, dataloader_pin_memory=True,
        ddp_find_unused_parameters=False, remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        seed=cfg.seed)

    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=collator,
                      callbacks=make_callbacks(cfg, tokenizer, eval_examples) if is_main() else None)

    ck = None
    if cfg.resume:
        cks = sorted(Path(cfg.out_dir).glob("checkpoint-*"),
                     key=lambda p:int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
        ck = str(cks[-1]) if cks else None
    log("resume from:", ck or "scratch")
    trainer.train(resume_from_checkpoint=ck)

    if is_main():
        final = os.path.join(cfg.out_dir, "final")
        trainer.save_model(final); tokenizer.save_pretrained(final)
        with open(os.path.join(final, "grammora_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
        log("✅ saved adapter ->", final)
        log("next: python train_grammora.py merge --out_dir", cfg.out_dir)


# =============================================================================
# MERGE + CHAT
# =============================================================================
def merge(cfg: Config):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    final = os.path.join(cfg.out_dir, "final")
    merged = os.path.join(cfg.out_dir, "merged")
    tok = load_tokenizer(cfg)
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.bfloat16, trust_remote_code=True)
    m = PeftModel.from_pretrained(base, final).merge_and_unload()
    m.config.use_cache = True
    m.save_pretrained(merged, safe_serialization=True); tok.save_pretrained(merged)
    print("✅ standalone model ->", merged)
    print("serve: python -m vllm.entrypoints.openai.api_server --model", merged)

def chat(cfg: Config, prompt: str):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    merged = os.path.join(cfg.out_dir, "merged")
    final  = os.path.join(cfg.out_dir, "final")
    tok = load_tokenizer(cfg)
    if os.path.exists(merged):
        model = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, final)
    model.eval()
    msgs = ([{"role":"system","content":cfg.system_prompt}] if cfg.add_system_prompt else []) + [{"role":"user","content":prompt}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    o = model.generate(ids, max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.9,
                       repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
    print("Q:", prompt)
    print("A:", tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip())


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["analyze","train","merge","chat"])
    ap.add_argument("--data_dir"); ap.add_argument("--out_dir"); ap.add_argument("--base_model")
    ap.add_argument("--train_mode", choices=["lora","qlora"])
    ap.add_argument("--max_steps", type=int); ap.add_argument("--micro_batch_size", type=int)
    ap.add_argument("--grad_accum_steps", type=int); ap.add_argument("--max_seq_len", type=int)
    ap.add_argument("--learning_rate", type=float); ap.add_argument("--prompt")
    a = ap.parse_args()
    cfg = Config()
    for k in ("data_dir","out_dir","base_model","train_mode","max_steps",
              "micro_batch_size","grad_accum_steps","max_seq_len","learning_rate"):
        v = getattr(a, k, None)
        if v is not None: setattr(cfg, k, v)
    if a.mode == "analyze": analyze(cfg)
    elif a.mode == "train": train(cfg)
    elif a.mode == "merge": merge(cfg)
    elif a.mode == "chat":
        if not a.prompt: raise SystemExit("--prompt required")
        chat(cfg, a.prompt)

if __name__ == "__main__":
    main()
