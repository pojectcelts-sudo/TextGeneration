#!/usr/bin/env python
# =============================================================================
#  GRAMMORA GPT  —  Urdu + English instruction-following, decoder-only LLM
#  Trained FROM SCRATCH on 100% of your JSONL datasets.
#
#  What this gives you (vs. your old paraphrasing code):
#    - GPT / decoder-only architecture (not encoder-decoder seq2seq)
#    - INSTRUCTION-BASED: one model handles grammar, paraphrasing, QA,
#      summarization, text-generation, translation AND raw language modeling,
#      selected by the user/assistant chat template.
#    - Uses EVERY record of EVERY file (no 20% / 30% sampling, no quality
#      filter that throws pairs away). Streaming loader => the 84 GB QA file
#      is never loaded into RAM.
#    - Same SentencePiece BPE trick you trust: byte_fallback=True guarantees
#      ZERO <unk> tokens, ever.
#    - bf16, FlashAttention (SDPA), gradient accumulation, cosine schedule,
#      full checkpoint/resume, and native multi-GPU (your 2x RTX PRO 6000).
#
#  ------------------------------------------------------------------ RUN IT
#  Single GPU (or notebook):     python grammora_gpt_train.py
#  Both GPUs (recommended):      torchrun --nproc_per_node=2 grammora_gpt_train.py
#
#  Everything is driven by the CONFIG block below — nothing else needs editing.
# =============================================================================

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# sentencepiece is imported lazily inside the tokenizer helpers so the file can
# at least be inspected without it installed.


# =============================================================================
# 1. CONFIG — the only block you normally touch
# =============================================================================

@dataclass
class Config:
    # ---- where your data lives --------------------------------------------
    # Point this at the folder that holds grammar.jsonl, paraphrasing.jsonl,
    # question_answering.jsonl, summarization.jsonl, text_generation.jsonl,
    # translation.jsonl, urdu_corpus.jsonl.
    #
    # On your Windows machine that is:
    #   D:\Kamran_Raza_Text_Generation_1_August_2026\jsonl_datasets
    # On the rented Linux GPU box, upload the folder and set it here, e.g.:
    #   /workspace/jsonl_datasets
    data_dir: str = os.environ.get("GRAMMORA_DATA_DIR", "/workspace/jsonl_datasets")
    out_dir: str = os.environ.get("GRAMMORA_OUT_DIR", "/workspace/grammora_out")

    # Which files to use and how to read each one.
    #   "chat" -> record has a "messages" list (user/assistant) => instruction data
    #   "text" -> record has a "text" field                     => raw LM data
    # Every listed file is used 100% end-to-end. Remove a line to skip a file.
    files: Dict[str, str] = field(default_factory=lambda: {
        "grammar.jsonl":            "chat",
        "paraphrasing.jsonl":       "chat",
        "summarization.jsonl":      "chat",
        "translation.jsonl":        "chat",
        "text_generation.jsonl":    "chat",
        "question_answering.jsonl": "chat",
        "urdu_corpus.jsonl":        "text",
    })
    # Optional per-file sampling weight for INTERLEAVING order only. This never
    # drops data — every record is still seen once per epoch. It only controls
    # how often each file is drawn so the giant QA file doesn't monopolise the
    # early part of an epoch. Higher weight = drawn more often. Set all equal
    # for pure round-robin. These are gentle defaults, tune freely.
    file_weights: Dict[str, float] = field(default_factory=lambda: {
        "grammar.jsonl":            1.0,
        "paraphrasing.jsonl":       1.0,
        "summarization.jsonl":      1.0,
        "translation.jsonl":        1.0,
        "text_generation.jsonl":    2.0,
        "question_answering.jsonl": 3.0,
        "urdu_corpus.jsonl":        1.0,
    })

    # ---- tokenizer (SentencePiece BPE, zero-UNK via byte_fallback) ---------
    vocab_size: int = 32000
    tokenizer_prefix: str = "grammora_bpe"      # -> grammora_bpe.model / .vocab
    tokenizer_sample_lines_per_file: int = 1_500_000   # cap ONLY for training
    #        the tokenizer (not the model). The model still trains on all data.

    # ---- model (decoder-only GPT). ~440M params at these defaults. ---------
    # Scale up n_embd / n_layer if you want a bigger model — you have the VRAM.
    block_size: int = 1024        # context length (tokens)
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    mlp_ratio: float = 8 / 3      # SwiGLU expansion (keeps param count ~4x)
    dropout: float = 0.0          # 0.0 is best for large-data pretraining
    rope_theta: float = 10000.0

    # ---- optimization ------------------------------------------------------
    micro_batch_size: int = 24    # per-GPU sequences per forward pass
    grad_accum_steps: int = 8     # effective batch = micro * accum * n_gpus
    max_steps: int = 200_000      # optimizer steps (raise for more passes)
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # ---- runtime -----------------------------------------------------------
    dtype: str = "bfloat16"       # "bfloat16" (Ada/Blackwell) or "float16"
    compile_model: bool = True    # torch.compile for speed (PyTorch 2.x)
    num_workers: int = 4          # dataloader workers per process
    shuffle_buffer: int = 50_000  # streaming shuffle buffer (records)
    seed: int = 3407

    # ---- checkpointing / logging ------------------------------------------
    log_every: int = 20
    eval_every: int = 2000
    save_every: int = 2000
    keep_last_k: int = 3
    resume: bool = True

    # ---- chat template special tokens -------------------------------------
    # Registered as user_defined_symbols so each is ONE token id.
    tok_system: str = "<|system|>"
    tok_user: str = "<|user|>"
    tok_assistant: str = "<|assistant|>"
    tok_eot: str = "<|eot|>"      # end-of-turn / end-of-sample

    default_system: str = "آپ Grammora ہیں، ایک مددگار اردو اور انگریزی معاون۔"  # "You are Grammora, a helpful Urdu/English assistant."


CFG = Config()
IGNORE_INDEX = -100  # label value that is skipped by the loss


# =============================================================================
# 2. DISTRIBUTED / DEVICE SETUP
# =============================================================================

def setup_distributed() -> Tuple[int, int, int, torch.device, bool]:
    """Returns (rank, local_rank, world_size, device, is_ddp)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return rank, local_rank, world_size, device, True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 0, 1, device, False


def is_master(rank: int) -> bool:
    return rank == 0


def log(rank: int, *args):
    if is_master(rank):
        print(*args, flush=True)


# =============================================================================
# 3. TOKENIZER  (SentencePiece BPE, byte_fallback => zero UNK)
# =============================================================================

class Tokenizer:
    """Thin wrapper over a SentencePiece BPE model with chat special tokens."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sp = None
        # ids filled after load()
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.system_id = self.user_id = self.assistant_id = self.eot_id = None

    # ---- training ---------------------------------------------------------
    def train(self, files: Dict[str, str]):
        import sentencepiece as spm
        cfg = self.cfg
        corpus = os.path.join(cfg.out_dir, "_tok_corpus.txt")
        os.makedirs(cfg.out_dir, exist_ok=True)

        print(f"[tok] building tokenizer corpus (up to "
              f"{cfg.tokenizer_sample_lines_per_file:,} lines/file)...")
        n_written = 0
        with open(corpus, "w", encoding="utf-8") as out:
            for fname, kind in files.items():
                path = os.path.join(cfg.data_dir, fname)
                if not os.path.exists(path):
                    print(f"[tok]   WARN missing {path}, skipping")
                    continue
                written = 0
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if written >= cfg.tokenizer_sample_lines_per_file:
                            break
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        for txt in _iter_texts(rec, kind):
                            txt = txt.strip()
                            if txt:
                                out.write(txt.replace("\n", " ") + "\n")
                        written += 1
                        n_written += 1
                print(f"[tok]   {fname}: sampled {written:,} records")
        print(f"[tok] corpus ready ({n_written:,} records). training SPM BPE...")

        prefix = os.path.join(cfg.out_dir, cfg.tokenizer_prefix)
        spm.SentencePieceTrainer.train(
            input=corpus,
            model_prefix=prefix,
            vocab_size=cfg.vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            byte_fallback=True,          # <-- guarantees ZERO <unk>
            pad_id=0, bos_id=1, eos_id=2, unk_id=3,
            pad_piece="<pad>", bos_piece="<s>", eos_piece="</s>", unk_piece="<unk>",
            user_defined_symbols=[cfg.tok_system, cfg.tok_user,
                                  cfg.tok_assistant, cfg.tok_eot],
            normalization_rule_name="identity",   # keep Urdu untouched
            split_by_whitespace=True,
            split_by_unicode_script=False,
            remove_extra_whitespaces=False,
            num_threads=os.cpu_count() or 8,
            train_extremely_large_corpus=True,
            input_sentence_size=20_000_000,
            shuffle_input_sentence=True,
            max_sentence_length=16384,
        )
        try:
            os.remove(corpus)
        except OSError:
            pass
        print(f"[tok] trained -> {prefix}.model")

    # ---- load / use -------------------------------------------------------
    def load(self):
        import sentencepiece as spm
        prefix = os.path.join(self.cfg.out_dir, self.cfg.tokenizer_prefix)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{prefix}.model")
        c = self.cfg
        self.system_id = self.sp.piece_to_id(c.tok_system)
        self.user_id = self.sp.piece_to_id(c.tok_user)
        self.assistant_id = self.sp.piece_to_id(c.tok_assistant)
        self.eot_id = self.sp.piece_to_id(c.tok_eot)
        assert self.sp.get_piece_size() == c.vocab_size, \
            f"vocab mismatch {self.sp.get_piece_size()} vs {c.vocab_size}"
        return self

    @property
    def exists(self) -> bool:
        prefix = os.path.join(self.cfg.out_dir, self.cfg.tokenizer_prefix)
        return os.path.exists(f"{prefix}.model")

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)


def _iter_texts(rec: dict, kind: str) -> Iterator[str]:
    """Yield the raw text fields of a record (used only for tokenizer corpus)."""
    if kind == "text":
        t = rec.get("text")
        if t:
            yield t
    else:  # chat
        for m in rec.get("messages", []):
            c = m.get("content")
            if c:
                yield c


# =============================================================================
# 4. EXAMPLE ENCODING  (chat template + loss masking)
# =============================================================================

def encode_example(rec: dict, kind: str, tok: Tokenizer,
                   cfg: Config) -> Optional[Tuple[List[int], List[int]]]:
    """
    Turn one raw record into (token_ids, labels).

    - For CHAT records we lay out:
          <s> <|system|> sys <|eot|>
              <|user|> u1 <|eot|> <|assistant|> a1 <|eot|>  (repeated per turn)
      and mask EVERYTHING except the assistant content + its <|eot|>. That is
      what makes the model INSTRUCTION-following: it only learns to produce the
      answer, never to parrot the prompt.
    - For TEXT records (urdu_corpus) it is plain language modeling: every token
      is a label (the model learns to continue Urdu text).
    """
    ids: List[int] = [tok.bos_id]
    labels: List[int] = [IGNORE_INDEX]

    def add(token_ids: List[int], supervised: bool):
        ids.extend(token_ids)
        labels.extend(token_ids if supervised else [IGNORE_INDEX] * len(token_ids))

    if kind == "text":
        body = (rec.get("text") or "").strip()
        if not body:
            return None
        add(tok.encode(body), supervised=True)
        add([tok.eos_id], supervised=True)
    else:
        messages = rec.get("messages") or []
        if not messages:
            return None
        # optional system preamble (adds task-agnostic persona; masked)
        add([tok.system_id] + tok.encode(cfg.default_system) + [tok.eot_id], False)
        saw_assistant = False
        for m in messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            enc = tok.encode(content)
            if role == "assistant":
                add([tok.assistant_id], supervised=False)   # the marker itself: no loss
                add(enc + [tok.eot_id], supervised=True)     # answer + end: LEARN this
                saw_assistant = True
            elif role == "system":
                add([tok.system_id] + enc + [tok.eot_id], supervised=False)
            else:  # user / anything else -> prompt, masked
                add([tok.user_id] + enc + [tok.eot_id], supervised=False)
        if not saw_assistant:
            return None

    return ids, labels


# =============================================================================
# 5. STREAMING DATASET  (100% of data, packed to fixed blocks, no padding waste)
# =============================================================================

class PackedStream(IterableDataset):
    """
    Streams every record of every file, interleaved by weight, tokenizes with
    loss-masking, and PACKS the token/label streams into fixed-length blocks of
    `block_size`. Packing means ~zero wasted compute on padding, i.e. maximum
    advantage from your data.

    Sharding: each (ddp_rank, dataloader_worker) reads a disjoint slice of the
    record stream, so nothing is trained on twice within an epoch.
    """

    def __init__(self, cfg: Config, tok: Tokenizer, rank: int, world_size: int,
                 seed_offset: int = 0):
        super().__init__()
        self.cfg = cfg
        self.tok = tok
        self.rank = rank
        self.world_size = world_size
        self.seed_offset = seed_offset

    # -- weighted round-robin over the open files ---------------------------
    def _record_iter(self, worker_id: int, num_workers: int) -> Iterator[Tuple[dict, str]]:
        cfg = self.cfg
        files = [(f, k) for f, k in cfg.files.items()
                 if os.path.exists(os.path.join(cfg.data_dir, f))]
        handles = {f: open(os.path.join(cfg.data_dir, f), "r", encoding="utf-8")
                   for f, _ in files}
        kinds = {f: k for f, k in files}
        weights = {f: cfg.file_weights.get(f, 1.0) for f, _ in files}
        # fractional accumulator for weighted draws
        credit = {f: 0.0 for f, _ in files}
        alive = set(handles)

        # global stride so rank*worker slices are disjoint
        stride = self.world_size * num_workers
        offset = self.rank * num_workers + worker_id
        counter = 0

        rng = random.Random(cfg.seed + self.seed_offset + offset)
        order = list(handles.keys())

        while alive:
            rng.shuffle(order)
            progressed = False
            for f in order:
                if f not in alive:
                    continue
                credit[f] += weights[f]
                # draw floor(credit) records from this file this cycle
                draws = int(credit[f])
                credit[f] -= draws
                for _ in range(max(1, draws)):
                    line = handles[f].readline()
                    if not line:
                        alive.discard(f)
                        handles[f].close()
                        break
                    progressed = True
                    counter += 1
                    if (counter % stride) != offset:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    yield rec, kinds[f]
            if not progressed:
                break

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        cfg = self.cfg
        B = cfg.block_size + 1  # +1 so we can shift for next-token targets

        buf_ids: List[int] = []
        buf_lbl: List[int] = []
        shuffle_buf: List[Tuple[List[int], List[int]]] = []
        rng = random.Random(cfg.seed + 12345 + worker_id + self.rank * 97)

        def emit(block_ids, block_lbl):
            x = torch.tensor(block_ids[:-1], dtype=torch.long)
            y = torch.tensor(block_lbl[1:], dtype=torch.long)  # shifted labels
            return x, y

        for rec, kind in self._record_iter(worker_id, num_workers):
            enc = encode_example(rec, kind, self.tok, cfg)
            if enc is None:
                continue
            ids, lbl = enc
            buf_ids.extend(ids)
            buf_lbl.extend(lbl)
            # carve full blocks out of the running buffer
            while len(buf_ids) >= B:
                block_ids = buf_ids[:B]
                block_lbl = buf_lbl[:B]
                buf_ids = buf_ids[B:]
                buf_lbl = buf_lbl[B:]
                shuffle_buf.append((block_ids, block_lbl))
                if len(shuffle_buf) >= cfg.shuffle_buffer:
                    j = rng.randrange(len(shuffle_buf))
                    bi, bl = shuffle_buf.pop(j)
                    yield emit(bi, bl)
        # drain
        rng.shuffle(shuffle_buf)
        for bi, bl in shuffle_buf:
            yield emit(bi, bl)


# =============================================================================
# 6. MODEL  (decoder-only GPT: RMSNorm + RoPE + SwiGLU + SDPA/flash)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)          # (T, head_dim/2)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)   # (1,1,T,D/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = xr1
    out[..., 1::2] = xr2
    return out


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.n_embd * cfg.mlp_ratio)
        hidden = 64 * ((hidden + 63) // 64)   # round to multiple of 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GrammoraGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight   # weight tying

        self._rope = {}   # cached per (device, dtype)
        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_cache(self, T, device, dtype):
        key = (device, dtype)
        cos, sin = self._rope.get(key, (None, None))
        if cos is None or cos.size(0) < T:
            cos, sin = build_rope_cache(max(T, self.cfg.block_size),
                                        self.cfg.n_embd // self.cfg.n_head,
                                        self.cfg.rope_theta, device, dtype)
            self._rope[key] = (cos, sin)
        return cos, sin

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope_cache(T, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=IGNORE_INDEX)
        return logits, loss

    def num_params(self) -> int:
        # subtract tied head so we don't double-count
        n = sum(p.numel() for p in self.parameters())
        return n - self.lm_head.weight.numel()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, tok: Tokenizer,
                 temperature=0.8, top_k=50, top_p=0.95):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            if top_p and top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                mask = cum - sp > top_p
                sp[mask] = 0.0
                sp = sp / sp.sum(dim=-1, keepdim=True)
                nxt = si.gather(-1, torch.multinomial(sp, 1))
            else:
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() == tok.eot_id or nxt.item() == tok.eos_id:
                break
        return idx


# =============================================================================
# 7. LR SCHEDULE
# =============================================================================

def lr_at(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# =============================================================================
# 8. CHECKPOINTING
# =============================================================================

def save_ckpt(cfg, model, opt, step, rank):
    if not is_master(rank):
        return
    os.makedirs(cfg.out_dir, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    # unwrap torch.compile
    raw = getattr(raw, "_orig_mod", raw)
    path = os.path.join(cfg.out_dir, f"ckpt_{step:07d}.pt")
    torch.save({
        "model": raw.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "config": asdict(cfg),
    }, path)
    latest = os.path.join(cfg.out_dir, "ckpt_latest.pt")
    torch.save({"path": os.path.basename(path), "step": step}, latest)
    # prune old
    cks = sorted(glob.glob(os.path.join(cfg.out_dir, "ckpt_[0-9]*.pt")))
    for old in cks[:-cfg.keep_last_k]:
        try:
            os.remove(old)
        except OSError:
            pass
    print(f"[ckpt] saved {path}", flush=True)


def load_latest(cfg, model, opt, device):
    latest = os.path.join(cfg.out_dir, "ckpt_latest.pt")
    if not (cfg.resume and os.path.exists(latest)):
        return 0
    meta = torch.load(latest, map_location="cpu")
    path = os.path.join(cfg.out_dir, meta["path"])
    ck = torch.load(path, map_location=device)
    raw = getattr(model, "_orig_mod", model)
    raw.load_state_dict(ck["model"])
    if opt is not None and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
    print(f"[ckpt] resumed from {path} @ step {ck['step']}", flush=True)
    return ck["step"]


# =============================================================================
# 9. TRAIN
# =============================================================================

def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    return torch.optim.AdamW(groups, lr=cfg.learning_rate,
                             betas=(cfg.beta1, cfg.beta2),
                             **({"fused": True} if fused and torch.cuda.is_available() else {}))


def train(cfg: Config):
    rank, local_rank, world_size, device, is_ddp = setup_distributed()
    torch.manual_seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- tokenizer --------------------------------------------------------
    tok = Tokenizer(cfg)
    if is_master(rank) and not tok.exists:
        tok.train(cfg.files)
    if is_ddp:
        import torch.distributed as dist
        dist.barrier()
    tok.load()
    log(rank, f"[tok] vocab={tok.sp.get_piece_size()} "
              f"specials: user={tok.user_id} asst={tok.assistant_id} eot={tok.eot_id}")

    # ---- model ------------------------------------------------------------
    model = GrammoraGPT(cfg).to(device)
    log(rank, f"[model] params ~ {model.num_params()/1e6:.1f}M | "
              f"layers={cfg.n_layer} d={cfg.n_embd} heads={cfg.n_head} ctx={cfg.block_size}")

    opt = build_optimizer(model, cfg)
    start_step = load_latest(cfg, model, opt, device)

    if cfg.compile_model:
        model = torch.compile(model)
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    # ---- data -------------------------------------------------------------
    ds = PackedStream(cfg, tok, rank, world_size, seed_offset=start_step)
    loader = DataLoader(ds, batch_size=cfg.micro_batch_size,
                        num_workers=cfg.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=cfg.num_workers > 0,
                        prefetch_factor=4 if cfg.num_workers > 0 else None)
    data_iter = iter(loader)

    amp_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.dtype == "float16")

    eff_batch = cfg.micro_batch_size * cfg.grad_accum_steps * world_size
    log(rank, f"[train] start@{start_step} effective_batch={eff_batch} seqs "
              f"(~{eff_batch*cfg.block_size:,} tokens/step) world_size={world_size}")

    model.train()
    t0 = time.time()
    running_loss = 0.0
    tokens_seen = 0

    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(cfg.grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # only sync grads on the last micro-step under DDP
            sync = (micro == cfg.grad_accum_steps - 1)
            ctx = model.no_sync() if (is_ddp and not sync) else _null_ctx()
            with ctx:
                with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                    enabled=torch.cuda.is_available()):
                    _, loss = model(x, y)
                    loss = loss / cfg.grad_accum_steps
                scaler.scale(loss).backward()
            loss_accum += loss.item()
            tokens_seen += x.numel()

        if cfg.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        running_loss += loss_accum

        if step % cfg.log_every == 0 and step > start_step:
            dt = time.time() - t0
            avg = running_loss / cfg.log_every
            tps = tokens_seen / dt
            log(rank, f"step {step:>7} | loss {avg:.4f} | ppl {math.exp(min(avg,20)):.1f} "
                      f"| lr {lr:.2e} | {tps/1e3:.0f}k tok/s | {dt:.1f}s")
            running_loss = 0.0
            tokens_seen = 0
            t0 = time.time()

        if step % cfg.save_every == 0 and step > start_step:
            save_ckpt(cfg, model, opt, step, rank)

        if step % cfg.eval_every == 0 and step > start_step and is_master(rank):
            _quick_sample(cfg, model, tok, device)
            model.train()

    save_ckpt(cfg, model, opt, cfg.max_steps, rank)
    log(rank, "[train] done.")
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@torch.no_grad()
def _quick_sample(cfg, model, tok, device):
    raw = model.module if hasattr(model, "module") else model
    raw = getattr(raw, "_orig_mod", raw)
    prompt = "اس کی گرامر درست کریں۔\n\nمیں اسکول جاتا ہوں کل۔"
    ids = [tok.bos_id, tok.system_id] + tok.encode(cfg.default_system) + [tok.eot_id]
    ids += [tok.user_id] + tok.encode(prompt) + [tok.eot_id, tok.assistant_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = raw.generate(x, max_new_tokens=120, tok=tok)
    gen = tok.decode(out[0, len(ids):].tolist())
    print(f"[sample] Q: {prompt}\n[sample] A: {gen}\n", flush=True)


# =============================================================================
# 10. INFERENCE (standalone, after training)
# =============================================================================

@torch.no_grad()
def chat(cfg: Config, prompt: str, system: Optional[str] = None,
         max_new_tokens: int = 256, temperature: float = 0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(cfg).load()
    model = GrammoraGPT(cfg).to(device)
    load_latest(cfg, model, None, device)
    model.eval()
    system = system or cfg.default_system
    ids = [tok.bos_id, tok.system_id] + tok.encode(system) + [tok.eot_id]
    ids += [tok.user_id] + tok.encode(prompt) + [tok.eot_id, tok.assistant_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens, tok, temperature=temperature)
    return tok.decode(out[0, len(ids):].tolist())


# =============================================================================
# 11. ENTRY POINT
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Grammora GPT — from-scratch Urdu/English chat LLM")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--mode", choices=["train", "tokenizer", "chat"], default="train")
    ap.add_argument("--prompt", default=None)
    # a few convenient overrides
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--micro_batch_size", type=int, default=None)
    ap.add_argument("--grad_accum_steps", type=int, default=None)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--no_compile", action="store_true")
    args = ap.parse_args()

    cfg = CFG
    if args.data_dir:            cfg.data_dir = args.data_dir
    if args.out_dir:             cfg.out_dir = args.out_dir
    if args.max_steps:           cfg.max_steps = args.max_steps
    if args.micro_batch_size:    cfg.micro_batch_size = args.micro_batch_size
    if args.grad_accum_steps:    cfg.grad_accum_steps = args.grad_accum_steps
    if args.block_size:          cfg.block_size = args.block_size
    if args.no_compile:          cfg.compile_model = False

    if args.mode == "tokenizer":
        Tokenizer(cfg).train(cfg.files)
    elif args.mode == "chat":
        assert args.prompt, "--prompt required for chat mode"
        print(chat(cfg, args.prompt))
    else:
        train(cfg)


if __name__ == "__main__":
    main()
