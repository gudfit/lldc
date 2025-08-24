# lldc/scripts/execution/stages.py

from __future__ import annotations
import hydra
import numpy as np
import matplotlib.pyplot as plt
import json, re, math, os, subprocess, tempfile, time, random, inspect, gc, logging, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, DefaultDict, Iterable
from collections import defaultdict, Counter
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    GPT2LMHeadModel,
    BertForMaskedLM,
    RobertaForMaskedLM,
)
from transformers.utils import logging as hf_logging
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from contextlib import nullcontext
from sentence_transformers import SentenceTransformer
from datasketch import MinHash, MinHashLSH
from scipy.stats import pearsonr
from statsmodels.stats.power import TTestIndPower

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.utils import wandb_log
from lldc.data.bootstrap import ensure_data
from lldc.data.probes import generate_factual_probes, ProbeSpec
from lldc.utils.determinism import set_determinism
from lldc.models.specialization import structured_prune
from lldc.utils.hydra_utils import resolve_auto
from lldc.compression.predictive_masking import pll_surprisal_scores
from lldc.compression.masking_policies import choose_mask
from lldc.compression.payload_codec import (
    arithmetic as ac,
    bitmask as bm,
    rle_elias as rle,
)
from lldc.compression.reconstruction import reconstruct_mlm_text
from lldc.compression.token_alignment import (
    kept_char_spans_from_offsets,
    select_oracle_token_ids_from_spans,
)
from lldc.compressors.pm.token_coder import encode_kept_stream_with_oracle
from lldc.models.vq.vq_trainer import (
    train_vq_joint,
    encode_indices,
    train_index_lm,
    cross_entropy_bits_index_stream,
    _IndexGRULM,
    _build_vq_wrapper,
    encode_index_stream_ac,
)

hf_logging.set_verbosity_error()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def _cfg_get(cfg: Any, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def run_prune_and_recover(cfg: Any) -> None:
    def _cfg_get_first(cfg: Any, keys: List[str], default: Any = None) -> Any:
        for k in keys:
            val = _cfg_get(cfg, k)
            if val is not None:
                return val
        return default

    def _is_auto(v: Any) -> bool:
        return isinstance(v, str) and v.lower() in ("auto", "automatic")

    def _cfg_arch(model_name_or_cfg: Any) -> str:
        name = (
            str(
                getattr(model_name_or_cfg.model, "pretrained_name", model_name_or_cfg)
            ).lower()
            if hasattr(model_name_or_cfg, "model")
            else str(model_name_or_cfg).lower()
        )
        ar_markers = ("gpt", "llama", "mistral", "opt", "phi", "qwen", "glm", "mpt")
        return "ar" if any(m in name for m in ar_markers) else "mlm"

    def _is_valid_hf_dir(p: Path) -> bool:
        return p.is_dir() and (p / "config.json").is_file() and (p / "READY").is_file()

    def _make_training_args(**kwargs) -> TrainingArguments:
        bs = kwargs.get("batch_size", 4)
        return TrainingArguments(
            output_dir=str(kwargs["output_dir"]),
            num_train_epochs=float(kwargs.get("epochs", 3.0)),
            learning_rate=float(kwargs.get("lr", 5e-5)),
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            warmup_ratio=float(kwargs.get("warmup_ratio", 0.1)),
            weight_decay=0.01,
            lr_scheduler_type=str(kwargs.get("scheduler", "linear")),
            logging_dir=str(Path(kwargs["output_dir"]) / "logs"),
            logging_steps=50,
            save_strategy="epoch",
            report_to="none",
            fp16=torch.cuda.is_available(),
            optim="adamw_torch",
            seed=kwargs.get("seed", 13),
            data_seed=kwargs.get("seed", 13),
            use_cpu=not torch.cuda.is_available(),
        )

    def _resolve_recovery_hparams(
        cfg, prune_level, arch, n_train_examples, per_device_batch_size
    ) -> Dict:
        base = {
            "epochs": 3,
            "lr": 5e-5,
            "scheduler": "linear",
            "warmup_ratio": 0.1,
            "batch_size": 2,
        }
        steps_per_epoch = n_train_examples // per_device_batch_size
        if steps_per_epoch < 50:
            base["lr"] = 1e-4
        if prune_level > 0.6:
            base["epochs"] = 5
            base["lr"] = 8e-5
        if arch == "ar":
            base["lr"] *= 0.5
        hparams = _cfg_get(cfg, "e2a.recovery.hparams", {})
        for k, v in hparams.items():
            if k in base and v is not None:
                base[k] = v
        return base

    def _update_config_after_pruning(model, model_name, arch, log) -> None:
        if arch == "mlm":
            if isinstance(model, BertForMaskedLM):
                n_h = model.bert.config.num_attention_heads
                d_ff = model.bert.config.intermediate_size
            else:
                n_h, d_ff = (
                    model.roberta.config.num_attention_heads,
                    model.roberta.config.intermediate_size,
                )
        else:
            n_h, d_ff = model.config.n_head, model.config.n_inner
        log.info(f"  > Post-pruning config: n_head={n_h}, d_ff={d_ff}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    log, paths = setup_logging(), Paths().ensure()
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_determinism(getattr(cfg, "seed", 13))

    model_name, arch = cfg.model.pretrained_name, _cfg_arch(model_name)
    level, seed = float(getattr(cfg, "prune_level", 0.0)), getattr(cfg, "seed", None)
    exp_name = str(_cfg_get(cfg, "experiment.name", "default"))
    outdir = paths.checkpoints / exp_name / f"{model_name}_pruned_{level}" + (
        f"_seed{seed}" if seed is not None else ""
    )

    if (
        bool(
            _cfg_get_first(
                cfg, ["recovery.skip_if_ready", "e2a.recovery.skip_if_ready"], True
            )
        )
        and not bool(
            _cfg_get_first(cfg, ["recovery.force", "e2a.recovery.force"], False)
        )
        and _is_valid_hf_dir(outdir)
    ):
        log.info(f"[prune] Checkpoint already READY — skipping retrain: {outdir}")
        return

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForMaskedLM.from_pretrained(model_name)
        if arch == "mlm"
        else AutoModelForCausalLM.from_pretrained(model_name)
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    drop_heads = bool(_cfg_get(cfg, "pruning.structured.drop_attention_heads", True))
    drop_ffn = bool(_cfg_get(cfg, "pruning.structured.drop_ffn_channels", True))
    dropped = structured_prune(
        model, level=level, drop_heads=drop_heads, drop_ffn=drop_ffn
    )
    log.info(f"[prune] {model_name} @ level={level} — dropped={dropped}")
    _update_config_after_pruning(model, model_name, arch, log)

    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    ds = ds.map(
        lambda b: tok(
            b[cfg.data.processing.text_field],
            truncation=True,
            max_length=cfg.data.processing.max_length,
        ),
        batched=True,
        remove_columns=[cfg.data.processing.text_field],
    )
    train_split = ds["train"].filter(lambda ex: len(ex["input_ids"]) > 0)
    if not train_split:
        raise RuntimeError("Empty training dataset for recovery.")

    train_dataset = train_split.select(range(min(4000, len(train_split))))
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=(arch == "mlm"))
    resolved = _resolve_recovery_hparams(
        cfg=cfg,
        prune_level=level,
        arch=arch,
        n_train_examples=len(train_dataset),
        per_device_batch_size=2,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "recovery_hparams.json").write_text(json.dumps(resolved, indent=2))
    args = _make_training_args(
        output_dir=str(outdir),
        epochs=resolved["epochs"],
        lr=resolved["lr"],
        scheduler=resolved["scheduler"],
        warmup_ratio=resolved["warmup_ratio"],
    )

    Trainer(
        model=model, args=args, data_collator=collator, train_dataset=train_dataset
    ).train()
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    (outdir / "READY").write_text("ok\n")
    log.info(f"[prune] Saved pruned+recovered checkpoint — {outdir}")


def run_stage1_specialise(cfg: Any) -> None:
    def _cfg_arch(cfg_obj: Any) -> str:
        name = str(getattr(cfg_obj.model, "pretrained_name", "")).lower()
        ar_markers = ("gpt", "llama", "mistral", "opt", "phi", "qwen", "glm", "mpt")
        return "ar" if any(m in name for m in ar_markers) else "mlm"

    def _is_valid_hf_dir(p: Path) -> bool:
        return p.is_dir() and (p / "config.json").is_file() and (p / "READY").is_file()

    def _auto_or(v: Any, default: Any) -> Any:
        return (
            default
            if (v is None or (isinstance(v, str) and v.lower() == "auto"))
            else v
        )

    def _update_config_after_pruning(model, model_name, arch, log) -> None:
        if arch == "mlm":
            if isinstance(model, BertForMaskedLM):
                n_h = model.bert.config.num_attention_heads
                d_ff = model.bert.config.intermediate_size
            else:
                n_h, d_ff = (
                    model.roberta.config.num_attention_heads,
                    model.roberta.config.intermediate_size,
                )
        else:
            n_h, d_ff = model.config.n_head, model.config.n_inner
        log.info(f"  > Post-pruning config: n_head={n_h}, d_ff={d_ff}")

    @dataclass
    class _AdaptiveMaskCollator:
        tok: Any
        prefer_ids: torch.Tensor
        p_high: float
        p_low: float
        special_ids: List[int]
        max_len: int

        def __call__(self, batch: List[Dict]) -> Dict:
            ids = [torch.tensor(ex["input_ids"]) for ex in batch]
            ids = torch.nn.utils.rnn.pad_sequence(
                ids, batch_first=True, padding_value=self.tok.pad_token_id
            )
            labels = ids.clone()
            special_mask = torch.isin(ids, torch.tensor(self.special_ids))
            is_preferred = torch.isin(ids, self.prefer_ids.to(ids.device))
            prob_matrix = torch.full_like(
                ids, self.p_low, device=ids.device, dtype=torch.float32
            )
            prob_matrix[is_preferred] = self.p_high
            mask = torch.bernoulli(prob_matrix).bool() & ~special_mask
            ids[mask] = self.tok.mask_token_id
            labels[~mask] = -100
            return {
                "input_ids": ids,
                "attention_mask": (ids != self.tok.pad_token_id).long(),
                "labels": labels,
            }

    @torch.no_grad()
    def _collect_token_loss_stats(model, tok, dataset, batch_size, mask_prob, device):
        collator = DataCollatorForLanguageModeling(
            tokenizer=tok, mlm=True, mlm_probability=mask_prob
        )
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
        sum_loss = torch.zeros(tok.vocab_size, device=device)
        counts = torch.zeros(tok.vocab_size, device=device)
        for batch in loader:
            labels = batch.pop("labels").to(device)
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            losses = F.cross_entropy(
                outputs.logits.view(-1, tok.vocab_size),
                labels.view(-1),
                reduction="none",
            ).view(labels.shape)
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    tok_id = labels[i, j].item()
                    if tok_id != -100:
                        sum_loss[tok_id] += losses[i, j]
                        counts[tok_id] += 1
        mean_loss = sum_loss / (counts + 1e-9)
        return mean_loss, counts, sum_loss

    def _derive_prefer_set_from_stats(mean_loss, counts, rate, min_count_frac):
        min_count = int(min_count_frac * counts.sum().item() / max(1, counts.numel()))
        eligible = (counts > min_count) & (mean_loss > 0)
        eligible_losses = mean_loss[eligible]
        k = int(eligible_losses.numel() * (1.0 - rate))
        thr = torch.kthvalue(eligible_losses, k=k).values.item()
        prefer_ids = torch.where(eligible & (mean_loss >= thr))[0]
        p_high = 0.8 * rate
        p_low = 0.2 * rate
        return prefer_ids, p_high, p_low, thr

    def _split_finetune_policy_indices(n_total, policy_frac, seed):
        indices = list(range(n_total))
        random.Random(seed).shuffle(indices)
        split = int(n_total * policy_frac)
        return indices[split:], indices[:split]

    def _epoch_policy_indices(all_policy_indices, start_frac, end_frac, epoch, seed):
        n = len(all_policy_indices)
        frac = start_frac + (end_frac - start_frac) * epoch
        size = max(1, int(n * frac))
        random.Random(seed + epoch).shuffle(all_policy_indices)
        return all_policy_indices[:size]

    log, cfg = setup_logging(), resolve_auto(cfg)
    paths = Paths().ensure()
    torch.use_deterministic_algorithms(True, warn_only=False)
    set_determinism(getattr(cfg, "seed", 13))

    model_name, arch = cfg.model.pretrained_name, _cfg_arch(cfg)
    level, seed = _auto_or(_cfg_get(cfg, "prune_level", 0.0), 0.0), getattr(
        cfg, "seed", 13
    )
    exp_name = str(_cfg_get(cfg, "experiment.name", "default"))
    outdir = paths.checkpoints / exp_name / f"{model_name}_pruned_{level}" + (
        f"_seed{seed}" if seed is not None else ""
    )

    if (
        bool(_auto_or(_cfg_get(cfg, "stage1.train.skip_if_ready", True), True))
        and not bool(_auto_or(_cfg_get(cfg, "stage1.train.force", False), False))
        and _is_valid_hf_dir(outdir)
    ):
        log.info(
            f"[stage1_specialise] READY checkpoint found - skipping Stage1 retrain: {outdir}"
        )
        return

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForMaskedLM.from_pretrained(model_name)
        if arch == "mlm"
        else AutoModelForCausalLM.from_pretrained(model_name)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if float(level) > 0.0:
        structured_prune(model, level=float(level), drop_heads=True, drop_ffn=True)
        _update_config_after_pruning(model, model_name, arch, log)

    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    text_field = cfg.data.processing.text_field
    ds_tok = ds.map(
        lambda b: tok(
            b[text_field], truncation=True, max_length=cfg.data.processing.max_length
        ),
        batched=True,
        remove_columns=[text_field],
    )
    train_tok = ds_tok[cfg.data.source.split_map.train].filter(
        lambda ex: len(ex["input_ids"]) > 1
    )
    max_train = int(
        _auto_or(_cfg_get(cfg, "stage1.train.max_train_samples", 1000), 1000)
    )
    base_train = train_tok.select(range(min(max_train, len(train_tok))))

    epochs = int(_auto_or(_cfg_get(cfg, "stage1.train.epochs", 3), 3))
    batch_size = int(_auto_or(_cfg_get(cfg, "stage1.train.batch_size", 8), 8))
    lr = float(_auto_or(_cfg_get(cfg, "stage1.train.lr", 5e-5), 5e-5))

    if arch == "ar":
        coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
        loader = DataLoader(
            base_train, batch_size=batch_size, shuffle=True, collate_fn=coll
        )
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        for epoch in range(epochs):
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                loss = model(**batch).loss
                loss.backward()
                opt.step()
    else:
        policy_frac = float(
            _auto_or(_cfg_get(cfg, "stage1.curriculum.policy_frac", 0.10), 0.10)
        )
        finetune_idx, policy_idx = _split_finetune_policy_indices(
            len(base_train), policy_frac, seed
        )
        finetune_set = Subset(base_train, finetune_idx)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            cur_rate = float(0.2 + (0.8 - 0.2) * (epoch / max(1, epochs - 1)))
            policy_subset = Subset(
                base_train, _epoch_policy_indices(policy_idx, 0.05, 0.05, epoch, seed)
            )
            model.eval()
            mean_loss, counts, _ = _collect_token_loss_stats(
                model, tok, policy_subset, batch_size, 0.15, device
            )
            prefer_ids, p_high, p_low, _ = _derive_prefer_set_from_stats(
                mean_loss, counts, cur_rate, 0.05
            )
            collator = _AdaptiveMaskCollator(
                tok,
                prefer_ids.cpu(),
                p_high,
                p_low,
                tok.all_special_ids,
                cfg.data.processing.max_length,
            )
            loader = DataLoader(
                finetune_set, batch_size=batch_size, shuffle=True, collate_fn=collator
            )
            model.train()
            for batch in tqdm(loader, desc=f"Training epoch {epoch+1}/{epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                loss = model(**batch).loss
                loss.backward()
                opt.step()

    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    (outdir / "READY").write_text("ok\n")
    log.info(f"[stage1] Saved specialised checkpoint -> {outdir}")


def run_stage2_compress_pm(cfg: Any) -> None:
    def _position_codec(flags: List[bool]):
        use_rle = sum(flags) / max(1, len(flags)) < 0.7
        if use_rle:
            runs = rle.encode_rle(flags)
            payload, bits = rle.elias_gamma_encode_stream(runs)
            return ("rle_elias", payload, bits)
        else:
            payload, bits = bm.encode_bitmask(flags)
            return ("bitmask", payload, bits)

    def _amp_ctx():
        return (
            torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
            if torch.cuda.is_available()
            else nullcontext()
        )

    log = setup_logging()
    paths = Paths().ensure()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mlm_name = _cfg_get(cfg, "model_ckpt", cfg.model.pretrained_name)
    keep_fraction = float(_cfg_get(cfg, "experiment.stage2.pm.keep_fraction", 0.4))
    payload_dir = paths.payloads / f"pm_{Path(mlm_name).name}_mask{keep_fraction:.2f}"
    payload_dir.mkdir(parents=True, exist_ok=True)
    recons_path, summary_path = (
        payload_dir / "recons.jsonl",
        payload_dir / "summary.json",
    )

    if summary_path.exists() and not _cfg_get(
        cfg, "experiment.stage2.pm.force_rerun", False
    ):
        log.info(f"[stage2_pm] Found existing outputs at {payload_dir}, skipping.")
        return

    tok_mlm = AutoTokenizer.from_pretrained(mlm_name)
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    mlm = (
        AutoModelForMaskedLM.from_pretrained(mlm_name, torch_dtype=model_dtype)
        .to(device)
        .eval()
    )

    oracle_name = _cfg_get(
        cfg, "experiment.stage2.pm.arithmetic_coder.oracle_model", "gpt2-large"
    )
    oracle_tok = AutoTokenizer.from_pretrained(oracle_name)
    oracle_model = (
        AutoModelForCausalLM.from_pretrained(oracle_name, torch_dtype=model_dtype)
        .to(device)
        .eval()
    )

    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    test_split = ds[cfg.data.source.split_map.test]
    max_eval = int(_cfg_get(cfg, "data.limits.max_eval_samples", 200) or 200)

    total_token_bits, total_position_bits, total_chars, n_docs = 0, 0, 0, 0
    with recons_path.open("w", encoding="utf-8") as fout:
        for ex in tqdm(
            test_split.select(range(min(max_eval, len(test_split)))),
            desc="PM recon+metrics",
        ):
            text = (ex.get(cfg.data.processing.text_field) or "").strip()
            if not text:
                continue
            ids = tok_mlm(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.data.processing.max_length,
                add_special_tokens=False,
            )["input_ids"][0].to(device)
            if ids.numel() == 0:
                continue

            s_bits = pll_surprisal_scores(ids, mlm, tok_mlm, tok_mlm.mask_token_id)
            keep_flags_t = choose_mask(
                _cfg_get(cfg, "experiment.stage2.pm.policy", "topk_global"),
                s_bits,
                keep_fraction,
            )
            _, pos_payload, pos_bits = _position_codec(keep_flags_t.tolist())
            total_position_bits += pos_bits

            spans = kept_char_spans_from_offsets(tok_mlm, text, keep_flags_t.tolist())
            kept_oracle_ids = select_oracle_token_ids_from_spans(
                oracle_tok, text, spans
            )
            token_bits = 0
            if kept_oracle_ids:
                syms, probs, _ = encode_kept_stream_with_oracle(
                    oracle_tok.decode(kept_oracle_ids), oracle_tok, oracle_model
                )
                payload = ac.encode_with_probs(syms, probs)
                token_bits = ac.payload_num_bits(payload)
            total_token_bits += token_bits

            recon_text = reconstruct_mlm_text(tok_mlm, mlm, ids, keep_flags_t)

            doc = {
                "original": text,
                "reconstruction": recon_text,
                "orig_chars": len(text),
                "token_bits": token_bits,
                "position_bits": pos_bits,
            }
            fout.write(json.dumps(doc) + "\n")
            total_chars += len(text)
            n_docs += 1

    summary = {
        "method": "PM",
        "model": Path(mlm_name).name,
        "mask_rate": 1.0 - keep_fraction,
        "n_docs": n_docs,
        "total_token_bits": total_token_bits,
        "total_position_bits": total_position_bits,
        "total_chars": total_chars,
        "bpc": (total_token_bits + total_position_bits) / max(1, total_chars),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"[stage2_pm] Wrote {recons_path} | bpc={summary['bpc']:.6f}")


def run_stage2_compress_vq(cfg: Any) -> None:
    def _limit(v: Any, default: int) -> int:
        return default if v is None or v == "auto" else int(v)

    def _resolve_base_model_name(cfg, paths: Paths) -> str:
        if _cfg_get(cfg, "model_ckpt"):
            return str(_cfg_get(cfg, "model_ckpt"))
        mname, seed = _cfg_get(cfg, "model.pretrained_name"), _cfg_get(cfg, "seed")
        exp = _cfg_get(cfg, "experiment.name")
        prune = _cfg_get(cfg, "prune_level", 0.0)
        cand = paths.checkpoints / exp / f"{mname}_pruned_{prune}_seed{seed}"
        if cand.exists():
            return str(cand)
        return mname

    log, paths = setup_logging(), Paths().ensure()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model_name = _resolve_base_model_name(cfg, paths)
    K = int(_cfg_get(cfg, "experiment.stage2.vq.codebook_sizes.0", 256))
    exp_name = str(_cfg_get(cfg, "experiment.name", "default"))
    safe_model_name = (
        Path(base_model_name).name
        if Path(base_model_name).exists()
        else base_model_name.replace("/", "-")
    )
    cache_dir = paths.checkpoints / f"vq_cache/{exp_name}/{safe_model_name}_K{K}"
    vq_model_path, idx_lm_path, tok_path = (
        cache_dir / "vq_model.pt",
        cache_dir / "idx_lm.pt",
        cache_dir,
    )

    payload_dir = paths.payloads / f"vq_{safe_model_name}_K{K}"
    payload_dir.mkdir(parents=True, exist_ok=True)
    recons_path, summary_path = (
        payload_dir / "recons.jsonl",
        payload_dir / "summary.json",
    )

    if (
        not _cfg_get(cfg, "experiment.stage2.vq.force_retrain", False)
        and summary_path.exists()
    ):
        log.info(f"[stage2_vq] Found existing outputs at {payload_dir}, skipping.")
        return

    model_vq, tok, lm = None, None, None
    if (
        not _cfg_get(cfg, "experiment.stage2.vq.force_retrain", False)
        and vq_model_path.exists()
        and idx_lm_path.exists()
    ):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model_vq = _build_vq_wrapper(base_model, layer_after=6, codebook_size=K)
        model_vq.load_state_dict(torch.load(vq_model_path, map_location=device))
        model_vq.to(device).eval()
        tok = AutoTokenizer.from_pretrained(tok_path)
        lm = _IndexGRULM(K=K, hidden=512, layers=1)
        lm.load_state_dict(torch.load(idx_lm_path, map_location=device))
        lm.to(device).eval()
    else:
        max_train = _limit(_cfg_get(cfg, "data.limits.max_train_samples"), 1000)
        model_vq, tok = train_vq_joint(
            base_model_name,
            cfg.data.source.hf_dataset,
            cfg.data.source.hf_config,
            cfg.data.processing.text_field,
            cfg.data.processing.max_length,
            6,
            K,
            5e-5,
            3,
            0.25,
            max_train,
        )
        model_vq.eval().to(device)
        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        train_split = ds[cfg.data.source.split_map.train].select(
            range(min(max_train, len(ds[cfg.data.source.split_map.train])))
        )
        idx_train = [
            encode_indices(
                model_vq,
                tok(
                    ex[cfg.data.processing.text_field],
                    return_tensors="pt",
                    truncation=True,
                    max_length=cfg.data.processing.max_length,
                )["input_ids"].to(device),
            )[0].tolist()
            for ex in train_split
            if ex[cfg.data.processing.text_field].strip()
        ]
        lm = (
            train_index_lm(
                [i for i in idx_train if i], K=K, hidden=512, layers=1, epochs=2
            )
            .to(device)
            .eval()
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_vq.state_dict(), vq_model_path)
        torch.save(lm.state_dict(), idx_lm_path)
        tok.save_pretrained(tok_path)

    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    test_split = ds[cfg.data.source.split_map.test]
    max_eval = _limit(_cfg_get(cfg, "data.limits.max_eval_samples"), 2000)

    total_bits, total_chars, n_docs = 0.0, 0, 0
    with recons_path.open("w", encoding="utf-8") as fout:
        for ex in tqdm(
            test_split.select(range(min(max_eval, len(test_split)))),
            desc="VQ recon+metrics",
        ):
            txt = ex.get(cfg.data.processing.text_field) or ""
            if not txt.strip():
                continue
            ids = tok(
                txt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.data.processing.max_length,
                add_special_tokens=False,
            )["input_ids"].to(device)
            if ids.numel() == 0:
                continue
            seq_idx = encode_indices(model_vq, ids)[0].tolist()
            if not seq_idx:
                continue

            bits = cross_entropy_bits_index_stream(lm, seq_idx)
            total_bits += bits
            total_chars += len(txt)
            n_docs += 1

            with torch.no_grad():
                idx_t = torch.tensor(
                    seq_idx, dtype=torch.long, device=device
                ).unsqueeze(0)
                toks_pred = model_vq.decode_from_indices(idx_t)[0].tolist()
            recon = tok.decode(toks_pred, skip_special_tokens=True)
            doc = {
                "original": txt,
                "reconstruction": recon,
                "orig_chars": len(txt),
                "token_bits": int(bits),
            }
            fout.write(json.dumps(doc) + "\n")

    bpc = total_bits / max(1, total_chars)
    summary = {
        "method": "VQ",
        "model": safe_model_name,
        "codebook_K": K,
        "bpc": bpc,
        "n_docs": n_docs,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"[stage2_vq] Wrote {recons_path} and summary: bpc={bpc:.6f}")
