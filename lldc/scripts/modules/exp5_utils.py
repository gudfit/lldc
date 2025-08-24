# lldc/scripts/modules/exp5_utils.py

from __future__ import annotations
import torch
import numpy as np
import re
import random
import math
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Set, Iterable
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2Config,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from omegaconf import DictConfig


@dataclass
class Item:
    q: str
    a: str
    aliases: List[str]
    bucket: str = ""


def _strip_accents_lower(s: str) -> str:
    try:
        import unicodedata

        s = "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
    except Exception:
        pass
    return s.lower()


def normalize_ans(s: str) -> str:
    s = _strip_accents_lower(s)
    s = re.sub(r"[^\w\s\-\/\.]", " ", s)
    s = re.sub(r"\b(?:a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_final_answer(text: str) -> str:
    s = text.strip()
    m = re.findall(r"(?:^|\n)\s*final\s*:\s*(.+)", s, flags=re.I)
    if m:
        cand = m[-1]
    else:
        m2 = re.findall(r"(?:^|\n)\s*answer\s*[:\-]\s*(.+)", s, flags=re.I)
        if m2:
            cand = m2[-1]
        else:
            lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
            cand = lines[-1] if lines else ""
    cand = re.split(r"[\n\r]", cand)[0]
    cand = re.sub(r"[.;:,!?\"']+$", "", cand).strip()
    return normalize_ans(cand)


def exact_match_with_aliases(gold: str, aliases: Iterable[str], pred_text: str) -> bool:
    pred = extract_final_answer(pred_text)
    gset = {normalize_ans(gold)} | {normalize_ans(x) for x in (aliases or [])}
    return (pred != "") and (pred in gset)


def _fit_power_law_numeric(x: List[float], y: List[float]) -> Tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    x_safe = [max(v, 1e-9) for v in x]
    y_safe = [max(v, 1e-9) for v in y]
    log_x = np.log(x_safe)
    log_y = np.log(y_safe)
    m, b = np.polyfit(log_x, log_y, 1)
    alpha = -float(m)
    c = float(np.exp(b))
    return alpha, c


@torch.inference_mode()
def _ar_len_norm_logprob(model, tok, text: str, device: str) -> float:
    ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    if ids.shape[-1] < 2:
        return -1e9
    out = model(ids, labels=ids)
    log_likelihood = -out.loss.item() * (ids.shape[-1] - 1)
    return log_likelihood / max(1, ids.shape[-1])


def _score_storage_mc(model, tok, q: str, options: List[str], device: str) -> int:
    scores = [
        _ar_len_norm_logprob(model, tok, f"Question: {q}\nAnswer: {opt}", device)
        for opt in options
    ]
    return int(np.argmax(scores))


def create_gpt2_config_for_size(target_params_b: float) -> GPT2Config:
    target_params = int(target_params_b * 1e9)
    config = GPT2Config.from_pretrained("gpt2")
    if target_params <= 60e6:
        config.n_layer, config.n_head, config.n_embd = 6, 8, 512
    elif target_params <= 160e6:
        config.n_layer, config.n_head, config.n_embd = 12, 12, 768
    elif target_params <= 550e6:
        config.n_layer, config.n_head, config.n_embd = 24, 16, 1024
    else:
        config.n_layer, config.n_head, config.n_embd = 36, 20, 1600
    return config


def finetune_modular_model(
    cfg: DictConfig,
    bucket_name: str,
    base_model_name: str,
    dataset: Dataset,
    sample_count: int,
    device: str,
    log,
) -> str:
    safe_base = base_model_name.replace("/", "_")
    model_dir = (
        Path(cfg.outputs.cache_dir)
        / f"s{int(sample_count)}"
        / f"base_{safe_base}"
        / bucket_name
    )
    if bool(cfg.finetuning.get("clear_cache", False)) and model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)
    if model_dir.exists() and (model_dir / "config.json").exists():
        log.info(f"[cache] Using fine-tuned model for '{bucket_name}' at {model_dir}")
        return str(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    trust = bool(cfg.storage_axis.get("trust_remote_code", True))
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, trust_remote_code=trust
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    available = len(dataset)
    limit = min(int(cfg.finetuning.max_train_samples_per_bucket), int(sample_count))
    use_n = min(available, limit)
    train_ds = dataset.select(range(use_n))

    def tok_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=int(cfg.finetuning.max_length)
        )

    tokenized = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=float(cfg.finetuning.epochs),
        per_device_train_batch_size=int(cfg.finetuning.batch_size),
        learning_rate=float(cfg.finetuning.lr),
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized, data_collator=collator
    )
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return str(model_dir)


def build_prompt(q: str, with_cot: bool = False) -> str:
    if with_cot:
        return f"You are a concise QA assistant.\nQuestion: {q}\nThink step by step briefly.\nOn a new line, write 'Final:' followed by ONLY the short answer (no punctuation).\nReasoning:"
    else:
        return f"You are a concise QA assistant.\nQuestion: {q}\nWrite 'Final:' followed by ONLY the short answer (no punctuation).\n"


@torch.inference_mode()
def run_generation(
    model, tok, q: str, regime: DictConfig, device: str
) -> Tuple[str, int, int]:
    method = str(regime.get("method"))
    temperature = float(regime.get("temperature", 0.0))
    top_p = float(regime.get("top_p", 1.0))
    if method == "direct":
        prompt = build_prompt(q, with_cot=False)
        max_new = min(int(regime.get("max_new_tokens", 8)), 8)
        inputs = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
        )
        seq = out.sequences[0]
        new_tokens = seq.shape[-1] - inputs["input_ids"].shape[-1]
        return tok.decode(seq, skip_special_tokens=True), int(new_tokens), 1
    if method == "cot_short":
        prompt = build_prompt(q, with_cot=True)
        max_new = min(int(regime.get("max_new_tokens", 64)), 64)
        inputs = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
        )
        seq = out.sequences[0]
        new_tokens = seq.shape[-1] - inputs["input_ids"].shape[-1]
        return tok.decode(seq, skip_special_tokens=True), int(new_tokens), 1
    if method == "self_consistency":
        prompt = build_prompt(q, with_cot=False)
        max_new = min(int(regime.get("max_new_tokens", 16)), 16)
        chains = max(2, int(regime.get("chains", 16)))
        inputs = tok(prompt, return_tensors="pt").to(device)
        votes, tot_new = {}, 0
        for _ in range(chains):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                do_sample=True,
                temperature=max(temperature, 0.7),
                top_p=min(top_p, 0.9),
                num_beams=1,
                return_dict_in_generate=True,
            )
            seq = out.sequences[0]
            tot_new += seq.shape[-1] - inputs["input_ids"].shape[-1]
            ans = extract_final_answer(tok.decode(seq, skip_special_tokens=True))
            if ans:
                votes[ans] = votes.get(ans, 0) + 1
        if votes:
            text = f"Final: {max(votes.items(), key=lambda kv: kv[1])[0]}"
        else:
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
            )
            seq = out.sequences[0]
            tot_new += seq.shape[-1] - inputs["input_ids"].shape[-1]
            text = tok.decode(seq, skip_special_tokens=True)
        return text, max(1, int(round(tot_new / chains))), chains
    return run_generation(model, tok, q, DictConfig({"method": "direct"}), device)
