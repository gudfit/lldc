# lldc/eval/entropy_tradeoff.py

from __future__ import annotations
from typing import List, Dict, Optional, Any
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _geomean_prob(probs: List[float]) -> float:
    if not probs:
        return 0.0
    if any(p <= 0.0 for p in probs):
        return 0.0
    s = 0.0
    for p in probs:
        s += math.log(p)
    return math.exp(s / len(probs))


@torch.no_grad()
def _calculate_ar_metrics(
    model, tokenizer, prompt: str, completion: str, device: str
) -> tuple[float, float]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0].tolist()
    comp_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0].tolist()
    if not comp_ids:
        return 0.0, 0.0
    seq = prompt_ids + comp_ids
    max_len = int(getattr(tokenizer, "model_max_length", 1024) or 1024)
    if max_len is None or max_len > 1000000:
        max_len = 2048
    comp_start = len(prompt_ids)
    probs_list: List[float] = []
    ents: List[float] = []
    for t in range(comp_start, len(seq)):
        start = max(0, t - max_len + 1)
        window = seq[start : t + 1]
        if len(window) < 2:
            continue
        input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(input_ids=input_ids).logits[0]
        pvec = torch.softmax(logits[-2], dim=-1)
        target_id = int(window[-1])
        pr = float(pvec[target_id].item()) if 0 <= target_id < pvec.size(0) else 0.0
        probs_list.append(pr)
        ents.append(float(-(pvec * torch.log2(pvec + 1e-12)).sum().item()))
    avg_prob = _geomean_prob(probs_list)
    avg_ent = float(sum(ents) / len(ents)) if ents else 0.0
    return avg_prob, avg_ent


@torch.no_grad()
def _calculate_mlm_metrics(
    model, tokenizer, prompt: str, completion: str, device: str
) -> tuple[float, float]:
    if tokenizer.mask_token_id is None:
        return 0.0, 0.0
    max_len = int(getattr(tokenizer, "model_max_length", 512) or 512)
    if max_len is None or max_len > 1000000:
        max_len = 512
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ][0].tolist()
    comp_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ][0].tolist()
    if not comp_ids:
        return 0.0, 0.0
    base = prompt_ids + comp_ids
    pos0 = len(prompt_ids)
    probs_list: List[float] = []
    ents: List[float] = []
    for i in range(len(comp_ids)):
        pos = pos0 + i
        end = min(len(base), pos + 1)
        start = max(0, end - max_len)
        window = base[start:end]
        if not window:
            continue
        local_idx = pos - start
        if local_idx < 0 or local_idx >= len(window):
            continue
        masked = list(window)
        masked[local_idx] = int(tokenizer.mask_token_id)
        input_ids = torch.tensor(masked, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(input_ids=input_ids).logits[0, local_idx]
        p = torch.softmax(logits, dim=-1)
        sym = int(comp_ids[i])
        pr = float(p[sym].item()) if 0 <= sym < p.size(0) else 0.0
        probs_list.append(pr)
        ents.append(float(-(p * torch.log2(p + 1e-12)).sum().item()))
    avg_prob = _geomean_prob(probs_list)
    avg_ent = float(sum(ents) / len(ents)) if ents else 0.0
    return avg_prob, avg_ent


def calculate_model_metrics(
    model,
    tokenizer,
    eval_data: List[Dict[str, str]],
    device: str,
    arch: str,
    progress_desc: str,
) -> Dict[str, Any]:
    results = {
        "avg_success_prob": 0.0,
        "avg_pred_entropy": 0.0,
        "completions": [],
        "success_probabilities": [],
    }
    n = 0
    for ex in tqdm(eval_data, desc=progress_desc):
        prompt = ex["prompt"]
        completion = ex["completion"]
        if arch == "mlm":
            prob, ent = _calculate_mlm_metrics(
                model, tokenizer, prompt, completion, device
            )
        else:
            prob, ent = _calculate_ar_metrics(
                model, tokenizer, prompt, completion, device
            )
        results["avg_success_prob"] += prob
        results["avg_pred_entropy"] += ent
        results["completions"].append(completion)
        results["success_probabilities"].append(prob)
        n += 1
    if n > 0:
        results["avg_success_prob"] /= n
        results["avg_pred_entropy"] /= n
    return results


def calculate_kenlm_metrics(
    kenlm_model, eval_data: List[Dict[str, str]], progress_desc: str
) -> Dict[str, Any]:
    results = {
        "avg_success_prob": 0.0,
        "avg_pred_entropy": None,
        "completions": [],
        "success_probabilities": [],
    }
    n = 0
    for ex in tqdm(eval_data, desc=progress_desc):
        prompt = ex["prompt"]
        completion = ex["completion"]
        full = (prompt.strip() + " " + completion.strip()).strip()
        try:
            lp = kenlm_model.score(full) - kenlm_model.score(prompt)
        except Exception:
            lp = float("-inf")
        words = completion.split()
        m = len(words)
        if m <= 0 or lp == float("-inf"):
            avgp = 0.0
        else:
            avgp = 10 ** (lp / m)
        results["avg_success_prob"] += avgp
        results["completions"].append(completion)
        results["success_probabilities"].append(avgp)
        n += 1
    if n > 0:
        results["avg_success_prob"] /= n
    return results


def compute_capability_entropy(
    completions: List[str],
    success_probabilities: List[float],
    p: float,
    tokenizer: Optional[Any] = None,
) -> float:
    sel = [c for c, pr in zip(completions, success_probabilities) if pr > p]
    if not sel:
        return 0.0
    if tokenizer is None:
        from lldc.oracles.measuring_tokenizer import get_measurement_tokenizer

        tokenizer = get_measurement_tokenizer()
    prefixes = set()
    for c in sel:
        ids = tokenizer(c, add_special_tokens=False)["input_ids"]
        for t in range(1, len(ids) + 1):
            prefixes.add(tuple(ids[:t]))
    n = len(prefixes)
    return math.log2(n) if n > 0 else 0.0


__all__ = [
    "calculate_model_metrics",
    "calculate_kenlm_metrics",
    "compute_capability_entropy",
]
