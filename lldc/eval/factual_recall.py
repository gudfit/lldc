# lldc/eval/factual_recall.py

from __future__ import annotations
from typing import Dict, List
import json
import math

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import evaluate


@torch.no_grad()
def _generate_answer(
    model: PreTrainedModel, tok: PreTrainedTokenizerBase, prompt: str, device: str
) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    if hasattr(model, "generate"):
        out = model.generate(**enc, max_new_tokens=32, do_sample=False, num_beams=1)
        gen = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return gen.strip()
    return ""


def evaluate_factual_recall(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    probe_dataset_path: str,
    device: str | None = None,
    bleurt_threshold: float = 0.9,
) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    metric = evaluate.load("bleurt")
    preds: List[str] = []
    refs: List[str] = []
    with open(probe_dataset_path, "r", encoding="utf-8") as f:
        for ln in f:
            j = json.loads(ln)
            q, a = (j.get("q") or "").strip(), (j.get("a") or "").strip()
            if not q or not a:
                continue
            prompt = f"Question: {q}\nAnswer briefly:"
            gen = _generate_answer(model, tokenizer, prompt, device)
            preds.append(gen)
            refs.append(a)
    if not preds:
        return {"accuracy_bleurt": 0.0, "count": 0, "threshold": bleurt_threshold}
    scores = metric.compute(predictions=preds, references=refs)["scores"]
    correct = sum(1 for s in scores if float(s) >= float(bleurt_threshold))
    acc = correct / max(1, len(scores))
    return {
        "accuracy_bleurt": float(acc),
        "count": int(len(scores)),
        "threshold": float(bleurt_threshold),
    }
