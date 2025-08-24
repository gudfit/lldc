# lldc/eval/hallucination.py

from __future__ import annotations
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@torch.no_grad()
def evaluate_nli_hallucination(
    original_texts: List[str],
    reconstructed_texts: List[str],
    nli_model_name: str = "roberta-large-mnli",
    batch_size: int = 8,
    device: str = "cuda",
) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
    model.to(device).eval()

    non_entailment_count = 0
    total_count = 0

    for i in range(0, len(original_texts), batch_size):
        batch_originals = original_texts[i : i + batch_size]
        batch_recons = reconstructed_texts[i : i + batch_size]

        inputs = tokenizer(
            batch_originals,
            batch_recons,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1)

        entailment_label = model.config.label2id.get("entailment", 0)
        non_entailment_count += (preds != entailment_label).sum().item()
        total_count += len(batch_originals)

    hallucination_rate = (non_entailment_count / max(1, total_count)) * 100.0
    return {
        "hallucination_rate_nli_pct": float(hallucination_rate),
        "non_entailment_count": int(non_entailment_count),
        "total_samples": int(total_count),
        "nli_model": nli_model_name,
    }
