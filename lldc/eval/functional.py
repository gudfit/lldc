# lldc/eval/functional.py

from __future__ import annotations
from typing import List, Dict
import numpy as np
import torch
from datasets import load_dataset
import evaluate


def evaluate_superglue_zero_shot(
    model, tok, tasks: List[str], n: int = 100, device="cuda"
) -> Dict:
    model.to(device).eval()
    specs = {
        "rte": (
            "super_glue",
            "rte",
            ("premise", "hypothesis"),
            ["entailment", "not_entailment"],
        ),
        "cb": (
            "super_glue",
            "cb",
            ("premise", "hypothesis"),
            ["entailment", "contradiction", "neutral"],
        ),
        "boolq": ("super_glue", "boolq", ("passage", "question"), ["yes", "no"]),
    }
    macro = evaluate.load("f1", "multiclass")
    out_task, f1s = {}, []

    max_len = getattr(tok, "model_max_length", 512)

    with torch.no_grad():
        for t in tasks:
            if t not in specs:
                continue
            name, subset, fields, labels = specs[t]
            ds = load_dataset(name, subset)["validation"]
            k = min(n, len(ds))
            preds, refs = [], []
            for i in range(k):
                a, b = ds[i][fields[0]], ds[i][fields[1]]
                prompt = f"{a}\n{b}\nAnswer with one of: {', '.join(labels)}."

                if hasattr(model, "generate"):
                    enc = tok(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_len - 16,
                    ).to(device)
                    out_ids = model.generate(
                        **enc, max_new_tokens=8, pad_token_id=tok.eos_token_id
                    )
                    gen_text = tok.decode(
                        out_ids[0][enc["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    ).lower()

                else:
                    scores = []
                    for label_text in labels:
                        full_text = prompt + " " + label_text
                        inp = tok(
                            full_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_len,
                        ).to(device)

                        loss = model(**inp, labels=inp["input_ids"]).loss
                        scores.append(-loss.item())

                    best_label_idx = int(np.argmax(scores))
                    gen_text = labels[best_label_idx].lower()

                if t == "boolq":
                    pred = 0 if "no" in gen_text else 1
                    ref = int(ds[i]["label"])
                elif t == "rte":
                    pred = 1 if "not_entailment" in gen_text else 0
                    ref = int(ds[i]["label"])
                else:
                    if "entailment" in gen_text:
                        pred = 0
                    elif "contradiction" in gen_text:
                        pred = 1
                    else:
                        pred = 2
                    ref = int(ds[i]["label"])

                preds.append(pred)
                refs.append(ref)

            f1 = macro.compute(predictions=preds, references=refs, average="macro")[
                "f1"
            ]
            out_task[t] = float(f1)
            f1s.append(f1)

    return {"per_task_f1": out_task, "macro_f1": float(np.mean(f1s) if f1s else 0.0)}
