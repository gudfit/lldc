# lldc/data/ood.py

from __future__ import annotations
from pathlib import Path
import json
import random
from typing import List
from datasets import load_dataset

from .ood_sources import (
    generate_math_qa,
    generate_code_qa,
    to_text_only,
    to_jsonl_dicts,
)


def synth_math(n: int, rnd: random.Random) -> list[str]:
    qa = generate_math_qa(n, seed=rnd.randint(1, 10_000))
    return to_text_only(qa)


def synth_code(n: int, rnd: random.Random) -> list[str]:
    qa = generate_code_qa(n, seed=rnd.randint(1, 10_000))
    return to_text_only(qa)


def from_stack(n: int) -> list[str]:
    try:
        ds = load_dataset(
            "bigcode/the-stack", data_dir="data/python", split="train[:2000]"
        )
        out: List[str] = []
        for r in ds:
            t = (r.get("content") or "").strip()
            if not t:
                continue
            lines = t.splitlines()
            if 2 <= len(lines) <= 80:
                out.append(t)
            if len(out) >= n:
                break
        return out[:n] if out else synth_code(n, random.Random(17))
    except Exception:
        return synth_code(n, random.Random(17))


def from_math(n: int) -> list[str]:
    tried = 0
    out: List[str] = []
    for spec in [
        ("openai/gsm8k", "main", "train[:2000]", "question"),
        (
            "lukaemon/mathematics_dataset",
            "algebra__linear_1d",
            "train[:3000]",
            "question",
        ),
        ("hendrycks/competition_math", None, "train[:2000]", "problem"),
    ]:
        name, cfg, split, field = spec
        try:
            ds = load_dataset(name, cfg, split=split)
            for r in ds:
                q = (r.get(field) or "").strip()
                if q:
                    out.append(q if q.endswith("?") else (q + ""))
                if len(out) >= n:
                    break
            if len(out) >= n:
                break
        except Exception:
            pass
        tried += 1
    if len(out) < n:
        out.extend(synth_math(n - len(out), random.Random(17)))
    return out[:n]


def ensure_ood(
    out: Path,
    n: int = 500,
    mode: str = "synthetic",
    seed: int = 17,
    write_qa_path: Path | None = None,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)

    if mode == "stack":
        lines = from_stack(n)
        qa = generate_code_qa(n, seed=seed)
    elif mode == "math":
        lines = from_math(n)
        qa = generate_math_qa(n, seed=seed)
    else:
        a = n // 2
        code = synth_code(a, rnd)
        math_only = synth_math(n - a, rnd)
        lines = code + math_only
        qa = generate_code_qa(a, seed=seed) + generate_math_qa(n - a, seed=seed + 1)

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if write_qa_path is not None:
        write_qa_path.parent.mkdir(parents=True, exist_ok=True)
        with write_qa_path.open("w", encoding="utf-8") as f:
            for rec in to_jsonl_dicts(qa):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return out
