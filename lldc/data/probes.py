# lldc/data/probes.py

from __future__ import annotations
import json, random, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any
from datasets import load_dataset

SPLIT_PATS = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ProbeSpec:
    out_path: Path
    n: int = 500
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    seed: int = 17


def _mask_capital_spans(sent: str) -> Dict[str, str] | None:
    tokens = sent.split()
    best = (-1, -1)
    i = 0
    while i < len(tokens):
        if re.match(r"^[A-Z][A-Za-z\-]+", tokens[i]):
            j = i
            while j < len(tokens) and re.match(r"^[A-Z][A-Za-z\-]+", tokens[j]):
                j += 1
            if (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    if best == (-1, -1):
        return None
    i, j = best
    ans = " ".join(tokens[i:j])
    q = " ".join(tokens[:i] + ["[MASK]"] + tokens[j:])
    return {"q": q, "a": ans}


def _generate_wikitext_probes(spec: ProbeSpec) -> List[Dict[str, str]]:
    ds = load_dataset(spec.dataset_name, spec.dataset_config)
    pool: List[Dict[str, str]] = []
    for row in ds["train"]:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        for sent in SPLIT_PATS.split(text):
            sent = sent.strip()
            if 20 <= len(sent) <= 240:
                m = _mask_capital_spans(sent)
                if m and 2 <= len(m["a"].split()) <= 4:
                    pool.append(m)
    return pool


_CODE_NAME_PAT = re.compile(
    r"""(?x)
    (?:
        def\s+([A-Za-z_][A-Za-z0-9_]*)
        \s*\(
      | class\s+([A-Za-z_][A-Za-z0-9_]*)
      | ([A-Za-z_][A-Za-z0-9_]*)\s*=\s*
    )
    """
)


def _generate_stack_probes(spec: ProbeSpec) -> List[Dict[str, str]]:
    ds = load_dataset("bigcode/the-stack", spec.dataset_config or "python")
    split = ds["train"]
    pool: List[Dict[str, str]] = []
    for row in split:
        text = (row.get("content") or "").strip()
        if not text:
            continue
        for line in text.splitlines():
            m = _CODE_NAME_PAT.search(line)
            if not m:
                continue
            name = next((g for g in m.groups() if g), None)
            if not name or len(name) < 2:
                continue
            q = line.replace(name, "[MASK]", 1)
            ctx = "\n".join([q] + text.splitlines()[1:4])
            pool.append({"q": f"Complete the code:\n{ctx}\nName:", "a": name})
            break
        if len(pool) >= spec.n * 3:
            break
    return pool


def _generate_math_probes(spec: ProbeSpec) -> List[Dict[str, str]]:
    name = "lukaemon/mathematics_dataset"
    cfg = spec.dataset_config or "all"
    ds = load_dataset(name, cfg)
    pool: List[Dict[str, str]] = []
    split = ds.get("train") or ds.get("validation") or list(ds.values())[0]
    for row in split:
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or not a:
            continue
        if 10 <= len(q) <= 240 and 1 <= len(a.split()) <= 8:
            pool.append({"q": q, "a": a})
        if len(pool) >= spec.n * 3:
            break
    return pool


def generate_factual_probes(spec: ProbeSpec) -> Path:
    rnd = random.Random(spec.seed)

    ds_name = (spec.dataset_name or "").lower()
    if "bigcode/the-stack" in ds_name or "the-stack" == ds_name:
        pool = _generate_stack_probes(spec)
    elif "lukaemon/mathematics_dataset" in ds_name or "mathematics_dataset" in ds_name:
        pool = _generate_math_probes(spec)
    else:
        pool = _generate_wikitext_probes(spec)

    rnd.shuffle(pool)
    spec.out_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.out_path.open("w", encoding="utf-8") as f:
        for item in pool[: spec.n]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return spec.out_path
