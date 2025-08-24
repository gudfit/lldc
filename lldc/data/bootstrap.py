# FILE: lldc/data/bootstrap.py
from __future__ import annotations
from pathlib import Path
from .probes import generate_factual_probes, ProbeSpec
from .ood import ensure_ood


def ensure_data(root: Path) -> dict:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    probes = data_dir / "factual_probes.jsonl"
    ood_text = data_dir / "ood_texts.txt"
    ood_qa = data_dir / "ood_qa.jsonl"
    if not probes.exists():
        generate_factual_probes(ProbeSpec(out_path=probes, n=800))
    if not ood_text.exists():
        ensure_ood(ood_text, n=500, mode="synthetic", write_qa_path=ood_qa)
    else:
        if not ood_qa.exists():
            ensure_ood(ood_text, n=500, mode="synthetic", write_qa_path=ood_qa)

    return {"probes": probes, "ood": ood_text, "ood_qa": ood_qa}
