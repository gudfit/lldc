# lldc/post/amortised_bpc.py

from __future__ import annotations
import json, math
from pathlib import Path
from typing import List, Tuple


def amortised_bpc(
    bpc_runtime: float,
    static_bits: int,
    base_chars: int,
    copies: int = 1,
    scale: int = 1,
) -> float:
    denom = max(1, base_chars * copies * scale)
    return static_bits / denom + bpc_runtime


def write_breakeven_table(
    out_dir: Path,
    static_bits: int,
    test_chars_hint: int,
    pm_points_json: Path,
    baselines: List[Tuple[str, float]],
) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pm_points_json.exists() or not baselines:
        return None
    arr = json.loads(pm_points_json.read_text())
    bpcs = [pt["bpc"] for pt in arr if pt.get("method") == "PM"]
    if not bpcs:
        return None
    bpc_pm = float(min(bpcs))
    rows = []
    for name, bpc_ref in baselines:
        delta = bpc_ref - bpc_pm
        nc = math.inf if delta <= 0 else static_bits / max(1, test_chars_hint * delta)
        rows.append((name, bpc_ref, bpc_pm, nc))
    out_tsv = out_dir / "breakeven_table.tsv"
    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("baseline\tbpc_ref\tpm_bpc\tNc_break_even\n")
        for name, b_ref, b_pm, nc in rows:
            f.write(f"{name}\t{b_ref:.6f}\t{b_pm:.6f}\t{nc}\n")
    return out_tsv
