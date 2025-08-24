# lldc/scripts/execution/misc_tasks.py

from __future__ import annotations
import os
import json
import math
import gc
import shutil
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from datasets import load_dataset

from lldc.utils.paths import Paths
from lldc.utils.logging import setup_logging
from lldc.metrics.latency_rate import (
    aggregate_cpu_decode_ms_from_payloads,
    estimate_transformer_flops_per_seq,
)
from lldc.eval.hallucination import evaluate_nli_hallucination
from lldc.metrics.fidelity import character_level_fidelity, chrf_score, semantic_span_fidelity
from lldc.metrics.pm_bpt import pm_bpt_bpc_from_fraction

@dataclass
class _MeasureCfg:
    max_examples: int = 128
    batch_size: int = 8
    max_length: Optional[int] = None

def _arch_from_cfg(cfg: Any) -> str:
    arch = getattr(cfg.model, "arch", None)
    if arch:
        return str(arch)
    name = str(getattr(cfg.model, "pretrained_name", "")).lower()
    ar_markers = ("gpt", "llama", "mistral", "opt", "phi", "qwen", "glm", "mpt")
    return "ar" if any(m in name for m in ar_markers) else "mlm"

@torch.no_grad()
def _time_encode_gpu(model, tok, texts: List[str], batch_size: int, max_length: Optional[int]) -> Dict[str, float]:
    device = next(model.parameters()).device
    t_enc: List[float] = []
    def batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]
    for batch in batches(texts, batch_size):
        enc = tok(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length or getattr(tok, "model_max_length", 1024),
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if device.type == "cuda":
            t0.record()
            out = model(**enc)
            _ = out.logits if hasattr(out, "logits") else out
            t1.record()
            torch.cuda.synchronize(device)
            dt = t0.elapsed_time(t1)
        else:
            import time
            st = time.perf_counter()
            out = model(**enc)
            _ = out.logits if hasattr(out, "logits") else out
            dt = (time.perf_counter() - st) * 1000.0
        t_enc.append(dt)
    if not t_enc:
        return {"encode_ms_mean": 0.0, "encode_ms_std": 0.0}
    mean = sum(t_enc) / len(t_enc)
    var = sum((x - mean) ** 2 for x in t_enc) / max(1, len(t_enc) - 1)
    return {"encode_ms_mean": float(mean), "encode_ms_std": float(math.sqrt(var))}

def run_measure_latency_flops(cfg: Any) -> None:
    log = setup_logging()
    paths = Paths().ensure()
    mc = _MeasureCfg(
        max_examples=int(getattr(getattr(cfg, "evaluation", {}), "latency_samples", 128) or 128),
        batch_size=int(getattr(cfg.data.loader, "batch_size", 8) or 8),
        max_length=getattr(cfg.data.processing, "max_length", None),
    )
    arch = _arch_from_cfg(cfg)
    name = cfg.model.pretrained_name
    model = AutoModelForMaskedLM.from_pretrained(name) if arch == "mlm" else AutoModelForCausalLM.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    tf = cfg.data.processing.text_field
    test = ds[cfg.data.source.split_map.test]
    texts: List[str] = []
    for i in range(min(mc.max_examples, len(test))):
        texts.append(test[i][tf])
    enc_stats = _time_encode_gpu(model, tok, texts, mc.batch_size, mc.max_length)
    cpu_dec = aggregate_cpu_decode_ms_from_payloads(paths.payloads)
    seq_len = mc.max_length or getattr(tok, "model_max_length", 1024)
    flops_seq = estimate_transformer_flops_per_seq(model, int(seq_len))
    out = {
        "model": name,
        "arch": arch,
        "device": device,
        "encode_gpu": enc_stats,
        "decode_cpu_aggregate": cpu_dec,
        "estimate_flops_per_seq": float(flops_seq),
        "seq_len_assumed": int(seq_len),
        "notes": "FLOPs approximation uses quadratic attention and standard projections",
    }
    out_path = paths.results / "latency_flops.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info(f"[latency/flops] Wrote {out_path}")

def _load_recons(path: Path) -> tuple[List[str], List[str]]:
    originals, recons = [], []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                j = json.loads(ln)
            except Exception:
                continue
            originals.append(j.get("original", "") or "")
            recons.append(j.get("reconstruction", "") or "")
    return originals, recons

def run_evaluate_reconstructions(cfg: Any) -> None:
    log = setup_logging()
    paths = Paths().ensure()
    recon_path = Path(str(getattr(cfg, "recon_path", "")))
    if not recon_path.exists():
        cands = list((paths.payloads).glob("*/recons.jsonl"))
        if not cands:
            raise FileNotFoundError("No reconstructions found. Provide cfg.recon_path=path/to/recons.jsonl.")
        recon_path = cands[0]
        log.info(f"[eval_recons] Using found recon file: {recon_path}")
    orig, rec = _load_recons(recon_path)
    if not orig or not rec:
        raise RuntimeError("Empty reconstructions set – nothing to evaluate.")
    res = evaluate_nli_hallucination(orig, rec)
    out = paths.results / "reconstruction_functional.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info(f"[eval_recons] Wrote {out}")

def _avg_bpc_list(p):
    if not isinstance(p, list) or not p:
        return None
    return sum(d.get("bpc", 0.0) for d in p) / len(p)

def run_plot_crossover_vs_size(cfg: Any) -> None:
    root = Path(getattr(getattr(cfg, "experiment", {}), "crossover_root", "results/subsets"))
    sizes, pm_bpc, vq_bpc = [], [], []
    for d in sorted([p for p in Path(root).iterdir() if p.is_dir()], key=lambda x: int(x.name)):
        pm = json.loads((d / "pm_points.json").read_text()) if (d / "pm_points.json").exists() else []
        vq = json.loads((d / "vq_points.json").read_text()) if (d / "vq_points.json").exists() else []
        apm = _avg_bpc_list(pm)
        avq = _avg_bpc_list(vq)
        if apm is None or avq is None:
            continue
        sizes.append(int(d.name))
        pm_bpc.append(apm)
        vq_bpc.append(avq)
    if not sizes:
        return
    plt.figure()
    plt.plot(sizes, pm_bpc, marker="o", label="PM (avg BPC)")
    plt.plot(sizes, vq_bpc, marker="s", label="VQ (avg BPC)")
    plt.xlabel("Dataset size (docs)")
    plt.ylabel("Average BPC")
    plt.title("Crossover vs dataset size")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out = Paths().ensure().results / "crossover_vs_size.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)

def _try_load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _collect_points(results_dir: Path) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {"PM": [], "VQ": [], "BASELINE": [], "AR": []}
    pm = _try_load_json(results_dir / "pm_points.json")
    if isinstance(pm, list):
        for d in pm:
            d["method"] = "PM"
        out["PM"] = pm
    vq = _try_load_json(results_dir / "vq_points.json")
    if isinstance(vq, list):
        for d in vq:
            d["method"] = "VQ"
        out["VQ"] = vq
    for candidate in ["baselines.json", "lossless_baselines.json", "compress_baselines.json"]:
        j = _try_load_json(results_dir / candidate)
        if isinstance(j, list):
            for d in j:
                d["method"] = d.get("method", "BASELINE")
                d["label"] = d.get("label") or d.get("name") or d["method"]
            out["BASELINE"].extend(j)
    ar = _try_load_json(results_dir / "ar_points.json")
    if isinstance(ar, list):
        for d in ar:
            d["method"] = "AR"
        out["AR"] = ar
    return out

def _entropy_line(results_dir: Path) -> Optional[float]:
    for fn in ["dataset_stats.json", "data_stats.json"]:
        p = results_dir / fn
        j = _try_load_json(p)
        if isinstance(j, dict):
            for k in ["h_infty_bits_per_char", "H_infty_bits_per_char", "H_bits_per_char"]:
                if k in j and isinstance(j[k], (int, float)):
                    return float(j[k])
    return None

def _extract_rd(points: List[Dict], metric: str) -> Tuple[List[float], List[float], List[str]]:
    X, Y, labels = [], [], []
    for d in points:
        bpc = d.get("bpc")
        if bpc is None:
            continue
        if metric == "charF":
            dist = 1.0 - float(d.get("charF_mean", d.get("char_fidelity", 0.0)))
        elif metric == "chrf":
            dist = 1.0 - float(d.get("chrf_mean", d.get("chrf", 0.0)))
        elif metric == "bertscore_f1":
            dist = 1.0 - float(d.get("bertscore_f1_mean", d.get("bertscore_f1", 0.0)))
        elif metric == "sem_span":
            dist = 1.0 - float(d.get("sem_span_fid_mean", d.get("semantic_span_fid", 0.0)))
        else:
            dist = 1.0 - float(d.get("charF_mean", d.get("char_fidelity", 0.0)))
        X.append(float(bpc))
        Y.append(float(dist))
        lbl = d.get("label") or str(d.get("codebook_size", "")) or ""
        labels.append(lbl)
    return X, Y, labels

def run_rd_collect_and_plot(cfg: Any) -> None:
    results_dir = Paths().ensure().results
    metric = getattr(getattr(cfg, "experiment", {}), "rd_metric", "charF")
    out = getattr(getattr(cfg, "experiment", {}), "rd_out", "")
    pts = _collect_points(results_dir)
    ent = _entropy_line(results_dir)
    fig, ax = plt.subplots()
    for name, marker in [("PM", "o"), ("VQ", "s")]:
        X, Y, _ = _extract_rd(pts[name], metric)
        if X:
            ax.plot(X, Y, marker=marker, linestyle="-", label=name)
    for d in pts["BASELINE"]:
        bpc = d.get("bpc")
        if bpc is None:
            continue
        ax.scatter([float(bpc)], [0.0], marker="^", label=d.get("label", d.get("method", "baseline")))
    X, Y, _ = _extract_rd(pts["AR"], metric)
    if X:
        ax.scatter(X, Y, marker="x", label="AR")
    ax.set_xlabel("Bits per Character (BPC)")
    ax.set_ylabel(f"Distortion (1 - {metric})")
    ax.set_title("Rate–Distortion (all methods)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if ent is not None:
        ax.axvline(ent, linestyle=":", label=f"entropy ~ {ent:.3f} bpc")
    handles, labels = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        H.append(h)
        L.append(l)
    ax.legend(H, L, loc="best")
    out_png = Path(out) if out else (results_dir / f"rd_plot_{metric}.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)

def _read_lines(p: Path) -> List[str]:
    with p.open("r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def _pairwise_charF(orig: List[str], recon: List[str]) -> float:
    return _mean([character_level_fidelity(o, r) for o, r in zip(orig, recon)])

def _load_points_from_config(cfg: Dict, defaults: Dict) -> Tuple[List[Dict], Dict[str, str]]:
    series_colors: Dict[str, str] = {}
    for s in cfg.get("series", []):
        nm = s["name"]
        if "color" in s:
            series_colors[nm] = s["color"]
    pts = []
    for p in cfg.get("points", []):
        point = {**defaults, **p}
        pts.append(point)
    return pts, series_colors

def build_rd_points(points_cfg: List[Dict]) -> Tuple[Dict[str, List[Tuple[float, float, str, Optional[float]]]], List[str]]:
    series_points: Dict[str, List[Tuple[float, float, str, Optional[float]]]] = {}
    series_order: List[str] = []
    for p in points_cfg:
        series = p["series"]
        label = p.get("label", "")
        orig = _read_lines(Path(p["orig"]))
        recon = _read_lines(Path(p["recon"]))
        if len(orig) != len(recon):
            raise ValueError(f"Length mismatch for {label}: {len(orig)} vs {len(recon)}")
        charF = _pairwise_charF(orig, recon)
        distortion = 1.0 - charF
        rate_bpc: Optional[float] = p.get("rate_bpc", None)
        bpt_to_annotate: Optional[float] = None
        if rate_bpc is None:
            keep_fraction = p.get("keep_fraction", None)
            vocab_size = p.get("vocab_size", None)
            if keep_fraction is None or vocab_size is None:
                raise ValueError(f"Point '{label}' needs either `rate_bpc` or both `keep_fraction` and `vocab_size`.")
            entropy_coded = bool(p.get("entropy_coded", False))
            tokenizer_name = p.get("tokenizer", None)
            bpt, bpc, tpc = pm_bpt_bpc_from_fraction(
                orig,
                float(keep_fraction),
                int(vocab_size),
                tokenizer_name,
                entropy_coded=entropy_coded,
            )
            rate_bpc = bpc
            if p.get("annotate_bpt", True):
                bpt_to_annotate = bpt
        tup = (float(rate_bpc), float(distortion), label, bpt_to_annotate)
        series_points.setdefault(series, []).append(tup)
        if series not in series_order:
            series_order.append(series)
    for s in series_points:
        series_points[s].sort(key=lambda x: x[0])
    return series_points, series_order

def plot_unified_rd(series_points: Dict[str, List[Tuple[float, float, str, Optional[float]]]], series_order: List[str], series_colors: Dict[str, str], out_path: Path, title: str = "Unified Rate–Distortion"):
    plt.figure(figsize=(8, 5.5))
    for s in series_order:
        pts = series_points[s]
        xs = [x for (x, _, _, _) in pts]
        ys = [y for (_, y, _, _) in pts]
        color = series_colors.get(s, None)
        if color is None:
            plt.plot(xs, ys, marker="o", label=s)
        else:
            plt.plot(xs, ys, marker="o", label=s, color=color)
        for x, y, lbl, bpt in pts:
            suffix = f" [BPT={bpt:.3f}]" if (bpt is not None) else ""
            plt.annotate(f"{lbl}{suffix}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    plt.xlabel("Rate (bits per character, BPC)")
    plt.ylabel("Distortion (1 − charF)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png = out_path.with_suffix(".png")
    pdf = out_path.with_suffix(".pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=160)
    plt.savefig(pdf)

def run_unified_rd_plot(cfg: Any) -> None:
    log = setup_logging()
    cfg_obj = getattr(getattr(cfg, "experiment", {}), "unified_rd", {})
    cfg_path = Path(cfg_obj.get("config", getattr(cfg, "config", "")))
    out_path = Path(cfg_obj.get("out", getattr(cfg, "out", "results/unified_rd")))
    title = cfg_obj.get("title", getattr(cfg, "title", "Unified Rate–Distortion"))
    if not cfg_path.exists():
        raise FileNotFoundError("Unified RD config not found")
    try:
        import yaml
    except Exception:
        yaml = None
    if cfg_path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("pyyaml not installed")
        cfg_file = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg_file = json.loads(cfg_path.read_text(encoding="utf-8"))
    defaults = cfg_file.get("_defaults", {})
    points_cfg, series_colors = _load_points_from_config(cfg_file, defaults)
    series_points, series_order = build_rd_points(points_cfg)
    plot_unified_rd(series_points, series_order, series_colors, out_path, title=title)
    log.info(f"[unified_rd] Saved to {out_path.with_suffix('.png')} and {out_path.with_suffix('.pdf')}")

def run_analyze_pruning_correlation(cfg: Any) -> None:
    import re
    import pandas as pd
    from scipy.stats import pearsonr
    from statsmodels.stats.power import TTestIndPower
    log = setup_logging()
    paths = Paths().ensure()
    def _float(x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None
    def _scan_results(root: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        def _prune_level_from_name(name: str) -> Optional[float]:
            for pat in [r"pruned[_-](0\.\d+|1(?:\.0+)?)", r"prune[_-](0\.\d+|1(?:\.0+)?)", r"level(0\.\d+|1(?:\.0+)?)"]:
                m = re.search(pat, name)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        pass
            return None
        for sub in (root / "artifacts").glob("*"):
            if not sub.is_dir():
                continue
            level = _prune_level_from_name(sub.name)
            if level is None:
                run_root = None
                for deeper in sub.glob("*"):
                    if not deeper.is_dir():
                        continue
                    level = _prune_level_from_name(deeper.name)
                    run_root = deeper
                    if level is None:
                        continue
                    break
                if run_root is None or level is None:
                    continue
            else:
                run_root = sub
            res_dir = run_root / "results"
            rd_dir = run_root / "rd_curves"
            ar = res_dir / "abstract_reasoning.json"
            fr = res_dir / "factual_recall.json"
            pm = rd_dir / "pm_points.json"
            glue = None
            fac = None
            tcm = None
            pcm = None
            try:
                if ar.exists():
                    j = json.loads(ar.read_text())
                    glue = _float(j.get("macro_f1"))
            except Exception:
                pass
            try:
                if fr.exists():
                    j = json.loads(fr.read_text())
                    fac = _float(j.get("accuracy_bleurt"))
            except Exception:
                pass
            try:
                if pm.exists():
                    arr = json.loads(pm.read_text())
                    if isinstance(arr, list) and arr:
                        tcm = _float(arr[0].get("tcm_mean"))
                        pcm = _float(arr[0].get("pcm_mean"))
            except Exception:
                pass
            rows.append({"run_dir": str(run_root), "pruning_level": level, "glue_macro_f1": glue, "factual_recall_bleurt": fac, "tcm_bits": tcm, "pcm_bits": pcm})
        return rows
    def _pearson(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        x2 = x.dropna()
        y2 = y.dropna()
        df = pd.concat([x2, y2], axis=1).dropna()
        if df.shape[0] < 3:
            return float("nan"), float("nan")
        r, p = pearsonr(df.iloc[:, 0], df.iloc[:, 1])
        return float(r), float(p)
    def _perform_power_analysis(effect_size=0.5, alpha=0.05, power=0.8) -> int:
        analysis = TTestIndPower()
        required_n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0, alternative="two-sided")
        return int(math.ceil(required_n))
    rows = _scan_results(paths.root)
    if not rows:
        log.warning("[correlation] No pruning runs found under artifacts/.")
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    out_dir = paths.results / "pruning_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "raw_table.tsv", sep="\t", index=False)
    required_sample_size = _perform_power_analysis(effect_size=0.5, alpha=0.05, power=0.8)
    metrics = []
    for struct_col in ["tcm_bits", "pcm_bits"]:
        for func_col in ["glue_macro_f1", "factual_recall_bleurt"]:
            r, p = _pearson(df[struct_col], df[func_col])
            actual_n = int(df[[struct_col, func_col]].dropna().shape[0])
            metrics.append({"x_struct": struct_col, "y_func": func_col, "pearson_r": r, "p_value": p, "n_actual": actual_n, "n_required_for_power": required_sample_size, "is_powered": bool(actual_n >= required_sample_size)})
    (out_dir / "summary.json").write_text(json.dumps(metrics, indent=2))
    try:
        import matplotlib.pyplot as plt
        for m in metrics:
            x = m["x_struct"]
            y = m["y_func"]
            sub = df[[x, y]].dropna()
            if sub.shape[0] < 3:
                continue
            plt.figure()
            plt.scatter(sub[x], sub[y], marker="o")
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"{x} vs {y} (r={m['pearson_r']:.3f}, p={m['p_value']:.3g}, n={m['n_actual']}, powered={m['is_powered']})")
            plt.tight_layout()
            plt.savefig(out_dir / f"scatter_{x}_vs_{y}.png", dpi=170)
            plt.close()
    except Exception:
        pass

def run_post_hoc_cost_analysis(cfg: Any) -> None:
    import tempfile
    from lldc.metrics.entropy_mi import mutual_information_adjacent
    from lldc.post.amortized_bpc import write_breakeven_table
    log = setup_logging()
    paths = Paths().ensure()
    def _files_matching(root: Path, patterns: Tuple[str, ...]) -> set[Path]:
        return {p.resolve() for pat in patterns for p in root.rglob(pat) if p.is_file()}
    def _sum_bytes(files: set[Path]) -> int:
        return sum(p.stat().st_size for p in files if p.exists())
    _MODEL_PATTERNS = ("*.bin", "*.pt", "*.safetensors")
    _VQ_BASE_MODEL_PATTERNS = ("*base_model.pt",)
    _CODEBOOK_PATTERNS = ("*codebook*.pt", "*vq_model.pt")
    _GRU_PATTERNS = ("*index_lm*.pt", "*gru*.pt")
    def _compute_static_bits(paths: Paths) -> Dict[str, int]:
        roots = [paths.checkpoints, paths.artifacts]
        all_model_files, vq_base_files, codebook_files, gru_files = set(), set(), set(), set()
        for r in roots:
            all_model_files.update(_files_matching(r, _MODEL_PATTERNS))
            vq_base_files.update(_files_matching(r, _VQ_BASE_MODEL_PATTERNS))
            codebook_files.update(_files_matching(r, _CODEBOOK_PATTERNS))
            gru_files.update(_files_matching(r, _GRU_PATTERNS))
        gru_bits = _sum_bytes(gru_files) * 8
        codebook_bits = _sum_bytes(codebook_files) * 8
        model_files = all_model_files.difference(codebook_files).difference(gru_files).difference(vq_base_files)
        model_bits = (_sum_bytes(model_files) + _sum_bytes(vq_base_files)) * 8
        total_bits = model_bits + codebook_bits + gru_bits
        return {"static_bits_total": total_bits, "model_bits": model_bits, "codebook_bits": codebook_bits, "gru_bits": gru_bits}
    def _export_amortised_curves(out_dir: Path, best_pm_bpc: float, static_bits: int, base_chars: int) -> None:
        Ns = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
        scales = [1, 2, 4, 8, 16, 32]
        curves = []
        for n in Ns:
            for s in scales:
                denom = max(1, base_chars * n * s)
                amort_bpc = (static_bits / denom) + best_pm_bpc
                curves.append({"method": "PM", "N_copies": n, "scale": s, "amortised_bpc": amort_bpc})
        (out_dir / "amortised_bpc_curves.json").write_text(json.dumps(curves, indent=2))
    static_bits_results = _compute_static_bits(paths)
    out_path_static = paths.results / "static_size.json"
    out_path_static.write_text(json.dumps(static_bits_results, indent=2))
    total_chars = 0
    dataset_stats_path = paths.results / "dataset_stats" / f"{cfg.data.name}.json"
    if dataset_stats_path.exists():
        try:
            stats_data = json.loads(dataset_stats_path.read_text())
            total_chars = stats_data["dataset"]["total_test_set_chars"]
        except Exception:
            pass
    if total_chars == 0:
        try:
            ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
            test_split = ds[cfg.data.source.split_map.test]
            text_field = cfg.data.processing.text_field
            test_texts = [ex.get(text_field, "") or "" for ex in test_split]
            total_chars = sum(len(t) for t in test_texts)
        except Exception:
            total_chars = 0
    if total_chars > 0 and static_bits_results["static_bits_total"] > 0:
        pm_points_path = paths.results / "pm_points.json"
        baselines_path = paths.results / "baselines.json"
        if pm_points_path.exists() and baselines_path.exists():
            try:
                pm_points = json.loads(pm_points_path.read_text())
                baselines_data = json.loads(baselines_path.read_text())
                best_pm_bpc = min(p["bpc"] for p in pm_points if "bpc" in p and p["bpc"] is not None)
                baselines_for_table = []
                if isinstance(baselines_data, dict):
                    for name, data in baselines_data.items():
                        if isinstance(data, dict) and data.get("status") == "ok" and "bpc" in data:
                            baselines_for_table.append((name, data["bpc"]))
                write_breakeven_table(out_dir=paths.results, static_bits=static_bits_results["static_bits_total"], test_chars_hint=total_chars, pm_points_json=pm_points_path, baselines=baselines_for_table)
                _export_amortised_curves(paths.results, best_pm_bpc, static_bits_results["static_bits_total"], total_chars)
            except Exception:
                pass

def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not (x != x)

def _cuda_device_str() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def _compute_bertscore_gpu(preds: List[str], refs: List[str], model_type: str, device: str, log: logging.Logger) -> List[float]:
    import evaluate
    metric = evaluate.load("bertscore")
    nonempty_idx = [i for i, p in enumerate(preds) if isinstance(p, str) and p.strip()]
    out = [0.0] * len(preds)
    if not nonempty_idx:
        return out
    preds_ne = [preds[i] for i in nonempty_idx]
    refs_ne  = [refs[i]  for i in nonempty_idx]
    candidate_bs = [256, 128, 64, 32, 16, 8]
    last_err = None
    for bs in candidate_bs:
        try:
            scores = metric.compute(predictions=preds_ne, references=refs_ne, lang="en", device=device, model_type=model_type, batch_size=bs)
            vals = [float(x) * 100.0 for x in scores["f1"]]
            for j, i in enumerate(nonempty_idx):
                out[i] = vals[j]
            return out
        except RuntimeError as e:
            msg = str(e)
            last_err = e
            if "CUDA" in msg:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
                continue
            raise
        except Exception as e:
            last_err = e
            break
    scores = metric.compute(predictions=preds_ne, references=refs_ne, lang="en", device="cpu", model_type=model_type, batch_size=8)
    vals = [float(x) * 100.0 for x in scores["f1"]]
    for j, i in enumerate(nonempty_idx):
        out[i] = vals[j]
    return out

def run_backfill_metrics(cfg: Any) -> None:
    log = setup_logging()
    log.setLevel(logging.DEBUG)
    bert_dev = _cuda_device_str()
    sbert_dev = bert_dev
    bertscore_model = os.environ.get("LLDC_BERTSCORE_MODEL", "roberta-large")
    paths = Paths().ensure()
    payload_root = paths.payloads
    sbert_model = None
    try:
        from sentence_transformers import SentenceTransformer
        model_kwargs = {"torch_dtype": torch.float16} if sbert_dev.startswith("cuda") else {}
        sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=sbert_dev, model_kwargs=model_kwargs)
    except Exception:
        sbert_model = None
    recon_files: List[Path] = list(payload_root.glob("**/recons.jsonl"))
    for recon_file in recon_files:
        try:
            with recon_file.open("r", encoding="utf-8") as f:
                docs = [json.loads(line) for line in f]
        except Exception:
            continue
        missing_bertscore = sum(1 for d in docs if not _is_number(d.get("bertscore_f1")))
        missing_chrf      = sum(1 for d in docs if not _is_number(d.get("chrf")))
        missing_ssf       = sum(1 for d in docs if not _is_number(d.get("semantic_span_fid")))
        if not (missing_bertscore or missing_chrf or missing_ssf):
            continue
        originals = [d.get("original", "") for d in docs]
        recons    = [d.get("reconstruction", "") for d in docs]
        if missing_chrf:
            for i in range(len(docs)):
                v = docs[i].get("chrf")
                if not _is_number(v):
                    docs[i]["chrf"] = chrf_score(originals[i], recons[i])
        if missing_bertscore:
            try:
                bsf = _compute_bertscore_gpu(preds=recons, refs=originals, model_type=bertscore_model, device=bert_dev, log=log)
                for i in range(len(docs)):
                    docs[i]["bertscore_f1"] = bsf[i]
            except Exception:
                for i in range(len(docs)):
                    if not _is_number(docs[i].get("bertscore_f1")):
                        docs[i]["bertscore_f1"] = None
                gc.collect()
        if missing_ssf and sbert_model is not None:
            from tqdm import tqdm as _tqdm
            for i in _tqdm(range(len(docs)), desc="SSF", leave=False):
                v = docs[i].get("semantic_span_fid")
                if not _is_number(v):
                    docs[i]["semantic_span_fid"] = semantic_span_fidelity(originals[i], recons[i], sbert_model_obj=sbert_model)
        try:
            backup_file = recon_file.with_suffix(".jsonl.bak")
            shutil.copy2(recon_file, backup_file)
            with recon_file.open("w", encoding="utf-8") as f:
                for d in docs:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        except Exception:
            pass
