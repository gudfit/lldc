# lldc/scripts/execution/tasks.py

from __future__ import annotations
import hydra
import numpy as np
import matplotlib.pyplot as plt
import json, re, math, os, subprocess, tempfile, time, random, inspect, gc, logging, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, DefaultDict, Iterable, Optional
from collections import defaultdict, Counter
from copy import deepcopy
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
from transformers.utils import logging as hf_logging
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from datasketch import MinHash, MinHashLSH
from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.metrics.fidelity import (
    character_level_fidelity,
)
from lldc.eval.perplexity import ar_bpc as compute_ar_bpc
from lldc.baselines.kenlm_subsample import kenlm_ngram_bpc
from lldc.eval.functional import evaluate_superglue_zero_shot
from lldc.eval.factual_recall import evaluate_factual_recall
from lldc.data.bootstrap import ensure_data
from lldc.data.probes import generate_factual_probes, ProbeSpec
from lldc.metrics.crumpled_paper import OracleEnsemble, tcm_pcm_from_texts
from lldc.metrics.entropy_mi import (
    unigram_entropy_bits_per_symbol,
    avg_token_length_bytes,
    entropy_per_byte,
    mutual_information_adjacent,
)
from lldc.post.amortized_bpc import write_breakeven_table

hf_logging.set_verbosity_error()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def _cfg_get(cfg: Any, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


@dataclass
class RDPoint:
    method: str
    model: str | None
    mask_rate: float | None
    codebook_K: int | None
    bpc: float
    bpt: float | None
    fidelity: float | None
    chrf: float | None
    berts_f1: float | None
    sem_span_fid: float | None
    cpu_decode_ms: float | None
    n_docs: int


def run_channel_analysis_script(cfg: Any) -> None:
    log = setup_logging()

    def _ngram_topk_coverage(texts: List[str], n: int, top_k: int) -> float:
        grams = []
        for t in texts:
            toks = t.split()
            grams += [
                " ".join(toks[i : i + n]) for i in range(0, max(0, len(toks) - n + 1))
            ]
        cnt = Counter(grams)
        total = sum(cnt.values())
        if total == 0:
            return 0.0
        top = sum(c for _, c in cnt.most_common(top_k))
        return 100.0 * top / total

    def _semantic_dedup_coverage(
        texts: List[str],
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        thr: float = 0.9,
        num_perm: int = 256,
    ) -> float:
        if len(texts) < 2:
            return 0.0
        log.info(f"[semantic_dedup] Encoding {len(texts)} texts with {model_name}...")
        enc = SentenceTransformer(model_name)
        embs = enc.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        minhashes = []
        for vec in embs:
            m = MinHash(num_perm=num_perm)
            for i, val in enumerate(vec):
                if val > 0:
                    m.update(f"dim_{i}_pos".encode("utf8"))
                else:
                    m.update(f"dim_{i}_neg".encode("utf8"))
            minhashes.append(m)
        lsh = MinHashLSH(threshold=(1 + thr) / 2, num_perm=num_perm)
        for i, m in enumerate(minhashes):
            lsh.insert(f"doc_{i}", m)
        duplicate_pairs = set()
        for i, m in enumerate(minhashes):
            result = lsh.query(m)
            for res_key in result:
                j = int(res_key.split("_")[1])
                if i < j:
                    cosine_sim = float((embs[i] * embs[j]).sum())
                    if cosine_sim >= thr:
                        duplicate_pairs.add((i, j))
        total_possible_pairs = len(texts) * (len(texts) - 1) / 2
        if total_possible_pairs == 0:
            return 0.0
        return 100.0 * len(duplicate_pairs) / max(1, total_possible_pairs)

    n_list = list(
        _cfg_get(
            cfg,
            "deduplication.ngram.n_list",
            _cfg_get(cfg, "experiment.deduplication.ngram.n_list", []),
        )
    )
    top_k = int(
        _cfg_get(
            cfg,
            "deduplication.ngram.top_k",
            _cfg_get(cfg, "experiment.deduplication.ngram.top_k", 10),
        )
    )
    sbert_model = _cfg_get(
        cfg,
        "deduplication.semantic.sbert_model",
        _cfg_get(
            cfg,
            "experiment.deduplication.semantic.sbert_model",
            "sentence-transformers/all-mpnet-base-v2",
        ),
    )

    paths = Paths()
    recon_texts, orig_texts = [], []
    recon_file_cand = list(paths.payloads.glob("**/recons.jsonl"))
    if recon_file_cand:
        with recon_file_cand[0].open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                recon_texts.append(data.get("reconstruction", ""))
                orig_texts.append(data.get("original", ""))

    cov = {}
    for n in n_list:
        cov[f"ngram{n}_top{top_k}_coverage_pct"] = _ngram_topk_coverage(
            recon_texts, n, top_k
        )
    cov["semantic_dup_pct"] = _semantic_dedup_coverage(recon_texts, sbert_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    oracle = OracleEnsemble(model_names=["gpt2-large"], device=device)
    tcms, pcms = [], []
    for orig, recon in zip(orig_texts, recon_texts):
        if orig and recon:
            res = tcm_pcm_from_texts(orig, recon, oracle)
            tcms.append(res["tcm_bits"])
            pcms.append(res["pcm_bits"])

    cov["tcm_bits_mean"] = float(np.mean(tcms)) if tcms else 0.0
    cov["pcm_bits_mean"] = float(np.mean(pcms)) if pcms else 0.0

    log.info(f"Channel Analysis Results: {json.dumps(cov, indent=2)}")
    (paths.results / "channel_analysis.json").write_text(json.dumps(cov, indent=2))


def run_compute_baselines(cfg: Any) -> None:
    log = setup_logging()
    paths = Paths().ensure()
    out_dir = paths.results
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    text_field = cfg.data.processing.text_field
    train_texts = [ex[text_field] for ex in ds[cfg.data.source.split_map.train]]
    test_texts = [ex[text_field] for ex in ds[cfg.data.source.split_map.test]]
    fair_train_limit = (
        getattr(getattr(cfg.data, "limits", {}), "max_train_samples", 10000) or 10000
    )
    if len(train_texts) > fair_train_limit:
        train_texts = train_texts[:fair_train_limit]
    total_chars = sum(len(t) for t in test_texts)
    results: Dict[str, Dict] = {}

    for order in [2, 3, 4, 5, 6, 8]:
        key = f"kenlm_{order}gram"
        try:
            bpc, decode_ms = kenlm_ngram_bpc(
                train_texts,
                test_texts,
                workdir=f"artifacts/runs/{key}_baseline",
                order=order,
            )
            results[key] = {
                "bpc": float(bpc),
                "cpu_decode_ms": float(decode_ms),
                "status": "ok",
            }
        except Exception as e:
            results[key] = {"status": f"error: {e}"}

    def _try_run_cmd(cmd: List[str]) -> tuple[bool, float, str]:
        try:
            t0 = time.perf_counter()
            out = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            dt = (time.perf_counter() - t0) * 1000.0
            return True, dt, out.stdout.decode("utf-8", errors="ignore")
        except Exception as e:
            return False, 0.0, str(e)

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "test.txt"
        raw.write_text("\n".join(test_texts), encoding="utf-8")
        comp = Path(td) / "test.zst"
        ok, _, _ = _try_run_cmd(["zstd", "-f", "-22", "-q", str(raw), "-o", str(comp)])
        if ok:
            out_zstd = Path(td) / "out.txt"
            ok2, dec_ms, _ = _try_run_cmd(
                ["zstd", "-d", "-q", "-f", str(comp), "-o", str(out_zstd)]
            )
            if ok2:
                comp_bits = comp.stat().st_size * 8
                results["zstd_22"] = {
                    "bpc": float(comp_bits / max(1, total_chars)),
                    "cpu_decode_ms": float(dec_ms / max(1, len(test_texts))),
                    "status": "ok",
                }
            else:
                results["zstd_22"] = {"status": "zstd decode failed"}
        else:
            results["zstd_22"] = {"status": "zstd not available"}

    if not _cfg_get(cfg, "experiment.baselines.skip_cmix", False):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "test.txt"
            raw.write_text("\n".join(test_texts), encoding="utf-8")
            comp = Path(td) / "test.cmix"
            ok, _, _ = _try_run_cmd(["cmix", "-c", str(raw), str(comp)])
            if ok:
                out_cmix = Path(td) / "out.txt"
                ok2, dec_ms, _ = _try_run_cmd(["cmix", "-d", str(comp), str(out_cmix)])
                if ok2:
                    comp_bits = comp.stat().st_size * 8
                    results["cmix"] = {
                        "bpc": float(comp_bits / max(1, total_chars)),
                        "cpu_decode_ms": float(dec_ms),
                        "status": "ok",
                    }
                else:
                    results["cmix"] = {"status": "cmix decode failed"}
            else:
                results["cmix"] = {"status": "cmix not available"}
    else:
        results["cmix"] = {"status": "skipped"}

    (out_dir / "baselines.json").write_text(json.dumps(results, indent=2))
    log.info(f"[baselines] Wrote results -> {out_dir / 'baselines.json'}")


def run_dataset_stats(cfg: Any) -> None:
    log = setup_logging()
    paths = Paths().ensure()
    ds_name, ds_cfg = cfg.data.source.hf_dataset, getattr(
        cfg.data.source, "hf_config", None
    )
    tf, split_map = cfg.data.processing.text_field, cfg.data.source.split_map
    streaming = bool(getattr(cfg.data.source, "streaming", False))
    ds = load_dataset(ds_name, ds_cfg, streaming=streaming)
    train_split, test_split = ds[split_map.train], ds[split_map.test]
    s_cfg = _cfg_get(cfg, "data.stats", {})
    max_train = _cfg_get(cfg, "data.limits.max_train_samples")
    max_test = _cfg_get(cfg, "data.limits.max_eval_samples")

    def _safe_list(ds_split, limit):
        texts = []
        it = iter(ds_split)
        for _ in range(
            limit
            if limit is not None
            else len(ds_split) if hasattr(ds_split, "__len__") else 1_000_000
        ):
            try:
                texts.append(next(it)[tf])
            except StopIteration:
                break
        return texts

    train_texts, test_texts = _safe_list(train_split, max_train), _safe_list(
        test_split, max_test
    )
    total_test_chars = sum(len(t) for t in test_texts)

    out: Dict[str, Any] = {
        "dataset": {
            "name": ds_name,
            "config": ds_cfg,
            "num_train_samples_used": len(train_texts),
            "num_test_samples_used": len(test_texts),
            "total_test_set_chars": total_test_chars,
        }
    }

    def _word_tokens(texts):
        return [tok for t in texts for tok in (t or "").split()]

    if s_cfg.get("compute_unigram_entropy", True):
        tokens = _word_tokens(train_texts)
        H_uni, avg_bytes = unigram_entropy_bits_per_symbol(
            tokens
        ), avg_token_length_bytes(tokens)
        out["unigram"] = {
            "entropy_bits_per_token": float(H_uni),
            "avg_token_len_bytes": float(avg_bytes),
            "entropy_bits_per_byte": entropy_per_byte(H_uni, avg_bytes),
        }

    if s_cfg.get("compute_ngram_entropy"):
        out["ngram"] = {}
        for n in s_cfg["compute_ngram_entropy"]:
            try:
                bpc_n, _ = kenlm_ngram_bpc(
                    train_texts[:10000],
                    test_texts,
                    workdir=str(paths.artifacts / "runs" / f"kenlm_{n}gram_stats"),
                    order=int(n),
                )
                out["ngram"][f"order_{n}"] = {"bpc": float(bpc_n)}
            except Exception as e:
                out["ngram"][f"order_{n}"] = {"error": str(e)}

    if s_cfg.get("compute_mi", False):
        out["mutual_information"] = {
            "adjacent_token_bits": float(
                mutual_information_adjacent(_word_tokens(test_texts))
            )
        }

    stats_dir = paths.results / "dataset_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    json_path = stats_dir / f"{cfg.data.name}.json"
    json_path.write_text(json.dumps(out, indent=2))
    log.info(f"[dataset-stats] Wrote {json_path}")


def run_evaluate_all(cfg: Any, name_filter: Optional[str] = None) -> None:
    _MASK_PAT = re.compile(r"mask(?:_|-)?(0\.\d+|1(?:\.0+)?)", re.I)
    _K_PAT = re.compile(r"(?:k|codebook|K)(?:_|-)?(\d{2,6})", re.I)

    def _scan_payload_dir(
        payload_root: Path, name_filter: Optional[str] = None
    ) -> List[RDPoint]:
        points: List[RDPoint] = []
        for sub in sorted(payload_root.glob("*")):
            if not sub.is_dir():
                continue
            if name_filter and name_filter not in sub.name:
                continue
            method = (
                "PM"
                if sub.name.lower().startswith("pm_")
                else "VQ" if sub.name.lower().startswith("vq_") else None
            )
            if method is None:
                continue
            recons = sub / "recons.jsonl"
            if not recons.exists():
                continue
            mask_rate, codebook_K = None, None
            m, k = _MASK_PAT.search(sub.name), _K_PAT.search(sub.name)
            if m:
                mask_rate = float(m.group(1))
            if k:
                codebook_K = int(k.group(1))
            model = sub.name.split("_mask")[0].split("_K")[0]
            total_bits, n_docs, total_chars = 0, 0, 0
            fids, chrf, berts, sems, decms = [], [], [], [], []
            with recons.open("r", encoding="utf-8") as f:
                for ln in f:
                    try:
                        j = json.loads(ln)
                    except Exception:
                        continue
                    n_docs += 1
                    total_bits += int(j.get("position_bits", 0)) + int(
                        j.get("token_bits", 0)
                    )
                    orig = j.get("original", "")
                    total_chars += max(1, len(orig))
                    fids.append(float(j.get("char_fidelity", 0.0)) / 100.0)
                    chrf.append(float(j.get("chrf", 0.0)))
                    berts.append(float(j.get("bertscore_f1", 0.0)))
                    sems.append(float(j.get("semantic_span_fid", 0.0)))
                    decms.append(float(j.get("cpu_decode_ms", 0.0)))
            if total_chars == 0:
                continue
            bpc = total_bits / float(total_chars)
            points.append(
                RDPoint(
                    method=method,
                    model=model,
                    mask_rate=mask_rate,
                    codebook_K=codebook_K,
                    bpc=bpc,
                    bpt=None,
                    fidelity=np.mean(fids) if fids else None,
                    chrf=np.mean(chrf) if chrf else None,
                    berts_f1=np.mean(berts) if berts else None,
                    sem_span_fid=np.mean(sems) if sems else None,
                    cpu_decode_ms=np.mean(decms) if decms else None,
                    n_docs=n_docs,
                )
            )
        return points

    def _compute_static_bits(paths: Paths, cfg: Any) -> Dict:
        model_bytes = 0
        try:
            with tempfile.TemporaryDirectory() as td:
                m = AutoModelForMaskedLM.from_pretrained(
                    cfg.model.pretrained_name, cache_dir=td
                )
                for p in Path(td).rglob("*"):
                    if p.is_file():
                        model_bytes += p.stat().st_size
        except Exception:
            pass
        return {"static_bits_total": model_bytes * 8}

    def _write_static_and_rd_exports(
        paths: Paths,
        static: Dict,
        rd_points: List[Dict],
        file_suffix: str = "",
        write_static: bool = True,
    ) -> None:
        out_dir = paths.results
        if write_static:
            (out_dir / f"static_size{file_suffix}.json").write_text(
                json.dumps(static, indent=2)
            )
        pm_points = [p for p in rd_points if p["method"] == "PM"]
        (out_dir / f"pm_points{file_suffix}.json").write_text(
            json.dumps(pm_points, indent=2)
        )
        vq_points = [p for p in rd_points if p["method"] == "VQ"]
        (out_dir / f"vq_points{file_suffix}.json").write_text(
            json.dumps(vq_points, indent=2)
        )

    def _aggregate_mean_std(points: List[RDPoint]) -> Dict:
        grouped = defaultdict(lambda: defaultdict(list))
        for p in points:
            key = f"{p.method}_{p.model}"
            for metric in ["bpc", "fidelity", "chrf", "berts_f1", "sem_span_fid"]:
                val = getattr(p, metric)
                if val is not None:
                    grouped[key][metric].append(val)
        agg = {}
        for key, metrics in grouped.items():
            agg[key] = {
                f"{m}_mean": np.mean(vals) for m, vals in metrics.items() if vals
            }
            agg[key].update(
                {
                    f"{m}_std": np.std(vals)
                    for m, vals in metrics.items()
                    if len(vals) > 1
                }
            )
        return agg

    def _arch_from_cfg(cfg: Any) -> str:
        name = str(getattr(cfg.model, "pretrained_name", "")).lower()
        ar_markers = ("gpt", "llama", "mistral", "opt", "phi", "qwen", "glm", "mpt")
        return "ar" if any(m in name for m in ar_markers) else "mlm"

    log = setup_logging()
    paths = Paths().ensure()
    file_suffix = f"_{name_filter}" if name_filter else ""
    rd_points_objects = _scan_payload_dir(paths.payloads, name_filter=name_filter)
    rd_points = [p.__dict__ for p in rd_points_objects]

    if name_filter:
        _write_static_and_rd_exports(
            paths, {}, rd_points, file_suffix=file_suffix, write_static=False
        )
        log.info(f"Filtered evaluation for '{name_filter}' complete.")
        return

    static = _compute_static_bits(paths, cfg)
    _write_static_and_rd_exports(paths, static, rd_points, file_suffix="")
    agg = _aggregate_mean_std(rd_points_objects)
    (paths.results / "rd_aggregates.json").write_text(json.dumps(agg, indent=2))
    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    test_texts = [
        ex[cfg.data.processing.text_field] for ex in ds[cfg.data.source.split_map.test]
    ]

    try:
        tok = AutoTokenizer.from_pretrained("gpt2-large")
        if tok.pad_token is None and tok.eos_token:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained("gpt2-large")
        ar_bpc_val = compute_ar_bpc(
            m, tok, test_texts, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        (paths.results / "ar_baseline.json").write_text(
            json.dumps({"model": "gpt2-large", "bpc": ar_bpc_val}, indent=2)
        )
    except Exception as e:
        (paths.results / "ar_baseline.json").write_text(
            json.dumps({"error": str(e)}, indent=2)
        )

    ensure_data(paths.root)

    try:
        spec = ProbeSpec(
            out_path=paths.root / f"data/factual_probes_{cfg.data.name}.jsonl",
            n=800,
            dataset_name=str(cfg.data.source.hf_dataset),
            dataset_config=str(getattr(cfg.data.source, "hf_config", "")),
        )
        generate_factual_probes(spec)
        tok = AutoTokenizer.from_pretrained(cfg.model.pretrained_name)
        if tok.pad_token is None and tok.eos_token:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(cfg.model.pretrained_name)
        fact = evaluate_factual_recall(
            m,
            tok,
            probe_dataset_path=str(spec.out_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        (paths.results / "factual_recall.json").write_text(json.dumps(fact, indent=2))
    except Exception as e:
        (paths.results / "factual_recall.json").write_text(
            json.dumps({"error": str(e)}, indent=2)
        )

    try:
        model_name = cfg.model.pretrained_name
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None and tok.eos_token:
            tok.pad_token = tok.eos_token
        arch = _arch_from_cfg(cfg)
        m = (
            AutoModelForMaskedLM.from_pretrained(model_name)
            if arch == "mlm"
            else AutoModelForCausalLM.from_pretrained(model_name)
        )
        sg = evaluate_superglue_zero_shot(
            m,
            tok,
            tasks=["rte", "cb", "boolq"],
            n=int(getattr(cfg, "superglue_n", 100) or 100),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        (paths.results / "abstract_reasoning.json").write_text(json.dumps(sg, indent=2))
    except Exception as e:
        (paths.results / "abstract_reasoning.json").write_text(
            json.dumps({"error": str(e)}, indent=2)
        )

    log.info("[evaluate_all] Completed all evaluation steps.")
