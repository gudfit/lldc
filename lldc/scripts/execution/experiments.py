# lldc/scripts/execution/experiments.py

from __future__ import annotations
from typing import Any, List, Dict, Tuple, Callable
import json, math, random
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset, DatasetDict, IterableDatasetDict
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from lldc.utils.paths import Paths
from lldc.utils.logging import setup_logging
from lldc.utils.hydra_utils import resolve_auto
from lldc.utils.determinism import set_determinism
from lldc.models.loaders import load_model, load_tokenizer
from lldc.oracles.measuring_tokenizer import get_measurement_tokenizer
from lldc.eval.entropy_tradeoff import (
    calculate_model_metrics,
    compute_capability_entropy,
)
from lldc.scripts.modules.exp5_utils import (
    Item,
    _fit_power_law_numeric,
    _score_storage_mc,
    finetune_modular_model,
    run_generation,
    exact_match_with_aliases,
)
from .pipeline import (
    run_stage1_specialise,
    run_stage2_compress_pm,
    run_stage2_compress_vq,
    _cfg_get,
)

try:
    import spacy
except ImportError:
    spacy = None


def run_e3(cfg: DictConfig) -> None:
    log = setup_logging()
    paths = Paths().ensure()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ecfg = cfg.experiment
    results_dir = Path(ecfg.outputs.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    meas_tok = get_measurement_tokenizer()

    def _prepare_domain_data(dcfg, max_samples, split_ratio):
        ds = load_dataset(
            dcfg.source.hf_dataset, getattr(dcfg.source, "hf_config", None)
        )
        split = ds[dcfg.source.split_map.test]
        samples = []
        for ex in split.select(range(min(max_samples, len(split)))):
            text = ex.get(dcfg.processing.text_field)
            if not text or not isinstance(text, str):
                continue
            tokens = text.split()
            if len(tokens) < 2:
                continue
            split_point = max(1, int(len(tokens) * float(split_ratio)))
            prompt, completion = " ".join(tokens[:split_point]), " ".join(
                tokens[split_point:]
            )
            if prompt and completion:
                samples.append({"prompt": prompt, "completion": completion})
        return samples

    for domain, max_samples in ecfg.evaluation.domain_samples.items():
        domain_cfg = hydra.compose(config_name=f"data/{domain}").data
        for split_ratio in ecfg.evaluation.prompt_completion_split:
            eval_data = _prepare_domain_data(
                domain_cfg, int(max_samples), float(split_ratio)
            )
            models_to_run = list(ecfg.model_groups.get("mlm", [])) + list(
                ecfg.model_groups.get("ar", [])
            )
            for model_name in tqdm(
                models_to_run, desc=f"{domain} @ split={split_ratio}"
            ):
                model_cfg = hydra.compose(config_name=f"model/{model_name}").model
                model = load_model(model_cfg.arch, model_cfg.pretrained_name)
                tokenizer = load_tokenizer(model_cfg.tokenizer)
                metrics = calculate_model_metrics(
                    model, tokenizer, eval_data, device, arch=model_cfg.arch
                )
                cap_entropies = {
                    f"p_{p}": compute_capability_entropy(
                        metrics["completions"],
                        metrics["success_probabilities"],
                        p,
                        meas_tok,
                    )
                    for p in ecfg.evaluation.reliability_thresholds
                }
                all_results.append(
                    {
                        "domain": domain,
                        "model": model_name,
                        "split": float(split_ratio),
                        "avg_pred_entropy": metrics["avg_pred_entropy"],
                        "capability_entropy": cap_entropies,
                    }
                )

    results_path = results_dir / "entropy_tradeoff_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    df = pd.json_normalize(all_results, sep="_")
    if not df.empty:
        for domain in df["domain"].unique():
            for split_ratio in df["split"].unique():
                domain_df = df[(df["domain"] == domain) & (df["split"] == split_ratio)]
                if domain_df.empty:
                    continue
                plt.figure(figsize=(12, 8))
                melted = domain_df.melt(
                    id_vars=["model", "avg_pred_entropy"],
                    value_vars=[
                        f"capability_entropy_p_{p}"
                        for p in ecfg.evaluation.reliability_thresholds
                    ],
                    var_name="p_threshold",
                    value_name="H_G",
                )
                sns.lineplot(
                    data=melted,
                    x="avg_pred_entropy",
                    y="H_G",
                    hue="model",
                    style="p_threshold",
                    markers=True,
                )
                plt.title(f"Entropy Tradeoff in {domain.upper()} (split={split_ratio})")
                plt.xlabel("Predictive Entropy (H_pred)")
                plt.ylabel("Capability Entropy (H_G)")
                plt.savefig(
                    results_dir / f"tradeoff_{domain}_all_p_split{split_ratio}.png"
                )
                plt.close()


def run_e4(cfg: DictConfig) -> None:
    from .pipeline import (
        _items_from_cfg,
        _build_foils,
        _score_retrieval_mc,
        _wilson_ci,
        _fit_power,
    )

    log = setup_logging()
    paths = Paths().ensure()
    exp_cfg = cfg.get("experiment", cfg)
    out_dir = paths.root / exp_cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"by_dataset": {}}
    storage_oracle_rg = {
        "method": "mc_oracle",
        "chains": 8,
        "max_new_tokens": 64,
        "temperature": 0.9,
        "top_p": 0.95,
    }

    for dcfg in tqdm(exp_cfg.datasets, desc="Processing datasets"):
        ds_key = str(dcfg["name"])
        items = _items_from_cfg(dcfg, int(exp_cfg.evaluation.max_per_dataset), log)
        if not items:
            continue
        foils_map = _build_foils(items, 4)
        results["by_dataset"][ds_key] = {}

        for model_name in tqdm(
            exp_cfg.storage_axis.models, desc=f"Models on {ds_key}", leave=False
        ):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = (
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else None,
                    trust_remote_code=True,
                )
                .to(device)
                .eval()
            )

            rec_ok, total, stored_mask = 0, 0, []
            for i, it in enumerate(items):
                options = [it.a] + foils_map.get(i, [])
                ok = (
                    _score_retrieval_mc(
                        model, tok, it.q, options, device, storage_oracle_rg
                    )
                    == 0
                )
                stored_mask.append(ok)
                if ok:
                    rec_ok += 1
                total += 1
            Ls = 1.0 - (rec_ok / max(1, total))

            gamma_res = {}
            for rg_name, rg in exp_cfg.retrieval_axis.regimes.items():
                succ, tot, succ_on_stored = 0, 0, 0
                for i, it in enumerate(items):
                    options = [it.a] + foils_map.get(i, [])
                    if _score_retrieval_mc(model, tok, it.q, options, device, rg) == 0:
                        succ += 1
                        if stored_mask[i]:
                            succ_on_stored += 1
                    tot += 1
                Ltot = 1.0 - (succ / max(1, tot))
                gamma_res[rg_name] = {
                    "total_loss": Ltot,
                    "retrieval_loss": max(0.0, Ltot - Ls),
                }
            results["by_dataset"][ds_key][model_name] = {
                "storage_loss": Ls,
                "retrieval": gamma_res,
            }

    out_json = out_dir / "exp4a_summary.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info(f"[exp4] Wrote {out_json}")


def run_e5(cfg: DictConfig) -> None:
    log = setup_logging()
    paths = Paths().ensure()
    exp_cfg = cfg.get("experiment", cfg)
    out_dir = paths.root / exp_cfg.outputs.dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _partitioner(items, p_cfg):
        if spacy is None:
            raise ImportError("Spacy is required for POS partitioning.")
        nlp = spacy.load(p_cfg.spacy_model, disable=["parser", "ner"])
        partitions = {k: [] for k in p_cfg.buckets}
        partitions["Other"] = []
        for it in items:
            tag = nlp(it.a)[0].pos_
            bucket = next(
                (b for b, tags in p_cfg.buckets.items() if tag in tags), "Other"
            )
            it.bucket = bucket
            partitions[bucket].append({"text": f"Question: {it.q}\nAnswer: {it.a}"})
        return items, {k: Dataset.from_list(v) for k, v in partitions.items() if v}

    items = [
        Item(q=ex["question"], a=ex["answer"], aliases=[])
        for ex in load_dataset(
            exp_cfg.dataset.hf_dataset, exp_cfg.dataset.get("hf_config")
        )[exp_cfg.dataset.split].select(range(exp_cfg.evaluation.max_per_dataset))
    ]
    all_items, bucket_ds = _partitioner(items, exp_cfg.partitioning)
    num_buckets = len(bucket_ds)
    enc = SentenceTransformer(
        exp_cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"), device=device
    )
    master_summary = {}

    for sample_count in exp_cfg.finetuning.sample_sweep:
        results_5a, results_5b = {"storage_loss_vs_beta": []}, {
            "retrieval_loss_vs_gamma": []
        }
        storage_losses, betas_total_b = [], []
        stored_ids_per_model = {}

        for base_model_name in exp_cfg.storage_axis.models:
            correct, total, stored_ids = 0, 0, set()
            for bucket, data in bucket_ds.items():
                model_path = finetune_modular_model(
                    exp_cfg, bucket, base_model_name, data, sample_count, device, log
                )
                model = (
                    AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
                )
                tok = AutoTokenizer.from_pretrained(model_path)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token

                bucket_items = [it for it in all_items if it.bucket == bucket]
                answers = [it.a for it in bucket_items]
                embs = enc.encode(answers, normalize_embeddings=True)
                for i, it in enumerate(bucket_items):
                    foils = [
                        answers[j] for j in np.argsort(-np.dot(embs, embs[i])) if j != i
                    ][:4]
                    if _score_storage_mc(model, tok, it.q, [it.a] + foils, device) == 0:
                        correct += 1
                        stored_ids.add(id(it))
                    total += 1

            loss = 1.0 - (correct / total) if total > 0 else 1.0
            beta_b = float(
                dict(exp_cfg.storage_axis.model_params_b).get(base_model_name, 1.0)
            )
            betas_total_b.append(beta_b * num_buckets)
            storage_losses.append(loss)
            stored_ids_per_model[base_model_name] = stored_ids
        results_5a["alpha_s_mod"], results_5a["c_s_mod"] = _fit_power_law_numeric(
            betas_total_b, storage_losses
        )

        model_for_retrieval = str(exp_cfg.retrieval_axis.model_for_retrieval)
        known_ids = stored_ids_per_model.get(model_for_retrieval, set())
        models, toks = {}, {}
        for bucket in bucket_ds.keys():
            model_path = (
                Path(exp_cfg.outputs.cache_dir)
                / f"s{sample_count}/base_{model_for_retrieval.replace('/', '_')}/{bucket}"
            )
            if not model_path.exists():
                continue
            models[bucket] = (
                AutoModelForCausalLM.from_pretrained(str(model_path)).to(device).eval()
            )
            toks[bucket] = AutoTokenizer.from_pretrained(str(model_path))
            if toks[bucket].pad_token is None:
                toks[bucket].pad_token = toks[bucket].eos_token

        gamma_values, retrieval_losses = [], []
        for name, regime in exp_cfg.retrieval_axis.regimes.items():
            success, count, token_counts, fanouts = 0, 0, [], []
            for it in all_items:
                if id(it) not in known_ids or it.bucket not in models:
                    continue
                text, new_tokens, fanout = run_generation(
                    models[it.bucket], toks[it.bucket], it.q, regime, device
                )
                if exact_match_with_aliases(it.a, it.aliases, text):
                    success += 1
                count += 1
                token_counts.append(new_tokens)
                fanouts.append(fanout)

            retrieval_loss = 1.0 - (success / count) if count > 0 else 1.0
            gamma_measured = (
                np.mean(token_counts) * np.mean(fanouts) if token_counts else 1.0
            )
            retrieval_losses.append(retrieval_loss)
            gamma_values.append(gamma_measured)
        results_5b["alpha_r_mod"], results_5b["c_r_mod"] = _fit_power_law_numeric(
            gamma_values, retrieval_losses
        )

        summary_path = out_dir / f"exp5_{exp_cfg.name}_summary_s{sample_count}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "sample_count": sample_count,
                    "experiment_5a": results_5a,
                    "experiment_5b": results_5b,
                },
                indent=2,
            )
        )
        master_summary[str(sample_count)] = {"results_path": str(summary_path)}

    (out_dir / f"exp5_{exp_cfg.name}_master_summary.json").write_text(
        json.dumps(master_summary, indent=2)
    )
