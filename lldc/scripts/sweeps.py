# lldc/scripts/sweeps.py

from __future__ import annotations
import hydra
from omegaconf import OmegaConf
from typing import Any, List
import sys
from copy import deepcopy
from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.data.bootstrap import ensure_data
from lldc.utils.seed import DEFAULT_SEEDS
from lldc.scripts.execution.stages import (
    run_stage1_specialise,
    run_stage2_compress_pm,
    run_stage2_compress_vq,
    run_prune_and_recover,
)
from lldc.scripts.execution.tasks import (
    run_compute_baselines,
    run_evaluate_all,
    run_dataset_stats,
    run_channel_analysis_script,
)
from lldc.scripts.execution.experiments import run_e3, run_e4, run_e5
from lldc.scripts.execution.misc_tasks import (
    run_measure_latency_flops,
    run_evaluate_reconstructions,
    run_rd_collect_and_plot,
    run_unified_rd_plot,
    run_plot_crossover_vs_size,
    run_analyze_pruning_correlation,
    run_post_hoc_cost_analysis,
    run_backfill_metrics,
)

EXPERIMENT_RUNNERS = {
    "e3a_entropy_tradeoff": run_e3,
    "exp3a_custom_config": run_e3,
    "e4a_decomposition": run_e4,
    "e4a_decomposition_fast_it": run_e4,
    "e5ab_modular_decomposition": run_e5,
    "e5c_advanced_modular": run_e5,
    "e5_pos_modular": run_e5,
    "measure_latency_flops": run_measure_latency_flops,
    "evaluate_reconstructions": run_evaluate_reconstructions,
    "rd_collect_and_plot": run_rd_collect_and_plot,
    "unified_rd_plot": run_unified_rd_plot,
    "plot_crossover_vs_size": run_plot_crossover_vs_size,
    "analyze_pruning_correlation": run_analyze_pruning_correlation,
    "post_hoc_cost_analysis": run_post_hoc_cost_analysis,
    "backfill_metrics": run_backfill_metrics,
}


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: Any) -> None:
    log = setup_logging()
    paths = Paths().ensure()
    ensure_data(paths.root)
    exp = cfg.experiment.name
    mg = cfg.experiment.get("model_groups", {})
    seed_list = list(DEFAULT_SEEDS)[:2]

    if exp == "e1a_wiki103":
        for seed in seed_list:
            for m in mg.get("mlm", []):
                step_cfg = deepcopy(cfg)
                OmegaConf.update(step_cfg, "model.pretrained_name", m)
                OmegaConf.update(step_cfg, "seed", seed, force_add=True)
                run_stage1_specialise(step_cfg)
        for seed in seed_list:
            for m in mg.get("mlm", []):
                for rate in cfg.experiment.stage2.pm.mask_rates:
                    step_cfg = deepcopy(cfg)
                    OmegaConf.update(step_cfg, "model.pretrained_name", m)
                    OmegaConf.update(step_cfg, "seed", seed, force_add=True)
                    OmegaConf.update(
                        step_cfg, "experiment.stage2.pm.keep_fraction", 1.0 - rate
                    )
                    run_stage2_compress_pm(step_cfg)
        for seed in seed_list:
            for m in mg.get("ar", []):
                for k in cfg.experiment.stage2.vq.codebook_sizes:
                    step_cfg = deepcopy(cfg)
                    OmegaConf.update(step_cfg, "model.pretrained_name", m)
                    OmegaConf.update(step_cfg, "seed", seed, force_add=True)
                    OmegaConf.update(
                        step_cfg, "experiment.stage2.vq.codebook_sizes", [k]
                    )
                    run_stage2_compress_vq(step_cfg)
        run_compute_baselines(cfg)
        run_evaluate_all(cfg)

    elif exp == "e2a_pruning":
        for seed in seed_list:
            for m in mg.get("mlm", []) + mg.get("ar", []):
                arch = "mlm" if m in mg.get("mlm", []) else "ar"
                for lvl in cfg.experiment.pruning.schedule.levels:
                    prune_cfg = deepcopy(cfg)
                    OmegaConf.update(prune_cfg, "model.pretrained_name", m)
                    OmegaConf.update(prune_cfg, "prune_level", lvl, force_add=True)
                    OmegaConf.update(prune_cfg, "seed", seed, force_add=True)
                    run_prune_and_recover(prune_cfg)
                    ckpt_path = paths.checkpoints / exp / f"{m}_pruned_{lvl}_seed{seed}"
                    if arch == "mlm":
                        for rate in cfg.experiment.stage2.pm.mask_rates:
                            pm_cfg = deepcopy(cfg)
                            OmegaConf.update(pm_cfg, "model.pretrained_name", m)
                            OmegaConf.update(pm_cfg, "seed", seed, force_add=True)
                            OmegaConf.update(
                                pm_cfg, "experiment.stage2.pm.keep_fraction", 1.0 - rate
                            )
                            if ckpt_path.exists():
                                OmegaConf.update(
                                    pm_cfg, "model_ckpt", str(ckpt_path), force_add=True
                                )
                            run_stage2_compress_pm(pm_cfg)
                    else:
                        for k in cfg.experiment.stage2.vq.codebook_sizes:
                            vq_cfg = deepcopy(cfg)
                            OmegaConf.update(vq_cfg, "model.pretrained_name", m)
                            OmegaConf.update(vq_cfg, "seed", seed, force_add=True)
                            OmegaConf.update(
                                vq_cfg, "experiment.stage2.vq.codebook_sizes", [k]
                            )
                            if ckpt_path.exists():
                                OmegaConf.update(
                                    vq_cfg, "model_ckpt", str(ckpt_path), force_add=True
                                )
                            run_stage2_compress_vq(vq_cfg)
        run_compute_baselines(cfg)
        run_evaluate_all(cfg)

    elif exp == "e2b_channel":
        run_channel_analysis_script(cfg)

    elif exp == "e1a_evaluate_only":
        run_evaluate_all(cfg, name_filter="e1a")

    elif exp == "compute_all_dataset_stats":
        datasets_to_run = [
            "wikitext-103",
            "wikitext-2",
            "the-stack",
            "mathematics",
            "tinystories",
        ]
        for ds_name in datasets_to_run:
            ds_cfg_obj = hydra.compose(config_name=f"data/{ds_name}")
            step_cfg = OmegaConf.merge(cfg, ds_cfg_obj)
            run_dataset_stats(step_cfg)
        log.info("--- All dataset statistics computed successfully. ---")

    elif exp in EXPERIMENT_RUNNERS:
        log.info(f"[sweeps] Delegating to consolidated experiment runner for '{exp}'")
        runner = EXPERIMENT_RUNNERS[exp]
        runner(cfg)

    else:
        log.error(f"Unknown experiment '{exp}'")
        sys.exit(2)


if __name__ == "__main__":
    main()
