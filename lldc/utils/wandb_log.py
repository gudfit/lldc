# lldc/utils/wandb_log.py

from __future__ import annotations
from typing import Optional, Iterable, Dict, Any

try:
    import wandb as _wandb
    from omegaconf import OmegaConf
except Exception:
    _wandb = None


def _cfg_get(cfg, path, default=None):
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, key, None) if not isinstance(cur, dict) else cur.get(key)
    return default if cur is None else cur


def start(cfg, run_name: str, tags: Optional[Iterable[str]] = None):
    enabled = bool(_cfg_get(cfg, "logging.wandb.enabled", False))
    if not enabled or _wandb is None:
        return None
    project = _cfg_get(cfg, "logging.wandb.project", "lldc")
    entity = _cfg_get(cfg, "logging.wandb.entity", None)
    config = None
    try:
        config = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return _wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=list(tags or []),
        config=config,
    )


def init_wandb(
    cfg: Any,
    run_name: str,
    tags: Optional[Iterable[str]] = None,
    config_dict: Optional[Dict] = None,
):
    if _wandb is None:
        return None
    enabled = bool(_cfg_get(cfg, "logging.wandb.enabled", False))
    if not enabled:
        return None
    project = _cfg_get(cfg, "logging.wandb.project", "lldc")
    entity = _cfg_get(cfg, "logging.wandb.entity", None)
    return _wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=list(tags or []),
        config=config_dict or {},
    )


def log(metrics: dict, step: int | None = None):
    if _wandb is None:
        return
    if _wandb.run is None:
        return
    _wandb.log(metrics, step=step)


def log_metrics(wb, step: int | None = None, **kwargs):
    log(kwargs, step=step)


def log_artifact_file(path: str, name: str, type_: str = "artifact"):
    if _wandb is None or _wandb.run is None:
        return
    art = _wandb.Artifact(name, type=type_)
    art.add_file(path)
    _wandb.run.log_artifact(art)


def finish():
    if _wandb is None:
        return
    if _wandb.run is not None:
        _wandb.finish()
