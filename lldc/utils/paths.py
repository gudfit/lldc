# lldc/utils/paths.py

from __future__ import annotations
from pathlib import Path
import os


class Paths:

    def __init__(self, root: str | Path | None = None):
        self.root = self._resolve_root(root)

        self.artifacts = self.root / "artifacts"
        self.checkpoints = self.artifacts / "checkpoints"
        self.logs = self.artifacts / "logs"
        self.runs = self.artifacts / "runs"
        self.payloads = self.artifacts / "payloads"
        self.position_codecs = self.artifacts / "position_codecs"
        self.rd_curves = self.artifacts / "rd_curves"
        self.results = self.artifacts / "results"

    def _resolve_root(self, root_arg: str | Path | None) -> Path:
        env_root = os.environ.get("LLDC_ROOT", "").strip()
        if env_root:
            return Path(env_root).resolve()

        if root_arg is not None:
            return Path(root_arg).resolve()

        here = Path(__file__).resolve()
        for cand in [here] + list(here.parents):
            try:
                if (cand / "lldc").is_dir() and (cand / "configs").is_dir():
                    return cand
            except Exception:
                pass

        return Path(".").resolve()

    def ensure(self):
        for p in [
            self.artifacts,
            self.checkpoints,
            self.logs,
            self.runs,
            self.payloads,
            self.position_codecs,
            self.rd_curves,
            self.results,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        return self

