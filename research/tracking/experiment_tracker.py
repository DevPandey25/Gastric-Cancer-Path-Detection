from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from research.reproducibility import get_environment_provenance


class ExperimentRun:
    """Represents a single research experiment run with metric tracking and artifact logging."""

    def __init__(
        self,
        run_id: str,
        name: str,
        run_dir: Path,
        config: dict[str, Any],
        dataset_info: dict[str, Any],
        model_architecture: str,
        seed: int,
    ) -> None:
        self.run_id = run_id
        self.name = name
        self.run_dir = run_dir
        self.config = config
        self.dataset_info = dataset_info
        self.model_architecture = model_architecture
        self.seed = seed
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: str | None = None
        self.status = "RUNNING"
        self.metrics_summary: dict[str, Any] = {}
        self.artifacts: list[dict[str, str]] = []
        self.provenance = get_environment_provenance()

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.manifest_file = self.run_dir / "run_manifest.json"

        self._save_manifest()

    def _save_manifest(self) -> None:
        payload = {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "seed": self.seed,
            "model_architecture": self.model_architecture,
            "dataset_info": self.dataset_info,
            "provenance": self.provenance,
            "config": self.config,
            "metrics_summary": self.metrics_summary,
            "artifacts": self.artifacts,
        }
        self.manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a single scalar metric with timestamp and optional step/epoch."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "value": float(value),
            "step": step,
        }
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.metrics_summary[name] = float(value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at the same step/epoch."""
        ts = datetime.now(timezone.utc).isoformat()
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            for name, val in metrics.items():
                entry = {
                    "timestamp": ts,
                    "name": name,
                    "value": float(val),
                    "step": step,
                }
                f.write(json.dumps(entry) + "\n")
                self.metrics_summary[name] = float(val)

    def log_artifact(self, source_path: Path | str, artifact_name: str | None = None) -> Path:
        """Copies an artifact (plot, weights, report) to the experiment artifact folder."""
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact source file does not exist: {src}")

        dest_name = artifact_name or src.name
        dest = self.artifacts_dir / dest_name
        shutil.copy2(src, dest)

        self.artifacts.append({
            "name": dest_name,
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
        })
        self._save_manifest()
        return dest

    def finish(self, status: str = "COMPLETED", summary: dict[str, Any] | None = None) -> None:
        """Concludes the experiment run and records final status and summary."""
        self.status = status
        self.completed_at = datetime.now(timezone.utc).isoformat()
        if summary:
            self.metrics_summary.update(summary)
        self._save_manifest()


class ExperimentTracker:
    """Manages the creation, indexing, and querying of local research experiments."""

    def __init__(self, experiment_dir: Path | str) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.experiment_dir / "experiments_index.json"

    def _update_index(self, run: ExperimentRun) -> None:
        index_data: list[dict[str, Any]] = []
        if self.index_file.exists():
            try:
                index_data = json.loads(self.index_file.read_text(encoding="utf-8"))
            except Exception:
                index_data = []

        # Update or append
        existing = [idx for idx, entry in enumerate(index_data) if entry.get("run_id") == run.run_id]
        summary_entry = {
            "run_id": run.run_id,
            "name": run.name,
            "status": run.status,
            "model_architecture": run.model_architecture,
            "created_at": run.created_at,
            "completed_at": run.completed_at,
            "seed": run.seed,
            "metrics_summary": run.metrics_summary,
            "run_dir": str(run.run_dir),
        }
        if existing:
            index_data[existing[0]] = summary_entry
        else:
            index_data.insert(0, summary_entry)

        self.index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")

    def start_experiment(
        self,
        name: str,
        config: dict[str, Any],
        dataset_info: dict[str, Any] | None = None,
        model_architecture: str = "GastricCancerRNN",
        seed: int = 42,
    ) -> ExperimentRun:
        """Initializes and returns a new ExperimentRun instance."""
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_id = uuid4().hex[:8]
        run_id = f"exp_{date_str}_{short_id}"
        run_dir = self.experiment_dir / run_id

        dataset_payload = dataset_info or {"name": "GCHTID", "classes": 8}

        run = ExperimentRun(
            run_id=run_id,
            name=name,
            run_dir=run_dir,
            config=config,
            dataset_info=dataset_payload,
            model_architecture=model_architecture,
            seed=seed,
        )
        self._update_index(run)
        return run

    def finish_experiment(
        self,
        run: ExperimentRun,
        status: str = "COMPLETED",
        summary: dict[str, Any] | None = None,
    ) -> None:
        """Finishes the run and updates the index."""
        run.finish(status=status, summary=summary)
        self._update_index(run)

    def list_experiments(self) -> list[dict[str, Any]]:
        """Returns the list of all tracked experiments."""
        if not self.index_file.exists():
            return []
        try:
            return json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def get_run_manifest(self, run_id: str) -> dict[str, Any] | None:
        """Retrieves the full manifest for a given run ID."""
        run_dir = self.experiment_dir / run_id
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return None

