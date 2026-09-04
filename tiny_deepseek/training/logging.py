"""CSV and TensorBoard logging with a single scalar interface."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable

from torch.utils.tensorboard import SummaryWriter


class StructuredLogger:
    def __init__(
        self,
        experiment_dir: str | Path,
        fields: Iterable[str],
        purge_step: int | None = None,
    ) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.fields = list(dict.fromkeys(fields))
        self.csv_path = self.experiment_dir / "training_metrics.csv"
        if purge_step is not None and self.csv_path.exists():
            with self.csv_path.open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows = [
                row for row in rows
                if not row.get("step") or int(float(row["step"])) < purge_step
            ]
            with self.csv_path.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=self.fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        existed = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self.handle = self.csv_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields, extrasaction="ignore")
        if not existed:
            self.writer.writeheader()
            self.handle.flush()
        self.tensorboard = SummaryWriter(
            str(self.experiment_dir / "tensorboard"), purge_step=purge_step
        )

    def log(self, values: Dict[str, Any], step: int) -> None:
        row = {field: values.get(field, "") for field in self.fields}
        row["step"] = step
        self.writer.writerow(row)
        self.handle.flush()
        split = values.get("split", "train")
        for key, value in values.items():
            if key in {"step", "split"} or not isinstance(value, (int, float)):
                continue
            self.tensorboard.add_scalar(f"{split}/{key}", value, step)

    def close(self) -> None:
        self.tensorboard.flush()
        self.tensorboard.close()
        self.handle.close()
