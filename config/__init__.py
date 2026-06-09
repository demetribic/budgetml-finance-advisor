"""
config/__init__.py — Typed project settings loaded from config/settings.yaml.

Usage
-----
    from config import Settings

    cfg = Settings.load()          # reads config/settings.yaml
    cfg.training.lr                # 0.001
    cfg.model("anomaly").lambda_cls  # 0.5
    cfg.dataset("personaledger").has_bulk_buy_labels  # False
    cfg.save_dir / "preprocessor.pkl"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_CFG_FILE = Path(__file__).parent / "settings.yaml"


# ── Per-dataset descriptor ─────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Documents which labels a dataset provides and its load defaults."""
    has_anomaly_labels: bool = False
    has_bulk_buy_labels: bool = False
    default_hf_config: str = "default"
    default_split: str = "train"
    default_max_rows: int = 500_000


# ── Synthetic augmentation config ──────────────────────────────────────────────

@dataclass
class SyntheticAugCfg:
    """
    How many synthetic users to generate when a task lacks positive labels.
    Set to 0 to disable augmentation for that task.
    """
    bulk_buy_num_users: int = 200
    anomaly_num_users: int = 0


# ── Preprocessor config ────────────────────────────────────────────────────────

@dataclass
class PrepCfg:
    seq_len: int = 60
    forecast_horizon: int = 30
    max_merchants: int = 512


# ── Model architecture config ──────────────────────────────────────────────────

@dataclass
class ModelCfg:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    lambda_cls: float = 0.0   # classification loss weight (anomaly model only)


# ── Training hyperparameter config ─────────────────────────────────────────────

@dataclass
class TrainingCfg:
    epochs: int = 100           # anomaly detector default
    forecast_epochs: int = 30
    bulk_buy_epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    grad_clip: float = 1.0
    lr_eta_min_factor: float = 0.01


# ── Top-level Settings ─────────────────────────────────────────────────────────

@dataclass
class Settings:
    """
    Project-wide configuration loaded from config/settings.yaml.

    All training scripts and evaluation go through here — no more
    magic numbers or argparse defaults scattered across files.
    """
    categories: list[str]
    datasets: dict[str, DatasetConfig]
    synthetic_augmentation: SyntheticAugCfg
    preprocessor: PrepCfg
    models: dict[str, ModelCfg]
    training: TrainingCfg
    save_dir: Path

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        """Load settings from YAML.  Defaults to config/settings.yaml."""
        if path is None:
            path = _CFG_FILE
        with open(path) as fh:
            raw = yaml.safe_load(fh)

        datasets = {
            name: DatasetConfig(**cfg)
            for name, cfg in raw.get("datasets", {}).items()
        }

        aug = SyntheticAugCfg(**raw.get("synthetic_augmentation", {}))
        prep = PrepCfg(**raw.get("preprocessor", {}))

        _model_fields = ModelCfg.__dataclass_fields__
        models = {
            name: ModelCfg(**{k: v for k, v in cfg.items() if k in _model_fields})
            for name, cfg in raw.get("models", {}).items()
        }

        training = TrainingCfg(**raw.get("training", {}))

        save_dir = Path(raw.get("paths", {}).get("save_dir", "models/saved"))
        if not save_dir.is_absolute():
            # Resolve relative to project root (parent of this config/ package)
            save_dir = Path(__file__).parent.parent / save_dir

        return cls(
            categories=raw.get("categories", []),
            datasets=datasets,
            synthetic_augmentation=aug,
            preprocessor=prep,
            models=models,
            training=training,
            save_dir=save_dir,
        )

    # ── Convenience accessors ─────────────────────────────────────────────────

    def dataset(self, source: str) -> DatasetConfig:
        """Return DatasetConfig for *source*, or a default (all-False) config."""
        return self.datasets.get(source, DatasetConfig())

    def model(self, task: str) -> ModelCfg:
        """Return ModelCfg for *task* (forecast / anomaly / bulk_buy)."""
        return self.models.get(task, ModelCfg())
