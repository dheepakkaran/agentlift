"""Thin MLflow wrapper. MLflow is imported lazily so CI never needs it."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

TRACKING_URI = "file:./mlruns"
EXPERIMENT = "agentlift"


@contextlib.contextmanager
def run(name: str, params: dict | None = None, enabled: bool = True) -> Iterator[object]:
    """Open an MLflow run, or a no-op context when MLflow is unavailable or disabled."""
    if not enabled:
        yield None
        return
    # The local file store is in maintenance mode upstream and raises unless opted into.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
    except ImportError:
        yield None
        return

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as active:
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        yield active


def log_metrics(metrics: dict, active: object | None) -> None:
    if active is None:
        return
    import mlflow

    numeric = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    numeric.update({k: float(v) for k, v in metrics.items() if isinstance(v, bool)})
    if numeric:
        mlflow.log_metrics(numeric)


def log_artifact(path: Path, active: object | None) -> None:
    if active is None or not Path(path).exists():
        return
    import mlflow

    mlflow.log_artifact(str(path))


def flat_metrics(report: dict) -> dict:
    """Pull the scalar fidelity metrics worth tracking out of a report."""
    out = {
        "mean_ate_abs_error_pp": report["primary_metric_mean_ate_abs_error_pp"],
        "baseline_calibration_error_pp": report["baseline_calibration_error_pp"],
        "arm_ranking_match": report["arm_ranking_match"],
        "kendall_tau": report["kendall_tau"],
        "segment_spearman_rho": report["segment_lift_spearman"]["spearman_rho"],
        "reason_citation_share": report["mechanism_plausibility"]["reason_citation_share"],
        "anchored_mean_ate_abs_error_pp": report["anchored_to_real_control"]["mean_ate_abs_error_pp"],
    }
    for arm, values in report["per_arm"].items():
        key = arm.lower().replace(" ", "_").replace("-", "")
        out[f"{key}_ate_abs_error_pp"] = values["ate_abs_error_pp"]
        out[f"{key}_sim_ate_pp"] = values["sim_ate_pp"]
    return out
