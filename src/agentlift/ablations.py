"""Ablation variants run on a subset of the main sample."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Dataset
from .fidelity import evaluate
from .llm import ResponseCache
from .simulate import SimConfig, output_dir, run_simulation

#: Each variant is (ablation id, label, the SimConfig fields it overrides).
VARIANTS: tuple[tuple[str, str, dict], ...] = (
    ("baseline", "baseline", {}),
    ("A1_persona", "minimal persona", {"persona_level": "minimal"}),
    ("A2_model", "small model", {"model_size": "small"}),
    ("A3_temperature", "temperature 0.0", {"temperature": 0.0}),
    ("A4_framing", "realism framing", {"framing": "realism"}),
)


def run_ablations(
    ds: Dataset,
    customers: pd.DataFrame,
    base_config: SimConfig,
    real: dict,
    real_segments: pd.DataFrame,
    models: tuple[str, str],
    cache: ResponseCache | None = None,
    concurrency: int = 8,
    requests_per_minute: int = 28,
    only: tuple[str, ...] | None = None,
    min_segment_n: int = 15,
) -> tuple[pd.DataFrame, dict]:
    """Run each variant on the ablation subset and score it against the real experiment."""
    large, small = models
    owns_cache = cache is None
    cache = cache or ResponseCache()
    rows, reports, frames = [], {}, {}
    try:
        for ablation_id, label, overrides in VARIANTS:
            if only and ablation_id not in only and ablation_id != "baseline":
                continue
            fields = dict(overrides)
            if "model_size" in fields:
                fields["model"] = small if fields["model_size"] == "small" else large
            config = replace(base_config, **fields)
            sim = run_simulation(
                ds,
                customers,
                config,
                cache=cache,
                concurrency=concurrency,
                requests_per_minute=requests_per_minute,
            )
            report = evaluate(sim, real, real_segments, ds.frame, min_segment_n=min_segment_n)
            reports[ablation_id] = report
            frames[ablation_id] = sim
            rows.append(
                {
                    "ablation": ablation_id,
                    "variant": label,
                    "config": config.label(),
                    "mean_ate_abs_error_pp": report["primary_metric_mean_ate_abs_error_pp"],
                    "baseline_calibration_error_pp": report["baseline_calibration_error_pp"],
                    "arm_ranking_match": report["arm_ranking_match"],
                    "segment_spearman_rho": report["segment_lift_spearman"]["spearman_rho"],
                    "reason_citation_share": report["mechanism_plausibility"][
                        "reason_citation_share"
                    ],
                    **{
                        f"sim_ate_pp[{arm}]": values["sim_ate_pp"]
                        for arm, values in report["per_arm"].items()
                    },
                }
            )
    finally:
        if owns_cache:
            cache.close()

    table = pd.DataFrame(rows)
    ate_columns = [c for c in table.columns if c.startswith("sim_ate_pp[")]
    robustness = {
        "n_variants": int(len(table)),
        "ate_std_pp": {col: float(table[col].std(ddof=1)) for col in ate_columns},
        "max_ate_spread_pp": float(
            max(table[col].max() - table[col].min() for col in ate_columns)
        ),
        "primary_metric_std_pp": float(table["mean_ate_abs_error_pp"].std(ddof=1)),
        "best_variant": table.loc[table["mean_ate_abs_error_pp"].idxmin(), "ablation"],
        "worst_variant": table.loc[table["mean_ate_abs_error_pp"].idxmax(), "ablation"],
    }
    return table, {"variants": reports, "robustness": robustness, "sim_frames": frames}


def save(table: pd.DataFrame, summary: dict, provider: str, mode: str) -> tuple[Path, Path]:
    out = output_dir(provider)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"ablations_{mode}.csv"
    json_path = out / f"ablations_{mode}.json"
    table.to_csv(csv_path, index=False)
    payload = {
        "robustness": summary["robustness"],
        "variants": {
            k: {
                "primary_metric_mean_ate_abs_error_pp": v["primary_metric_mean_ate_abs_error_pp"],
                "baseline_calibration_error_pp": v["baseline_calibration_error_pp"],
                "arm_ranking_match": v["arm_ranking_match"],
                "segment_spearman_rho": v["segment_lift_spearman"]["spearman_rho"],
                "per_arm": v["per_arm"],
            }
            for k, v in summary["variants"].items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    return csv_path, json_path


def spread_note(summary: dict) -> str:
    values = summary["robustness"]["ate_std_pp"].values()
    return f"ATE std across variants: {np.mean(list(values)):.2f} pp"
