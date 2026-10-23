"""Causal analysis of the real Hillstrom randomized experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from .data import ARM_CONTROL, ARM_MENS, ARM_WOMENS, CATEGORICAL, Dataset

SEGMENT_VARS = ("zip_code", "channel", "newbie", "buyer_category", "history_segment")
SMD_FLAG = 0.1
ALPHA = 0.05
POWER = 0.80


@dataclass
class ArmEffect:
    """Difference in proportions for one treatment arm against the control arm."""

    arm: str
    n_treat: int
    n_control: int
    rate_treat: float
    rate_control: float
    ate: float
    ci_low: float
    ci_high: float
    z_stat: float
    p_value: float

    def as_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) and k not in ("arm",) else v)
                for k, v in self.__dict__.items()}


def _proportion_effect(treat: np.ndarray, control: np.ndarray, arm: str) -> ArmEffect:
    n_t, n_c = len(treat), len(control)
    p_t, p_c = float(treat.mean()), float(control.mean())
    diff = p_t - p_c
    se = float(np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))
    z_crit = float(stats.norm.ppf(1 - ALPHA / 2))
    z, p = proportions_ztest(
        count=[int(treat.sum()), int(control.sum())], nobs=[n_t, n_c], alternative="two-sided"
    )
    return ArmEffect(
        arm=arm,
        n_treat=n_t,
        n_control=n_c,
        rate_treat=p_t,
        rate_control=p_c,
        ate=diff,
        ci_low=diff - z_crit * se,
        ci_high=diff + z_crit * se,
        z_stat=float(z),
        p_value=float(p),
    )


def arm_effects(ds: Dataset, outcome: str = "visit") -> dict[str, ArmEffect]:
    """ATE of each e-mail arm versus No E-Mail on `outcome`."""
    control = ds.arm(ARM_CONTROL)[outcome].to_numpy()
    return {
        arm: _proportion_effect(ds.arm(arm)[outcome].to_numpy(), control, arm)
        for arm in (ARM_MENS, ARM_WOMENS)
    }


def standardized_mean_differences(ds: Dataset) -> pd.DataFrame:
    """SMD of every covariate between each e-mail arm and the control arm.

    Categorical covariates are expanded to one indicator per level, so each row is
    a single comparable quantity.
    """
    frame = ds.frame
    design = pd.get_dummies(
        frame[list(ds.covariates)], columns=[c for c in CATEGORICAL if c in ds.covariates]
    ).astype(float)
    control = design[frame["segment"] == ARM_CONTROL]
    rows = []
    for arm in (ARM_MENS, ARM_WOMENS):
        treat = design[frame["segment"] == arm]
        pooled = np.sqrt((treat.var(ddof=1) + control.var(ddof=1)) / 2).replace(0.0, np.nan)
        smd = (treat.mean() - control.mean()) / pooled
        for name, value in smd.items():
            rows.append(
                {
                    "arm": arm,
                    "covariate": name,
                    "smd": float(value),
                    "imbalanced": bool(abs(value) > SMD_FLAG),
                }
            )
    return pd.DataFrame(rows)


def power_analysis(effect: ArmEffect) -> dict:
    """Sample size needed for the observed lift, and the MDE at the actual arm size."""
    analysis = NormalIndPower()
    observed_h = proportion_effectsize(effect.rate_treat, effect.rate_control)
    ratio = effect.n_control / effect.n_treat
    n_needed = float(
        analysis.solve_power(effect_size=abs(observed_h), alpha=ALPHA, power=POWER, ratio=ratio)
    )
    mde_h = float(
        analysis.solve_power(
            nobs1=effect.n_treat, alpha=ALPHA, power=POWER, ratio=ratio, effect_size=None
        )
    )
    # Convert the detectable effect size back to a lift in percentage points.
    phi_c = 2 * np.arcsin(np.sqrt(effect.rate_control))
    mde_rate = float(np.sin((mde_h + phi_c) / 2) ** 2)
    return {
        "arm": effect.arm,
        "observed_effect_size_h": float(observed_h),
        "n_per_arm_for_80pct_power": n_needed,
        "actual_n_treat": effect.n_treat,
        "mde_effect_size_h": mde_h,
        "mde_pp": (mde_rate - effect.rate_control) * 100,
    }


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def segment_effects(ds: Dataset, outcome: str = "visit") -> pd.DataFrame:
    """Lift within each level of each segment variable, Holm-corrected across all tests."""
    frame = ds.frame
    rows = []
    for var in SEGMENT_VARS:
        for level, group in frame.groupby(var, observed=True):
            control = group[group["segment"] == ARM_CONTROL][outcome].to_numpy()
            for arm in (ARM_MENS, ARM_WOMENS):
                treat = group[group["segment"] == arm][outcome].to_numpy()
                if len(treat) < 2 or len(control) < 2:
                    continue
                eff = _proportion_effect(treat, control, arm)
                rows.append(
                    {
                        "segment_var": var,
                        "segment_level": str(level),
                        "arm": arm,
                        "n_treat": eff.n_treat,
                        "n_control": eff.n_control,
                        "rate_treat": eff.rate_treat,
                        "rate_control": eff.rate_control,
                        "lift_pp": eff.ate * 100,
                        "ci_low_pp": eff.ci_low * 100,
                        "ci_high_pp": eff.ci_high * 100,
                        "p_value": eff.p_value,
                    }
                )
    out = pd.DataFrame(rows)
    out["p_holm"] = _holm(out["p_value"].tolist())
    out["significant_holm"] = out["p_holm"] < ALPHA
    return out


def t_learner_cate(ds: Dataset, outcome: str = "visit", seed: int = 42) -> pd.DataFrame:
    """Per-customer CATE from a T-learner, averaged back up to each segment level."""
    from sklearn.ensemble import GradientBoostingClassifier

    frame = ds.frame
    design = pd.get_dummies(
        frame[list(ds.covariates)], columns=[c for c in CATEGORICAL if c in ds.covariates]
    ).astype(float)
    control_mask = frame["segment"] == ARM_CONTROL
    rows = []
    for arm in (ARM_MENS, ARM_WOMENS):
        treat_mask = frame["segment"] == arm
        models = {}
        for label, mask in (("treat", treat_mask), ("control", control_mask)):
            model = GradientBoostingClassifier(random_state=seed, n_estimators=100, max_depth=3)
            model.fit(design[mask], frame.loc[mask, outcome])
            models[label] = model
        subset = treat_mask | control_mask
        cate = (
            models["treat"].predict_proba(design[subset])[:, 1]
            - models["control"].predict_proba(design[subset])[:, 1]
        )
        block = frame.loc[subset].copy()
        block["cate"] = cate
        for var in SEGMENT_VARS:
            grouped = block.groupby(var, observed=True)["cate"].mean()
            for level, value in grouped.items():
                rows.append(
                    {
                        "segment_var": var,
                        "segment_level": str(level),
                        "arm": arm,
                        "tlearner_cate_pp": float(value) * 100,
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class RealResults:
    overall: dict = field(default_factory=dict)
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)


def run(ds: Dataset, outcome: str = "visit", with_t_learner: bool = True, seed: int = 42) -> RealResults:
    effects = arm_effects(ds, outcome)
    smd = standardized_mean_differences(ds)
    segments = segment_effects(ds, outcome)
    if with_t_learner:
        cate = t_learner_cate(ds, outcome, seed=seed)
        segments = segments.merge(cate, on=["segment_var", "segment_level", "arm"], how="left")

    rates = {arm: float(ds.arm(arm)[outcome].mean()) for arm in ds.frame["segment"].unique()}
    conversion = {
        arm: float(ds.arm(arm)["conversion"].mean()) for arm in ds.frame["segment"].unique()
    }
    overall = {
        "outcome": outcome,
        "n_total": int(len(ds.frame)),
        "arm_sizes": {arm: int(len(ds.arm(arm))) for arm in rates},
        "arm_visit_rates": rates,
        "arm_conversion_rates": conversion,
        "effects": {arm: eff.as_dict() for arm, eff in effects.items()},
        "power": {arm: power_analysis(eff) for arm, eff in effects.items()},
        "balance": {
            "max_abs_smd": float(smd["smd"].abs().max()),
            "n_imbalanced": int(smd["imbalanced"].sum()),
            "threshold": SMD_FLAG,
            "by_covariate": smd.to_dict(orient="records"),
        },
        "n_segment_tests": int(len(segments)),
        "n_segment_significant_holm": int(segments["significant_holm"].sum()),
    }
    return RealResults(overall=overall, segments=segments)


def save(results: RealResults, out_dir: Path = Path("results")) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "real_experiment.json"
    csv_path = out_dir / "real_segments.csv"
    json_path.write_text(json.dumps(results.overall, indent=2, sort_keys=True))
    results.segments.to_csv(csv_path, index=False)
    return json_path, csv_path
