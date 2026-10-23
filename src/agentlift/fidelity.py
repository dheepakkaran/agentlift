"""Compare a simulated experiment against the real one."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import ARM_CONTROL, ARM_MENS, ARM_WOMENS
from .experiment import SEGMENT_VARS

CONDITION_TO_ARM = {"mens": ARM_MENS, "womens": ARM_WOMENS, "none": ARM_CONTROL}
EMAIL_CONDITIONS = ("mens", "womens")
BOOTSTRAP = 2000
MIN_SEGMENT_N = 15

#: Words that mark a reason as citing the email or a persona feature rather than
#: being generic. Documented here so the mechanism metric is reproducible.
EMAIL_KEYWORDS = (
    "email",
    "e-mail",
    "message",
    "promotion",
    "offer",
    "newsletter",
    "campaign",
    "inbox",
)
PERSONA_KEYWORDS = (
    "bought",
    "purchase",
    "order",
    "spent",
    "shopped",
    "recent",
    "month",
    "loyal",
    "regular",
    "browsing",
    "rural",
    "suburb",
    "city",
    "online",
    "phone",
    "new to",
    "men",
    "women",
)


def _pivot(sim: pd.DataFrame, value: str = "visit_probability") -> pd.DataFrame:
    """One row per customer, one column per condition."""
    return sim.pivot_table(index="customer_id", columns="condition", values=value, aggfunc="mean")


def _bootstrap_ci(values: np.ndarray, resamples: int = BOOTSTRAP, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sim_ate(sim: pd.DataFrame, value: str = "visit_probability", seed: int = 42) -> dict[str, dict]:
    """Paired within-subject ATE per e-mail condition, with a bootstrap CI."""
    wide = _pivot(sim, value).dropna()
    out: dict[str, dict] = {}
    for condition in EMAIL_CONDITIONS:
        paired = (wide[condition] - wide["none"]).to_numpy(dtype=float)
        low, high = _bootstrap_ci(paired, seed=seed)
        out[condition] = {
            "arm": CONDITION_TO_ARM[condition],
            "n_customers": int(len(paired)),
            "ate": float(paired.mean()),
            "ate_pp": float(paired.mean()) * 100,
            "ci_low_pp": low * 100,
            "ci_high_pp": high * 100,
        }
    return out


def sim_rates(sim: pd.DataFrame, value: str = "visit_probability") -> dict[str, float]:
    return {c: float(g[value].mean()) for c, g in sim.groupby("condition")}


def mechanism_plausibility(sim: pd.DataFrame, real_segments: pd.DataFrame) -> dict:
    """Do agents give reasons that cite a cause, and do they rank segments like reality?"""
    visited = sim[sim["visit"]]
    def cites(reason: str) -> bool:
        low = str(reason).lower()
        return any(k in low for k in EMAIL_KEYWORDS) or any(k in low for k in PERSONA_KEYWORDS)

    share = float(visited["reason"].map(cites).mean()) if len(visited) else 0.0

    directional = []
    for var in SEGMENT_VARS:
        for condition in EMAIL_CONDITIONS:
            arm = CONDITION_TO_ARM[condition]
            real = real_segments[
                (real_segments["segment_var"] == var) & (real_segments["arm"] == arm)
            ]
            sim_lift = segment_lifts(sim, var)[condition]
            shared = [lvl for lvl in sim_lift if lvl in set(real["segment_level"])]
            if len(shared) < 2:
                continue
            real_best = real.set_index("segment_level").loc[shared, "lift_pp"].idxmax()
            sim_best = max(shared, key=lambda lvl: sim_lift[lvl]["lift_pp"])
            directional.append(
                {
                    "segment_var": var,
                    "condition": condition,
                    "real_most_responsive": real_best,
                    "sim_most_responsive": sim_best,
                    "match": bool(real_best == sim_best),
                }
            )
    matches = [d["match"] for d in directional]
    return {
        "reason_citation_share": share,
        "n_visit_reasons": int(len(visited)),
        "directional_checks": directional,
        "directional_match_rate": float(np.mean(matches)) if matches else float("nan"),
        "email_keywords": list(EMAIL_KEYWORDS),
        "persona_keywords": list(PERSONA_KEYWORDS),
    }


def segment_lifts(
    sim: pd.DataFrame, var: str, covariates: pd.DataFrame | None = None, min_n: int = MIN_SEGMENT_N
) -> dict[str, dict[str, dict]]:
    """Simulated lift per level of one segment variable, keyed by condition then level."""
    frame = sim
    if var not in frame.columns:
        if covariates is None:
            raise ValueError(f"{var} is not in the simulation frame and no covariates were given")
        frame = frame.merge(covariates[["customer_id", var]], on="customer_id", how="left")
    out: dict[str, dict[str, dict]] = {c: {} for c in EMAIL_CONDITIONS}
    for level, group in frame.groupby(var, observed=True):
        wide = _pivot(group).dropna()
        if len(wide) < min_n:
            continue
        for condition in EMAIL_CONDITIONS:
            paired = (wide[condition] - wide["none"]).to_numpy(dtype=float)
            out[condition][str(level)] = {"n": int(len(paired)), "lift_pp": float(paired.mean()) * 100}
    return out


def segment_correlation(
    sim: pd.DataFrame, covariates: pd.DataFrame, real_segments: pd.DataFrame, min_n: int = MIN_SEGMENT_N
) -> dict:
    """Spearman correlation of simulated vs real segment lift, pooled over all segments."""
    pairs = []
    skipped = []
    for var in SEGMENT_VARS:
        lifts = segment_lifts(sim, var, covariates, min_n=min_n)
        for condition in EMAIL_CONDITIONS:
            arm = CONDITION_TO_ARM[condition]
            real = real_segments[
                (real_segments["segment_var"] == var) & (real_segments["arm"] == arm)
            ].set_index("segment_level")
            for level, stats_ in lifts[condition].items():
                if level not in real.index:
                    continue
                pairs.append(
                    {
                        "segment_var": var,
                        "segment_level": level,
                        "condition": condition,
                        "n_sim": stats_["n"],
                        "sim_lift_pp": stats_["lift_pp"],
                        "real_lift_pp": float(real.loc[level, "lift_pp"]),
                    }
                )
        all_levels = set(sim.merge(covariates[["customer_id", var]], on="customer_id")[var].astype(str))
        kept = set(lifts["mens"])
        skipped.extend(f"{var}={lvl}" for lvl in sorted(all_levels - kept))

    table = pd.DataFrame(pairs)
    if len(table) >= 3:
        rho, p = stats.spearmanr(table["sim_lift_pp"], table["real_lift_pp"])
    else:
        rho, p = float("nan"), float("nan")
    return {
        "n_segments_compared": int(len(table)),
        "min_segment_n": min_n,
        "skipped_small_segments": sorted(set(skipped)),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "pairs": table.to_dict(orient="records"),
    }


MATCHING_CATEGORIES = {"mens": {"mens only", "both"}, "womens": {"womens only", "both"}}


def persona_match_effects(sim: pd.DataFrame, covariates: pd.DataFrame) -> dict:
    """Split the simulated lift by whether the e-mail matched the customer's buying history.

    This uses simulated responses and covariates only, never the real outcomes, so it is
    available before the blind comparison.
    """
    merged = sim.merge(covariates[["customer_id", "buyer_category"]], on="customer_id", how="left")
    wide = merged.pivot_table(
        index=["customer_id", "buyer_category"], columns="condition", values="visit_probability"
    ).reset_index()
    out: dict = {}
    for condition in EMAIL_CONDITIONS:
        lift = (wide[condition] - wide["none"]) * 100
        matched = wide["buyer_category"].isin(MATCHING_CATEGORIES[condition])
        out[condition] = {
            "matched_lift_pp": float(lift[matched].mean()) if matched.any() else float("nan"),
            "n_matched": int(matched.sum()),
            "mismatched_lift_pp": float(lift[~matched].mean()) if (~matched).any() else float("nan"),
            "n_mismatched": int((~matched).sum()),
        }
    gaps = [
        v["matched_lift_pp"] - v["mismatched_lift_pp"]
        for v in out.values()
        if not (np.isnan(v["matched_lift_pp"]) or np.isnan(v["mismatched_lift_pp"]))
    ]
    out["mean_match_gap_pp"] = float(np.mean(gaps)) if gaps else float("nan")
    return out


def _ranking(rates: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(rates.items(), key=lambda kv: -kv[1])]


def evaluate(
    sim: pd.DataFrame,
    real: dict,
    real_segments: pd.DataFrame,
    covariates: pd.DataFrame,
    seed: int = 42,
    min_segment_n: int = MIN_SEGMENT_N,
) -> dict:
    """All fidelity metrics for one simulation run against the real experiment."""
    real_rates = real["arm_visit_rates"]
    sim_prob_rates = sim_rates(sim, "visit_probability")
    sim_binary_rates = sim_rates(sim, "visit")

    sim_by_arm = {CONDITION_TO_ARM[c]: v for c, v in sim_prob_rates.items()}
    real_order = _ranking(real_rates)
    sim_order = _ranking(sim_by_arm)
    tau, tau_p = stats.kendalltau(
        [real_rates[a] for a in real_order], [sim_by_arm[a] for a in real_order]
    )

    ate_prob = sim_ate(sim, "visit_probability", seed=seed)
    ate_binary = sim_ate(sim, "visit", seed=seed)

    per_arm = {}
    for condition in EMAIL_CONDITIONS:
        arm = CONDITION_TO_ARM[condition]
        real_ate_pp = real["effects"][arm]["ate"] * 100
        sim_ate_pp = ate_prob[condition]["ate_pp"]
        per_arm[arm] = {
            "real_ate_pp": real_ate_pp,
            "sim_ate_pp": sim_ate_pp,
            "sim_ate_binary_pp": ate_binary[condition]["ate_pp"],
            "sim_ci_low_pp": ate_prob[condition]["ci_low_pp"],
            "sim_ci_high_pp": ate_prob[condition]["ci_high_pp"],
            "ate_abs_error_pp": abs(sim_ate_pp - real_ate_pp),
            "sign_agreement": bool(np.sign(sim_ate_pp) == np.sign(real_ate_pp)),
        }

    baseline_error = abs(sim_prob_rates["none"] - real_rates[ARM_CONTROL]) * 100

    # Anchored: rescale so the simulated control equals the real control rate.
    anchor = real_rates[ARM_CONTROL] / sim_prob_rates["none"] if sim_prob_rates["none"] else np.nan
    anchored_sim = sim.copy()
    anchored_sim["visit_probability"] = (anchored_sim["visit_probability"] * anchor).clip(0, 1)
    anchored_ate = sim_ate(anchored_sim, "visit_probability", seed=seed)
    anchored = {
        CONDITION_TO_ARM[c]: {
            "sim_ate_pp": anchored_ate[c]["ate_pp"],
            "ate_abs_error_pp": abs(
                anchored_ate[c]["ate_pp"] - real["effects"][CONDITION_TO_ARM[c]]["ate"] * 100
            ),
        }
        for c in EMAIL_CONDITIONS
    }

    segments = segment_correlation(sim, covariates, real_segments, min_n=min_segment_n)
    mechanism = mechanism_plausibility(
        sim.merge(covariates[["customer_id", *SEGMENT_VARS]], on="customer_id", how="left"),
        real_segments,
    )

    return {
        "n_customers": int(sim["customer_id"].nunique()),
        "n_responses": int(len(sim)),
        "real_arm_visit_rates": real_rates,
        "sim_arm_rates_probability": sim_by_arm,
        "sim_arm_rates_binary": {CONDITION_TO_ARM[c]: v for c, v in sim_binary_rates.items()},
        "arm_ranking_match": bool(real_order == sim_order),
        "real_ranking": real_order,
        "sim_ranking": sim_order,
        "kendall_tau": float(tau),
        "kendall_tau_p": float(tau_p),
        "per_arm": per_arm,
        "primary_metric_mean_ate_abs_error_pp": float(
            np.mean([v["ate_abs_error_pp"] for v in per_arm.values()])
        ),
        "baseline_calibration_error_pp": float(baseline_error),
        "segment_lift_spearman": segments,
        "mechanism_plausibility": mechanism,
        "persona_match_effects": persona_match_effects(sim, covariates),
        "anchored_to_real_control": {
            "note": "Uses the real control rate, so it is NOT part of the blind prediction.",
            "scale_factor": float(anchor),
            "per_arm": anchored,
            "mean_ate_abs_error_pp": float(
                np.mean([v["ate_abs_error_pp"] for v in anchored.values()])
            ),
        },
    }


def save(report: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float))
    return path
