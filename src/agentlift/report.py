"""Generate the findings memo and the README results table from real results files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import ARM_CONTROL, ARM_MENS, ARM_WOMENS

RESULTS = Path("results")
REPORTS = Path("reports")
README = Path("README.md")
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


class MissingResults(RuntimeError):
    """Raised when a real results file the report needs is absent."""


def _require(path: Path) -> Path:
    if not path.exists():
        raise MissingResults(f"missing real results file: {path}")
    return path


def _load(mode: str) -> tuple[dict, dict, pd.DataFrame]:
    real = json.loads(_require(RESULTS / "real_experiment.json").read_text())
    fidelity = json.loads(_require(RESULTS / f"fidelity_{mode}.json").read_text())
    ablations = pd.read_csv(_require(RESULTS / f"ablations_{mode}.csv"))
    return real, fidelity, ablations


def results_table(real: dict, fid: dict) -> str:
    lines = [
        "| Arm | Real ATE (pp) | Real 95% CI | Simulated ATE (pp) | Absolute error (pp) | Sign agrees |",
        "| --- | ---: | :---: | ---: | ---: | :---: |",
    ]
    for arm in (ARM_MENS, ARM_WOMENS):
        eff = real["effects"][arm]
        sim = fid["per_arm"][arm]
        lines.append(
            f"| {arm} | {eff['ate'] * 100:+.2f} | "
            f"{eff['ci_low'] * 100:+.2f} to {eff['ci_high'] * 100:+.2f} | "
            f"{sim['sim_ate_pp']:+.2f} | {sim['ate_abs_error_pp']:.2f} | "
            f"{'yes' if sim['sign_agreement'] else 'no'} |"
        )
    return "\n".join(lines)


def headline_lines(real: dict, fid: dict, abl: pd.DataFrame) -> list[str]:
    seg = fid["segment_lift_spearman"]
    best = abl.loc[abl["mean_ate_abs_error_pp"].idxmin()]
    worst = abl.loc[abl["mean_ate_abs_error_pp"].idxmax()]
    return [
        f"- Agents: **{fid['n_customers']}** simulated customers, each run under all three conditions "
        f"({fid['n_responses']} agent responses).",
        f"- Winning arm: the simulation {'reproduces' if fid['arm_ranking_match'] else 'does not reproduce'} "
        f"the real ordering of the three arms (Kendall tau {fid['kendall_tau']:.2f}).",
        f"- Primary metric: mean absolute ATE error **{fid['primary_metric_mean_ate_abs_error_pp']:.2f} pp** "
        f"across the two e-mail arms.",
        f"- Baseline calibration: the simulated no-e-mail visit rate is "
        f"**{fid['baseline_calibration_error_pp']:.2f} pp** away from the real control rate of "
        f"{real['arm_visit_rates'][ARM_CONTROL] * 100:.2f}%.",
        f"- Segment fidelity: Spearman rho **{seg['spearman_rho']:.2f}** over {seg['n_segments_compared']} "
        f"segment-by-arm comparisons with at least {seg['min_segment_n']} simulated customers.",
        f"- Anchored to the real control rate (uses real data, so not a blind prediction): mean absolute ATE error "
        f"**{fid['anchored_to_real_control']['mean_ate_abs_error_pp']:.2f} pp**.",
        f"- Ablations: best variant `{best['ablation']}` at {best['mean_ate_abs_error_pp']:.2f} pp, "
        f"worst `{worst['ablation']}` at {worst['mean_ate_abs_error_pp']:.2f} pp.",
    ]


def memo(real: dict, fid: dict, abl: pd.DataFrame, mode: str) -> str:
    mech = fid["mechanism_plausibility"]
    best = abl.loc[abl["mean_ate_abs_error_pp"].idxmin()]
    worst = abl.loc[abl["mean_ate_abs_error_pp"].idxmax()]
    spread = abl["mean_ate_abs_error_pp"].std(ddof=1)
    reliable, unreliable = _reliability(fid)

    return "\n".join(
        [
            "# Can LLM agents stand in for a real A/B test?",
            "",
            "## Question",
            "",
            "Before running a real e-mail experiment, can a population of LLM customer agents tell you "
            "which arm wins and roughly how large the lift will be?",
            "",
            "## Setup",
            "",
            f"The real experiment is the Hillstrom MineThatData e-mail test: {real['n_total']:,} customers "
            f"randomized across {ARM_MENS}, {ARM_WOMENS} and {ARM_CONTROL}, with website visits over the "
            "following two weeks as the outcome. Randomization checks out (max |SMD| "
            f"{real['balance']['max_abs_smd']:.3f}, {real['balance']['n_imbalanced']} covariates over the "
            "0.1 threshold).",
            "",
            f"Each of {fid['n_customers']} sampled customers becomes an agent, built from covariates only, "
            "and is asked the same question once per condition in separate calls. Outcomes and the real arm "
            "label never enter a prompt. The simulation is hashed and locked before it is compared with the "
            "real results, so the prediction is blind.",
            "",
            "## Results",
            "",
            results_table(real, fid),
            "",
            *headline_lines(real, fid, abl),
            "",
            "## Where the simulation is reliable",
            "",
            *[f"- {line}" for line in reliable],
            "",
            "## Where it is not",
            "",
            *[f"- {line}" for line in unreliable],
            "",
            "## Recommendation",
            "",
            _recommendation(fid),
            "",
            "## Ablations",
            "",
            abl.drop(columns=[c for c in abl.columns if c.startswith("sim_ate_pp[")]).to_markdown(
                index=False, floatfmt=".3f"
            ),
            "",
            f"Spread of the primary metric across the {len(abl)} variants is {spread:.2f} pp, from "
            f"`{best['ablation']}` to `{worst['ablation']}`.",
            "",
            "## Limitations",
            "",
            f"- The simulation covers {fid['n_customers']} customers against 64,000 real ones, so its "
            "confidence intervals are wide.",
            "- One dataset, one domain. E-mail response is not evidence about pricing, UI or other channels.",
            "- The Hillstrom data is public and may sit in the model's training data, which would flatter "
            "these results rather than hurt them.",
            "- Anchored numbers rescale the simulation using the real control rate, so they are a "
            "diagnostic, not a prediction.",
            f"- Agent reasons were scored with a fixed keyword list; {mech['reason_citation_share'] * 100:.0f}% "
            "of visit reasons cite the e-mail or a persona feature.",
            "",
            f"_Mode: {mode}. Every number above comes from the committed files in `results/`._",
            "",
        ]
    )


def _reliability(fid: dict) -> tuple[list[str], list[str]]:
    reliable, unreliable = [], []
    target = reliable if fid["arm_ranking_match"] else unreliable
    target.append(
        f"Arm ranking: the simulated ordering {'matches' if fid['arm_ranking_match'] else 'differs from'} "
        f"the real one, so picking the winner {'is' if fid['arm_ranking_match'] else 'is not'} safe here."
    )
    signs = all(v["sign_agreement"] for v in fid["per_arm"].values())
    (reliable if signs else unreliable).append(
        f"Direction of effect: every arm's simulated sign {'agrees' if signs else 'does not agree'} with reality."
    )
    error = fid["primary_metric_mean_ate_abs_error_pp"]
    (reliable if error < 3 else unreliable).append(
        f"Effect magnitude: mean absolute ATE error is {error:.2f} pp, "
        f"{'close enough to size an experiment' if error < 3 else 'too large to size an experiment from'}."
    )
    base = fid["baseline_calibration_error_pp"]
    (reliable if base < 5 else unreliable).append(
        f"Baseline level: the simulated no-e-mail rate is off by {base:.2f} pp"
        f"{'' if base < 5 else ', so absolute rates cannot be read off the simulation'}."
    )
    rho = fid["segment_lift_spearman"]["spearman_rho"]
    (reliable if rho > 0.3 else unreliable).append(
        f"Segment ordering: Spearman rho {rho:.2f} against the real segment lifts."
    )
    return reliable, unreliable


def _recommendation(fid: dict) -> str:
    error = fid["primary_metric_mean_ate_abs_error_pp"]
    if fid["arm_ranking_match"] and error < 3:
        return (
            "Use the simulation to shortlist which arm to run and to rule out variants, then size the "
            "real experiment from the real control rate. Do not quote simulated absolute rates to "
            "stakeholders."
        )
    if fid["arm_ranking_match"]:
        return (
            f"Use the simulation for direction only. It picked the winning arm but missed the effect size "
            f"by {error:.2f} pp on average, which is too coarse for powering a test. Treat it as a "
            "hypothesis filter before the real experiment, never as a substitute for it."
        )
    return (
        f"Do not use this simulation to choose an arm. It got the ordering wrong and missed the effect "
        f"size by {error:.2f} pp on average. Its value here is as a prompt-and-persona testbed, not a "
        "prediction."
    )


def update_readme(real: dict, fid: dict, abl: pd.DataFrame, path: Path = README) -> Path:
    body = "\n".join(
        [
            START,
            "## Results",
            "",
            results_table(real, fid),
            "",
            *headline_lines(real, fid, abl),
            "",
            "Full write-up: [reports/findings.md](reports/findings.md).",
            END,
        ]
    )
    text = path.read_text()
    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        text = head + body + tail
    else:
        text = text.rstrip() + "\n\n" + body + "\n"
    path.write_text(text)
    return path


def generate(mode: str = "quick") -> list[Path]:
    """Write reports/findings.md and the README results section from results/."""
    real, fid, abl = _load(mode)
    REPORTS.mkdir(parents=True, exist_ok=True)
    memo_path = REPORTS / "findings.md"
    memo_path.write_text(memo(real, fid, abl, mode))
    readme_path = update_readme(real, fid, abl)
    return [memo_path, readme_path]
