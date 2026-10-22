"""AgentLift dashboard. Reads results/ only; no API key and no network needed."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REAL_DIR = Path("results")
MOCK_DIR = REAL_DIR / "mock"
ARM_ORDER = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]


def _first_existing(name: str) -> tuple[Path | None, bool]:
    """Prefer real results; fall back to mock and say so."""
    real = REAL_DIR / name
    if real.exists():
        return real, False
    mock = MOCK_DIR / name
    if mock.exists():
        return mock, True
    return None, False


@st.cache_data
def load_results(mode: str) -> dict:
    out: dict = {"is_mock": False}
    real_path = REAL_DIR / "real_experiment.json"
    out["real"] = json.loads(real_path.read_text()) if real_path.exists() else None
    seg_path = REAL_DIR / "real_segments.csv"
    out["real_segments"] = pd.read_csv(seg_path) if seg_path.exists() else None

    for key, name in (
        ("fidelity", f"fidelity_{mode}.json"),
        ("sim", f"sim_main_{mode}.jsonl"),
        ("ablations", f"ablations_{mode}.csv"),
    ):
        path, is_mock = _first_existing(name)
        out["is_mock"] = out["is_mock"] or is_mock
        if path is None:
            out[key] = None
        elif path.suffix == ".json":
            out[key] = json.loads(path.read_text())
        elif path.suffix == ".csv":
            out[key] = pd.read_csv(path)
        else:
            out[key] = pd.DataFrame(
                [json.loads(line) for line in path.read_text().splitlines() if line]
            )
    return out


def tab_real(data: dict) -> None:
    real = data["real"]
    if real is None:
        st.info("Run `python -m agentlift.cli real` to populate this tab.")
        return
    cols = st.columns(3)
    for col, arm in zip(cols, ARM_ORDER, strict=False):
        col.metric(arm, f"{real['arm_visit_rates'][arm] * 100:.2f}%", f"n={real['arm_sizes'][arm]:,}")

    fig = go.Figure()
    for arm, eff in real["effects"].items():
        fig.add_trace(
            go.Bar(
                x=[arm],
                y=[eff["ate"] * 100],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [(eff["ci_high"] - eff["ate"]) * 100],
                    "arrayminus": [(eff["ate"] - eff["ci_low"]) * 100],
                },
                name=arm,
            )
        )
    fig.update_layout(title="Real ATE vs No E-Mail (percentage points, 95% CI)", yaxis_title="pp")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Randomization balance")
    balance = pd.DataFrame(real["balance"]["by_covariate"])
    st.write(
        f"Max |SMD| {real['balance']['max_abs_smd']:.3f}; "
        f"{real['balance']['n_imbalanced']} of {len(balance)} comparisons exceed "
        f"{real['balance']['threshold']}."
    )
    st.dataframe(balance, width="stretch", hide_index=True)

    st.subheader("Power")
    st.dataframe(pd.DataFrame(real["power"]).T, width="stretch")
    st.caption(
        "Conversion is reported for the real data only: it is too rare to measure in a small "
        "simulated sample."
    )
    st.write({k: round(v * 100, 3) for k, v in real["arm_conversion_rates"].items()})


def tab_simulation(data: dict) -> None:
    fid = data["fidelity"]
    if fid is None:
        st.info("Run `simulate`, `lock` and `evaluate` to populate this tab.")
        return
    st.metric(
        "Mean absolute ATE error",
        f"{fid['primary_metric_mean_ate_abs_error_pp']:.2f} pp",
        "ranking match" if fid["arm_ranking_match"] else "ranking mismatch",
    )
    real_rates = fid["real_arm_visit_rates"]
    sim_rates = fid["sim_arm_rates_probability"]
    fig = go.Figure(
        [
            go.Bar(name="Real", x=ARM_ORDER, y=[real_rates[a] * 100 for a in ARM_ORDER]),
            go.Bar(name="Simulated", x=ARM_ORDER, y=[sim_rates[a] * 100 for a in ARM_ORDER]),
        ]
    )
    fig.update_layout(barmode="group", title="Visit rate by arm (%)", yaxis_title="%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(fid["per_arm"]).T, width="stretch")
    st.caption(
        "Anchored error (rescales the simulation with the real control rate, so not a blind "
        f"prediction): {fid['anchored_to_real_control']['mean_ate_abs_error_pp']:.2f} pp."
    )


def tab_segments(data: dict) -> None:
    fid = data["fidelity"]
    if fid is None:
        st.info("Run `evaluate` to populate this tab.")
        return
    pairs = pd.DataFrame(fid["segment_lift_spearman"]["pairs"])
    if pairs.empty:
        st.info("No segment reached the minimum simulated sample size.")
        return
    fig = go.Figure()
    for condition, group in pairs.groupby("condition"):
        fig.add_trace(
            go.Scatter(
                x=group["real_lift_pp"],
                y=group["sim_lift_pp"],
                mode="markers",
                name=condition,
                text=group["segment_var"] + "=" + group["segment_level"],
            )
        )
    low = float(min(pairs["real_lift_pp"].min(), pairs["sim_lift_pp"].min()))
    high = float(max(pairs["real_lift_pp"].max(), pairs["sim_lift_pp"].max()))
    fig.add_trace(
        go.Scatter(x=[low, high], y=[low, high], mode="lines", name="y = x", line={"dash": "dash"})
    )
    fig.update_layout(
        title=f"Segment lift, simulated vs real (Spearman rho "
        f"{fid['segment_lift_spearman']['spearman_rho']:.2f})",
        xaxis_title="Real lift (pp)",
        yaxis_title="Simulated lift (pp)",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pairs, width="stretch", hide_index=True)


def tab_ablations(data: dict) -> None:
    abl = data["ablations"]
    if abl is None:
        st.info("Run `ablate` to populate this tab.")
        return
    st.dataframe(abl, width="stretch", hide_index=True)
    fig = go.Figure(
        [go.Bar(x=abl["ablation"], y=abl["mean_ate_abs_error_pp"], text=abl["variant"])]
    )
    fig.update_layout(
        title="Primary metric by variant (lower is better)",
        yaxis_title="Mean |ATE error| (pp)",
    )
    st.plotly_chart(fig, width="stretch")


def tab_reasons(data: dict) -> None:
    sim = data["sim"]
    if sim is None:
        st.info("Run `simulate` to populate this tab.")
        return
    condition = st.selectbox("Condition", ["all", *sorted(sim["condition"].unique())])
    visited = st.selectbox("Visit", ["all", "true", "false"])
    view = sim
    if condition != "all":
        view = view[view["condition"] == condition]
    if visited != "all":
        view = view[view["visit"] == (visited == "true")]
    search = st.text_input("Search reasons")
    if search:
        view = view[view["reason"].str.contains(search, case=False, na=False)]
    st.write(f"{len(view)} responses")
    st.dataframe(
        view[["customer_id", "condition", "visit", "visit_probability", "reason"]],
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="AgentLift", layout="wide")
    st.title("AgentLift")
    st.caption(
        "LLM customer agents simulate a real randomized A/B test; the tabs below measure how "
        "faithfully they reproduce it."
    )
    mode = st.sidebar.radio("Run mode", ["quick", "full"], index=0)
    data = load_results(mode)
    if data["is_mock"]:
        st.error(
            "MOCK DATA — these simulation numbers come from the deterministic offline provider "
            "in `results/mock/` and mean nothing about real agent behaviour."
        )
    tabs = st.tabs(
        ["Real A/B test", "Simulation vs real", "Segments", "Ablations", "Agent reasons"]
    )
    for tab, render in zip(
        tabs, (tab_real, tab_simulation, tab_segments, tab_ablations, tab_reasons), strict=True
    ):
        with tab:
            render(data)


main()
