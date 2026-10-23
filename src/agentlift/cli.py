"""Command line entry point: python -m agentlift.cli {real,simulate,lock,evaluate,ablate,report,all}."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from . import ablations, experiment, fidelity, lock, report, tracking
from .data import load
from .simulate import (
    SAMPLE_SIZES,
    SimConfig,
    main_sim_path,
    output_dir,
    read_jsonl,
    resolve_models,
    run_simulation,
    sample_customers,
    write_jsonl,
)

CONFIG_PATH = Path("config.yaml")
REAL_JSON = Path("results/real_experiment.json")
REAL_SEGMENTS = Path("results/real_segments.csv")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _mode(args) -> str:
    return "quick" if args.quick else "full"


def _sizes(cfg: dict, mode: str) -> dict:
    return cfg.get("sample", {}).get(mode, SAMPLE_SIZES[mode])


def _sim_config(cfg: dict, args, models: tuple[str, str]) -> SimConfig:
    sim = cfg.get("simulation", {})
    return SimConfig(
        provider=args.provider,
        model=models[0],
        model_size="large",
        persona_level=sim.get("persona_level", "full"),
        framing=sim.get("framing", "neutral"),
        temperature=float(sim.get("temperature", 0.7)),
        seed=int(cfg.get("seed", 42)),
    )


def _load_real() -> tuple[dict, pd.DataFrame]:
    if not REAL_JSON.exists():
        raise SystemExit("run `real` first: results/real_experiment.json is missing")
    return json.loads(REAL_JSON.read_text()), pd.read_csv(REAL_SEGMENTS)


def cmd_real(args) -> None:
    cfg = load_config()
    ds = load()
    results = experiment.run(
        ds, outcome=cfg.get("data", {}).get("outcome", "visit"), with_t_learner=not args.no_t_learner
    )
    paths = experiment.save(results)
    with tracking.run("real_experiment", {"n": results.overall["n_total"]}, args.track) as active:
        tracking.log_metrics(
            {
                f"real_ate_pp[{arm}]": v["ate"] * 100
                for arm, v in results.overall["effects"].items()
            },
            active,
        )
        for path in paths:
            tracking.log_artifact(path, active)
    for arm, eff in results.overall["effects"].items():
        print(
            f"[real] {arm}: {eff['ate'] * 100:+.2f} pp "
            f"(95% CI {eff['ci_low'] * 100:+.2f} to {eff['ci_high'] * 100:+.2f}, p={eff['p_value']:.2e})"
        )
    print(f"[real] max |SMD| {results.overall['balance']['max_abs_smd']:.3f}; wrote {paths[0]}")


def cmd_simulate(args) -> None:
    cfg = load_config()
    mode = _mode(args)
    ds = load()
    models = resolve_models(args.provider, seed=int(cfg.get("seed", 42)))
    config = _sim_config(cfg, args, models)
    sizes = _sizes(cfg, mode)
    customers = sample_customers(ds, sizes["main"], seed=config.seed)
    llm_cfg = cfg.get("llm", {})
    sim = run_simulation(
        ds,
        customers,
        config,
        concurrency=int(llm_cfg.get("concurrency", 8)),
        requests_per_minute=int(llm_cfg.get("requests_per_minute", 28)),
    )
    path = write_jsonl(sim, main_sim_path(mode, args.provider))
    with tracking.run(f"simulate_{mode}", asdict(config), args.track) as active:
        tracking.log_metrics(
            {f"sim_rate[{c}]": float(g["visit_probability"].mean()) for c, g in sim.groupby("condition")},
            active,
        )
        tracking.log_artifact(path, active)
    print(f"[simulate] {len(customers)} customers x 3 conditions -> {path}")


def cmd_lock(args) -> None:
    cfg = load_config()
    mode = _mode(args)
    out = output_dir(args.provider)
    sim_path = main_sim_path(mode, args.provider)
    models = resolve_models(args.provider, seed=int(cfg.get("seed", 42)))
    path = lock.write(sim_path, out, mode, asdict(_sim_config(cfg, args, models)))
    print(f"[lock] {path} -> {json.loads(path.read_text())['sha256'][:16]}")


def cmd_evaluate(args) -> None:
    cfg = load_config()
    mode = _mode(args)
    out = output_dir(args.provider)
    sim_path = main_sim_path(mode, args.provider)
    locked = lock.verify(sim_path, out, mode)
    real, real_segments = _load_real()
    ds = load()
    sim = read_jsonl(sim_path)
    report_dict = fidelity.evaluate(
        sim,
        real,
        real_segments,
        ds.frame,
        seed=int(cfg.get("seed", 42)),
        min_segment_n=int(cfg.get("fidelity", {}).get("min_segment_n", 15)),
    )
    report_dict["lock"] = locked
    path = fidelity.save(report_dict, out / f"fidelity_{mode}.json")
    with tracking.run(f"evaluate_{mode}", locked["config"], args.track) as active:
        tracking.log_metrics(tracking.flat_metrics(report_dict), active)
        tracking.log_artifact(path, active)
    print(
        f"[evaluate] ranking match: {report_dict['arm_ranking_match']}; "
        f"mean |ATE error|: {report_dict['primary_metric_mean_ate_abs_error_pp']:.2f} pp; "
        f"baseline error: {report_dict['baseline_calibration_error_pp']:.2f} pp -> {path}"
    )


def cmd_ablate(args) -> None:
    cfg = load_config()
    mode = _mode(args)
    ds = load()
    real, real_segments = _load_real()
    models = resolve_models(args.provider, seed=int(cfg.get("seed", 42)))
    config = _sim_config(cfg, args, models)
    sizes = _sizes(cfg, mode)
    main = sample_customers(ds, sizes["main"], seed=config.seed)
    subset = main.head(sizes["ablation"])
    llm_cfg = cfg.get("llm", {})
    table, summary = ablations.run_ablations(
        ds,
        subset,
        config,
        real,
        real_segments,
        models,
        concurrency=int(llm_cfg.get("concurrency", 8)),
        requests_per_minute=int(llm_cfg.get("requests_per_minute", 28)),
        min_segment_n=int(cfg.get("fidelity", {}).get("min_segment_n", 15)),
    )
    csv_path, json_path = ablations.save(table, summary, args.provider, mode)
    for _, row in table.iterrows():
        with tracking.run(f"ablation_{row['ablation']}_{mode}", {"config": row["config"]}, args.track) as active:
            tracking.log_metrics(
                {
                    "mean_ate_abs_error_pp": row["mean_ate_abs_error_pp"],
                    "segment_spearman_rho": row["segment_spearman_rho"],
                },
                active,
            )
            tracking.log_artifact(csv_path, active)
    print(table.to_string(index=False))
    print(f"[ablate] {ablations.spread_note(summary)} -> {csv_path}")


def cmd_report(args) -> None:
    mode = _mode(args)
    paths = report.generate(mode=mode)
    for path in paths:
        print(f"[report] wrote {path}")


def cmd_all(args) -> None:
    cmd_real(args)
    cmd_simulate(args)
    cmd_lock(args)
    cmd_evaluate(args)
    cmd_ablate(args)
    if args.provider != "mock":
        cmd_report(args)
    else:
        print("[all] skipping report: mock numbers never reach committed documents")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--quick", action="store_true", help="small samples (100 main / 30 ablation)")
    common.add_argument("--provider", default="mock", choices=["mock", "groq", "gemini"])
    common.add_argument("--track", action="store_true", help="log the run to MLflow")
    common.add_argument("--no-t-learner", action="store_true", help="skip the T-learner CATE model")

    parser = argparse.ArgumentParser(prog="agentlift", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (
        ("real", cmd_real),
        ("simulate", cmd_simulate),
        ("lock", cmd_lock),
        ("evaluate", cmd_evaluate),
        ("ablate", cmd_ablate),
        ("report", cmd_report),
        ("all", cmd_all),
    ):
        sub.add_parser(name, parents=[common]).set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
