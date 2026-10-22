"""Run LLM customer agents under each experimental condition."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .data import Dataset
from .llm import AgentRunner, CallSpec, Provider, ResponseCache, get_provider
from .persona import Condition, Framing, PersonaLevel, build_prompt

CONDITIONS: tuple[Condition, ...] = ("mens", "womens", "none")
RESULTS_DIR = Path("results")
MOCK_DIR = RESULTS_DIR / "mock"

SAMPLE_SIZES = {
    "quick": {"main": 100, "ablation": 30},
    "full": {"main": 300, "ablation": 60},
}


@dataclass(frozen=True)
class SimConfig:
    """One simulation configuration; every field is part of the cache key."""

    provider: str = "mock"
    model: str = "mock-large"
    model_size: str = "large"
    persona_level: PersonaLevel = "full"
    framing: Framing = "neutral"
    temperature: float = 0.7
    seed: int = 42

    def label(self) -> str:
        return (
            f"{self.model_size}/{self.persona_level}/{self.framing}/t{self.temperature:g}"
        )


def output_dir(provider: str) -> Path:
    """Mock runs are kept apart so their numbers never reach the committed results."""
    return MOCK_DIR if provider == "mock" else RESULTS_DIR


def sample_customers(ds: Dataset, n: int, seed: int = 42, stratify_by: str = "history_segment") -> pd.DataFrame:
    """Draw a stratified sample of customers, proportional to the strata sizes."""
    frame = ds.frame
    fraction = n / len(frame)
    parts = [
        group.sample(max(1, round(len(group) * fraction)), random_state=seed)
        for _, group in frame.groupby(stratify_by, observed=True)
    ]
    sampled = pd.concat(parts).sort_values("customer_id")
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed).sort_values("customer_id")
    return sampled.reset_index(drop=True)


async def _run_async(
    customers: pd.DataFrame,
    covariates: list[str],
    config: SimConfig,
    provider: Provider,
    cache: ResponseCache,
    concurrency: int,
    requests_per_minute: int,
) -> list[dict]:
    runner = AgentRunner(
        provider,
        cache=cache,
        concurrency=concurrency,
        requests_per_minute=requests_per_minute,
    )
    rows = customers[[*covariates, "customer_id"]].to_dict(orient="records")

    async def one(row: dict, condition: Condition) -> dict:
        customer_id = int(row["customer_id"])
        prompt = build_prompt(row, condition, level=config.persona_level, framing=config.framing)
        spec = CallSpec(
            provider=config.provider,
            model=config.model,
            temperature=config.temperature,
            persona_level=config.persona_level,
            framing=config.framing,
            condition=condition,
            customer_id=customer_id,
            seed=config.seed,
        )
        response = await runner.call(prompt, spec)
        return {
            "customer_id": customer_id,
            "condition": condition,
            **asdict(config),
            "config_label": config.label(),
            "visit": bool(response.visit),
            "visit_probability": float(response.visit_probability),
            "reason": response.reason,
        }

    tasks = [one(row, condition) for row in rows for condition in CONDITIONS]
    results = await asyncio.gather(*tasks)
    results = sorted(results, key=lambda r: (r["customer_id"], r["condition"]))
    print(
        f"[simulate] {config.label()}: {runner.stats.provider_calls} provider calls, "
        f"{runner.stats.cache_hits} cache hits, {runner.stats.retries} retries"
    )
    return results


def run_simulation(
    ds: Dataset,
    customers: pd.DataFrame,
    config: SimConfig,
    cache: ResponseCache | None = None,
    concurrency: int = 8,
    requests_per_minute: int = 28,
) -> pd.DataFrame:
    """Run every sampled customer under all three conditions (within-subject)."""
    provider = get_provider(config.provider, seed=config.seed)
    if config.provider == "mock":
        # The mock provider makes no network calls, so the RPM limit would only waste time.
        requests_per_minute = 1_000_000
    owns_cache = cache is None
    cache = cache or ResponseCache()
    try:
        rows = asyncio.run(
            _run_async(
                customers,
                ds.covariates,
                config,
                provider,
                cache,
                concurrency,
                requests_per_minute,
            )
        )
    finally:
        if owns_cache:
            cache.close()
    return pd.DataFrame(rows)


def write_jsonl(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(json.dumps(record) + "\n")
    return path


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in Path(path).read_text().splitlines() if line])


def main_sim_path(mode: str, provider: str) -> Path:
    return output_dir(provider) / f"sim_main_{mode}.jsonl"


def resolve_models(config_provider: str, seed: int = 42) -> tuple[str, str]:
    """Current (large, small) model ids for the provider."""
    return get_provider(config_provider, seed=seed).models()
