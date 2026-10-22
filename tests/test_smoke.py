"""End-to-end smoke test: ten customers through every CLI step on the mock provider."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from agentlift import fidelity, lock
from agentlift.data import ARM_CONTROL, ARM_MENS, ARM_WOMENS, Dataset
from agentlift.simulate import SimConfig, read_jsonl, run_simulation, write_jsonl

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def tiny_dataset() -> Dataset:
    rows = []
    for i in range(10):
        rows.append(
            {
                "customer_id": i,
                "recency": 1 + i % 12,
                "history_segment": ["1) $0 - $100", "3) $200 - $350"][i % 2],
                "history": 50.0 + 90 * i,
                "mens": i % 2,
                "womens": (i + 1) % 2,
                "zip_code": ["Rural", "Urban", "Surburban"][i % 3],
                "newbie": i % 2,
                "channel": ["Web", "Phone", "Multichannel"][i % 3],
                "buyer_category": ["mens only", "womens only"][i % 2],
                "segment": [ARM_CONTROL, ARM_MENS, ARM_WOMENS][i % 3],
                "visit": i % 4 == 0,
                "conversion": 0,
                "spend": 0.0,
            }
        )
    return Dataset(frame=pd.DataFrame(rows))


@pytest.fixture
def real_stub() -> tuple[dict, pd.DataFrame]:
    real = {
        "arm_visit_rates": {ARM_CONTROL: 0.106, ARM_MENS: 0.183, ARM_WOMENS: 0.151},
        "effects": {ARM_MENS: {"ate": 0.0766}, ARM_WOMENS: {"ate": 0.0452}},
    }
    segments = pd.DataFrame(
        [
            {"segment_var": "zip_code", "segment_level": level, "arm": arm, "lift_pp": lift}
            for level, lift in (("Rural", 7.3), ("Urban", 6.1), ("Surburban", 7.4))
            for arm, _ in ((ARM_MENS, 0), (ARM_WOMENS, 0))
        ]
    )
    return real, segments


def test_simulate_lock_evaluate_end_to_end(tmp_path, tiny_dataset, real_stub, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, real_segments = real_stub
    config = SimConfig(provider="mock", model="mock-large", seed=42)

    sim = run_simulation(tiny_dataset, tiny_dataset.frame, config, requests_per_minute=6000)
    assert len(sim) == 30
    assert set(sim["condition"]) == {"mens", "womens", "none"}

    sim_path = write_jsonl(sim, tmp_path / "results" / "mock" / "sim_main_quick.jsonl")
    out_dir = sim_path.parent

    with pytest.raises(lock.LockError):
        lock.verify(sim_path, out_dir, "quick")

    lock.write(sim_path, out_dir, "quick", {"provider": "mock"})
    lock.verify(sim_path, out_dir, "quick")

    report = fidelity.evaluate(
        read_jsonl(sim_path), real, real_segments, tiny_dataset.frame, min_segment_n=1
    )
    assert report["n_customers"] == 10
    assert set(report["per_arm"]) == {ARM_MENS, ARM_WOMENS}
    assert 0.0 <= report["mechanism_plausibility"]["reason_citation_share"] <= 1.0

    path = fidelity.save(report, out_dir / "fidelity_quick.json")
    assert json.loads(path.read_text())["arm_ranking_match"] in (True, False)


def test_simulation_is_deterministic_for_a_seed(tmp_path, tiny_dataset, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = SimConfig(provider="mock", seed=42)
    first = run_simulation(tiny_dataset, tiny_dataset.frame, config, requests_per_minute=6000)
    (tmp_path / "data" / "raw" / "llm_cache.sqlite").unlink()
    second = run_simulation(tiny_dataset, tiny_dataset.frame, config, requests_per_minute=6000)
    pd.testing.assert_series_equal(first["visit_probability"], second["visit_probability"])
