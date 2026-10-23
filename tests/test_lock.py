import json

import pytest

from agentlift import lock


def _sim(tmp_path):
    path = tmp_path / "sim_main_quick.jsonl"
    path.write_text(json.dumps({"customer_id": 1, "condition": "mens"}) + "\n")
    return path


def test_evaluate_refuses_when_lock_is_missing(tmp_path):
    with pytest.raises(lock.LockError, match="no prediction lock"):
        lock.verify(_sim(tmp_path), tmp_path, "quick")


def test_lock_round_trips(tmp_path):
    sim = _sim(tmp_path)
    lock.write(sim, tmp_path, "quick", {"provider": "mock"})
    payload = lock.verify(sim, tmp_path, "quick")
    assert payload["n_rows"] == 1
    assert payload["config"] == {"provider": "mock"}
    assert payload["locked_at_utc"].endswith("+00:00")


def test_evaluate_refuses_on_hash_mismatch(tmp_path):
    sim = _sim(tmp_path)
    lock.write(sim, tmp_path, "quick", {})
    sim.write_text(json.dumps({"customer_id": 2, "condition": "none"}) + "\n")
    with pytest.raises(lock.LockError, match="has changed since it was locked"):
        lock.verify(sim, tmp_path, "quick")


def test_cannot_lock_a_missing_simulation(tmp_path):
    with pytest.raises(lock.LockError, match="does not exist"):
        lock.write(tmp_path / "nope.jsonl", tmp_path, "quick", {})
