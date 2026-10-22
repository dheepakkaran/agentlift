import asyncio
import json

import pytest

from agentlift.llm import AgentRunner, CallSpec, Provider, ResponseCache, parse_response

SPEC = CallSpec(
    provider="mock",
    model="mock-large",
    temperature=0.0,
    persona_level="full",
    framing="neutral",
    condition="mens",
    customer_id=1,
    seed=42,
)


def test_valid_json_parses():
    response = parse_response('{"visit": true, "visit_probability": 0.4, "reason": "ok"}')
    assert response.visit is True
    assert response.visit_probability == 0.4


def test_json_wrapped_in_prose_still_parses():
    response = parse_response('Sure!\n{"visit": false, "visit_probability": 0.1, "reason": "no"}\n')
    assert response.visit is False


def test_invalid_json_is_rejected():
    with pytest.raises(ValueError):
        parse_response("not json at all")


def test_probability_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        parse_response('{"visit": true, "visit_probability": 1.7, "reason": "too high"}')
    with pytest.raises(ValueError):
        parse_response('{"visit": true, "visit_probability": -0.2, "reason": "negative"}')


def test_reason_is_capped_at_40_words():
    long_reason = " ".join(["word"] * 80)
    response = parse_response(
        json.dumps({"visit": True, "visit_probability": 0.5, "reason": long_reason})
    )
    assert len(response.reason.split()) == 40


class _FlakyProvider(Provider):
    """Returns junk once, then a valid payload."""

    name = "flaky"

    def __init__(self):
        self.calls = 0

    async def complete(self, prompt, model, temperature, spec):
        self.calls += 1
        if self.calls == 1:
            return "I cannot answer that."
        return '{"visit": true, "visit_probability": 0.33, "reason": "fine"}'


class _CountingProvider(Provider):
    name = "counting"

    def __init__(self):
        self.calls = 0

    async def complete(self, prompt, model, temperature, spec):
        self.calls += 1
        return '{"visit": true, "visit_probability": 0.25, "reason": "counted"}'


def test_invalid_output_triggers_a_retry(tmp_path):
    provider = _FlakyProvider()
    runner = AgentRunner(provider, cache=ResponseCache(tmp_path / "c.sqlite"), requests_per_minute=6000)
    response = asyncio.run(runner.call("prompt", SPEC))
    assert provider.calls == 2
    assert response.visit_probability == 0.33
    runner.close()


def test_second_identical_call_makes_zero_provider_calls(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    provider = _CountingProvider()
    runner = AgentRunner(provider, cache=cache, requests_per_minute=6000)
    asyncio.run(runner.call("prompt", SPEC))
    assert provider.calls == 1

    second = AgentRunner(_CountingProvider(), cache=cache, requests_per_minute=6000)
    asyncio.run(second.call("prompt", SPEC))
    assert second.provider.calls == 0
    assert second.stats.cache_hits == 1
    second.close()


def test_cache_key_changes_with_every_field():
    base = SPEC.key()
    for field, value in [
        ("provider", "groq"),
        ("model", "other"),
        ("temperature", 0.7),
        ("persona_level", "minimal"),
        ("framing", "realism"),
        ("condition", "none"),
        ("customer_id", 2),
        ("seed", 7),
    ]:
        assert CallSpec(**{**SPEC.__dict__, field: value}).key() != base
