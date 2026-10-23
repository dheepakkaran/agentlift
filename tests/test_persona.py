import pytest

from agentlift.persona import FORBIDDEN_FIELDS, build_persona, build_prompt

SENTINEL = "ZZLEAKZZ"

ROW = {
    "customer_id": 7,
    "recency": 6,
    "history_segment": "3) $200 - $350",
    "history": 329.08,
    "mens": 1,
    "womens": 1,
    "zip_code": "Rural",
    "newbie": 1,
    "channel": "Web",
    "buyer_category": "both",
}


def test_persona_never_contains_outcomes_or_arm():
    row = dict(ROW)
    leaky = {f: SENTINEL for f in FORBIDDEN_FIELDS}
    row.update(leaky)
    with pytest.raises(ValueError):
        build_persona(row)


def test_prompt_built_from_clean_row_has_no_sentinel_or_arm_label():
    for condition in ("mens", "womens", "none"):
        prompt = build_prompt(ROW, condition)
        assert SENTINEL not in prompt
        for label in ("Mens E-Mail", "Womens E-Mail", "No E-Mail", "conversion", "spend"):
            assert label not in prompt


def test_prompt_never_reveals_the_experiment():
    prompt = build_prompt(ROW, "mens").lower()
    for banned in ("experiment", "a/b", "base rate", "hillstrom", "minethatdata", "dataset", "2008"):
        assert banned not in prompt


def test_minimal_persona_drops_purchase_memory():
    minimal = build_persona(ROW, level="minimal")
    full = build_persona(ROW, level="full")
    assert "329" in full and "329" not in minimal
    assert "rural" in minimal.lower()


def test_each_condition_is_isolated():
    """The agent sees exactly one condition line and no trace of the others."""
    mens = build_prompt(ROW, "mens")
    assert "email from the retailer featuring men's merchandise" in mens
    assert "email from the retailer featuring women's merchandise" not in mens
    assert "not received any email" not in mens

    none = build_prompt(ROW, "none")
    assert "not received any email" in none
    assert "you received an email" not in none
