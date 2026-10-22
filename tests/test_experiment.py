import numpy as np
import pandas as pd
import pytest

from agentlift.data import ARM_CONTROL, ARM_MENS, ARM_WOMENS, Dataset
from agentlift.experiment import arm_effects, standardized_mean_differences


def _synthetic(true_lift: float, n: int = 20000, seed: int = 0) -> Dataset:
    """Balanced three-arm trial where the Mens arm has a known lift on visit."""
    rng = np.random.default_rng(seed)
    arms = np.repeat([ARM_CONTROL, ARM_MENS, ARM_WOMENS], n)
    base = 0.10
    rates = {ARM_CONTROL: base, ARM_MENS: base + true_lift, ARM_WOMENS: base + true_lift / 2}
    visit = np.array([rng.binomial(1, rates[a]) for a in arms])
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(len(arms)),
            "recency": rng.integers(1, 13, len(arms)),
            "history_segment": rng.choice(["1) $0 - $100", "2) $100 - $200"], len(arms)),
            "history": rng.uniform(10, 500, len(arms)),
            "mens": rng.integers(0, 2, len(arms)),
            "womens": rng.integers(0, 2, len(arms)),
            "zip_code": rng.choice(["Rural", "Urban"], len(arms)),
            "newbie": rng.integers(0, 2, len(arms)),
            "channel": rng.choice(["Web", "Phone"], len(arms)),
            "buyer_category": rng.choice(["mens only", "womens only"], len(arms)),
            "segment": arms,
            "visit": visit,
            "conversion": np.zeros(len(arms), dtype=int),
            "spend": np.zeros(len(arms)),
        }
    )
    return Dataset(frame=frame)


def test_ate_recovers_known_effect():
    true_lift = 0.05
    effects = arm_effects(_synthetic(true_lift))
    mens = effects[ARM_MENS]
    assert mens.ate == pytest.approx(true_lift, abs=0.01)
    assert mens.ci_low < true_lift < mens.ci_high
    assert mens.p_value < 1e-6


def test_ate_ci_covers_zero_for_null_effect():
    effects = arm_effects(_synthetic(0.0))
    mens = effects[ARM_MENS]
    assert mens.ci_low < 0 < mens.ci_high
    assert mens.p_value > 0.01


def test_randomized_arms_are_balanced():
    smd = standardized_mean_differences(_synthetic(0.05))
    assert not smd["imbalanced"].any()
