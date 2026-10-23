"""Turn a customer row into a second-person persona prompt.

A persona is built from covariates only. Outcomes (visit, conversion, spend) and the
real arm label are never read, so an agent cannot be told the answer it is asked for.
"""

from __future__ import annotations

from typing import Literal

from .data import COVARIATES

PersonaLevel = Literal["full", "minimal"]
Framing = Literal["neutral", "realism"]
Condition = Literal["mens", "womens", "none"]

MINIMAL_FIELDS = ("recency", "zip_code", "channel")

#: Fields a persona must never contain, enforced by tests.
FORBIDDEN_FIELDS = ("visit", "conversion", "spend", "segment")

CONDITION_TEXT: dict[str, str] = {
    "mens": "A few days ago you received an email from the retailer featuring men's merchandise.",
    "womens": "A few days ago you received an email from the retailer featuring women's merchandise.",
    "none": "You have not received any email from the retailer.",
}

REALISM_LINE = (
    "Respond as this person realistically would, including ignoring messages or doing nothing."
)

QUESTION = (
    "Over the next two weeks, do you visit the retailer's website?"
)

SCHEMA_LINE = (
    'Answer with JSON only: {"visit": true or false, "visit_probability": number between 0 and 1, '
    '"reason": string of at most 40 words}.'
)

_ZIP_TEXT = {
    "Rural": "You live in a rural area.",
    "Surburban": "You live in the suburbs.",
    "Urban": "You live in a city.",
}
_CHANNEL_TEXT = {
    "Web": "You have shopped with this retailer online.",
    "Phone": "You have shopped with this retailer by phone.",
    "Multichannel": "You have shopped with this retailer both online and by phone.",
}
_BUYER_TEXT = {
    "mens only": "Everything you have bought from them has been men's merchandise.",
    "womens only": "Everything you have bought from them has been women's merchandise.",
    "both": "You have bought both men's and women's merchandise from them.",
    "neither": "Your past purchases are not tied to a particular department.",
}


def _recency_text(months: int) -> str:
    if months <= 1:
        return "You bought something from an online retailer within the last month."
    if months <= 3:
        return f"You last bought something from an online retailer {months} months ago."
    return f"It has been {months} months since you last bought anything from this retailer."


def _spend_memory(history: float, segment: str) -> str:
    """Write the dollar history as a short purchase memory rather than a number dump."""
    del segment  # the bucket label is redundant once the amount is narrated
    if history < 100:
        scale = "a couple of small orders"
    elif history < 350:
        scale = "a handful of orders of ordinary size"
    elif history < 750:
        scale = "a steady run of orders, some of them substantial"
    else:
        scale = "many orders, several of them large"
    return (
        f"Looking back over the past year with them, you placed {scale}, "
        f"about ${history:,.0f} in total."
    )


def build_persona(row: dict, level: PersonaLevel = "full") -> str:
    """Build the persona paragraph for one customer row."""
    unknown = set(row) - set(COVARIATES) - {"customer_id"}
    leaked = unknown & set(FORBIDDEN_FIELDS)
    if leaked:
        raise ValueError(f"persona input must not carry outcome fields: {sorted(leaked)}")

    if level == "minimal":
        parts = [
            _recency_text(int(row["recency"])),
            _ZIP_TEXT.get(str(row["zip_code"]), ""),
            _CHANNEL_TEXT.get(str(row["channel"]), ""),
        ]
        return " ".join(p for p in parts if p)

    parts = [
        _recency_text(int(row["recency"])),
        _spend_memory(float(row["history"]), str(row["history_segment"])),
        _BUYER_TEXT.get(str(row["buyer_category"]), ""),
        _ZIP_TEXT.get(str(row["zip_code"]), ""),
        _CHANNEL_TEXT.get(str(row["channel"]), ""),
        "You are new to this retailer, having started buying only recently."
        if int(row["newbie"]) == 1
        else "You have been buying from this retailer for a while.",
    ]
    return " ".join(p for p in parts if p)


def build_prompt(
    row: dict,
    condition: Condition,
    level: PersonaLevel = "full",
    framing: Framing = "neutral",
) -> str:
    """Assemble the full single-condition prompt sent to one agent."""
    blocks = [
        "You are a shopper. Here is your situation.",
        build_persona(row, level=level),
        CONDITION_TEXT[condition],
    ]
    if framing == "realism":
        blocks.append(REALISM_LINE)
    blocks.append(QUESTION)
    blocks.append(SCHEMA_LINE)
    return "\n\n".join(blocks)


def persona_fields(level: PersonaLevel) -> tuple[str, ...]:
    """Covariates a persona at this level is allowed to read."""
    if level == "minimal":
        return MINIMAL_FIELDS
    return tuple(c for c in COVARIATES)
