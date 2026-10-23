"""Load and clean the Hillstrom MineThatData e-mail experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
FALLBACK_CSV_URL = (
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_"
    "DataMiningChallenge_2008.03.20.csv"
)

ARM_CONTROL = "No E-Mail"
ARM_MENS = "Mens E-Mail"
ARM_WOMENS = "Womens E-Mail"
ARMS = (ARM_CONTROL, ARM_MENS, ARM_WOMENS)

#: Covariates an agent persona may see. Outcomes and the arm label are never included.
COVARIATES = (
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "buyer_category",
)
OUTCOMES = ("visit", "conversion", "spend")
ARM_COL = "segment"

CATEGORICAL = ("history_segment", "zip_code", "channel", "buyer_category")
NUMERIC = ("recency", "history")
BINARY = ("mens", "womens", "newbie")


@dataclass(frozen=True)
class Dataset:
    """The cleaned experiment table plus the column roles used downstream."""

    frame: pd.DataFrame

    @property
    def covariates(self) -> list[str]:
        return [c for c in COVARIATES if c in self.frame.columns]

    def arm(self, name: str) -> pd.DataFrame:
        return self.frame[self.frame[ARM_COL] == name]


def _derive_buyer_category(frame: pd.DataFrame) -> pd.Series:
    mens = frame["mens"].astype(bool)
    womens = frame["womens"].astype(bool)
    out = pd.Series("neither", index=frame.index, dtype="object")
    out[mens & ~womens] = "mens only"
    out[~mens & womens] = "womens only"
    out[mens & womens] = "both"
    return out


def _download_fallback() -> pd.DataFrame:
    """Read the original challenge CSV directly if scikit-uplift is unavailable."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached = RAW_DIR / "hillstrom.csv"
    if not cached.exists():
        pd.read_csv(FALLBACK_CSV_URL).to_csv(cached, index=False)
    return pd.read_csv(cached)


def _fetch_raw() -> pd.DataFrame:
    try:
        from sklift.datasets import fetch_hillstrom
    except ImportError:
        return _download_fallback()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    bunch = fetch_hillstrom(target_col="all", data_home=str(RAW_DIR))
    frame = bunch.data.copy()
    target = bunch.target
    for col in target.columns:
        frame[col] = target[col].to_numpy()
    frame[ARM_COL] = bunch.treatment.to_numpy()
    return frame


def load(validate: bool = True) -> Dataset:
    """Return the cleaned 64k-row experiment table with typed columns."""
    frame = _fetch_raw()
    frame.columns = [c.strip() for c in frame.columns]

    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    for col in (*BINARY, "visit", "conversion"):
        frame[col] = frame[col].astype(int)
    frame["spend"] = pd.to_numeric(frame["spend"], errors="raise").astype(float)
    for col in ("history_segment", "zip_code", "channel", ARM_COL):
        frame[col] = frame[col].astype(str).str.strip()

    frame["buyer_category"] = _derive_buyer_category(frame)
    frame = frame.reset_index(drop=True)
    frame.insert(0, "customer_id", frame.index.astype(int))

    if validate:
        missing = set(ARMS) - set(frame[ARM_COL].unique())
        if missing:
            raise ValueError(f"missing experiment arms: {sorted(missing)}")
        if frame[list(COVARIATES)].isna().any().any():
            raise ValueError("covariates contain missing values")
    return Dataset(frame=frame)
