"""LLM providers, output validation, rate limiting and an on-disk response cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

DEFAULT_CACHE = Path("data/raw/llm_cache.sqlite")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class AgentResponse(BaseModel):
    """The only shape an agent reply may take."""

    visit: bool
    visit_probability: float = Field(ge=0.0, le=1.0)
    reason: str

    @field_validator("reason")
    @classmethod
    def _at_most_40_words(cls, value: str) -> str:
        words = value.split()
        if len(words) > 40:
            return " ".join(words[:40])
        return value.strip()


def parse_response(text: str) -> AgentResponse:
    """Parse a raw completion into a validated response, or raise ValueError."""
    match = _JSON_BLOCK.search(text or "")
    if not match:
        raise ValueError("no JSON object in model output")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    try:
        return AgentResponse.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"response failed schema validation: {exc}") from exc


@dataclass(frozen=True)
class CallSpec:
    """Everything that identifies one agent call, and therefore its cache key."""

    provider: str
    model: str
    temperature: float
    persona_level: str
    framing: str
    condition: str
    customer_id: int
    seed: int

    def key(self) -> str:
        raw = "|".join(
            [
                self.provider,
                self.model,
                f"{self.temperature:.4f}",
                self.persona_level,
                self.framing,
                self.condition,
                str(self.customer_id),
                str(self.seed),
            ]
        )
        return hashlib.sha256(raw.encode()).hexdigest()


class ResponseCache:
    """SQLite-backed cache so re-runs make zero provider calls and runs resume."""

    def __init__(self, path: Path = DEFAULT_CACHE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> AgentResponse | None:
        row = self._conn.execute("SELECT payload FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return AgentResponse.model_validate_json(row[0])

    def put(self, key: str, response: AgentResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
            (key, response.model_dump_json(), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class RateLimiter:
    """Requests-per-minute limiter shared by all workers of one run."""

    def __init__(self, requests_per_minute: int):
        self._interval = 60.0 / max(requests_per_minute, 1)
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self._interval
        if wait:
            await asyncio.sleep(wait)


class Provider:
    """Interface every provider implements."""

    name = "base"

    def models(self) -> tuple[str, str]:
        raise NotImplementedError

    async def complete(self, prompt: str, model: str, temperature: float, spec: CallSpec) -> str:
        raise NotImplementedError


class MockProvider(Provider):
    """Deterministic offline provider.

    Response probabilities are a seeded function of the persona features and the
    condition, so the whole pipeline, tests and dashboard run with no network.
    """

    name = "mock"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def models(self) -> tuple[str, str]:
        return ("mock-large", "mock-small")

    @staticmethod
    def _features(prompt: str) -> dict[str, float]:
        lowered = prompt.lower()
        return {
            "recent": 1.0 if "within the last month" in lowered else 0.0,
            "lapsed": 1.0 if "it has been" in lowered else 0.0,
            "big_spender": 1.0 if "several of them large" in lowered else 0.0,
            "newbie": 1.0 if "new to this retailer" in lowered else 0.0,
            "urban": 1.0 if "in a city" in lowered else 0.0,
            "multichannel": 1.0 if "both online and by phone" in lowered else 0.0,
        }

    async def complete(self, prompt: str, model: str, temperature: float, spec: CallSpec) -> str:
        f = self._features(prompt)
        base = 0.16
        base += 0.10 * f["recent"] - 0.05 * f["lapsed"]
        base += 0.06 * f["big_spender"] + 0.04 * f["multichannel"]
        base += 0.02 * f["urban"] - 0.03 * f["newbie"]
        if spec.condition == "mens":
            base += 0.11 if "men's merchandise" in prompt.lower() else 0.08
        elif spec.condition == "womens":
            base += 0.07
        if model.endswith("small"):
            base += 0.03
        if spec.framing == "realism":
            base -= 0.04
        rng = random.Random(f"{spec.key()}:{self.seed}")
        noise = rng.gauss(0.0, 0.03 + 0.05 * temperature)
        prob = min(0.98, max(0.01, base + noise))
        visit = rng.random() < prob
        if spec.condition == "none":
            reason = "No message arrived, so I would only browse if I happened to need something."
        else:
            reason = "The email reminded me of the retailer and I would click through to look."
        return json.dumps({"visit": visit, "visit_probability": round(prob, 4), "reason": reason})


class GroqProvider(Provider):
    """Groq chat completions in JSON mode."""

    name = "groq"

    def __init__(self, api_key: str | None = None):
        from groq import AsyncGroq  # imported lazily so CI never needs the SDK

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self._client = AsyncGroq(api_key=key)
        self._models: tuple[str, str] | None = None

    def models(self) -> tuple[str, str]:
        """Pick one large and one small current chat model from the live model list."""
        if self._models is not None:
            return self._models
        override = (os.environ.get("GROQ_MODEL_LARGE"), os.environ.get("GROQ_MODEL_SMALL"))
        if all(override):
            self._models = (override[0], override[1])  # type: ignore[assignment]
            return self._models

        from groq import Groq

        listed = Groq(api_key=self._client.api_key).models.list().data
        chat = [
            m.id
            for m in listed
            if getattr(m, "active", True)
            and not any(
                tag in m.id.lower()
                for tag in ("whisper", "tts", "guard", "embed", "vision-preview", "compound")
            )
        ]

        def size_of(model_id: str) -> float:
            found = re.findall(r"(\d+(?:\.\d+)?)\s*b\b", model_id.lower())
            return max((float(x) for x in found), default=0.0)

        ranked = sorted(chat, key=size_of)
        if not ranked:
            raise RuntimeError("no usable chat models returned by the Groq API")
        large = override[0] or ranked[-1]
        small = override[1] or ranked[0]
        self._models = (large, small)
        return self._models

    async def complete(self, prompt: str, model: str, temperature: float, spec: CallSpec) -> str:
        del spec
        completion = await self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content or ""


class GeminiProvider(Provider):
    """Fallback provider, used only when a Gemini key is present."""

    name = "gemini"

    def __init__(self, api_key: str | None = None):
        from google import genai  # imported lazily

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self._genai = genai
        self._client = genai.Client(api_key=key)

    def models(self) -> tuple[str, str]:
        return (
            os.environ.get("GEMINI_MODEL_LARGE", "gemini-2.5-flash"),
            os.environ.get("GEMINI_MODEL_SMALL", "gemini-2.5-flash-lite"),
        )

    async def complete(self, prompt: str, model: str, temperature: float, spec: CallSpec) -> str:
        del spec
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": temperature, "response_mime_type": "application/json"},
        )
        return response.text or ""


def get_provider(name: str, seed: int = 42) -> Provider:
    if name == "mock":
        return MockProvider(seed=seed)
    if name == "groq":
        return GroqProvider()
    if name == "gemini":
        return GeminiProvider()
    raise ValueError(f"unknown provider: {name}")


def _retry_after_seconds(error: Exception) -> float | None:
    for attr in ("response", "body"):
        obj = getattr(error, attr, None)
        headers = getattr(obj, "headers", None)
        if headers:
            value = headers.get("retry-after") or headers.get("Retry-After")
            if value:
                try:
                    return float(value)
                except ValueError:
                    continue
    return None


def _is_rate_limit(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    return status == 429 or "429" in str(error) or "rate limit" in str(error).lower()


@dataclass
class CallStats:
    provider_calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0


class AgentRunner:
    """Runs agent calls with caching, a RPM limit, bounded concurrency and retries."""

    def __init__(
        self,
        provider: Provider,
        cache: ResponseCache | None = None,
        concurrency: int = 8,
        requests_per_minute: int = 28,
        max_retries: int = 3,
    ):
        self.provider = provider
        self.cache = cache if cache is not None else ResponseCache()
        self.limiter = RateLimiter(requests_per_minute)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.stats = CallStats()

    async def call(self, prompt: str, spec: CallSpec) -> AgentResponse:
        key = spec.key()
        cached = self.cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        last_error: Exception | None = None
        async with self.semaphore:
            for attempt in range(self.max_retries):
                await self.limiter.acquire()
                try:
                    raw = await self.provider.complete(
                        prompt, spec.model, spec.temperature, spec
                    )
                    self.stats.provider_calls += 1
                    response = parse_response(raw)
                    self.cache.put(key, response)
                    return response
                except Exception as exc:  # noqa: BLE001 - every failure mode retries alike
                    last_error = exc
                    self.stats.retries += 1
                    if _is_rate_limit(exc):
                        wait = _retry_after_seconds(exc) or (2**attempt)
                    else:
                        wait = min(8.0, 2**attempt) * (0.5 + random.random())
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(wait)
        self.stats.failures += 1
        raise RuntimeError(f"agent call failed after {self.max_retries} attempts: {last_error}")

    def close(self) -> None:
        self.cache.close()
