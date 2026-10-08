#!/usr/bin/env python3
"""LLM-Nutzungs- und Kostenerfassung für den Terminal-Report (USD-Schätzung)."""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any


# Preise in USD pro 1M Token (Input, Output); Stand 2025, per ENV überschreibbar:
# PRICE_GPT_4O_MINI_IN, PRICE_GPT_4O_MINI_OUT, ...
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "qwen3-32b": (0.08, 0.28),
    "qwen3.8-27b": (0.265, 0.65),
}


def normalize_model(model: str) -> str:
    """'openai/gpt-4o-mini' -> 'gpt-4o-mini'."""
    name = (model or "").strip().lower()
    if "/" in name:
        name = name.split("/")[-1]
    return name or "unknown"


def _env_key(model: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", normalize_model(model).upper()).strip("_")


def prices_for(model: str) -> tuple[float, float]:
    name = normalize_model(model)
    default = DEFAULT_PRICES.get(name, (0.0, 0.0))
    key = _env_key(model)
    try:
        price_in = float(os.getenv(f"PRICE_{key}_IN", str(default[0])))
    except ValueError:
        price_in = default[0]
    try:
        price_out = float(os.getenv(f"PRICE_{key}_OUT", str(default[1])))
    except ValueError:
        price_out = default[1]
    return price_in, price_out


def estimate_tokens(model: str, text: str) -> int:
    """Token-Schätzung via tiktoken (für Stellen ohne echte Usage)."""
    content = text or ""
    try:
        import tiktoken
    except ImportError:
        return max(1, len(content) // 4)
    try:
        encoding = tiktoken.encoding_for_model(normalize_model(model))
    except KeyError:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return max(1, len(content) // 4)
    return len(encoding.encode(content))


class UsageTracker:
    """Zählt pro Modell Calls sowie Prompt-/Completion-Tokens (exakt + geschätzt)."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.prompt_tokens: Counter[str] = Counter()
        self.completion_tokens: Counter[str] = Counter()
        self.estimated_prompt_tokens: Counter[str] = Counter()
        self.estimated_completion_tokens: Counter[str] = Counter()

    def add(self, model: str, prompt_tokens: int = 0, completion_tokens: int = 0,
            estimated: bool = False) -> None:
        name = normalize_model(model)
        self.calls[name] += 1
        if estimated:
            self.estimated_prompt_tokens[name] += max(0, int(prompt_tokens))
            self.estimated_completion_tokens[name] += max(0, int(completion_tokens))
        else:
            self.prompt_tokens[name] += max(0, int(prompt_tokens))
            self.completion_tokens[name] += max(0, int(completion_tokens))

    def add_response(self, model: str, response: Any) -> bool:
        """Usage aus einer OpenAI-Response übernehmen. True bei Erfolg."""
        try:
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        except (TypeError, ValueError):
            return False
        if usage is None or (not prompt_tokens and not completion_tokens):
            return False
        self.add(model, prompt_tokens, completion_tokens)
        return True

    def merge(self, other: "UsageTracker") -> None:
        for counter_name in ("calls", "prompt_tokens", "completion_tokens",
                             "estimated_prompt_tokens", "estimated_completion_tokens"):
            mine = getattr(self, counter_name)
            for key, value in getattr(other, counter_name).items():
                mine[key] += value

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": dict(self.calls),
            "prompt_tokens": dict(self.prompt_tokens),
            "completion_tokens": dict(self.completion_tokens),
            "estimated_prompt_tokens": dict(self.estimated_prompt_tokens),
            "estimated_completion_tokens": dict(self.estimated_completion_tokens),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "UsageTracker":
        tracker = cls()
        if not isinstance(data, dict):
            return tracker
        for counter_name in ("calls", "prompt_tokens", "completion_tokens",
                             "estimated_prompt_tokens", "estimated_completion_tokens"):
            values = data.get(counter_name, {})
            if isinstance(values, dict):
                for key, value in values.items():
                    try:
                        getattr(tracker, counter_name)[str(key)] += int(value)
                    except (TypeError, ValueError):
                        continue
        return tracker

    def cost_usd(self, model: str) -> float:
        name = normalize_model(model)
        price_in, price_out = prices_for(name)
        prompt = self.prompt_tokens[name] + self.estimated_prompt_tokens[name]
        completion = self.completion_tokens[name] + self.estimated_completion_tokens[name]
        return (prompt * price_in + completion * price_out) / 1_000_000

    def total_usd(self) -> float:
        models = set(self.calls) | set(self.prompt_tokens) | set(self.estimated_prompt_tokens)
        return sum(self.cost_usd(model) for model in models)

    def render(self, title: str = "LLM-Nutzung") -> str:
        models = sorted(
            set(self.calls) | set(self.prompt_tokens) | set(self.estimated_prompt_tokens)
        )
        lines = [f"--- {title} ---"]
        if not models:
            lines.append("keine LLM-Aufrufe")
            return "\n".join(lines)
        for model in models:
            calls = self.calls.get(model, 0)
            prompt = self.prompt_tokens.get(model, 0)
            completion = self.completion_tokens.get(model, 0)
            est_prompt = self.estimated_prompt_tokens.get(model, 0)
            est_completion = self.estimated_completion_tokens.get(model, 0)
            parts = [f"{model}: {calls} Calls"]
            if prompt or completion:
                parts.append(f"{prompt} in / {completion} out")
            if est_prompt or est_completion:
                parts.append(f"~{est_prompt} in / ~{est_completion} out (geschätzt)")
            price_in, price_out = prices_for(model)
            if price_in or price_out:
                parts.append(f"~${self.cost_usd(model):.2f}")
            else:
                parts.append("(kein Preis hinterlegt)")
            lines.append(" | ".join(parts))
        lines.append(f"GESAMT ~${self.total_usd():.2f} (Schätzung)")
        return "\n".join(lines)


def record_usage(state: dict[str, Any], tracker: UsageTracker) -> None:
    """Tracker-Stand in den Run-State mischen (für das Pipeline-Total)."""
    stored = UsageTracker.from_dict(state.get("usage"))
    stored.merge(tracker)
    state["usage"] = stored.to_dict()
    from progress import write_run_state
    write_run_state(state)


def load_usage(state: dict[str, Any]) -> UsageTracker:
    return UsageTracker.from_dict(state.get("usage") if isinstance(state, dict) else None)
