#!/usr/bin/env python3
"""Gemeinsame LLM-Client-Konfiguration (OpenAI-kompatibel, z.B. OpenRouter).

Umgebungsvariablen:
  LLM_BASE_URL -> OPENAI_BASE_URL (leer = SDK-Default api.openai.com)
  LLM_REASONING_EFFORT (optional, z.B. "low" für Qwen-Thinking-Modelle;
    unset = kein reasoning-Parameter wird gesendet)

Key-Wahl (provider-bewusst, damit kein OpenAI-Key an OpenRouter geht):
  1. LLM_API_KEY gewinnt immer (explizit).
  2. Zeigt Base-URL/Modell auf OpenRouter ("openrouter" in Host oder
     Modell-Prefix), wird OPENROUTER_API_KEY bevorzugt.
  3. Sonst OPENAI_API_KEY, dann OPENROUTER_API_KEY als Fallback.
"""
from __future__ import annotations

import os
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


OPENROUTER_URL = "https://openrouter.ai/api/v1"


def is_openrouter_model(model: str = "") -> bool:
    """'qwen/...' und 'openrouter/...' laufen über OpenRouter; 'gpt-4o-mini' und 'openai/...' über OpenAI."""
    parts = (model or "").lower().split("/")
    return len(parts) > 1 and parts[0] != "openai"


def _is_openrouter(model: str = "", url: str | None = None) -> bool:
    haystack = f"{model or ''} {url or ''}".lower()
    return "openrouter" in haystack or is_openrouter_model(model)


def api_key(model: str = "", url: str | None = None) -> str:
    """Provider-bewusste Key-Wahl (verhindert 401 durch falschen Key)."""
    explicit = (os.getenv("LLM_API_KEY") or "").strip()
    if explicit:
        return explicit
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if _is_openrouter(model, url if url is not None else base_url(model)):
        return openrouter_key or openai_key
    return openai_key or openrouter_key


def base_url(model: str = "") -> str | None:
    """LLM_BASE_URL erzwingt eine URL für alle Modelle; sonst Routing pro Modell."""
    explicit = (os.getenv("LLM_BASE_URL") or "").strip()
    if explicit:
        return explicit
    if is_openrouter_model(model):
        return OPENROUTER_URL
    return (os.getenv("OPENAI_BASE_URL") or "").strip() or None


def require_api_key(model: str = "", url: str | None = None) -> str:
    key = api_key(model, url)
    if not key:
        raise SystemExit(
            "Kein LLM-API-Key gefunden (LLM_API_KEY, OPENAI_API_KEY oder "
            "OPENROUTER_API_KEY in .env oder Umgebung)"
        )
    return key


def make_client(model: str = "") -> Any:
    from openai import AsyncOpenAI
    url = base_url(model)
    return AsyncOpenAI(api_key=require_api_key(model, url), base_url=url)


class _RoutedCompletions:
    def __init__(self, router: "ClientRouter") -> None:
        self._router = router

    async def create(self, **kwargs: Any) -> Any:
        client = self._router.client_for(kwargs.get("model", ""))
        return await client.chat.completions.create(**kwargs)


class _RoutedChat:
    def __init__(self, router: "ClientRouter") -> None:
        self.completions = _RoutedCompletions(router)


class ClientRouter:
    """Client-Fassade: wählt pro Modell Provider (OpenAI/OpenRouter), Clients werden lazy erzeugt."""

    def __init__(self) -> None:
        self._clients: dict[str | None, Any] = {}
        self.chat = _RoutedChat(self)

    def client_for(self, model: str) -> Any:
        url = base_url(model)
        if url not in self._clients:
            self._clients[url] = make_client(model)
        return self._clients[url]

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()


def reasoning_kwargs(model: str = "") -> dict[str, Any]:
    """extra_body für Thinking-Modelle; leer wenn nicht konfiguriert.

    Mit Modellangabe nur für OpenRouter-Modelle (gpt-4o-mini lehnt den Parameter ab)."""
    if model and not is_openrouter_model(model) and not (os.getenv("LLM_BASE_URL") or "").strip():
        return {}
    effort = (os.getenv("LLM_REASONING_EFFORT") or "").strip().lower()
    if effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        return {}
    return {"extra_body": {"reasoning": {"effort": effort}}}
