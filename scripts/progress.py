"""Compact terminal progress and per-run timing helpers."""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUN_STATE_PATH = ROOT / "OUTPUT" / ".pipeline-progress.json"
_T80 = (0, 3.078, 1.886, 1.638, 1.533, 1.476, 1.440, 1.415, 1.397, 1.383)


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def _read_run_state() -> dict[str, Any] | None:
    try:
        state = json.loads(RUN_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) and "started_at" in state else None


def start_run() -> dict[str, Any]:
    state = {"started_at": time.time(), "steps": {}}
    write_run_state(state)
    return state


def load_or_start_run() -> dict[str, Any]:
    return _read_run_state() or start_run()


def write_run_state(state: dict[str, Any]) -> None:
    RUN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RUN_STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state), encoding="utf-8")
    temporary.replace(RUN_STATE_PATH)


def record_step(state: dict[str, Any], name: str, seconds: float) -> None:
    steps = state.setdefault("steps", {})
    steps[name] = seconds
    write_run_state(state)


def finish_run(state: dict[str, Any]) -> None:
    elapsed = time.time() - float(state["started_at"])
    steps = state.get("steps", {})
    step_names = {
        "crawl_orgas": "Orgas",
        "crawl_details": "Details",
        "crawl": "Crawl",
        "enrich": "Enrich",
        "convert": "Convert",
    }
    summary = " | ".join(
        f"{step_names.get(name, name)} {format_duration(seconds)}"
        for name, seconds in steps.items()
    )
    print(f"Gesamt {format_duration(elapsed)}" + (f" | {summary}" if summary else ""))
    try:
        RUN_STATE_PATH.unlink()
    except FileNotFoundError:
        pass


class Progress:
    _SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(
        self,
        label: str,
        total: int,
        run_started_at: float,
        parallelism: int = 1,
    ) -> None:
        self.label = label
        self.total = total
        self.run_started_at = run_started_at
        self.parallelism = max(1, parallelism)
        self.started = time.monotonic()
        self.durations: list[float] = []
        self._last_render = 0.0
        self._completed = 0
        self._tick = 0
        self._lock = threading.Lock()
        self._beat: threading.Thread | None = None
        self._stopped = False

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def _ensure_beat(self) -> None:
        """Sekündlicher Heartbeat, damit die Zeile auch zwischen Einheiten lebt."""
        if self._beat is not None or not sys.stdout.isatty() or self.total <= 0:
            return
        self._beat = threading.Thread(target=self._heartbeat, daemon=True)
        self._beat.start()

    def _heartbeat(self) -> None:
        while True:
            time.sleep(1.0)
            with self._lock:
                if self._stopped or self._completed >= self.total:
                    return
                self._tick += 1
                self._render(self._completed, animate=True)

    def update(self, completed: int, unit_seconds: float) -> None:
        with self._lock:
            self.durations.append(max(0.0, unit_seconds))
            self._completed = completed
            now = time.monotonic()
            if not sys.stdout.isatty() and completed < self.total and now - self._last_render < 5:
                return
            self._last_render = now
            self._render(completed, animate=False)
        self._ensure_beat()

    def _render(self, completed: int, animate: bool) -> None:
        remaining = max(0, self.total - completed)
        eta, interval = self._estimate_remaining(remaining)
        total_elapsed = time.time() - self.run_started_at
        spin = f"{self._SPINNER[self._tick % len(self._SPINNER)]} " if animate else ""
        line = (
            f"{spin}{self.label} {completed}/{self.total} | {format_duration(self.elapsed)} | "
            f"gesamt {format_duration(total_elapsed)}"
        )
        if remaining > 0 and eta is not None:
            end_clock = datetime.fromtimestamp(time.time() + eta).astimezone().strftime("%H:%M")
            line += f" | Phase ~{end_clock}"
        elif remaining > 0:
            line += " | ETA —"
        if remaining > 0 and eta is not None and interval is not None:
            end_at = time.time() + eta
            low = datetime.fromtimestamp(end_at - interval).astimezone().strftime("%H:%M")
            high = datetime.fromtimestamp(end_at + interval).astimezone().strftime("%H:%M")
            line += f" (80% {low}–{high})"
        print(line, end="\r" if sys.stdout.isatty() and completed < self.total else "\n", flush=True)

    def finish(self) -> None:
        with self._lock:
            self._stopped = True
        if self._beat is not None:
            self._beat.join(timeout=2.0)
        print(
            f"{self.label} fertig | {format_duration(self.elapsed)}",
            flush=True,
        )

    def _estimate_remaining(self, remaining: int) -> tuple[float | None, float | None]:
        if remaining == 0:
            return 0.0, None
        if not self.durations:
            return None, None

        count = len(self.durations)
        mean = sum(self.durations) / count
        work_units = remaining / self.parallelism
        estimate = work_units * mean
        if count < 3:
            return estimate, None

        variance = sum((duration - mean) ** 2 for duration in self.durations) / (count - 1)
        df = count - 1
        critical = _T80[df] if df < len(_T80) else 1.282
        prediction_std = math.sqrt(
            (remaining * variance + remaining**2 * variance / count)
            / self.parallelism**2
        )
        return estimate, critical * prediction_std
