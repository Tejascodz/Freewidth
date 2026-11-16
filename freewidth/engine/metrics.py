"""Minimal thread-safe Prometheus metrics (text exposition format, no extra dependency)."""
from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self, prefix: str = "freewidth"):
        self.prefix = prefix
        self._lock = threading.Lock()
        self._counters: dict = defaultdict(float)
        self._gauges: dict = {}
        self._help: dict = {}
        # latency histogram buckets (seconds)
        self._buckets = [0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, float("inf")]
        self._hist = defaultdict(lambda: [0] * len(self._buckets))
        self._hist_sum: dict = defaultdict(float)
        self._hist_count: dict = defaultdict(int)

    @staticmethod
    def _key(name: str, labels: dict | None) -> tuple:
        return (name, tuple(sorted((labels or {}).items())))

    def inc(self, name: str, value: float = 1.0, labels: dict | None = None, help: str = "") -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += value
            if help:
                self._help[name] = help

    def set(self, name: str, value: float, labels: dict | None = None, help: str = "") -> None:
        with self._lock:
            self._gauges[self._key(name, labels)] = value
            if help:
                self._help[name] = help

    def observe(self, name: str, value: float, labels: dict | None = None, help: str = "") -> None:
        k = self._key(name, labels)
        with self._lock:
            for i, b in enumerate(self._buckets):
                if value <= b:
                    self._hist[k][i] += 1
            self._hist_sum[k] += value
            self._hist_count[k] += 1
            if help:
                self._help[name] = help

    @staticmethod
    def _lbl(items, extra: str = "") -> str:
        parts = [f'{k}="{v}"' for k, v in items]
        if extra:
            parts.append(extra)
        return "{" + ",".join(parts) + "}" if parts else ""

    def render(self) -> str:
        p = self.prefix
        out = []
        with self._lock:
            seen = set()
            for (name, items), v in sorted(self._counters.items()):
                if name not in seen:
                    out.append(f"# HELP {p}_{name} {self._help.get(name, name)}")
                    out.append(f"# TYPE {p}_{name} counter")
                    seen.add(name)
                out.append(f"{p}_{name}{self._lbl(items)} {v}")
            for (name, items), v in sorted(self._gauges.items()):
                if name not in seen:
                    out.append(f"# HELP {p}_{name} {self._help.get(name, name)}")
                    out.append(f"# TYPE {p}_{name} gauge")
                    seen.add(name)
                out.append(f"{p}_{name}{self._lbl(items)} {v}")
            for (name, items), counts in sorted(self._hist.items()):
                if name not in seen:
                    out.append(f"# HELP {p}_{name} {self._help.get(name, name)}")
                    out.append(f"# TYPE {p}_{name} histogram")
                    seen.add(name)
                for b, c in zip(self._buckets, counts):
                    le = "+Inf" if b == float("inf") else str(b)
                    lbl = self._lbl(items, 'le="' + le + '"')
                    out.append(f"{p}_{name}_bucket{lbl} {c}")
                out.append(f"{p}_{name}_sum{self._lbl(items)} {self._hist_sum[(name, items)]}")
                out.append(f"{p}_{name}_count{self._lbl(items)} {self._hist_count[(name, items)]}")
        return "\n".join(out) + "\n"
