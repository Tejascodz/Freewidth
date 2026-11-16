"""Device helpers: GPU energy via NVML (Volta and newer, incl. T4)."""
from __future__ import annotations


class EnergyMeter:
    """Reads cumulative board energy (mJ) from NVML. No-op where unsupported."""

    def __init__(self, device_indices: list[int] | None = None):
        self.ok = False
        self._handles = []
        self._start = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            idx = device_indices if device_indices is not None else [0]
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in idx]
            for h in self._handles:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            self._nvml = pynvml
            self.ok = True
        except Exception:
            self.ok = False

    def _read(self) -> float:
        return sum(self._nvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in self._handles)

    def start(self) -> None:
        self._start = self._read() if self.ok else None

    def stop(self) -> float | None:
        if not self.ok or self._start is None:
            return None
        return (self._read() - self._start) / 1000.0


def cuda_index(device: str) -> int | None:
    if device.startswith("cuda"):
        return int(device.split(":")[1]) if ":" in device else 0
    return None
