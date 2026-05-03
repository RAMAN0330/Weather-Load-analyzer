"""SLDC schedule validation and guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

BLOCKS_PER_DAY = 96
HARYANA_MUST_RUN_MW = 3500.0
HARYANA_MAX_RAMP_MW = 350.0
HARYANA_PEAK_CAP_MW = 14000.0


@dataclass
class SLDCValidationResult:
    forecast: np.ndarray
    flags: list[str]
    metadata: dict[str, Any]


def _as_96(values: Any, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
    out = np.full(BLOCKS_PER_DAY, float(fill), dtype=float)
    if arr.size:
        n = min(BLOCKS_PER_DAY, arr.size)
        out[:n] = arr[:n]
        if n < BLOCKS_PER_DAY:
            out[n:] = out[n - 1]
    return out


def validate_sldc_schedule(
    forecast_96: Any,
    baseline_96: Any,
    *,
    season: str = "summer",
    floor_mw: float = HARYANA_MUST_RUN_MW,
    peak_cap_mw: float = HARYANA_PEAK_CAP_MW,
    max_ramp_mw: float = HARYANA_MAX_RAMP_MW,
    max_energy_gap_pct: float = 5.0,
    max_baseline_deviation_pct: float = 15.0,
    enforce: bool = True,
) -> SLDCValidationResult:
    f = _as_96(forecast_96)
    baseline = _as_96(baseline_96, fill=float(np.nanmean(f) if f.size else 0.0))
    flags: list[str] = []
    adjusted = f.copy()

    floor = float(floor_mw) * (0.90 if str(season).lower().strip() == "winter" else 1.0)
    low = np.where(adjusted < floor)[0]
    if low.size:
        flags.append(f"FLOOR_BREACH: {int(low.size)} blocks below {floor:.0f} MW")
        if enforce:
            adjusted[low] = floor

    high = np.where(adjusted > float(peak_cap_mw))[0]
    if high.size:
        flags.append(f"CEILING_BREACH: {int(high.size)} blocks above {float(peak_cap_mw):.0f} MW")
        if enforce:
            adjusted[high] = float(peak_cap_mw)

    ramps = np.abs(np.diff(adjusted))
    ramp_idx = np.where(ramps > float(max_ramp_mw))[0]
    if ramp_idx.size:
        flags.append(f"RAMP_BREACH: {int(ramp_idx.size)} transitions exceed {float(max_ramp_mw):.0f} MW/15min")

    dev_pct = np.abs(adjusted - baseline) / np.maximum(np.abs(baseline), 1.0) * 100.0
    extreme = np.where(dev_pct > float(max_baseline_deviation_pct))[0]
    if extreme.size:
        flags.append(
            f"LARGE_DEVIATION: {int(extreme.size)} blocks >{float(max_baseline_deviation_pct):.0f}% from baseline"
        )

    baseline_energy = float(np.sum(baseline) * 0.25)
    schedule_energy = float(np.sum(adjusted) * 0.25)
    energy_gap_pct = abs(schedule_energy - baseline_energy) / max(abs(baseline_energy), 1.0) * 100.0
    if energy_gap_pct > float(max_energy_gap_pct):
        flags.append(f"ENERGY_GAP: {energy_gap_pct:.1f}% vs baseline")

    return SLDCValidationResult(
        forecast=adjusted,
        flags=flags,
        metadata={
            "enabled": True,
            "enforced": bool(enforce),
            "status": "ok" if not flags else "flagged",
            "flags": flags,
            "floor_mw": float(floor),
            "peak_cap_mw": float(peak_cap_mw),
            "max_ramp_mw": float(max_ramp_mw),
            "schedule_energy_mwh": round(schedule_energy, 3),
            "baseline_energy_mwh": round(baseline_energy, 3),
            "energy_gap_pct": round(float(energy_gap_pct), 3),
        },
    )
