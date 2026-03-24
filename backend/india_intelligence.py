"""
India-Specific Load Intelligence Module.

Provides state-level behavioural models, festival calendar engine,
and event-driven load adjustment features for Indian power utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, timedelta
import logging

logger = logging.getLogger(__name__)

# ── State-Level Agricultural Load Profiles ──────────────────────────
# Agricultural pump-set timings vary by state and crop season.
# Rabi (Oct–Mar): wheat/mustard, Kharif (Jun–Oct): rice/cotton
AGRICULTURAL_PROFILES = {
    "punjab": {
        "rabi":   {"months": [10, 11, 12, 1, 2, 3], "peak_blocks": (20, 40), "mw_add": 300},  # wheat irrigation, morning
        "kharif": {"months": [6, 7, 8, 9, 10],      "peak_blocks": (60, 80), "mw_add": 450},  # paddy, evening pump
    },
    "haryana": {
        "rabi":   {"months": [10, 11, 12, 1, 2, 3], "peak_blocks": (20, 40), "mw_add": 250},
        "kharif": {"months": [6, 7, 8, 9, 10],      "peak_blocks": (60, 80), "mw_add": 400},
    },
    "uttar pradesh": {
        "rabi":   {"months": [10, 11, 12, 1, 2, 3], "peak_blocks": (24, 44), "mw_add": 500},
        "kharif": {"months": [6, 7, 8, 9, 10],      "peak_blocks": (56, 76), "mw_add": 600},
    },
    "madhya pradesh": {
        "rabi":   {"months": [10, 11, 12, 1, 2, 3], "peak_blocks": (20, 44), "mw_add": 350},
        "kharif": {"months": [6, 7, 8, 9, 10],      "peak_blocks": (56, 76), "mw_add": 400},
    },
    "rajasthan": {
        "rabi":   {"months": [10, 11, 12, 1, 2, 3], "peak_blocks": (20, 40), "mw_add": 200},
        "kharif": {"months": [6, 7, 8, 9, 10],      "peak_blocks": (60, 80), "mw_add": 300},
    },
    "gujarat": {
        "rabi":   {"months": [10, 11, 12, 1, 2, 3], "peak_blocks": (20, 40), "mw_add": 250},
        "kharif": {"months": [6, 7, 8, 9, 10],      "peak_blocks": (56, 76), "mw_add": 350},
    },
}

# ── Industrial Shift Patterns ───────────────────────────────────────
INDUSTRIAL_SHIFT_PROFILES = {
    "maharashtra": {"shifts": 3, "base_industrial_pct": 0.45},  # 3-shift heavy industry
    "gujarat":     {"shifts": 3, "base_industrial_pct": 0.40},
    "tamil nadu":  {"shifts": 3, "base_industrial_pct": 0.35},
    "karnataka":   {"shifts": 2, "base_industrial_pct": 0.30},
    "delhi":       {"shifts": 2, "base_industrial_pct": 0.15},  # mostly commercial
    "punjab":      {"shifts": 2, "base_industrial_pct": 0.25},
    "haryana":     {"shifts": 2, "base_industrial_pct": 0.30},
}


def get_agricultural_adjustment(
    state: str, target_date: str, baseline: np.ndarray
) -> np.ndarray:
    """Return 96-block MW adjustment for agricultural load based on state and crop season."""
    adj = np.zeros(96, dtype=float)
    state_lower = state.lower().strip()
    if state_lower not in AGRICULTURAL_PROFILES:
        return adj

    dt = pd.to_datetime(target_date)
    month = dt.month

    for season_name, profile in AGRICULTURAL_PROFILES[state_lower].items():
        if month in profile["months"]:
            start_b, end_b = profile["peak_blocks"]
            mw = profile["mw_add"]
            # Gaussian shape centered on peak blocks
            center = (start_b + end_b) / 2.0
            width = (end_b - start_b) / 2.0
            for b in range(96):
                adj[b] += mw * np.exp(-0.5 * ((b - center) / max(width, 1)) ** 2)
            break  # only one season active at a time

    return adj


# ── Festival Calendar Engine ────────────────────────────────────────
# Multi-day festival profiles: [Day-3, Day-2, Day-1, Day-0, Day+1, Day+2]
# Values are multiplicative factors on daily energy (1.0 = normal)
FESTIVAL_PROFILES = {
    "diwali": {
        "duration_days": 5,  # Dhanteras to Bhai Dooj
        "daily_factors": [1.05, 1.08, 1.12, 0.82, 0.88, 0.95],  # shopping surge → dip → recovery
        "block_shape": {  # intraday modulation on Day-0 (Diwali)
            "evening_boost": (72, 88, 1.15),  # lighting, 18:00-22:00
            "night_dip": (88, 96, 0.70),       # post-celebration
            "morning_dip": (0, 24, 0.75),      # late start
        },
        "states": "all",
    },
    "holi": {
        "duration_days": 3,
        "daily_factors": [1.02, 0.85, 0.92],  # pre-Holi normal, Holi dip, recovery
        "block_shape": {
            "morning_dip": (0, 32, 0.80),     # late start
            "afternoon_dip": (32, 56, 0.85),   # celebrations
        },
        "states": ["punjab", "haryana", "delhi", "uttar pradesh", "rajasthan", "madhya pradesh", "bihar"],
    },
    "pongal": {
        "duration_days": 4,
        "daily_factors": [1.03, 0.88, 0.85, 0.92],
        "states": ["tamil nadu"],
    },
    "onam": {
        "duration_days": 3,
        "daily_factors": [1.05, 0.82, 0.90],
        "states": ["kerala"],
    },
    "durga_puja": {
        "duration_days": 5,
        "daily_factors": [1.05, 1.10, 1.15, 0.85, 0.90],
        "states": ["west bengal", "odisha", "assam", "jharkhand"],
    },
    "ganesh_chaturthi": {
        "duration_days": 2,
        "daily_factors": [0.90, 0.95],
        "states": ["maharashtra", "goa", "karnataka"],
    },
    "baisakhi": {
        "duration_days": 2,
        "daily_factors": [0.88, 0.94],
        "states": ["punjab", "haryana"],
    },
    "eid": {
        "duration_days": 3,
        "daily_factors": [1.05, 0.80, 0.90],  # pre-Eid shopping, Eid dip, recovery
        "states": "all",
    },
    "navratri": {
        "duration_days": 9,
        "daily_factors": [1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.10, 0.85],
        "states": ["gujarat", "rajasthan", "madhya pradesh", "maharashtra"],
    },
}


def get_festival_adjustment(
    state: str, target_date: str, baseline: np.ndarray
) -> Tuple[np.ndarray, str]:
    """
    Return 96-block adjustment and festival name if target_date falls
    within a festival window for the given state.
    """
    adj = np.ones(96, dtype=float)
    festival_name = ""

    try:
        import holidays
        dt = pd.to_datetime(target_date)
        year = dt.year
        national = holidays.India(years=[year])

        # Check each festival profile
        for fest_key, profile in FESTIVAL_PROFILES.items():
            states = profile["states"]
            if states != "all" and state.lower().strip() not in states:
                continue

            # Find the festival date from holidays library
            fest_date = None
            for hdate, hname in national.items():
                name_lower = hname.lower()
                if fest_key.replace("_", " ") in name_lower or fest_key.split("_")[0] in name_lower:
                    fest_date = pd.to_datetime(hdate)
                    break

            if fest_date is None:
                continue

            # Check if target_date is within the festival window
            duration = profile["duration_days"]
            start_date = fest_date - timedelta(days=max(0, duration // 2))
            day_offset = (dt - start_date).days

            if 0 <= day_offset < len(profile["daily_factors"]):
                daily_factor = profile["daily_factors"][day_offset]
                adj = np.full(96, daily_factor, dtype=float)
                festival_name = fest_key.replace("_", " ").title()

                # Apply intraday block shape if this is the main festival day
                if day_offset == duration // 2 and "block_shape" in profile:
                    for shape_name, (start_b, end_b, factor) in profile["block_shape"].items():
                        adj[start_b:end_b] *= factor

                break

    except Exception as e:
        logger.warning(f"Festival adjustment failed: {e}")

    return adj, festival_name


# ── Event-Driven Features ───────────────────────────────────────────

def get_event_features(target_date: str) -> Dict[str, Any]:
    """
    Return binary/categorical features for events that affect load:
    - IPL cricket matches (evening load spikes)
    - Exam seasons (evening study load)
    - Ramadan (pre-dawn and evening cooking spikes)
    """
    dt = pd.to_datetime(target_date)
    month = dt.month
    dow = dt.weekday()

    features = {
        "is_ipl_season": 0,
        "is_exam_season": 0,
        "is_ramadan": 0,
        "is_weekend": int(dow >= 5),
    }

    # IPL season: typically Mar-May
    if month in (3, 4, 5):
        features["is_ipl_season"] = 1

    # Board exam season: Feb-Mar (CBSE/state boards)
    # University exams: May-Jun
    if month in (2, 3):
        features["is_exam_season"] = 1  # board exams
    elif month in (5, 6):
        features["is_exam_season"] = 1  # university exams

    # Ramadan: approximate (moves ~11 days earlier each year)
    # For 2026, Ramadan is approximately Feb 18 - Mar 19
    # This is a rough approximation; production should use hijri calendar
    year = dt.year
    # Base: Ramadan 2024 started ~Mar 12. Each year shifts -11 days.
    ramadan_start_2024 = pd.to_datetime("2024-03-12")
    years_diff = year - 2024
    approx_start = ramadan_start_2024 - timedelta(days=int(years_diff * 10.88))
    approx_end = approx_start + timedelta(days=29)
    if approx_start <= dt <= approx_end:
        features["is_ramadan"] = 1

    return features


def get_event_load_adjustment(
    target_date: str, baseline: np.ndarray
) -> np.ndarray:
    """Return 96-block MW adjustment for events (IPL, exams, Ramadan)."""
    adj = np.zeros(96, dtype=float)
    features = get_event_features(target_date)
    avg_load = float(np.mean(baseline)) if baseline.size > 0 else 4000.0

    # IPL match: evening blocks 68-80 see 3-8% residential spike (TV + lighting)
    if features["is_ipl_season"] and not features["is_weekend"]:
        # Weekday evening matches are more impactful (people come home and watch)
        for b in range(68, 80):
            adj[b] += avg_load * 0.05  # ~5% spike

    # Exam season: evening blocks 72-92 see study-related load increase
    if features["is_exam_season"]:
        for b in range(72, 92):
            adj[b] += avg_load * 0.02  # ~2% increase

    # Ramadan: Sehri (pre-dawn) and Iftar (evening) cooking spikes
    if features["is_ramadan"]:
        # Sehri: blocks 12-18 (03:00-04:30)
        for b in range(12, 18):
            adj[b] += avg_load * 0.03
        # Iftar: blocks 72-78 (18:00-19:30)
        for b in range(72, 78):
            adj[b] += avg_load * 0.04

    return adj


# ── Temperature Saturation for Extreme Heat ─────────────────────────

TEMP_SATURATION_PARAMS = {
    "delhi":     {"ceiling_temp": 46.0, "ac_saturation_temp": 44.0, "max_cooling_load_pct": 0.55},
    "rajasthan": {"ceiling_temp": 48.0, "ac_saturation_temp": 45.0, "max_cooling_load_pct": 0.50},
    "punjab":    {"ceiling_temp": 45.0, "ac_saturation_temp": 43.0, "max_cooling_load_pct": 0.45},
    "haryana":   {"ceiling_temp": 45.0, "ac_saturation_temp": 43.0, "max_cooling_load_pct": 0.45},
    "tamil nadu": {"ceiling_temp": 42.0, "ac_saturation_temp": 40.0, "max_cooling_load_pct": 0.40},
    "kerala":    {"ceiling_temp": 38.0, "ac_saturation_temp": 36.0, "max_cooling_load_pct": 0.30},
}


def get_temp_saturation_factor(state: str, temperature: float) -> float:
    """
    At extreme temperatures, AC load saturates (units running at max capacity).
    Returns a factor 0-1 indicating how much of the cooling potential is used.
    """
    params = TEMP_SATURATION_PARAMS.get(state.lower().strip(), {
        "ceiling_temp": 45.0, "ac_saturation_temp": 42.0, "max_cooling_load_pct": 0.45
    })
    sat_temp = params["ac_saturation_temp"]
    ceil_temp = params["ceiling_temp"]

    if temperature <= sat_temp:
        return 1.0  # linear regime, no saturation
    elif temperature >= ceil_temp:
        return 0.95  # near-flat at ceiling (slight decrease due to brownouts)
    else:
        # Logistic saturation between sat_temp and ceil_temp
        x = (temperature - sat_temp) / max(ceil_temp - sat_temp, 0.1)
        return 1.0 - 0.3 * x  # reduces effectiveness of each additional degree


def compute_india_adjustments(
    state: str,
    target_date: str,
    baseline: np.ndarray,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Master function: compute all India-specific load adjustments.
    Returns dict with individual components and total adjustment.
    """
    agri_adj = get_agricultural_adjustment(state, target_date, baseline)
    festival_adj, festival_name = get_festival_adjustment(state, target_date, baseline)
    event_adj = get_event_load_adjustment(target_date, baseline)
    event_features = get_event_features(target_date)

    # Festival is multiplicative, agriculture and events are additive
    # Total: baseline * festival_factor + agri_adj + event_adj
    total_adj = baseline * (festival_adj - 1.0) + agri_adj + event_adj

    return {
        "agricultural_adj_mw": agri_adj.tolist(),
        "festival_factor": festival_adj.tolist(),
        "festival_name": festival_name,
        "event_adj_mw": event_adj.tolist(),
        "event_features": event_features,
        "total_adj_mw": total_adj.tolist(),
        "total_adj_mean_mw": float(np.mean(total_adj)),
        "active_adjustments": [
            k for k, v in {
                "agriculture": float(np.sum(np.abs(agri_adj))) > 1.0,
                "festival": festival_name != "",
                "ipl": event_features["is_ipl_season"] == 1,
                "exams": event_features["is_exam_season"] == 1,
                "ramadan": event_features["is_ramadan"] == 1,
            }.items() if v
        ],
    }
