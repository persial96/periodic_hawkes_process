# -*- coding: utf-8 -*-
"""
Dataset-agnostic data prep for the periodic multivariate Hawkes model.

Everything here is deliberately unaware of Boston PD categories, file
layout, or any project-specific structure -- that lives in the paper
scripts under ``scripts/``. This module only knows how to turn a
DataFrame of (timestamp, category) events into the numpy arrays /
Stan-ready dict the model needs, including the leap-year-aware
annual seasonal calendar.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

# Fixed length of the seasonal calendar. Feb 29 is folded onto the
# Feb 28 seasonal position so every year maps onto the same 365-day
# grid; see `calendar_tau`.
SEASONAL_DAYS = 365


def build_category_mapping(category_order: Iterable[str]) -> tuple[dict, dict]:
    """Return (category_to_id, id_to_category) for a fixed category order.

    IDs start at 1 (Stan's `types` array is 1-indexed).
    """
    category_order = list(category_order)
    category_to_id = {cat: i + 1 for i, cat in enumerate(category_order)}
    id_to_category = {v: k for k, v in category_to_id.items()}
    return category_to_id, id_to_category


def map_categories(
    df: pd.DataFrame,
    category_col: str,
    category_to_id: Mapping[str, int],
) -> pd.DataFrame:
    """Filter to known categories and add an integer `hawkes_id` column."""
    known = df[df[category_col].isin(category_to_id.keys())].copy()
    known["hawkes_id"] = known[category_col].map(category_to_id).astype(int)
    return known


def check_no_ties(dates: pd.Series) -> None:
    """Raise if `dates` contains exact timestamp ties or non-positive gaps.

    The model assumes a strictly increasing, tie-free event sequence.
    Ties should already have been removed by upstream cleaning; this is
    a guard, not a fixer.
    """
    if dates.duplicated(keep=False).any():
        raise ValueError("Event timestamps contain exact ties.")

    gaps = dates.sort_values().diff().dt.total_seconds()
    if (gaps <= 0).sum() > 0:
        raise ValueError("Event timestamps contain non-positive gaps after sorting.")


def continuous_time(dates: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    """Convert timestamps to fractional days since `origin`."""
    return ((dates - origin).dt.total_seconds() / (24 * 3600)).to_numpy(dtype=float)


def calendar_tau(dates: pd.Series) -> np.ndarray:
    """Map each timestamp onto a fixed 365-day seasonal calendar.

    Leap years are folded onto the non-leap calendar:
      - Feb 29 is assigned the Feb 28 seasonal position.
      - Dates after Feb 29 in a leap year are shifted back by one day.

    Returns tau in [0, 365).
    """
    year_start = pd.to_datetime(dates.dt.year.astype(str) + "-01-01")
    tau = ((dates - year_start).dt.total_seconds() / (24 * 3600)).to_numpy(
        dtype=float, copy=True
    )

    is_leap = dates.dt.is_leap_year.to_numpy(dtype=bool, copy=True)
    month = dates.dt.month.to_numpy(dtype=int, copy=True)
    day = dates.dt.day.to_numpy(dtype=int, copy=True)

    is_feb29 = (month == 2) & (day == 29)
    after_feb29 = month > 2

    tau[is_leap & is_feb29] -= 1.0
    tau[is_leap & after_feb29] -= 1.0

    return np.mod(tau, float(SEASONAL_DAYS))


def seasonal_features(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One annual Fourier harmonic: cos/sin(2*pi*tau/365)."""
    angle = 2.0 * np.pi * tau / float(SEASONAL_DAYS)
    return np.cos(angle), np.sin(angle)


def seasonal_grid(G: int = SEASONAL_DAYS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint-of-day seasonal grid used for baseline evaluation.

    Returns (tau_grid, cos_grid, sin_grid), each length G.
    """
    tau_grid = np.arange(G, dtype=float) + 0.5
    angle = 2.0 * np.pi * tau_grid / float(SEASONAL_DAYS)
    return tau_grid, np.cos(angle), np.sin(angle)


def day_of_year_exposure(
    observation_start: pd.Timestamp,
    observation_end: pd.Timestamp,
    T_observation: float,
) -> np.ndarray:
    """Total training exposure (in days) for each day of the 365-day
    seasonal calendar, accounting for partial first/last days and
    leap-year folding. Sums to `T_observation`.
    """
    if observation_end <= observation_start:
        raise ValueError("observation_end must be after observation_start.")

    day_exposure = np.zeros(SEASONAL_DAYS, dtype=float)
    current = pd.Timestamp(observation_start)
    end = pd.Timestamp(observation_end)

    while current < end:
        next_day = current.normalize() + pd.Timedelta(days=1)
        segment_end = min(next_day, end)
        days_in_segment = (segment_end - current).total_seconds() / (24 * 3600)
        midpoint = current + (segment_end - current) / 2

        doy = midpoint.dayofyear
        if midpoint.is_leap_year:
            if midpoint.month == 2 and midpoint.day == 29:
                doy = 59
            elif midpoint.month > 2:
                doy = doy - 1

        day_exposure[doy - 1] += days_in_segment
        current = segment_end

    exposure_sum = float(day_exposure.sum())
    if not np.isclose(exposure_sum, T_observation, rtol=1e-10, atol=1e-8):
        raise ValueError(
            f"Day exposure does not sum to T_end. "
            f"sum(day_exposure)={exposure_sum}, T_end={T_observation}"
        )

    return day_exposure
