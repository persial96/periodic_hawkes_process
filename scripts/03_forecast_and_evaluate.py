# -*- coding: utf-8 -*-
"""
Rolling forecast evaluation: MHP (constant baseline) vs PMHP (seasonal
baseline), across a set of forecast horizons, producing one row per
(unit, model, horizon, forecast window, category) with predicted vs.
observed counts.

This is the forecasting core of the old `rocroc.py` (its
`read_unit_data` + main rolling-forecast loop). The ROC/PR curves,
alarm-fraction hit-rate tables and plotting further down that file are
not ported yet -- see the note at the bottom of this docstring.

Usage
-----
    python scripts/03_forecast_and_evaluate.py \
        --seasonal-results-root results/mcmc/seasonal \
        --constant-results-root results/mcmc/constant \
        --data-root results/train_test \
        --output-csv results/forecasts/full_city_rolling_forecast.csv

Note: alarm-fraction hit rates, ROC/PR curves and the figure/table
generation from the rest of the original `rocroc.py` and from
`new_reader.py` still need to be ported -- that's the next step
(04_make_figures_tables.py) once we agree which plots/tables you
actually want kept.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from categories import CATEGORY_ORDER
from pmhp.forecast import (
    advance_state_with_observed_events,
    fixed_calendar_tau,
    forecast_expected_counts,
    initial_excitation_state,
    read_posterior_mean_parameters,
)

# NOTE: this is 0-indexed (matches numpy array positions used below),
# unlike pmhp.fit's / categories.CATEGORY_TO_ID which is 1-indexed for
# Stan. Two different conventions for two different consumers.
CATEGORY_TO_ID = {c: i for i, c in enumerate(CATEGORY_ORDER)}
D = len(CATEGORY_ORDER)

DATE_COL = "OCCURRED_ON_DATE"
CATEGORY_COL = "hawkes_category"

TRAIN_START = pd.Timestamp("2020-01-01")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-04-01")

FORECAST_HORIZONS = [6, 24, 48]


def seasonal_grid(G: int = 365) -> tuple[np.ndarray, np.ndarray]:
    tau_grid = np.arange(G, dtype=float) + 0.5
    angle = 2.0 * np.pi * tau_grid / 365.0
    return np.cos(angle), np.sin(angle)


def common_forecast_end(test_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.Timestamp:
    """Largest test-period cutoff evenly divisible by every horizon, so
    all horizons evaluate over the exact same calendar span.
    """
    max_horizon_hours = max(FORECAST_HORIZONS)
    total_test_hours = int((test_end - test_start).total_seconds() / 3600)
    usable_test_hours = (total_test_hours // max_horizon_hours) * max_horizon_hours
    return test_start + pd.Timedelta(hours=usable_test_hours)


def read_unit_data(data_root: Path, unit: str) -> dict:
    """Load a unit's cleaned train/test CSVs and restrict to the
    training window / complete test window used for evaluation.
    """
    train_path = data_root / unit / f"{unit}_train_2020_2024.csv"
    test_path = data_root / unit / f"{unit}_test_2025_end.csv"

    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)

    for df in (train, test):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    train = train.dropna(subset=[DATE_COL, CATEGORY_COL]).copy()
    test = test.dropna(subset=[DATE_COL, CATEGORY_COL]).copy()

    train = train[(train[DATE_COL] >= TRAIN_START) & (train[DATE_COL] < TEST_START)].copy()
    test = test[(test[DATE_COL] >= TEST_START) & (test[DATE_COL] < TEST_END)].copy()

    train = train[train[CATEGORY_COL].isin(CATEGORY_ORDER)].copy()
    test = test[test[CATEGORY_COL].isin(CATEGORY_ORDER)].copy()

    train = train.sort_values(DATE_COL).reset_index(drop=True)
    test = test.sort_values(DATE_COL).reset_index(drop=True)

    train["type_id"] = train[CATEGORY_COL].map(CATEGORY_TO_ID).astype(int)
    test["type_id"] = test[CATEGORY_COL].map(CATEGORY_TO_ID).astype(int)

    if train[DATE_COL].duplicated().any():
        raise ValueError(f"{unit}: training timestamp ties.")
    if test[DATE_COL].duplicated().any():
        raise ValueError(f"{unit}: test timestamp ties.")

    train_times = (train[DATE_COL] - TRAIN_START).dt.total_seconds().to_numpy(dtype=float) / (
        24 * 3600
    )
    test_times = (test[DATE_COL] - TRAIN_START).dt.total_seconds().to_numpy(dtype=float) / (
        24 * 3600
    )
    train_types = train["type_id"].to_numpy(dtype=int)
    test_types = test["type_id"].to_numpy(dtype=int)

    return {
        "train": train,
        "test": test,
        "train_times": train_times,
        "test_times": test_times,
        "train_types": train_types,
        "test_types": test_types,
    }


def run_rolling_forecast(
    unit: str,
    data: dict,
    model_parameters: dict,
    forecast_end: pd.Timestamp,
) -> list[dict]:
    """Roll a non-overlapping forecast forward across the test period
    for each model and each horizon, scoring predicted vs. observed
    counts per category, updating history with only the events that
    actually occurred (a proper one-step-ahead rolling evaluation).
    """
    test_times = data["test_times"]
    test_types = data["test_types"]
    rows: list[dict] = []

    common_hours = (forecast_end - TEST_START).total_seconds() / 3600

    for horizon_hours in FORECAST_HORIZONS:
        if common_hours % horizon_hours != 0:
            raise ValueError(
                f"Common evaluation period is not divisible by {horizon_hours} hours."
            )

        for model_name, parameters in model_parameters.items():
            state = initial_excitation_state(
                parameters,
                data["train_times"],
                data["train_types"],
                D,
                test_start_days=(TEST_START - TRAIN_START).total_seconds() / (24 * 3600),
            )

            forecast_origins = pd.date_range(
                start=TEST_START,
                end=forecast_end - pd.Timedelta(hours=horizon_hours),
                freq=f"{horizon_hours}h",
            )

            print(f"{unit} | {model_name} | {horizon_hours}h | {len(forecast_origins)} windows")

            for forecast_number, origin in enumerate(forecast_origins, start=1):
                window_end = origin + pd.Timedelta(hours=horizon_hours)
                origin_days = (origin - TRAIN_START).total_seconds() / (24 * 3600)
                end_days = (window_end - TRAIN_START).total_seconds() / (24 * 3600)

                origin_tau = fixed_calendar_tau(origin)
                predicted_counts = forecast_expected_counts(
                    parameters, state, origin_tau, model_name, horizon_hours, D
                )

                lo = np.searchsorted(test_times, origin_days, side="left")
                hi = np.searchsorted(test_times, end_days, side="left")
                interval_times = test_times[lo:hi]
                interval_types = test_types[lo:hi]
                observed_counts = np.bincount(interval_types, minlength=D)

                for category_id, category_name in enumerate(CATEGORY_ORDER):
                    rows.append(
                        {
                            "unit": unit,
                            "model": model_name,
                            "horizon_hours": horizon_hours,
                            "forecast_origin": origin,
                            "forecast_end": window_end,
                            "category": category_name,
                            "predicted_expected_count": predicted_counts[category_id],
                            "observed_count": int(observed_counts[category_id]),
                            "event_window": int(observed_counts[category_id] > 0),
                        }
                    )

                state = advance_state_with_observed_events(
                    state, parameters, interval_times, interval_types, origin_days, end_days
                )

                if forecast_number % 250 == 0 or forecast_number == len(forecast_origins):
                    print(f"  {forecast_number:,} / {len(forecast_origins):,}")

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasonal-results-root", required=True, type=Path)
    p.add_argument("--constant-results-root", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--unit", default="full_city")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    forecast_end = common_forecast_end(TEST_START, TEST_END)
    print(f"Common operational forecast period: {TEST_START} to {forecast_end}")

    grid_cos, grid_sin = seasonal_grid()

    results_roots = {
        "MHP": args.constant_results_root,
        "PMHP": args.seasonal_results_root,
    }
    model_parameters = {
        model_name: read_posterior_mean_parameters(
            results_root, args.unit, model_name, D, grid_cos, grid_sin
        )
        for model_name, results_root in results_roots.items()
    }

    data = read_unit_data(args.data_root, args.unit)
    rows = run_rolling_forecast(args.unit, data, model_parameters, forecast_end)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(result_df):,} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
