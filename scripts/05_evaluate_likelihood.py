# -*- coding: utf-8 -*-
"""
Held-out point-process log-likelihood comparison: constant-baseline MHP vs. seasonal-baseline PMHP, 
across full_city and every district, with an equal-draw Monte Carlo robustness check on the LPPD difference.

Author: Persia Luca (2026), Università della Svizzera italiana, Lugano, Switzerland
Notes: additional revision used Claude Code (model Sonnet 5 and Opus 4.8) to improve code clarity and maintainability.
"""

from __future__ import annotations
import argparse, numpy as np, pandas as pd
from pathlib import Path
from categories import CATEGORY_ORDER
from pmhp.data import calendar_tau, day_of_year_exposure, seasonal_features, seasonal_grid
from pmhp.posterior import read_posterior_draws
from pmhp.likelihood import (
    equal_draw_mc_delta_lppd,
    evaluate_test_likelihood,
    extract_parameter_draws,
    log_mean_exp,
)

# 0-indexed to match numpy array positions
CATEGORY_TO_ID = {c: i for i, c in enumerate(CATEGORY_ORDER)}
D = len(CATEGORY_ORDER)

# constants for the held-out likelihood evaluation
DATE_COL = "OCCURRED_ON_DATE"
CATEGORY_COL = "hawkes_category"
TRAIN_START = pd.Timestamp("2020-01-01")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-04-01")
DEFAULT_UNITS = [
    "full_city",
    "district_A1", "district_A15", "district_A7",
    "district_B2", "district_B3",
    "district_C11", "district_C6",
    "district_D14", "district_D4",
    "district_E13", "district_E18", "district_E5",
]

# Monte Carlo parameters for equal-draw LPPD difference robustness check
N_MC_REPETITIONS = 100
RANDOM_SEED = 12345
BATCH_SIZE = 100


def read_unit_data(data_root: Path, unit: str) -> dict:
    """Load a unit's cleaned train/test CSVs; compute the seasonal covariates (cos/sin of calendar position) at each test event time.
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
        # drop any rows with invalid timestamps
        if df[DATE_COL].dt.tz is not None:
            df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)

    train = train.dropna(subset=[DATE_COL, CATEGORY_COL]).copy()
    test = test.dropna(subset=[DATE_COL, CATEGORY_COL]).copy()

    train = train[(train[DATE_COL] >= TRAIN_START) & (train[DATE_COL] < TEST_START)].copy()
    test = test[(test[DATE_COL] >= TEST_START) & (test[DATE_COL] < TEST_END)].copy()

    train = train[train[CATEGORY_COL].isin(CATEGORY_ORDER)].copy()
    test = test[test[CATEGORY_COL].isin(CATEGORY_ORDER)].copy()

    train = train.sort_values(DATE_COL).reset_index(drop=True)
    test = test.sort_values(DATE_COL).reset_index(drop=True)
    # map category names to integer ids for Stan
    train["type_id"] = train[CATEGORY_COL].map(CATEGORY_TO_ID).astype(int)
    test["type_id"] = test[CATEGORY_COL].map(CATEGORY_TO_ID).astype(int)

    if train[DATE_COL].duplicated().any():
        raise ValueError(f"{unit}: training data contain exact timestamp ties.")
    if test[DATE_COL].duplicated().any():
        raise ValueError(f"{unit}: test data contain exact timestamp ties.")

    train_times = (train[DATE_COL] - TRAIN_START).dt.total_seconds().to_numpy(dtype=float) / (24 * 3600)
    test_times = (test[DATE_COL] - TRAIN_START).dt.total_seconds().to_numpy(dtype=float) / (24 * 3600)
    train_types = train["type_id"].to_numpy(dtype=int)
    test_types = test["type_id"].to_numpy(dtype=int)

    tau_test = calendar_tau(test[DATE_COL])
    test_cos, test_sin = seasonal_features(tau_test)

    return {
        "train": train, "test": test,
        "train_times": train_times, "test_times": test_times,
        "train_types": train_types, "test_types": test_types,
        "test_cos": test_cos, "test_sin": test_sin,
    }


def evaluate_unit(
    unit: str,
    unit_number: int,
    data: dict,
    results_roots: dict[str, Path],
    grid_cos: np.ndarray,
    grid_sin: np.ndarray,
    test_day_exposure: np.ndarray,
    test_start_days: float,
    test_end_days: float,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    """Fit vs test likelihood comparison for one model unit. Returns (comparison row, list of Monte Carlo repetition rows).
    """
    N_train = len(data["train"])
    N_test = len(data["test"])
    if N_test == 0:
        raise ValueError(f"{unit}: no test events.")
    print(f"Training events: {N_train:,} | Test events: {N_test:,}")

    unit_results = {}
    for model_name, results_root in results_roots.items():
        print(f"\n{unit} | {model_name.upper()}")
        draws = read_posterior_draws(results_root, unit)
        parameters = extract_parameter_draws(draws, model_name, D)

        components = evaluate_test_likelihood(
            parameters, model_name, D,
            data["train_times"], data["train_types"],
            data["test_times"], data["test_types"],
            data["test_cos"], data["test_sin"],
            test_start_days, test_end_days,
            test_day_exposure, grid_cos, grid_sin,
            batch_size=BATCH_SIZE,
        )
        loglik = components["loglik"]
        lppd = log_mean_exp(loglik)
        unit_results[model_name] = {"draws": len(loglik), "loglik": loglik, "lppd": lppd}
        print(f"mean test log likelihood: {np.mean(loglik):,.3f}")
        print(f"LPPD:                     {lppd:,.3f}")

        # save the likelihood components for this model unit and model type
        component_df = pd.DataFrame(
            {
                "test_loglik": components["loglik"],
                "event_log_term": components["event_log_term"],
                "baseline_compensator": components["baseline_compensator"],
                "excitation_compensator": components["excitation_compensator"],
            }
        )
        component_df.to_csv(output_dir / f"{unit}_{model_name}_test_likelihood_components.csv", index=False)

    # compute the LPPD difference and Monte Carlo equal-draw robustness check
    constant_lppd = unit_results["constant"]["lppd"]
    seasonal_lppd = unit_results["seasonal"]["lppd"]
    delta_lppd = seasonal_lppd - constant_lppd
    delta_lppd_per_event = delta_lppd / N_test
    # also compute the mean log-likelihood difference across all draws (not just the LPPD)
    constant_ll = unit_results["constant"]["loglik"]
    seasonal_ll = unit_results["seasonal"]["loglik"]

    n_equal_draws = min(len(constant_ll), len(seasonal_ll))
    delta_lppd_rep = equal_draw_mc_delta_lppd(constant_ll, seasonal_ll, seed=RANDOM_SEED + unit_number, n_repetitions=N_MC_REPETITIONS)
    # compute summary statistics for the Monte Carlo repetitions
    mc_mean = float(np.mean(delta_lppd_rep))
    mc_sd = float(np.std(delta_lppd_rep, ddof=1))
    mc_q05 = float(np.quantile(delta_lppd_rep, 0.05))
    mc_q95 = float(np.quantile(delta_lppd_rep, 0.95))
    mc_fraction_positive = float(np.mean(delta_lppd_rep > 0))
    n_subsample = min(1000, len(constant_ll) // 2, len(seasonal_ll) // 2)

    print(f"\nDelta LPPD: {delta_lppd:,.3f} | MC 5-95%: [{mc_q05:,.3f}, {mc_q95:,.3f}] "
          f"| MC fraction > 0: {mc_fraction_positive:.3f}")

    comparison_row = {
        "unit": unit,
        "train_events": N_train,
        "test_events": N_test,
        "test_days": test_end_days - test_start_days,
        "mhp_draws": unit_results["constant"]["draws"],
        "pmhp_draws": unit_results["seasonal"]["draws"],
        "mhp_mean_test_loglik": np.mean(constant_ll),
        "pmhp_mean_test_loglik": np.mean(seasonal_ll),
        "mhp_lppd": constant_lppd,
        "pmhp_lppd": seasonal_lppd,
        "delta_lppd_pmhp_minus_mhp": delta_lppd,
        "delta_lppd_per_event": delta_lppd_per_event,
        "mc_equal_draws": n_subsample,
        "mc_mean_delta_lppd": mc_mean,
        "mc_sd_delta_lppd": mc_sd,
        "mc_q05_delta_lppd": mc_q05,
        "mc_q95_delta_lppd": mc_q95,
        "mc_fraction_positive": mc_fraction_positive,
    }
    mc_rows = [
        {"unit": unit, "repetition": i + 1, "equal_draws": n_equal_draws, "delta_lppd_pmhp_minus_mhp": v}
        for i, v in enumerate(delta_lppd_rep)
    ]
    return comparison_row, mc_rows


def build_comparison_table(comparison_rows: list[dict]) -> pd.DataFrame:
    comparison = pd.DataFrame(comparison_rows)
    comparison["display_unit"] = (comparison["unit"].replace({"full_city": "Full city"}).str.replace("district_", "", regex=False))
    comparison["_sort_full_city"] = (comparison["unit"] != "full_city").astype(int)
    comparison = (
        comparison.sort_values(["_sort_full_city", "display_unit"])
        .drop(columns=["_sort_full_city"])
        .reset_index(drop=True)
    )
    return comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasonal-results-root", required=True, type=Path)
    p.add_argument("--constant-results-root", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--units", nargs="+", default=DEFAULT_UNITS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_roots = {"constant": args.constant_results_root, "seasonal": args.seasonal_results_root}

    test_start_days = float((TEST_START - TRAIN_START).total_seconds() / (24 * 3600))
    test_end_days = float((TEST_END - TRAIN_START).total_seconds() / (24 * 3600))
    _, grid_cos, grid_sin = seasonal_grid()
    test_day_exposure = day_of_year_exposure(TEST_START, TEST_END, test_end_days - test_start_days)

    comparison_rows = []
    mc_rows = []
    for unit_number, unit in enumerate(args.units, start=1):
        print(f"\n{'=' * 70}\nUNIT {unit_number}/{len(args.units)}: {unit}\n{'=' * 70}")
        data = read_unit_data(args.data_root, unit)
        comparison_row, unit_mc_rows = evaluate_unit(
            unit, unit_number, data, results_roots, grid_cos, grid_sin,
            test_day_exposure, test_start_days, test_end_days, args.output_dir,
        )
        comparison_rows.append(comparison_row)
        mc_rows.extend(unit_mc_rows)

    comparison = build_comparison_table(comparison_rows)
    comparison.to_csv(args.output_dir / "heldout_model_comparison_all_units.csv", index=False)

    mc_results = pd.DataFrame(mc_rows)
    mc_results.to_csv(args.output_dir / "heldout_model_comparison_mc_repetitions.csv", index=False)

    paper_table = comparison[
        [
            "display_unit", "test_events", "mhp_lppd", "pmhp_lppd",
            "delta_lppd_pmhp_minus_mhp", "delta_lppd_per_event",
            "mc_q05_delta_lppd", "mc_q95_delta_lppd", "mc_fraction_positive",
        ]
    ].copy()
    paper_table.columns = [
        "Unit", "Test events", "MHP LPPD", "PMHP LPPD", "Delta LPPD",
        "Delta LPPD per event", "MC 5% Delta", "MC 95% Delta", "MC fraction positive",
    ]
    paper_table.to_csv(args.output_dir / "heldout_model_comparison_paper_table.csv", index=False)

    print(f"\n{'=' * 100}\nHELD-OUT MODEL COMPARISON: FULL CITY AND DISTRICTS\n{'=' * 100}")
    print(paper_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    n_units = len(comparison)
    n_positive = int((comparison["delta_lppd_pmhp_minus_mhp"] > 0).sum())
    n_mc_positive_all = int((comparison["mc_fraction_positive"] == 1.0).sum())
    print(f"\nPMHP has larger all-draw LPPD in {n_positive}/{n_units} model units.")
    print(
        f"Equal-draw Monte Carlo Delta LPPD is positive in every repetition for "
        f"{n_mc_positive_all}/{n_units} model units."
    )
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
