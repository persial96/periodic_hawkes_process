# -*- coding: utf-8 -*-
"""Small synthetic-data unit tests for pmhp.data / pmhp.forecast /
pmhp.likelihood -- the parts of the package that don't need CmdStan.
Run with `pytest` from the repo root.
"""

import numpy as np
import pandas as pd
import pytest

from pmhp import data as pmhp_data
from pmhp import forecast as pmhp_forecast
from pmhp import likelihood as pmhp_likelihood


def test_calendar_tau_leap_year_folding():
    dates = pd.Series(
        pd.to_datetime(["2020-01-01", "2020-03-01", "2020-12-31", "2021-03-01"])
    )
    tau = pmhp_data.calendar_tau(dates)

    assert tau[0] == pytest.approx(0.0)
    # 2020 is a leap year: Feb 29 is folded onto Feb 28's position, so
    # Mar 1 2020 lands on the same seasonal day as Mar 1 in a non-leap year.
    assert tau[1] == pytest.approx(59.0)
    assert tau[3] == pytest.approx(59.0)
    assert tau[2] == pytest.approx(364.0)


def test_seasonal_features_unit_circle():
    tau = np.array([0.0, 91.25, 182.5, 273.75])
    cos, sin = pmhp_data.seasonal_features(tau)
    assert np.allclose(cos**2 + sin**2, 1.0)


def test_day_of_year_exposure_sums_to_window_length():
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2021-01-01")  # 2020 is a leap year: 366 days
    T = float((end - start).total_seconds() / (24 * 3600))
    exposure = pmhp_data.day_of_year_exposure(start, end, T)
    assert exposure.sum() == pytest.approx(T)
    assert exposure.shape == (pmhp_data.SEASONAL_DAYS,)


def test_day_of_year_exposure_rejects_bad_window():
    start = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError):
        pmhp_data.day_of_year_exposure(start, start, 0.0)


def test_branching_prefix_detection():
    assert pmhp_forecast.branching_prefix(["alpha.1.1", "beta.1.1"]) == "alpha"
    assert (
        pmhp_forecast.branching_prefix(["branching_ratio.1.1", "beta.1.1"])
        == "branching_ratio"
    )
    with pytest.raises(KeyError):
        pmhp_forecast.branching_prefix(["mu.1", "beta.1.1"])


def test_forecast_expected_counts_zero_intensity_gives_zero_counts():
    D = 2
    parameters = {
        "mu": np.zeros(D),
        "alpha": np.zeros((D, D)),
        "beta": np.ones((D, D)),
        "gamma_c": 0.0,
        "gamma_s": 0.0,
        "season_log_norm": 0.0,
    }
    state = np.zeros((D, D))
    counts = pmhp_forecast.forecast_expected_counts(
        parameters, state, origin_tau=0.0, model_name="MHP", horizon_hours=24.0, D=D
    )
    assert np.allclose(counts, 0.0)


def test_forecast_expected_counts_constant_baseline_matches_closed_form():
    # With no excitation, expected counts over a horizon of h days under
    # a constant baseline mu is just mu * h (Poisson process).
    D = 1
    mu_value = 2.0
    parameters = {
        "mu": np.array([mu_value]),
        "alpha": np.zeros((D, D)),
        "beta": np.ones((D, D)),
        "gamma_c": 0.0,
        "gamma_s": 0.0,
        "season_log_norm": 0.0,
    }
    state = np.zeros((D, D))
    horizon_hours = 48.0
    counts = pmhp_forecast.forecast_expected_counts(
        parameters, state, origin_tau=0.0, model_name="MHP", horizon_hours=horizon_hours, D=D
    )
    expected = mu_value * (horizon_hours / 24.0)
    assert counts[0] == pytest.approx(expected, rel=1e-6)


def test_log_mean_exp_matches_naive_for_small_values():
    x = np.array([-1.0, 0.0, 1.0])
    naive = np.log(np.mean(np.exp(x)))
    assert pmhp_likelihood.log_mean_exp(x) == pytest.approx(naive)
