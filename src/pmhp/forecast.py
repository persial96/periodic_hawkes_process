# -*- coding: utf-8 -*-
"""
Forecasting from a fitted periodic MHP posterior.

This module dedups the six functions that used to be copy-pasted
between `rocroc.py` and the tail of `new_reader.py`:

    read_posterior_mean_parameters
    fixed_calendar_tau
    initial_excitation_state
    season_factor
    forecast_expected_counts
    advance_state_with_observed_events

plus `resolve_result_directory`, the small helper both call sites used
to locate a unit's CmdStan output folder.

The originals relied on module-level globals (`D`, `TRAIN_START`,
`TEST_START`, `grid_cos`, `grid_sin`) set once near the top of the
script. Here every function takes those as explicit arguments instead,
so this module has no hidden state and both `scripts/03_forecast_and_
evaluate.py` and `scripts/04_make_figures_tables.py` can call the same
code with different units/paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

ModelName = Literal["MHP", "PMHP"]


def branching_prefix(columns: Iterable[str]) -> str:
    """Some chains store the branching matrix as alpha.i.j, others as
    branching_ratio.i.j -- detect which one a set of draw columns uses.
    """
    columns = list(columns)
    if any(c.startswith("branching_ratio.") for c in columns):
        return "branching_ratio"
    if any(c.startswith("alpha.") for c in columns):
        return "alpha"
    raise KeyError("Could not find alpha.i.j or branching_ratio.i.j columns.")


def resolve_result_directory(results_root: Path, unit: str) -> Path:
    """Locate a unit's CmdStan output folder, allowing for the
    'district_A1' -> 'A1' naming variant some result trees use.
    """
    candidates = [results_root / unit]
    if unit.startswith("district_"):
        candidates.append(results_root / unit.replace("district_", ""))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find {unit} under:\n{results_root}")


def read_posterior_draws(results_root: Path, unit: str) -> pd.DataFrame:
    """All posterior draws (every chain, concatenated) for one unit,
    tagged with `chain` and `draw_within_chain`. Uses
    `resolve_result_directory` so both the `district_A1` and bare `A1`
    folder-naming conventions work.
    """
    unit_dir = resolve_result_directory(results_root, unit)
    chain_files = sorted(
        p for p in unit_dir.glob("*.csv") if p.name.lower().startswith("hawkes")
    )
    if not chain_files:
        raise FileNotFoundError(f"No Stan chain CSVs found in:\n{unit_dir}")

    chains = []
    for chain_number, file_path in enumerate(chain_files, start=1):
        chain = pd.read_csv(file_path, comment="#", low_memory=False)
        chain["chain"] = chain_number
        chain["draw_within_chain"] = np.arange(len(chain))
        chains.append(chain)

    return pd.concat(chains, ignore_index=True)


def read_posterior_mean_parameters(
    results_root: Path,
    unit: str,
    model_name: ModelName,
    D: int,
    grid_cos: np.ndarray,
    grid_sin: np.ndarray,
) -> dict:
    """Posterior means of mu, alpha, beta (and, for PMHP, the seasonal
    Fourier coefficients + log-normalizer) from a unit's chain CSVs.
    """
    unit_dir = resolve_result_directory(results_root, unit)

    chain_files = sorted(
        p for p in unit_dir.glob("*.csv") if p.name.lower().startswith("hawkes")
    )
    if not chain_files:
        raise FileNotFoundError(f"No chain CSVs in:\n{unit_dir}")

    draws = pd.concat(
        [pd.read_csv(f, comment="#", low_memory=False) for f in chain_files],
        ignore_index=True,
    )

    # Baseline mu
    mu = np.zeros(D, dtype=float)
    for i in range(D):
        mu[i] = pd.to_numeric(draws[f"mu.{i + 1}"], errors="raise").mean()

    # Branching matrix (alpha) -- some chains store it as branching_ratio
    alpha_prefix = branching_prefix(draws.columns)

    alpha = np.zeros((D, D), dtype=float)
    for i in range(D):
        for j in range(D):
            col = f"{alpha_prefix}.{i + 1}.{j + 1}"
            alpha[i, j] = pd.to_numeric(draws[col], errors="raise").mean()

    # Decay matrix
    beta = np.zeros((D, D), dtype=float)
    for i in range(D):
        for j in range(D):
            col = f"beta.{i + 1}.{j + 1}"
            beta[i, j] = pd.to_numeric(draws[col], errors="raise").mean()

    # Seasonal coefficients (PMHP only)
    if model_name == "PMHP":
        if "season_cos_coef" in draws.columns and "season_sin_coef" in draws.columns:
            gamma_c = pd.to_numeric(draws["season_cos_coef"], errors="raise").mean()
            gamma_s = pd.to_numeric(draws["season_sin_coef"], errors="raise").mean()
        elif "season_coef.1" in draws.columns and "season_coef.2" in draws.columns:
            gamma_c = pd.to_numeric(draws["season_coef.1"], errors="raise").mean()
            gamma_s = pd.to_numeric(draws["season_coef.2"], errors="raise").mean()
        else:
            raise KeyError("Seasonal Fourier coefficients not found.")

        # Normalize so season_factor integrates to 1 over the year
        f_grid = gamma_c * grid_cos + gamma_s * grid_sin
        f_max = np.max(f_grid)
        season_log_norm = f_max + np.log(np.mean(np.exp(f_grid - f_max)))
    else:
        gamma_c = 0.0
        gamma_s = 0.0
        season_log_norm = 0.0

    return {
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "gamma_c": gamma_c,
        "gamma_s": gamma_s,
        "season_log_norm": season_log_norm,
    }


def fixed_calendar_tau(timestamp: pd.Timestamp) -> float:
    """Scalar version of `pmhp.data.calendar_tau` for a single timestamp,
    used inside the forecasting loop (which advances one event at a time
    rather than vectorizing over a DataFrame).
    """
    timestamp = pd.Timestamp(timestamp)
    year_start = pd.Timestamp(year=timestamp.year, month=1, day=1)
    tau = (timestamp - year_start).total_seconds() / (24 * 3600)

    if timestamp.is_leap_year:
        march_1 = pd.Timestamp(year=timestamp.year, month=3, day=1)
        feb_29 = pd.Timestamp(year=timestamp.year, month=2, day=29)
        if timestamp >= march_1:
            tau -= 1.0
        elif timestamp >= feb_29:
            tau -= 1.0

    return tau % 365.0


def initial_excitation_state(
    parameters: dict,
    train_times: np.ndarray,
    train_types: np.ndarray,
    D: int,
    test_start_days: float,
) -> np.ndarray:
    """Excitation state (D x D) carried over from training history into
    the start of the test period, from every parent's still-decaying
    contribution to each child intensity.
    """
    alpha = parameters["alpha"]
    beta = parameters["beta"]

    state = np.zeros((D, D), dtype=float)

    for parent in range(D):
        parent_times = train_times[train_types == parent]
        if len(parent_times) == 0:
            continue

        ages = test_start_days - parent_times
        for child in range(D):
            state[child, parent] = (
                alpha[child, parent]
                * beta[child, parent]
                * np.exp(-beta[child, parent] * ages).sum()
            )

    return state


def season_factor(parameters: dict, tau: float, model_name: ModelName) -> float:
    """Seasonal multiplier on the baseline intensity at calendar
    position `tau` (days into the year). Always 1.0 for the plain MHP.
    """
    if model_name == "MHP":
        return 1.0

    gamma_c = parameters["gamma_c"]
    gamma_s = parameters["gamma_s"]
    season_log_norm = parameters["season_log_norm"]

    angle = 2.0 * np.pi * (tau % 365.0) / 365.0
    return np.exp(gamma_c * np.cos(angle) + gamma_s * np.sin(angle) - season_log_norm)


def forecast_expected_counts(
    parameters: dict,
    state_at_origin: np.ndarray,
    origin_tau: float,
    model_name: ModelName,
    horizon_hours: float,
    D: int,
) -> np.ndarray:
    """Expected event counts per category over the next `horizon_hours`,
    including expected secondary triggering generated during the
    forecast window. Integrated with 4th-order Runge-Kutta at 15-minute
    steps (4 steps/hour).
    """
    mu = parameters["mu"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]

    horizon_days = horizon_hours / 24.0
    ode_steps = int(horizon_hours * 4)

    x = state_at_origin.copy()
    cumulative_count = np.zeros(D, dtype=float)
    dt = horizon_days / ode_steps

    def derivative(state, u):
        s = season_factor(parameters, origin_tau + u, model_name)
        intensity = mu * s + np.sum(state, axis=1)
        dx = -beta * state + alpha * beta * intensity[None, :]
        return dx, intensity

    for step in range(ode_steps):
        u = step * dt

        k1_x, k1_n = derivative(x, u)
        k2_x, k2_n = derivative(x + 0.5 * dt * k1_x, u + 0.5 * dt)
        k3_x, k3_n = derivative(x + 0.5 * dt * k2_x, u + 0.5 * dt)
        k4_x, k4_n = derivative(x + dt * k3_x, u + dt)

        x = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        cumulative_count = cumulative_count + (dt / 6.0) * (
            k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n
        )

    return cumulative_count


def advance_state_with_observed_events(
    state: np.ndarray,
    parameters: dict,
    event_times: np.ndarray,
    event_types: np.ndarray,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """Update the excitation state with events actually observed during
    [start_time, end_time), decaying between them. Only observed events
    enter the history used for the *next* forecast window (a proper
    rolling one-step-ahead evaluation, not a simulation).
    """
    alpha = parameters["alpha"]
    beta = parameters["beta"]

    current_time = start_time
    for event_time, event_type in zip(event_times, event_types):
        dt = event_time - current_time
        if dt < 0:
            raise ValueError("Observed events not ordered.")
        if dt > 0:
            state = state * np.exp(-beta * dt)

        state[:, event_type] += alpha[:, event_type] * beta[:, event_type]
        current_time = event_time

    final_dt = end_time - current_time
    if final_dt > 0:
        state = state * np.exp(-beta * final_dt)

    return state
