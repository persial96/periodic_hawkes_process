# -*- coding: utf-8 -*-
"""
Held-out point-process log-likelihood for a fitted periodic MHP,
evaluated over the full posterior (not just the posterior mean) --
used to compare a constant-baseline model against a seasonal one via
each model's LPPD (log pointwise predictive density) on test-period
events.

Author: Persia Luca (2026), Università della Svizzera italiana, Lugano, Switzerland
Notes: additional revision used Claude (model Sonnet 5) to improve code clarity and maintainability.
"""

from __future__ import annotations
from typing import Literal
import numpy as np, pandas as pd

ModelName = Literal["constant", "seasonal"]


def log_mean_exp(x: np.ndarray) -> float:
    """Numerically stable log(mean(exp(x))).
    Used to compute LPPD = log(mean_m(exp(test_loglik_m))) (Equation 14)    
    """
    x = np.asarray(x, dtype=float)
    xmax = np.max(x)
    return xmax + np.log(np.mean(np.exp(x - xmax)))


def extract_parameter_draws(draws: pd.DataFrame, model_name: ModelName, D: int) -> dict:
    """Per-draw parameter arrays (mu: M x D, alpha/beta: M x D x D, gamma_c/gamma_s: M) from a unit's posterior draws -- unlike a
    posterior-mean-only extraction, this keeps every draw so the likelihood below can be Monte Carlo-averaged over the posterior
    rather than evaluated at the posterior mean.
    """
    M = len(draws)

    mu = np.zeros((M, D), dtype=float)
    for i in range(D):
        col = f"mu.{i + 1}"
        if col not in draws.columns:
            raise KeyError(f"{model_name}: missing posterior column {col}")
        mu[:, i] = pd.to_numeric(draws[col], errors="raise").to_numpy()

    # alpha (branching ratio); some chains store it as branching_ratio
    if "alpha.1.1" in draws.columns:
        alpha_prefix = "alpha"
    elif "branching_ratio.1.1" in draws.columns:
        alpha_prefix = "branching_ratio"
    else:
        raise KeyError(f"{model_name}: neither alpha nor branching_ratio was found.")

    alpha = np.zeros((M, D, D), dtype=float)
    beta = np.zeros((M, D, D), dtype=float)
    for i in range(D):
        for j in range(D):
            alpha_col = f"{alpha_prefix}.{i + 1}.{j + 1}"
            beta_col = f"beta.{i + 1}.{j + 1}"
            if alpha_col not in draws.columns:
                raise KeyError(f"{model_name}: missing posterior column {alpha_col}")
            if beta_col not in draws.columns:
                raise KeyError(f"{model_name}: missing posterior column {beta_col}")
            alpha[:, i, j] = pd.to_numeric(draws[alpha_col], errors="raise").to_numpy()
            beta[:, i, j] = pd.to_numeric(draws[beta_col], errors="raise").to_numpy()

    if model_name == "seasonal":
        if "season_cos_coef" in draws.columns and "season_sin_coef" in draws.columns:
            gamma_c = pd.to_numeric(draws["season_cos_coef"], errors="raise").to_numpy()
            gamma_s = pd.to_numeric(draws["season_sin_coef"], errors="raise").to_numpy()
        elif "season_coef.1" in draws.columns and "season_coef.2" in draws.columns:
            gamma_c = pd.to_numeric(draws["season_coef.1"], errors="raise").to_numpy()
            gamma_s = pd.to_numeric(draws["season_coef.2"], errors="raise").to_numpy()
        else:
            raise KeyError("Could not find the seasonal Fourier coefficients.")
    else:
        gamma_c = np.zeros(M, dtype=float)
        gamma_s = np.zeros(M, dtype=float)

    for name, values in (("mu", mu), ("alpha", alpha), ("beta", beta)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{model_name}: non-finite {name} draws.")
        if np.any(values <= 0):
            # mu, alpha, beta all have positive-support priors (Gamma or
            # Beta); a non-positive draw here means either a corrupted
            # chain file or a column mismatch above, not a valid posterior.
            raise ValueError(f"{model_name}: non-positive {name} draw.")

    return {"mu": mu, "alpha": alpha, "beta": beta, "gamma_c": gamma_c, "gamma_s": gamma_s}


def initial_excitation_draws(
    parameters: dict,
    train_times: np.ndarray,
    train_types: np.ndarray,
    D: int,
    test_start_days: float,
    batch_size: int = 100,
) -> np.ndarray:
    """Excitation state S[m, i, j] (posterior draw m, child i, parent
    j) carried from the full training history into TEST_START, for
    every posterior draw. Batches over draws to bound memory.

    S[m, i, j] is the total remaining intensity that category-j
    training events still contribute to category-i's intensity at the
    moment the test period begins -- i.e. sum over every parent event
    of alpha_ij * beta_ij * exp(-beta_ij * age), where age is how long
    ago that event happened relative to test_start_days. This lets the
    held-out likelihood account for excitation generated *before* the
    test period (per Section 3.6: "excitation generated before the
    beginning of the test period is retained").

    Batching over posterior draws (rather than vectorizing over all M
    draws at once) keeps peak memory bounded: the per-parent residual
    sum below is an (batch_size x n_parent_events) array, which for the
    full posterior across a busy category could otherwise be huge.
    """
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    M = len(alpha)

    S = np.zeros((M, D, D), dtype=float)

    for parent in range(D):
        parent_times = train_times[train_types == parent]
        if len(parent_times) == 0:
            continue

        ages = test_start_days - parent_times
        if np.any(ages <= 0):
            raise ValueError("Training event found at or after the test start.")

        for child in range(D):
            for start in range(0, M, batch_size):
                stop = min(start + batch_size, M)
                beta_batch = beta[start:stop, child, parent]
                alpha_batch = alpha[start:stop, child, parent]
                # sum_n exp(-beta * age_n), i.e. the surviving fraction
                # of every past parent-category event's excitation.
                residual_sum = np.exp(-beta_batch[:, None] * ages[None, :]).sum(axis=1)
                S[start:stop, child, parent] = alpha_batch * beta_batch * residual_sum

    return S


def evaluate_test_likelihood(
    parameters: dict,
    model_name: ModelName,
    D: int,
    train_times: np.ndarray,
    train_types: np.ndarray,
    test_times: np.ndarray,
    test_types: np.ndarray,
    test_cos: np.ndarray,
    test_sin: np.ndarray,
    test_start_days: float,
    test_end_days: float,
    test_day_exposure: np.ndarray,
    grid_cos: np.ndarray,
    grid_sin: np.ndarray,
    batch_size: int = 100,
) -> dict:
    """Held-out point-process log-likelihood on the test period, one
    value per posterior draw. Decomposed into the event log-intensity
    term and the baseline/excitation compensators (the integrated
    intensity subtracted per the point-process likelihood).

    This implements Equation (13): for each posterior draw theta^(m),
        ell_test(theta^(m)) = sum_n log(lambda_cn(t_n | H_t-, theta^(m)))
                               - sum_i integral_{Ttest0}^{Ttest1} lambda_i(t) dt
    The first (sum-of-logs) term is `event_log_term` below; the second
    (integrated intensity) term is split into `baseline_compensator`
    (the mu*s(t) part) and `excitation_compensator` (the triggering
    part), computed exactly via the exponential kernel's closed-form
    integral rather than numerically.

    test_day_exposure comes from `pmhp.data.day_of_year_exposure`
    evaluated on the *test* window (not the training window that
    `pmhp.fit.prepare_stan_data` uses).
    """
    mu = parameters["mu"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    gamma_c = parameters["gamma_c"]
    gamma_s = parameters["gamma_s"]

    M = len(mu)
    N_test = len(test_times)
    test_duration = test_end_days - test_start_days

    # --- Baseline compensator: integral of mu_i * s(t) over the whole
    # test window, summed across categories (the "S_T^(G)" term from
    # Appendix B, evaluated here on the test window's exposure grid). ---
    if model_name == "seasonal":
        # Same normalization as the Stan model: s(t) is exp(f(tau(t)))
        # divided by its average over the annual grid, so it has annual
        # mean 1. Computed per-draw since gamma_c/gamma_s vary by draw.
        f_grid = gamma_c[:, None] * grid_cos[None, :] + gamma_s[:, None] * grid_sin[None, :]
        f_max = np.max(f_grid, axis=1)
        season_log_norm = f_max + np.log(np.mean(np.exp(f_grid - f_max[:, None]), axis=1))
        season_factor_grid = np.exp(f_grid - season_log_norm[:, None])
        # Exposure-weighted sum over the 365-day grid approximates
        # integral s(t) dt over the test window (same midpoint
        # approximation the training likelihood uses).
        seasonal_exposure = season_factor_grid @ test_day_exposure
        baseline_compensator = np.sum(mu, axis=1) * seasonal_exposure
    else:
        # Constant baseline: integral of mu dt over the window is just mu * duration.
        season_log_norm = np.zeros(M, dtype=float)
        baseline_compensator = np.sum(mu, axis=1) * test_duration

    # --- Excitation state carried in from training history (see
    # initial_excitation_draws' docstring for what S[m, i, j] means). ---
    S = initial_excitation_draws(
        parameters, train_times, train_types, D, test_start_days, batch_size
    )

    event_log_term = np.zeros(M, dtype=float)
    excitation_compensator = np.zeros(M, dtype=float)
    previous_time = test_start_days

    # Walk the test events one at a time, maintaining S as a running
    # excitation state. This mirrors how the exponential kernel lets a
    # Hawkes process be evaluated in O(N) rather than by re-summing the
    # full event history at every step: between events, S decays
    # deterministically; at each event, S is updated once (see below).
    for event_number in range(N_test):
        current_time = test_times[event_number]
        event_type = test_types[event_number]
        dt = current_time - previous_time
        if dt < 0:
            raise ValueError("Test events are not ordered.")

        if dt > 0:
            # Accumulate the excitation compensator's contribution from
            # this gap: integral of S * exp(-beta*u) du from 0 to dt,
            # which has the closed form S * (1 - exp(-beta*dt)) / beta
            # (the exponential kernel's exact finite-window integral,
            # Equation 12's excitation term with tm -> "now" and
            # T -> "now + dt"). Then decay S itself to the new time.
            one_minus_decay = -np.expm1(-beta * dt)  # 1 - exp(-beta*dt), stable for small beta*dt
            excitation_compensator += np.sum(S * one_minus_decay / beta, axis=(1, 2))
            S = S * np.exp(-beta * dt)

        if model_name == "seasonal":
            season_event = np.exp(
                gamma_c * test_cos[event_number] + gamma_s * test_sin[event_number]
                - season_log_norm
            )
        else:
            season_event = 1.0

        # Conditional intensity of this event's own category at its own
        # time: background + whatever excitation has survived to now
        # (S[:, event_type, :] summed over parent categories j).
        baseline_event = mu[:, event_type] * season_event
        excitation_event = np.sum(S[:, event_type, :], axis=1)
        lambda_event = baseline_event + excitation_event

        if np.any(lambda_event <= 0):
            raise ValueError("Non-positive event intensity encountered.")

        event_log_term += np.log(lambda_event)

        # This event now becomes part of the history: it adds a fresh
        # alpha_ij * beta_ij contribution (at age 0) to every child
        # category i's excitation state, for parent category = event_type.
        S[:, :, event_type] += alpha[:, :, event_type] * beta[:, :, event_type]
        previous_time = current_time

    # Excitation compensator for the tail after the last test event, up
    # to the end of the test window (same closed-form integral as above).
    final_dt = test_end_days - previous_time
    if final_dt < 0:
        raise ValueError("Last test event occurs after the test window end.")
    if final_dt > 0:
        one_minus_decay = -np.expm1(-beta * final_dt)
        excitation_compensator += np.sum(S * one_minus_decay / beta, axis=(1, 2))

    loglik = event_log_term - baseline_compensator - excitation_compensator
    if not np.all(np.isfinite(loglik)):
        raise ValueError(f"{model_name}: non-finite test log likelihood.")

    return {
        "loglik": loglik,
        "event_log_term": event_log_term,
        "baseline_compensator": baseline_compensator,
        "excitation_compensator": excitation_compensator,
    }


def equal_draw_mc_delta_lppd(
    constant_loglik: np.ndarray,
    seasonal_loglik: np.ndarray,
    seed: int,
    n_repetitions: int = 100,
    max_subsample: int = 1000,
) -> np.ndarray:
    """Repeatedly subsample both models to the same number of draws
    and compute delta LPPD (seasonal minus constant), as a robustness
    check against the two models having different posterior draw
    counts.

    LPPD is computed via log_mean_exp, so a model with more retained
    draws isn't inherently biased toward a higher or lower LPPD --
    but subsampling both models to a common draw count and repeating
    many times (Section 5.5's Monte Carlo robustness check) confirms
    the reported delta-LPPD isn't an artifact of which specific draws
    happened to be sampled. Returns one delta value per repetition, so
    callers can report e.g. the 5-95% range and the fraction > 0.
    """
    rng = np.random.default_rng(seed)
    n_subsample = min(max_subsample, len(constant_loglik) // 2, len(seasonal_loglik) // 2)

    deltas = np.empty(n_repetitions, dtype=float)
    for repetition in range(n_repetitions):
        constant_idx = rng.choice(len(constant_loglik), size=n_subsample, replace=False)
        seasonal_idx = rng.choice(len(seasonal_loglik), size=n_subsample, replace=False)
        deltas[repetition] = log_mean_exp(seasonal_loglik[seasonal_idx]) - log_mean_exp(
            constant_loglik[constant_idx]
        )
    return deltas
