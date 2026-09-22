# -*- coding: utf-8 -*-
"""
Fitting the periodic multivariate Hawkes model with CmdStanPy.

This is the general-purpose core extracted from the old
`last_hpc_use_this.py` driver script: the data prep (seasonal Fourier
features, leap-year handling, priors) and the CmdStanPy `sample()` call
are here; SLURM array IDs, scratch-dir shuffling, and env-var reads are
NOT -- those are HPC-run-specific and live in `scripts/02_fit_models.py`
and `hpc/run_array.sh` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, CmdStanMCMC

from . import data as pmhp_data

DEFAULT_STAN_FILE = "hawkes_temporal_seasonal_beta_gamma.stan"


@dataclass
class Priors:
    """Beta-Gamma prior specification (Stan uses Gamma(shape, rate))."""

    mu_prior_shape: float = 2.0
    mu_prior_rate: float = 0.50
    alpha_prior_a: float = 1.0
    alpha_prior_b: float = 8.0
    beta_prior_shape: float = 2.0
    beta_prior_rate: float = 0.50
    season_sd: float = 0.50


@dataclass
class PreparedData:
    """Stan-ready arrays plus the bits needed to interpret them later."""

    stan_data: dict
    category_to_id: dict
    id_to_category: dict
    observation_start: pd.Timestamp
    observation_end: pd.Timestamp
    T_observation: float
    N_events: int
    D_dims: int


def prepare_stan_data(
    df: pd.DataFrame,
    *,
    date_col: str,
    category_col: str,
    category_order: list[str],
    observation_start: pd.Timestamp,
    observation_end: pd.Timestamp,
    baseline: str = "seasonal",
    priors: Optional[Priors] = None,
) -> PreparedData:
    """Turn a cleaned event DataFrame into the dict CmdStanPy expects.

    `df` must already be de-duplicated (no exact timestamp ties) and
    restricted to the categories in `category_order` -- see
    `pmhp.data.check_no_ties` / `map_categories` for the upstream checks.
    """
    if baseline not in ("seasonal", "constant"):
        raise ValueError("baseline must be 'seasonal' or 'constant'")
    use_seasonality = 1 if baseline == "seasonal" else 0
    priors = priors or Priors()

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)

    work = df.copy()
    work[date_col] = dates
    work = work.dropna(subset=[date_col, category_col]).copy()
    work = work[
        (work[date_col] >= observation_start) & (work[date_col] < observation_end)
    ].copy()
    work = work.sort_values(date_col).reset_index(drop=True)

    if work.empty:
        raise ValueError("No events remain after applying the observation window.")

    pmhp_data.check_no_ties(work[date_col])

    category_to_id, id_to_category = pmhp_data.build_category_mapping(category_order)
    work = pmhp_data.map_categories(work, category_col, category_to_id)

    types = work["hawkes_id"].to_numpy(dtype=int)
    D_dims = len(category_order)

    times = pmhp_data.continuous_time(work[date_col], observation_start)
    N_events = len(work)
    marks = np.ones(N_events, dtype=float)
    T_observation = float(
        (observation_end - observation_start).total_seconds() / (24 * 3600)
    )

    if times[0] < 0:
        raise ValueError("First event time is negative. Check observation_start.")
    if times[-1] >= T_observation:
        raise ValueError("Last event time is outside the observation window.")

    tau_event = pmhp_data.calendar_tau(work[date_col])
    season_cos_event, season_sin_event = pmhp_data.seasonal_features(tau_event)

    G = pmhp_data.SEASONAL_DAYS
    _, season_cos_grid, season_sin_grid = pmhp_data.seasonal_grid(G)

    day_exposure = pmhp_data.day_of_year_exposure(
        observation_start, observation_end, T_observation
    )

    stan_data = {
        "N": N_events,
        "D": D_dims,
        "times": times.tolist(),
        "types": [int(x) for x in types],
        "marks": marks.tolist(),
        "T_end": T_observation,
        "use_seasonality": int(use_seasonality),
        "season_cos_event": season_cos_event.tolist(),
        "season_sin_event": season_sin_event.tolist(),
        "G": int(G),
        "season_cos_grid": season_cos_grid.tolist(),
        "season_sin_grid": season_sin_grid.tolist(),
        "day_exposure": day_exposure.tolist(),
        "mu_prior_shape": float(priors.mu_prior_shape),
        "mu_prior_rate": float(priors.mu_prior_rate),
        "alpha_prior_a": float(priors.alpha_prior_a),
        "alpha_prior_b": float(priors.alpha_prior_b),
        "beta_prior_shape": float(priors.beta_prior_shape),
        "beta_prior_rate": float(priors.beta_prior_rate),
        "season_sd": float(priors.season_sd),
    }

    return PreparedData(
        stan_data=stan_data,
        category_to_id=category_to_id,
        id_to_category=id_to_category,
        observation_start=observation_start,
        observation_end=observation_end,
        T_observation=T_observation,
        N_events=N_events,
        D_dims=D_dims,
    )


def default_stan_file() -> Path:
    """Path to the bundled Stan model file."""
    return Path(resources.files("pmhp.models") / DEFAULT_STAN_FILE)


class PMHPModel:
    """Thin wrapper around a CmdStanModel for the periodic MHP.

    Example
    -------
    >>> model = PMHPModel()
    >>> prepared = prepare_stan_data(train_df, date_col=..., ...)
    >>> fit = model.sample(prepared, chains=1, threads_per_chain=8)
    """

    def __init__(self, stan_file: Optional[Path | str] = None, threads: bool = True):
        self.stan_file = Path(stan_file) if stan_file else default_stan_file()
        cpp_options = {"STAN_THREADS": "TRUE"} if threads else None
        self._model = CmdStanModel(stan_file=str(self.stan_file), cpp_options=cpp_options)

    def sample(
        self,
        prepared: PreparedData,
        *,
        chains: int = 1,
        chain_ids: Optional[list[int]] = None,
        threads_per_chain: int = 4,
        iter_warmup: int = 1000,
        iter_sampling: int = 500,
        adapt_delta: float = 0.95,
        output_dir: Optional[str] = None,
        show_progress: bool = True,
        show_console: bool = False,
    ) -> CmdStanMCMC:
        return self._model.sample(
            data=prepared.stan_data,
            chains=chains,
            chain_ids=chain_ids,
            threads_per_chain=threads_per_chain,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            adapt_delta=adapt_delta,
            output_dir=output_dir,
            show_progress=show_progress,
            show_console=show_console,
        )
