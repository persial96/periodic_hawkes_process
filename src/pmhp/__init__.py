# -*- coding: utf-8 -*-
"""pmhp: periodic multivariate Hawkes process models with seasonal baselines.

Companion package to "Periodic Multivariate Hawkes Process for Urban Crime" (Persia & Diouane). 
See the paper's Section 3 for the model definition and Appendix B for the seasonal normalization this package implements.

Submodules (each usable on its own, e.g. `from pmhp import data`):
    data        Data prep: seasonal Fourier features, calendar mapping, category mapping.
    fit         PMHPModel: builds the Stan-ready data dict and runs CmdStanPy sampling (Sections 3.2-3.5).
    posterior   Shared helpers for locating/reading a fitted unit's CmdStan chain output.
    likelihood  Held-out point-process log-likelihood / LPPD, evaluated over the full posterior (Section 3.6, Equations 13-14).

Author: Persia Luca (2026), Università della Svizzera italiana, Lugano, Switzerland
"""

from .fit import PMHPModel, Priors, PreparedData, prepare_stan_data, default_stan_file
from .likelihood import (
    evaluate_test_likelihood,
    extract_parameter_draws,
    log_mean_exp,
)

__all__ = [
    "PMHPModel",
    "Priors",
    "PreparedData",
    "prepare_stan_data",
    "default_stan_file",
    "evaluate_test_likelihood",
    "extract_parameter_draws",
    "log_mean_exp",
]

__version__ = "0.1.0"
