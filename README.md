# periodic-mhp

A periodic multivariate Hawkes process (PMHP) with a seasonal
Fourier baseline, fit with Stan/CmdStanPy, plus the full paper
reproduction pipeline (Boston PD crime data).

The `pmhp` package (`src/pmhp/`) is the reusable, dataset-agnostic
core: data prep, model fitting, rolling-count forecasting, and
held-out likelihood evaluation. Everything Boston-PD-specific --
column names, category taxonomy, file paths, figures, LaTeX tables --
lives in `scripts/`.

## Install

```bash
pip install -e .
# or, to also pull in the paper's plotting dependencies (geopandas, cartopy, matplotlib):
pip install -e ".[paper]"
```

Requires a working CmdStan installation for `pmhp.fit` (see
[CmdStanPy's install docs](https://mc-stan.org/cmdstanpy/installation.html)).

## Package layout

```
src/pmhp/
├── data.py         # seasonal Fourier features, leap-year calendar mapping, category mapping
├── fit.py          # PMHPModel: prepares Stan data and runs CmdStanPy sampling
├── forecast.py      # posterior-mean parameters, rolling-forecast intensity integration
├── likelihood.py    # held-out point-process log-likelihood, full-posterior evaluation
└── models/
    └── hawkes_temporal_seasonal_beta_gamma.stan
```

## Running the paper pipeline

Each script is a standalone CLI (`python scripts/NN_*.py --help` for
full options); run them in order:

```bash
# 1. Combine raw data, map crime categories, dedup, split full-city +
#    per-district train/test sets.
python scripts/01_clean_data.py \
    --raw-data-dir /path/to/raw/csvs \
    --output-dir results/train_test

# 2. Fit one model (unit x baseline) at a time.
python scripts/02_fit_models.py \
    --train-csv results/train_test/full_city/full_city_train_2020_2024.csv \
    --unit full_city --baseline seasonal \
    --output-dir results/mcmc/seasonal/full_city
# ... repeat with --baseline constant, and for each district, to fill out
#     results/mcmc/{seasonal,constant}/<unit>/

# 3. Rolling-forecast evaluation (predicted vs. observed counts per window).
python scripts/03_forecast_and_evaluate.py \
    --seasonal-results-root results/mcmc/seasonal \
    --constant-results-root results/mcmc/constant \
    --data-root results/train_test \
    --output-csv results/forecasts/full_city_rolling_forecast.csv

# 4. Figures: branching-ratio heatmap, seasonal-multiplier plot,
#    self-excitation/background/event-count maps.
python scripts/04_make_figures.py \
    --data-root results/train_test \
    --results-root results/mcmc/constant \
    --output-dir results/figures \
    --geojson-path /path/to/Boston_Police_Districts.geojson   # optional, skips maps if omitted

# 5. Appendix LaTeX tables (background intensities, branching ratios,
#    decay rates, half-lives, seasonal coefficients).
python scripts/05_make_tables.py \
    --results-root results/mcmc/seasonal \
    --output-dir results/tables/appendix_district

# 6. Held-out likelihood model comparison (constant vs. seasonal baseline),
#    full city + every district, with a Monte Carlo robustness check.
python scripts/06_evaluate_likelihood.py \
    --seasonal-results-root results/mcmc/seasonal \
    --constant-results-root results/mcmc/constant \
    --data-root results/train_test \
    --output-dir results/model_comparison
```

`scripts/categories.py` holds the shared crime-category taxonomy
(`CATEGORY_ORDER`) used across 01/02/03/06.

## Data

Raw and intermediate data are not tracked in git -- see
`data/README.md`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers the pure numpy/pandas logic in `pmhp.data`, `pmhp.forecast`,
and `pmhp.likelihood` with small synthetic inputs (no CmdStan needed).

## License

MIT for the code (see `LICENSE`). Data license depends on the Boston
PD open-data terms -- TODO once confirmed.
