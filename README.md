# Periodic Multivariate Hawkes Process for Urban Crime

Companion code for **"Periodic Multivariate Hawkes Process for Urban Crime"** by Luca Persia (Zurich University of Applied Sciences; Università della Svizzera italiana) and Youness Diouane (MIT).

This repository contains the `pmhp` Python package — a periodic multivariate Hawkes process (PMHP) with a seasonal Fourier background, estimated with Stan/CmdStanPy — together with the full pipeline used to reproduce the paper's results on Boston Police Department crime incident data (2020–2026).

> **Paper:** *Periodic Multivariate Hawkes Process for Urban Crime* (Persia & Diouane).

## What the model does

Crime events cluster in time — a burglary can make a follow-up burglary or vehicle theft more likely in the days after, a mechanism criminologists call *near-repeat victimization*. At the same time, crime has a genuine seasonality: more incidents in summer than winter, independent of any one event triggering another. A standard multivariate Hawkes process, with a constant background rate, can't tell these two things apart; it risks attributing predictable seasonal upswings to event-to-event triggering.

This project extends the multivariate Hawkes process with a **periodic background component**: each of four crime categories in our dataset (vehicle theft, vandalism, burglary, violent crime) gets its own average background rate, multiplied by a shared, normalized annual seasonal curve. The excitation structure (who triggers whom, and for how long) is then estimated net of that seasonal pattern, both city-wide and separately for each of Boston's 12 police districts. The model is fit with Bayesian inference in Stan, and compared against a constant-baseline benchmark using held-out predictive likelihood.

## Repository structure
```bash
periodic-mhp/
├── src/pmhp/                  # the installable, dataset-agnostic package
│   ├── data.py                 # seasonal Fourier features, leap-year-aware calendar mapping
│   ├── fit.py                  # builds Stan-ready data, runs CmdStanPy sampling
│   ├── posterior.py            # helpers for locating/reading a fitted unit's posterior draws
│   ├── likelihood.py           # held-out point-process log-likelihood (LPPD) evaluation
│   └── models/
│       └── hawkes_temporal_seasonal_beta_gamma.stan   # the Stan model itself
├── scripts/                    # the Boston-PD-specific reproduction pipeline
│   ├── 01_clean_data.py         # raw data -> cleaned, deduplicated, train/test-split events
│   ├── 02_fit_models.py         # fits one (unit, baseline) pair with CmdStanPy
│   ├── 03_make_figures.py       # branching-matrix heatmap, seasonal plot, district maps
│   ├── 04_make_tables.py        # LaTeX appendix tables (background, branching, decay, half-life)
│   ├── 05_evaluate_likelihood.py # held-out LPPD comparison, constant vs. seasonal baseline
│   └── categories.py            # shared crime-category taxonomy
├── tests/                      # unit tests, including closed-form checks of the likelihood math
├── data/README.md              # where to get the raw data (not included in this repo)
├── pyproject.toml
└── README.md
```

## Installation

```bash
pip install -e .
# to also install the plotting dependencies used by 03_make_figures.py (geopandas, cartopy, matplotlib):
pip install -e ".[paper]"
```

Fitting the model requires a working [CmdStan](https://mc-stan.org/cmdstanpy/installation.html) installation:

```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

## Reproducing the paper's results

Each script is a standalone command-line tool (`python scripts/NN_*.py --help` for the full option list). Run them in order:

### 1. Clean and split the data

```bash
python scripts/01_clean_data.py \
    --raw-data-dir /path/to/raw/csvs \
    --output-dir results/train_test
```

Combines the raw BPD incident CSVs, maps offense descriptions onto the four modeled crime categories, removes administrative duplicates and exact-timestamp ties, and splits each of full-city plus all 12 districts into training (2020–2024) and held-out (2025 onward) sequences.

### 2. Fit the model

```bash
python scripts/02_fit_models.py \
    --train-csv results/train_test/full_city/full_city_train_2020_2024.csv \
    --unit full_city --baseline seasonal \
    --output-dir results/mcmc/seasonal/full_city
```

Run this once per (unit, baseline) pair — `--baseline seasonal` for the proposed PMHP, `--baseline constant` for the benchmark MHP — to fill out `results/mcmc/{seasonal,constant}/<unit>/` for full-city and every district.

### 3. Figures

```bash
python scripts/03_make_figures.py \
    --data-root results/train_test \
    --results-root results/mcmc/constant \
    --output-dir results/figures \
    --geojson-path /path/to/Boston_Police_Districts.geojson
```

Produces the branching-ratio heatmap, the estimated-vs-observed seasonal profile, and the district-level self-excitation, background, and event-density maps.

### 4. Tables

```bash
python scripts/04_make_tables.py \
    --results-root results/mcmc/seasonal \
    --output-dir results/tables/appendix_district
```

Generates the appendix LaTeX tables: district posterior estimates of background intensities, branching ratios, decay rates, half-lives, and seasonal coefficients.

### 5. Predictive evaluation

```bash
python scripts/05_evaluate_likelihood.py \
    --seasonal-results-root results/mcmc/seasonal \
    --constant-results-root results/mcmc/constant \
    --data-root results/train_test \
    --output-dir results/model_comparison
```

Computes the held-out log posterior predictive density (LPPD) for both models, city-wide and per district, with a Monte Carlo robustness check on the difference — this is the comparison reported in the paper's predictive-evaluation section.

## Data

The raw and intermediate data files are not distributed in this repository (see `.gitignore` and `data/README.md`). The Boston Police Department incident reports used in the paper are publicly available through the City of Boston's open-data portal; see `data/README.md` for pointers on regenerating the analysis dataset.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test validates the core package logic against closed-form and independently-derived solutions. In particular, `pmhp.likelihood`'s held-out log-likelihood is checked against exact analytical solutions for a pure Poisson process, a single decaying excitation event, and 100+ randomized configurations.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{persia_diouane_pmhp,
  title   = {Periodic Multivariate Hawkes Process for Urban Crime},
  author  = {Persia, Luca and Diouane, Youness},
  journal = {TBD},
  year    = {2026}
}
```

## License
TODO

## Contact

Luca Persia — pess@zhaw.ch / luca.persia@usi.ch
Youness Diouane — diouane@mit.edu

