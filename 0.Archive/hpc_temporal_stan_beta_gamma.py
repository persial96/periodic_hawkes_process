# -*- coding: utf-8 -*-
"""
Final prior specification for the periodic multivariate Hawkes model.

This version:
    - Uses the normalized exponential kernel.
    - Interprets alpha directly as the branching ratio.
    - Uses a Beta prior for alpha on the branching-ratio scale.
    - Uses a Gamma prior for beta to regularize extreme decay rates.
    - Uses the same data-cleaning flow as the local script.
    - Keeps the HPC structure: one chain per SLURM array node,
      output written first to /tmp, then moved to the network folder.
"""

import os
import shutil
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

data_path = "./metacategories_df.csv"
stan_file_path = "./hawkes_temporal_seasonal_beta_gamma.stan"
output_dir = "./mcmc_results_beta_gamma"


# ------------------------------------------------------------
# Helper: compute day-of-year exposure
# ------------------------------------------------------------

def compute_day_exposure_days(start_timestamp, end_timestamp):
    """
    Compute total observation exposure in days for each day of a
    365-day seasonal calendar.

    Leap years are mapped onto a 365-day seasonal calendar:
        - Feb 29 is assigned to the Feb 28 seasonal position.
        - Dates after Feb 29 are shifted back by one day.
    """

    exposure = np.zeros(365, dtype=float)

    current = pd.Timestamp(start_timestamp)
    end = pd.Timestamp(end_timestamp)

    if end <= current:
        raise ValueError("end_timestamp must be after start_timestamp.")

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

        exposure[doy - 1] += days_in_segment

        current = segment_end

    return exposure


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("Loading and cleaning dataset...")

df = pd.read_csv(data_path, low_memory=False, na_values=["", " "])

print(df.info())
print(f"Initial dataset shape: {df.shape}")


# ------------------------------------------------------------
# First duplicate cleaning
# ------------------------------------------------------------
#
# Remove repeated records with the same offense/time description,
# keeping the first occurrence.

offense_time_duplicate_cols = [
    "OFFENSE_CODE",
    "OFFENSE_DESCRIPTION",
    "OCCURRED_ON_DATE",
    "YEAR",
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR",
]

df = df.drop_duplicates(
    subset=offense_time_duplicate_cols,
    keep="first"
).copy()

print(f"After dropping offense-time duplicate rows: {df.shape}")


# ------------------------------------------------------------
# Same event record cleaning
# ------------------------------------------------------------
#
# Keep one record when timestamp, category, district, reporting area,
# street, and exact coordinates are identical.

event_duplicate_cols = [
    "OCCURRED_ON_DATE",
    "hawkes_category",
    "DISTRICT",
    "REPORTING_AREA",
    "STREET",
    "Lat",
    "Long",
]

df = df.drop_duplicates(
    subset=event_duplicate_cols,
    keep="first"
).copy()

print(f"After keeping one exact event-location duplicate: {df.shape}")


# ------------------------------------------------------------
# Near-duplicate location cleaning
# ------------------------------------------------------------
#
# Coordinates are rounded to identify events occurring at essentially
# the same location. Four decimals correspond roughly to 8--11 meters
# in Boston, depending on latitude/longitude.

df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
df["Long"] = pd.to_numeric(df["Long"], errors="coerce")

df["Lat_round"] = df["Lat"].round(4)
df["Long_round"] = df["Long"].round(4)

near_duplicate_cols = [
    "OCCURRED_ON_DATE",
    "hawkes_category",
    "DISTRICT",
    "REPORTING_AREA",
    "Lat_round",
    "Long_round",
]

df = df.drop_duplicates(
    subset=near_duplicate_cols,
    keep="first"
).copy()

df = df.drop(columns=["Lat_round", "Long_round"])

print(f"After near-duplicate location cleaning: {df.shape}")


# ------------------------------------------------------------
# Datetime cleaning
# ------------------------------------------------------------

df["OCCURRED_ON_DATE"] = pd.to_datetime(
    df["OCCURRED_ON_DATE"],
    errors="coerce"
)

if df["OCCURRED_ON_DATE"].dt.tz is not None:
    df["OCCURRED_ON_DATE"] = df["OCCURRED_ON_DATE"].dt.tz_localize(None)


# ------------------------------------------------------------
# Drop missing critical values
# ------------------------------------------------------------

df_crimes = df.dropna(
    subset=["hawkes_category", "OCCURRED_ON_DATE"]
).copy()

print(f"After dropping missing category/date rows: {df_crimes.shape}")


# ------------------------------------------------------------
# Drop same-time same-category ties
# ------------------------------------------------------------
#
# The model is purely temporal. Events with exactly the same timestamp
# and category are indistinguishable in the likelihood, so we remove
# all observations involved in such ties.

print(f"Before dropping same-time same-category ties: {df_crimes.shape}")

tie_cols = ["OCCURRED_ON_DATE", "hawkes_category"]

tie_counts = (
    df_crimes
    .groupby(tie_cols)
    .size()
    .reset_index(name="n_ties")
)

n_tied_groups = int((tie_counts["n_ties"] > 1).sum())
n_tied_events = int(
    tie_counts.loc[tie_counts["n_ties"] > 1, "n_ties"].sum()
)

print(f"Number of tied timestamp-category groups: {n_tied_groups}")
print(f"Number of events involved in ties: {n_tied_events}")

df_crimes = df_crimes.drop_duplicates(
    subset=tie_cols,
    keep=False
).copy()

print(f"After dropping same-time same-category ties: {df_crimes.shape}")


# ------------------------------------------------------------
# Apply time cutoffs
# ------------------------------------------------------------

cutoff1 = pd.Timestamp("2023-11-01")
cutoff2 = pd.Timestamp("2025-12-31")

df_crimes = df_crimes[
    (df_crimes["OCCURRED_ON_DATE"] >= cutoff1) &
    (df_crimes["OCCURRED_ON_DATE"] <= cutoff2)
].copy()

print(f"After applying time cutoffs: {df_crimes.shape}")


# ------------------------------------------------------------
# Sort chronologically
# ------------------------------------------------------------

df_crimes = (
    df_crimes
    .sort_values("OCCURRED_ON_DATE")
    .reset_index(drop=True)
)

if df_crimes.empty:
    raise ValueError("No events remain after filtering. Check cutoffs and cleaning steps.")


# ------------------------------------------------------------
# Exclude Opportunity Larceny
# ------------------------------------------------------------

df_crimes = df_crimes[
    df_crimes["hawkes_category"] != "Opportunistic_Larceny"
].copy()

df_crimes = (
    df_crimes
    .sort_values("OCCURRED_ON_DATE")
    .reset_index(drop=True)
)

print(f"After dropping Opportunistic Larceny: {df_crimes.shape}")

if df_crimes.empty:
    raise ValueError("No events remain after excluding Opportunistic Larceny.")


# ------------------------------------------------------------
# Map crime categories to integer IDs
# ------------------------------------------------------------

unique_categories = df_crimes["hawkes_category"].unique()

D_dims = int(len(unique_categories))

category_to_id = {
    cat: i + 1
    for i, cat in enumerate(unique_categories)
}

types = (
    df_crimes["hawkes_category"]
    .map(category_to_id)
    .astype(int)
    .values
)

print("\nCategory mapping:")
for cat, idx in category_to_id.items():
    print(f"  {idx}: {cat}")


# ------------------------------------------------------------
# Convert timestamps to continuous time
# ------------------------------------------------------------
#
# Time is measured in days since the first event.

t0 = df_crimes["OCCURRED_ON_DATE"].iloc[0]

time_deltas = df_crimes["OCCURRED_ON_DATE"] - t0

times = (
    time_deltas
    .dt.total_seconds()
    .to_numpy(dtype=float)
    / (24 * 3600)
)

N_events = int(len(df_crimes))

marks = np.ones(N_events, dtype=float)

T_observation = float(times[-1] + 1.0)
observation_start = t0
observation_end = t0 + pd.Timedelta(days=T_observation)


# ------------------------------------------------------------
# Continuous annual seasonality inputs
# ------------------------------------------------------------
#
# One annual Fourier harmonic:
#
#   cos(2*pi*tau/365), sin(2*pi*tau/365)

dates = df_crimes["OCCURRED_ON_DATE"]

year_start = pd.to_datetime(dates.dt.year.astype(str) + "-01-01")

tau_event = np.array(
    ((dates - year_start).dt.total_seconds() / (24 * 3600)).to_numpy(dtype=float),
    dtype=float,
    copy=True
)

is_leap = dates.dt.is_leap_year.to_numpy(dtype=bool, copy=True)
month = dates.dt.month.to_numpy(dtype=int, copy=True)
day = dates.dt.day.to_numpy(dtype=int, copy=True)

is_feb29 = (month == 2) & (day == 29)
after_feb29 = month > 2

tau_event[is_leap & is_feb29] -= 1.0
tau_event[is_leap & after_feb29] -= 1.0

tau_event = np.mod(tau_event, 365.0)

season_angle_event = 2.0 * np.pi * tau_event / 365.0
season_cos_event = np.cos(season_angle_event)
season_sin_event = np.sin(season_angle_event)

G = 365

tau_grid = np.arange(G, dtype=float) + 0.5
season_angle_grid = 2.0 * np.pi * tau_grid / 365.0

season_cos_grid = np.cos(season_angle_grid)
season_sin_grid = np.sin(season_angle_grid)

day_exposure = compute_day_exposure_days(
    start_timestamp=observation_start,
    end_timestamp=observation_end
)

exposure_sum = float(day_exposure.sum())

if not np.isclose(exposure_sum, T_observation, rtol=1e-10, atol=1e-8):
    raise ValueError(
        f"Day exposure does not sum to T_end. "
        f"sum(day_exposure)={exposure_sum}, T_end={T_observation}"
    )

print("\nSeasonal exposure summary:")
print(f"  Total seasonal exposure: {exposure_sum:.4f} days")
print(f"  T_end:                   {T_observation:.4f} days")
print("  Continuous annual seasonality: one cosine and one sine term")


# ------------------------------------------------------------
# Priors
# ------------------------------------------------------------
#
# Final main specification:
#
#   mu[d]        ~ Exponential(0.20)
#   alpha[d,j]   ~ Beta(1,7)
#   beta[d,j]    ~ Gamma(2,0.5)
#   season terms ~ Normal(0,0.5)
#
# Stan uses Gamma(shape, rate).

mu_rate = 0.20

alpha_prior_a = 1.0
alpha_prior_b = 7.0

beta_prior_shape = 2.0
beta_prior_rate = 0.50

season_sd = 0.50


# ------------------------------------------------------------
# Build data dictionary for CmdStanPy
# ------------------------------------------------------------

stan_data = {
    "N": N_events,
    "D": D_dims,
    "times": times.tolist(),
    "types": [int(x) for x in types],
    "marks": marks.tolist(),
    "T_end": T_observation,

    "season_cos_event": season_cos_event.tolist(),
    "season_sin_event": season_sin_event.tolist(),
    "G": int(G),
    "season_cos_grid": season_cos_grid.tolist(),
    "season_sin_grid": season_sin_grid.tolist(),
    "day_exposure": day_exposure.tolist(),

    "mu_rate": float(mu_rate),
    "alpha_prior_a": float(alpha_prior_a),
    "alpha_prior_b": float(alpha_prior_b),
    "beta_prior_shape": float(beta_prior_shape),
    "beta_prior_rate": float(beta_prior_rate),
    "season_sd": float(season_sd),
}

print(f"\nData Prepared: {N_events} Total Crimes across {D_dims} Categories.")
print(f"Observation start: {observation_start}")
print(f"Observation end:   {observation_end}")
print(f"T_end in days:     {T_observation:.4f}")

print("\nPrior specification:")
print(f"  mu[d]      ~ Exponential({mu_rate})")
print(f"  alpha[d,j] ~ Beta({alpha_prior_a}, {alpha_prior_b})")
print(f"  beta[d,j]  ~ Gamma({beta_prior_shape}, {beta_prior_rate})")
print(f"  season     ~ Normal(0, {season_sd})")


# ------------------------------------------------------------
# Compile and sample
# ------------------------------------------------------------

if os.path.exists(stan_file_path):

    print(f"\nFound {stan_file_path}. Compiling model with Multi-Threading enabled...")

    model = CmdStanModel(
        stan_file=stan_file_path,
        cpp_options={"STAN_THREADS": "TRUE"}
    )

    array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))

    print(f"\n--- I AM NODE {array_task_id} ---")
    print("Starting MCMC sampling: 1 Chain using all 80 Threads on this Node!")

    os.makedirs(output_dir, exist_ok=True)

    scratch_dir = "/tmp"

    fit = model.sample(
        data=stan_data,
        chains=1,
        chain_ids=[array_task_id],
        threads_per_chain=80,
        iter_warmup=1000,
        iter_sampling=500,
        adapt_delta=0.95,
        output_dir=scratch_dir,
        show_progress=True,
        show_console=True
    )

    print(f"\nSUCCESS: Node {array_task_id} finished computing! Moving files from scratch to network drive...")

    for csv_file in fit.runset.csv_files:
        base_name = os.path.basename(csv_file)

        new_csv_name = base_name.replace(".csv", f"_node{array_task_id}.csv")
        shutil.move(csv_file, os.path.join(output_dir, new_csv_name))

        txt_file = csv_file.replace(".csv", ".txt")
        if os.path.exists(txt_file):
            new_txt_name = base_name.replace(".csv", f"_node{array_task_id}.txt")
            shutil.move(txt_file, os.path.join(output_dir, new_txt_name))

    print(f"Files safely stored in {output_dir}")

    category_mapping_path = os.path.join(
        output_dir,
        f"category_mapping_chain_{array_task_id}.csv"
    )

    category_mapping_df = pd.DataFrame({
        "hawkes_category": list(category_to_id.keys()),
        "stan_type_id": list(category_to_id.values())
    })

    category_mapping_df.to_csv(category_mapping_path, index=False)

    print(f"Saved category mapping to: {category_mapping_path}")

else:
    print(f"\nError: The file {stan_file_path} does not exist. Check your folder path.")