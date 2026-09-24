# -*- coding: utf-8 -*-
"""
Combine raw BPD crime CSVs into one dataset, map offense descriptions onto the four Hawkes categories,
remove administrative and event-location duplicates, and split into full-city + per-district train/test CSV.

Details on exact data cleaning logic are explained in the paper.

Author: Persia Luca (2026), Università della Svizzera italiana, Lugano, Switzerland
Notes: additional revision used Claude Code (model Sonnet 5 and Opus 4.8) to improve code clarity and maintainability.
"""

from __future__ import annotations
import argparse, glob, numpy as np, pandas as pd
from pathlib import Path
from categories import CATEGORY_ORDER, CATEGORY_TO_ID

DATE_COL = "OCCURRED_ON_DATE"

# raw OFFENSE_DESCRIPTION -> Hawkes category. Anything not listed here is dropped (see map_hawkes_categories)
CRIME_MAPPING = {
    # note there are some typos in the raw data (e.g. "COMMERICAL" instead of "COMMERCIAL", "NEGLIGIENT" instead of "NEGLIGENT")
    "BURGLARY - COMMERICAL": "Burglary",
    "BURGLARY - COMMERICAL - ATTEMPT": "Burglary",
    "BURGLARY - COMMERICAL - FORCE": "Burglary",
    "BURGLARY - COMMERICAL - NO FORCE": "Burglary",
    "BURGLARY - RESIDENTIAL": "Burglary",
    "BURGLARY - RESIDENTIAL - ATTEMPT": "Burglary",
    "BURGLARY - RESIDENTIAL - FORCE": "Burglary",
    "BURGLARY - RESIDENTIAL - NO FORCE": "Burglary",
    "BURGLARY - OTHER - ATTEMPT": "Burglary",
    "BURGLARY - OTHER - FORCE": "Burglary",
    "BURGLARY - OTHER - NO FORCE": "Burglary",
    "AUTO THEFT": "Vehicle_Theft",
    "AUTO THEFT - LEASED/RENTED VEHICLE": "Vehicle_Theft",
    "AUTO THEFT - MOTORCYCLE / SCOOTER": "Vehicle_Theft",
    "BREAKING AND ENTERING (B&E) MOTOR VEHICLE": "Vehicle_Theft",
    "BREAKING AND ENTERING (B&E) MOTOR VEHICLE (NO PROPERTY STOLEN)": "Vehicle_Theft",
    "LARCENY THEFT FROM MV - NON-ACCESSORY": "Vehicle_Theft",
    "LARCENY THEFT OF MV PARTS & ACCESSORIES": "Vehicle_Theft",
    "MURDER, NON-NEGLIGENT MANSLAUGHTER": "Violent_Retaliatory",
    "MURDER, NON-NEGLIGIENT MANSLAUGHTER": "Violent_Retaliatory",
    "ASSAULT - AGGRAVATED": "Violent_Retaliatory",
    "ROBBERY": "Violent_Retaliatory",
    "VANDALISM": "Vandalism_Disorder",
    "GRAFFITI": "Vandalism_Disorder",
    "ARSON": "Vandalism_Disorder",
}

# Administrative duplicate definitions, applied city-wide before the per unit split 
# (exact timestamp ties are handled separately, per unit -- see remove_timestamp_ties).
OFFENSE_TIME_DUPLICATE_COLS = [
    "OFFENSE_CODE", "OFFENSE_DESCRIPTION", "OCCURRED_ON_DATE", "YEAR", "MONTH", "DAY_OF_WEEK", "HOUR",
]
EVENT_LOCATION_DUPLICATE_COLS = [
    "OCCURRED_ON_DATE", "hawkes_category", "DISTRICT", "REPORTING_AREA", "STREET", "Lat", "Long",
]

def combine_raw_files(
        raw_data_dir: Path, 
        start: pd.Timestamp, 
        end: pd.Timestamp
    ) -> pd.DataFrame:
    """Concatenate every raw *.csv under raw_data_dir and normalize
    OCCURRED_ON_DATE (note that the source file don't always use the same
    timestamp format), restricted to [start, end).
    """
    csv_files = [
        f
        for f in glob.glob(str(raw_data_dir / "*.csv"))
        if Path(f).name not in ("combined_data.csv", "metacategories_df.csv")
    ]
    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found under {raw_data_dir}")
    df = pd.concat(
        [pd.read_csv(f, low_memory=False, na_values=["", " "]) for f in csv_files],
        ignore_index=True,
    )

    # normalizing OCCURRED_ON_DATE to UTC, then convert to naive (no timezone) for Stan
    clean_strings = df[DATE_COL].astype(str).str[:19]
    df[DATE_COL] = pd.to_datetime(clean_strings, errors="coerce", utc=True).dt.tz_convert(None)
    df = df[(df[DATE_COL] >= start) & (df[DATE_COL] < end)].copy()
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns, derive calendar fields..
    """
    df = df.drop(columns=[c for c in ("OFFENSE_CODE_GROUP", "UCR_PART") if c in df.columns])
    df["YEAR"] = df[DATE_COL].dt.year
    df["MONTH"] = df[DATE_COL].dt.month
    df["HOUR"] = df[DATE_COL].dt.hour
    df["DAY_OF_WEEK"] = df[DATE_COL].dt.day_name()
    df["YearMonth"] = df[DATE_COL].dt.to_period("M").astype(str)
    # drop rows missing the fields every later step depends on (date, offense description, district)
    df = df.dropna(subset=[DATE_COL, "OFFENSE_DESCRIPTION", "DISTRICT"]).copy()
    return df

def map_hawkes_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add `hawkes_category`; drop offense descriptions not in CRIME_MAPPING."""
    df = df.copy()
    df["hawkes_category"] = df["OFFENSE_DESCRIPTION"].map(CRIME_MAPPING)
    return df.dropna(subset=["hawkes_category"]).copy()

def drop_administrative_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated incident numbers, repeated offense time records, and exact event location duplicates.
    Exact timestamp ties (needed for the point-process model) are handled separately per model unit; see remove_timestamp_ties.
    """
    before = len(df)
    df = df.drop_duplicates(subset="INCIDENT_NUMBER", keep="first").copy()
    print(f"Dropped repeated INCIDENT_NUMBER rows: {before - len(df)}")
    before = len(df)
    df = df.drop_duplicates(subset=OFFENSE_TIME_DUPLICATE_COLS, keep="first").copy()
    print(f"Dropped offense-time duplicate rows: {before - len(df)}")
    before = len(df)
    df = df.drop_duplicates(subset=EVENT_LOCATION_DUPLICATE_COLS, keep="first").copy()
    print(f"Dropped exact event-location duplicate rows: {before - len(df)}")

    return df


def build_model_units(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """(unit_name, subset) for full_city plus every valid BPD district -- this excludes entries like nan, 'External', 'Outside of').
    Note: valid districts are defined as a single uppercase letter followed by one or more digits (e.g. A1, B2, C3, D4, E5, F6, G7, H8).
    """
    df = df.copy()
    df["DISTRICT"] = df["DISTRICT"].astype(str).str.upper()
    district_mask = df["DISTRICT"].str.fullmatch(r"[A-Z]\d+")
    valid_districts = sorted(df.loc[district_mask, "DISTRICT"].dropna().unique())
    print("Valid districts used:", valid_districts)
    units = [("full_city", df)]
    units += [(f"district_{d}", df[df["DISTRICT"] == d].copy()) for d in valid_districts]
    return units


def remove_timestamp_ties(df_model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop every event sharing an exact timestamp with another event in this model unit (regardless of category).
    Returns (deduplicated df, the removed rows, for inspection).
    """
    same_time_rows = (
        df_model[df_model.duplicated(subset=[DATE_COL], keep=False)]
        .sort_values(DATE_COL)
        .copy()
    )
    df_model = df_model.drop_duplicates(subset=[DATE_COL], keep=False).copy()
    df_model = df_model.sort_values(DATE_COL).reset_index(drop=True)
    return df_model, same_time_rows


def build_unit_train_test(
    unit_name: str,
    df_unit: pd.DataFrame,
    unit_dir: Path,
    train_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict:
    """Apply the train/test window, remove timestamp ties, map category ids, compute continuous event time, split, and save the
    four CSVs (_all/_train/_test/_removed_timestamp_ties) for one model unit. Returns a summary dict.
    """
    print(f"\n{'=' * 80}\nModel unit: {unit_name}\n{'=' * 80}")
    unit_dir.mkdir(parents=True, exist_ok=True)
    # applying the train/test window and sort by timestamp
    df_model = df_unit[(df_unit[DATE_COL] >= train_start) & (df_unit[DATE_COL] < test_end)].copy()
    df_model = df_model.sort_values(DATE_COL).reset_index(drop=True)
    print(f"After applying train/test analysis window: {df_model.shape}")
    # removing exact timestamp ties (regardless of category) and reporting summary stats
    before = len(df_model)
    df_model, same_time_rows = remove_timestamp_ties(df_model)
    n_tied_events = len(same_time_rows)
    n_tied_groups = same_time_rows[DATE_COL].nunique()
    print(f"Timestamp tie groups: {n_tied_groups}, events removed: {n_tied_events}")
    print(f"After dropping all exact timestamp ties: {df_model.shape}")
    # compute summary stats for the unit
    same_time_same_category = df_model.duplicated(subset=[DATE_COL, "hawkes_category"], keep=False).sum()
    time_gaps = df_model.sort_values(DATE_COL)[DATE_COL].diff().dt.total_seconds()
    non_positive_time_gaps = int((time_gaps <= 0).sum())
    # map category names to integer ids for Stan
    df_model = df_model[df_model["hawkes_category"].isin(CATEGORY_ORDER)].copy()
    df_model["hawkes_id"] = df_model["hawkes_category"].map(CATEGORY_TO_ID).astype(int)
    # compute continuous event time in days since train_start (for Stan)
    df_model["event_time_days"] = (df_model[DATE_COL] - train_start).dt.total_seconds() / (24 * 3600)
    df_model["sample_split"] = np.where(df_model[DATE_COL] < test_start, "train", "test")
    # split into train/test subsets
    df_train = df_model[df_model["sample_split"] == "train"].copy()
    df_test = df_model[df_model["sample_split"] == "test"].copy()
    # save the four CSVs for this model unit
    # doesn't handle cases where the years of analyis change (assume the train/test split is always 2020-2024 vs 2025-end to reproduce paper results)
    all_path = unit_dir / f"{unit_name}_all_2020_end.csv"
    train_path = unit_dir / f"{unit_name}_train_2020_2024.csv"
    test_path = unit_dir / f"{unit_name}_test_2025_end.csv"
    ties_path = unit_dir / f"{unit_name}_removed_timestamp_ties.csv"

    df_model.to_csv(all_path, index=False)
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    same_time_rows.to_csv(ties_path, index=False)

    print(
        f"N_events all/train/test: {len(df_model)}/{len(df_train)}/{len(df_test)}"
        if len(df_model) else "no events remain after filtering."
    )

    return {
        "unit_name": unit_name,
        "n_before_timestamp_tie_removal": before,
        "n_timestamp_tied_groups": n_tied_groups,
        "n_timestamp_tied_events": n_tied_events,
        "n_all": len(df_model),
        "n_train": len(df_train),
        "n_test": len(df_test),
        "same_time_same_category": int(same_time_same_category),
        "non_positive_time_gaps": non_positive_time_gaps,
        "first_event_all": df_model[DATE_COL].min() if len(df_model) else pd.NaT,
        "last_event_all": df_model[DATE_COL].max() if len(df_model) else pd.NaT,
        "category_counts": df_model["hawkes_category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0),
        "all_path": str(all_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "removed_ties_path": str(ties_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-data-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--combined-csv-path", type=Path, default=None,
        help="If given, also save the combined (pre-split) dataset here, "
        "so later runs can skip re-reading the raw files. Not needed for git.",
    )
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--test-end", default="2026-04-01", help="Exclusive upper bound.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_start = pd.Timestamp(args.train_start)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # load and combine raw CSVs, filter to the analysis window, and save the combined dataset if requested
    df = combine_raw_files(args.raw_data_dir, train_start, test_end)
    if args.combined_csv_path:
        args.combined_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.combined_csv_path, index=False)
        print(f"Saved combined dataset to: {args.combined_csv_path}")

    df = basic_clean(df)
    print(f"After basic cleaning: {df.shape}")

    df = map_hawkes_categories(df)
    print(f"After mapping to Hawkes categories: {df.shape}")

    df = drop_administrative_duplicates(df)

    model_units = build_model_units(df)

    summary_rows = []
    for unit_name, df_unit in model_units:
        summary = build_unit_train_test(unit_name, df_unit, args.output_dir / unit_name, train_start, test_start, test_end)
        category_counts = summary.pop("category_counts")
        print(category_counts)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "train_test_dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved all train/test datasets to: {args.output_dir}")
    print(f"Saved run summary to: {summary_path}")
    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    main()
