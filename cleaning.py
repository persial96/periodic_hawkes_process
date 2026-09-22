# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:16:29 2026

@author: pess
"""

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob

single_file_path = r'C:\Users\pess\Desktop\BC_Crime_Boston_Paper\data\*.csv' 
csv_files = [
    f for f in glob.glob(single_file_path)
    if os.path.basename(f) != "combined_data.csv" and
    os.path.basename(f) != "metacategories_df.csv" 
]
df = pd.concat(
    [pd.read_csv(f, low_memory=False, na_values=["", " "]) for f in csv_files],
    ignore_index=True
)
# fix the timestamp problem (format is not always the same)
clean_strings = df["OCCURRED_ON_DATE"].astype(str).str[:19]
df["OCCURRED_ON_DATE"] = pd.to_datetime(clean_strings, errors="coerce", utc=True)
df["OCCURRED_ON_DATE"] = df["OCCURRED_ON_DATE"].dt.tz_convert(None)
#df = df[(df['YEAR'] >= 2016) & (df['YEAR'] <= 2026)]
df = df[(df["OCCURRED_ON_DATE"] >= pd.Timestamp("2020-01-01"))
        & (df["OCCURRED_ON_DATE"] <= pd.Timestamp("2026-03-31"))].copy()

# # Make sure date is datetime
# df["OCCURRED_ON_DATE"] = pd.to_datetime(df["OCCURRED_ON_DATE"], errors="coerce")
# # Drop missing dates
# df_plot = df.dropna(subset=["OCCURRED_ON_DATE"]).copy()
# # Create year-month variable
# df_plot["YearMonth"] = df_plot["OCCURRED_ON_DATE"].dt.to_period("M").dt.to_timestamp()
# # Count events per month
# monthly_counts = (
#     df_plot
#     .groupby("YearMonth")
#     .size()
#     .reset_index(name="n_events")
# )
# # Plot
# plt.figure(figsize=(14, 5))
# plt.plot(
#     monthly_counts["YearMonth"],
#     monthly_counts["n_events"],
#     marker="o",
#     linewidth=2
# )
# plt.title("Monthly Number of Crime Incidents")
# plt.xlabel("Month")
# plt.ylabel("Number of incidents")
# plt.grid(True, alpha=0.3)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

df.to_csv("C:\\Users\\pess\\Desktop\\Boston_Project\\combined_data.csv", index=False)

### Load data
# data_path = r"C:\Users\pess\Desktop\Boston_Project\combined_data.csv"
# df = pd.read_csv(data_path, low_memory=False, na_values=["", " "])
cols_to_drop = ["OFFENSE_CODE_GROUP", "UCR_PART"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
print(f"Loading dataset shape: {df.shape}")

### CLEANING ------------------------------------------------------------------
df["YEAR"] = df["OCCURRED_ON_DATE"].dt.year
df["MONTH"] = df["OCCURRED_ON_DATE"].dt.month
df["HOUR"] = df["OCCURRED_ON_DATE"].dt.hour
df["my_DAY_OF_WEEK"] = df["OCCURRED_ON_DATE"].dt.day_name()
df["DAY_OF_WEEK"] = df["my_DAY_OF_WEEK"]
df["YearMonth"] = df["OCCURRED_ON_DATE"].dt.to_period("M").astype(str)
df = df.dropna(subset=["OCCURRED_ON_DATE", "OFFENSE_DESCRIPTION", "DISTRICT"]).copy()
print(f"After datetime basic cleaning: {df.shape}")

### CLEANING specific to Hawkes/Point Process model ---------------------------
crime_mapping = {
    "BURGLARY - COMMERICAL": "Burglary",
    "BURGLARY - COMMERICAL - ATTEMPT": "Burglary",
    "BURGLARY - COMMERICAL - FORCE": "Burglary",
    "BURGLARY - COMMERICAL - NO FORCE": "Burglary",
    "BURGLARY - RESIDENTIAL": "Burglary",
    'BURGLARY - RESIDENTIAL - ATTEMPT': "Burglary",
    'BURGLARY - RESIDENTIAL - FORCE': "Burglary",
    'BURGLARY - RESIDENTIAL - NO FORCE': "Burglary",
    'BURGLARY - OTHER - ATTEMPT': "Burglary",
    'BURGLARY - OTHER - FORCE': "Burglary",
    'BURGLARY - OTHER - NO FORCE': "Burglary",
    
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

    # "LARCENY ALL OTHERS": "Opportunistic_Larceny",
    # "LARCENY PICK-POCKET": "Opportunistic_Larceny",
    # "LARCENY PURSE SNATCH - NO FORCE": "Opportunistic_Larceny",
    # "LARCENY THEFT FROM BUILDING": "Opportunistic_Larceny",
    # "LARCENY THEFT OF BICYCLE": "Opportunistic_Larceny",
    # "LARCENY SHOPLIFTING": "Opportunistic_Larceny",

    "VANDALISM": "Vandalism_Disorder",
    "GRAFFITI": "Vandalism_Disorder",
    "ARSON": "Vandalism_Disorder",
}
df["hawkes_category"] = df["OFFENSE_DESCRIPTION"].map(crime_mapping)
df = df.dropna(subset=["hawkes_category"]).copy()
print(f"After mapping to Hawkes categories: {df.shape}")


### Duplicates (these are a problem for standard Hawkes likelihood) -----------
df = df.drop_duplicates(subset='INCIDENT_NUMBER', keep='first')

### Remove repeated offense-time records --------------------------------------
# These are rows with the same offense code, offense description, timestamp,
# year, month, day of week, and hour. These rows are likely repeated database 
# records for the same offense-time information. We keep the first occurrence-

offense_time_duplicate_cols = ["OFFENSE_CODE", "OFFENSE_DESCRIPTION",
                               "OCCURRED_ON_DATE", 
                               "YEAR", "MONTH", "DAY_OF_WEEK", "HOUR"
                               ]
df = df.drop_duplicates(subset=offense_time_duplicate_cols, keep="first").copy()
print(f"After dropping offense-time duplicate rows: {df.shape}")

### Remove exact event-location duplicates ------------------------------------
# These are rows with the same timestamp, category, district, reporting area,
# street, latitude, and longitude. These records are indistinguishable as 
# event-location observations. We keep the first occurrence.
event_duplicate_cols = ["OCCURRED_ON_DATE", "hawkes_category",
                        "DISTRICT", "REPORTING_AREA", "STREET", "Lat", "Long"]
df = df.drop_duplicates(subset=event_duplicate_cols, keep="first").copy()
print(f"After exact event-location duplicate cleaning: {df.shape}")

### Remove all exact timestamp ties in the final model data -------------------
# The Hawkes model is continuous-time and purely temporal. Events with exactly 
# the same timestamp cannot be ordered. Therefore, all events sharing the same exact
# timestamp are removed from the final model dataset, regardless of category.
same_time_rows = (df[df.duplicated(subset=["OCCURRED_ON_DATE"], keep=False)]
                  .sort_values("OCCURRED_ON_DATE")
                  )
n_tied_events = len(same_time_rows)
n_tied_groups = same_time_rows["OCCURRED_ON_DATE"].nunique()
df_model = df.drop_duplicates(subset=["OCCURRED_ON_DATE"], keep=False).copy()
df_model = df_model.sort_values("OCCURRED_ON_DATE").reset_index(drop=True)
print(f"Number of exact timestamp tie groups: {n_tied_groups}")
print(f"Number of events involved in exact timestamp ties: {n_tied_events}")
print(f"After dropping all exact timestamp ties: {df_model.shape}")

### Study Window --------------------------------------------------------------
cutoff1 = pd.Timestamp("2016-03-01")
cutoff2 = pd.Timestamp("2026-03-31")
df_model = df_model[(df_model["OCCURRED_ON_DATE"] >= cutoff1) &
                    (df_model["OCCURRED_ON_DATE"] <= cutoff2)].copy()
df_model = df_model.sort_values("OCCURRED_ON_DATE").reset_index(drop=True)
print(f"After applying observation window: {df_model.shape}")

### Final checks --------------------------------------------------------------
same_time_any_category = df_model.duplicated(subset=["OCCURRED_ON_DATE"], 
                                             keep=False).sum()
same_time_same_category = df_model.duplicated(subset=["OCCURRED_ON_DATE", "hawkes_category"],
                                              keep=False).sum()
tmp = df_model.sort_values("OCCURRED_ON_DATE").copy()
time_gaps = tmp["OCCURRED_ON_DATE"].diff().dt.total_seconds()
non_positive_time_gaps = int((time_gaps <= 0).sum())
print("\nFinal timestamp checks:")
print(f"Same timestamp, any category:       {same_time_any_category}")
print(f"Same timestamp, same category:      {same_time_same_category}")
print(f"Non-positive time gaps after sort:  {non_positive_time_gaps}")

### Final mapping to Hawkes events and categories ids -------------------------
unique_categories = df_model["hawkes_category"].unique()
category_to_id = {cat: i + 1 for i, cat in enumerate(unique_categories)}
id_to_category = {val: cat for cat, val in category_to_id.items()}
df_model["hawkes_id"] = df_model["hawkes_category"].map(category_to_id).astype(int)
print("\nFinal category mapping for Multivariate Process:")
for cat, idx in category_to_id.items():
    print(f"  {idx}: {cat}")
t0 = df_model["OCCURRED_ON_DATE"].iloc[0]
df_model["event_time_days"] = ((df_model["OCCURRED_ON_DATE"] - t0)
                               .dt.total_seconds()
                               / (24 * 3600)
                               )
print("\nFinal Hawkes data:")
print(f"  N_events: {len(df_model)}")
print(f"  D_dims:   {len(unique_categories)}")
print(f"  t0:       {t0}")
print(f"  max time: {df_model['event_time_days'].max():.4f} days")









import os
import numpy as np
import pandas as pd

# ============================================================
# Summary statistics for Hawkes model dataset
# ============================================================

out_dir = r"C:\Users\pess\Desktop\Boston_Project\summary_statistics_outputs"
os.makedirs(out_dir, exist_ok=True)

d = df_model.copy()

d["OCCURRED_ON_DATE"] = pd.to_datetime(d["OCCURRED_ON_DATE"], errors="coerce")
d = d.dropna(subset=["OCCURRED_ON_DATE", "hawkes_category", "event_time_days"]).copy()
d = d.sort_values("OCCURRED_ON_DATE").reset_index(drop=True)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def fmt(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"

def save_latex(df, filename, caption, label, index=False):
    tex = df.to_latex(
        index=index,
        escape=False,
        caption=caption,
        label=label,
        column_format="l" + "r" * (df.shape[1] - 1 if not index else df.shape[1]),
    )
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        f.write(tex)
    print("\n" + "=" * 80)
    print(filename)
    print("=" * 80)
    print(tex)

# ------------------------------------------------------------
# 1. Overall Hawkes dataset summary
# ------------------------------------------------------------

t0 = d["OCCURRED_ON_DATE"].min()
t1 = d["OCCURRED_ON_DATE"].max()
T_days = (t1 - t0).total_seconds() / (24 * 3600)

time_gaps_days = d["event_time_days"].diff().dropna()
time_gaps_hours = time_gaps_days * 24

overall_summary = pd.DataFrame({
    "Statistic": [
        "Number of events",
        "Number of crime categories",
        "First event",
        "Last event",
        "Observation window (days)",
        "Average events per day",
        "Median inter-event time (minutes)",
        "Mean inter-event time (minutes)",
        "5th percentile inter-event time (minutes)",
        "95th percentile inter-event time (minutes)",
        "Minimum inter-event time (seconds)",
        "Maximum inter-event time (hours)",
    ],
    "Value": [
        f"{len(d):,}",
        f"{d['hawkes_category'].nunique():,}",
        t0.strftime("%Y-%m-%d %H:%M"),
        t1.strftime("%Y-%m-%d %H:%M"),
        fmt(T_days, 2),
        fmt(len(d) / T_days, 2),
        fmt(time_gaps_days.median() * 24 * 60, 2),
        fmt(time_gaps_days.mean() * 24 * 60, 2),
        fmt(time_gaps_days.quantile(0.05) * 24 * 60, 2),
        fmt(time_gaps_days.quantile(0.95) * 24 * 60, 2),
        fmt(time_gaps_days.min() * 24 * 3600, 2),
        fmt(time_gaps_days.max() * 24, 2),
    ]
})

save_latex(
    overall_summary,
    "hawkes_overall_summary.tex",
    "Overall summary of the final continuous-time Hawkes event sequence.",
    "tab:hawkes_overall_summary"
)

# ------------------------------------------------------------
# 2. Category-level summary
# ------------------------------------------------------------

category_summary = (
    d.groupby("hawkes_category")
    .agg(
        events=("hawkes_category", "size"),
        first_event=("OCCURRED_ON_DATE", "min"),
        last_event=("OCCURRED_ON_DATE", "max"),
        mean_event_time_days=("event_time_days", "mean"),
    )
    .reset_index()
)

category_summary["share_percent"] = 100 * category_summary["events"] / len(d)
category_summary["events_per_day"] = category_summary["events"] / T_days

category_summary["first_event"] = category_summary["first_event"].dt.strftime("%Y-%m-%d")
category_summary["last_event"] = category_summary["last_event"].dt.strftime("%Y-%m-%d")

category_summary = category_summary[
    [
        "hawkes_category",
        "events",
        "share_percent",
        "events_per_day",
        "first_event",
        "last_event",
    ]
].copy()

category_summary["events"] = category_summary["events"].map(lambda x: f"{x:,}")
category_summary["share_percent"] = category_summary["share_percent"].map(lambda x: f"{x:.2f}")
category_summary["events_per_day"] = category_summary["events_per_day"].map(lambda x: f"{x:.2f}")

category_summary = category_summary.rename(columns={
    "hawkes_category": "Category",
    "events": "Events",
    "share_percent": "Share (\\%)",
    "events_per_day": "Events/day",
    "first_event": "First event",
    "last_event": "Last event",
})

save_latex(
    category_summary,
    "hawkes_category_summary.tex",
    "Category-level summary of the final Hawkes event sequence.",
    "tab:hawkes_category_summary"
)

# ------------------------------------------------------------
# 3. Inter-event gaps by category
# ------------------------------------------------------------

gap_rows = []

for cat, tmp in d.groupby("hawkes_category"):
    tmp = tmp.sort_values("OCCURRED_ON_DATE").copy()
    gaps = tmp["event_time_days"].diff().dropna() * 24 * 60  # minutes

    gap_rows.append({
        "Category": cat,
        "Mean gap (min)": gaps.mean(),
        "Median gap (min)": gaps.median(),
        "5\\% gap (min)": gaps.quantile(0.05),
        "95\\% gap (min)": gaps.quantile(0.95),
        "Min gap (sec)": gaps.min() * 60,
        "Max gap (hours)": gaps.max() / 60,
    })

gap_summary = pd.DataFrame(gap_rows)

for col in gap_summary.columns[1:]:
    gap_summary[col] = gap_summary[col].map(lambda x: f"{x:.2f}")

save_latex(
    gap_summary,
    "hawkes_gap_summary_by_category.tex",
    "Within-category inter-event time summaries. Gaps are computed between consecutive events of the same category.",
    "tab:hawkes_gap_summary_by_category"
)








###### Stats to save in another folder ----------------------------------------
import matplot2tikz
save_tikz = matplot2tikz.save
plot_dir = r"C:\Users\pess\Desktop\Boston_Project\summary_statistics_outputs"
os.makedirs(plot_dir, exist_ok=True)
### Prepare monthly data
plot_df = df_model.copy()
plot_df["OCCURRED_ON_DATE"] = pd.to_datetime(plot_df["OCCURRED_ON_DATE"], errors="coerce")

plot_df = plot_df.dropna(subset=["OCCURRED_ON_DATE", "hawkes_category"]).copy()
plot_df["month_start"] = (plot_df["OCCURRED_ON_DATE"]
                          .dt.to_period("M")
                          .dt.to_timestamp())
category_order = ["Vandalism_Disorder", "Vehicle_Theft", "Violent_Retaliatory", "Burglary"]
category_order = [c for c in category_order if c in plot_df["hawkes_category"].unique()]
category_labels = {"Vandalism_Disorder": "Vandalism/Disorder", "Vehicle_Theft": "Vehicle theft",
                    "Violent_Retaliatory": "Violent", "Burglary": "Burglary"} 
category_colors = {"Vandalism_Disorder": "#4C78A8", "Vehicle_Theft": "#F58518",
                   "Violent_Retaliatory": "#E45756", "Burglary": "#54A24B"}
monthly = (plot_df.groupby(["month_start", "hawkes_category"]).size()
           .unstack(fill_value=0)
           .reindex(columns=category_order, fill_value=0))
monthly_total = monthly.sum(axis=1)
x = np.arange(len(monthly))
year_ticks = [i for i, d in enumerate(monthly.index) if d.month == 1 or i == 0]
year_labels = [str(monthly.index[i].year) for i in year_ticks]

########################### ----------------------- ###########################
monthly_share = monthly.div(monthly.sum(axis=1), axis=0) * 100
share_stats = (monthly_share.agg(["mean", "std"])
               .T
               .loc[category_order]
               .rename(index=category_labels))
print(share_stats.round(2))
share_sentence = ("On average, the monthly composition is: "+ "; ".join(
        f"{cat}: {row['mean']:.1f}% "
        f"(sd = {row['std']:.1f})"
        for cat, row in share_stats.iterrows()) + ".")
print(share_sentence)

### Monthly total line plot over all sample years
plt.style.use("default")
monthly_ma3 = monthly_total.rolling(window=3, center=True, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(13, 5), facecolor="white")
ax.set_facecolor("white")
ax.plot(x, monthly_total.values, marker="o", markersize=2.5, linewidth=1.0,
        color="gray", alpha=0.45, label="Monthly counts")
ax.plot(x, monthly_ma3.values, linewidth=2.8, 
        color="#4C78A8", label="3-month moving average")
ax.set_xticks(year_ticks)
ax.set_xticklabels(year_labels, rotation=45)
ax.set_xlabel("Year")
ax.set_ylabel("Number of events")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
save_tikz(os.path.join(plot_dir, "monthly_total_line.tex"))
plt.show()

### Stacked monthly histogram by category
bottom = np.zeros(len(monthly))
fig, ax = plt.subplots(figsize=(13, 5.5))
for cat in category_order:
    ax.bar(x, monthly[cat].values, bottom=bottom, color=category_colors[cat],
           label=category_labels[cat], width=0.85)
    bottom += monthly[cat].values
ax.set_xticks(year_ticks)
ax.set_xticklabels(year_labels, rotation=45)
ax.set_xlabel("Year")
ax.set_ylabel("Number of events")
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_tikz(os.path.join(plot_dir, "monthly_stacked_by_category.tex"))
plt.show()





















############### new train-test cleaning



import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import re


# =============================================================================
# 0. Output folders and train-test split
# =============================================================================

output_dir = r"C:\Users\pess\Desktop\Boston_Project\hawkes_train_test_datasets"
os.makedirs(output_dir, exist_ok=True)

train_start = pd.Timestamp("2020-01-01")
test_start = pd.Timestamp("2025-01-01")
test_end = pd.Timestamp("2026-04-01")   # exclusive, keeps data through 2026-03-31

category_order = [
    "Vehicle_Theft",
    "Vandalism_Disorder",
    "Burglary",
    "Violent_Retaliatory",
]

category_to_id = {cat: i + 1 for i, cat in enumerate(category_order)}
id_to_category = {val: cat for cat, val in category_to_id.items()}


# =============================================================================
# 1. Load and combine raw data
# =============================================================================

single_file_path = r'C:\Users\pess\Desktop\BC_Crime_Boston_Paper\data\*.csv' 

csv_files = [
    f for f in glob.glob(single_file_path)
    if os.path.basename(f) != "combined_data.csv" and
    os.path.basename(f) != "metacategories_df.csv" 
]

df = pd.concat(
    [pd.read_csv(f, low_memory=False, na_values=["", " "]) for f in csv_files],
    ignore_index=True
)

# fix the timestamp problem (format is not always the same)
clean_strings = df["OCCURRED_ON_DATE"].astype(str).str[:19]

df["OCCURRED_ON_DATE"] = pd.to_datetime(
    clean_strings,
    errors="coerce",
    utc=True
)

df["OCCURRED_ON_DATE"] = df["OCCURRED_ON_DATE"].dt.tz_convert(None)

df = df[
    (df["OCCURRED_ON_DATE"] >= pd.Timestamp("2016-03-01")) &
    (df["OCCURRED_ON_DATE"] < pd.Timestamp("2026-04-01"))
].copy()

df.to_csv(
    r"C:\Users\pess\Desktop\Boston_Project\combined_data.csv",
    index=False
)


# =============================================================================
# 2. Basic cleaning
# =============================================================================

cols_to_drop = ["OFFENSE_CODE_GROUP", "UCR_PART"]

df = df.drop(
    columns=[c for c in cols_to_drop if c in df.columns],
    errors="ignore"
)

print(f"Loading dataset shape: {df.shape}")

df["YEAR"] = df["OCCURRED_ON_DATE"].dt.year
df["MONTH"] = df["OCCURRED_ON_DATE"].dt.month
df["HOUR"] = df["OCCURRED_ON_DATE"].dt.hour
df["my_DAY_OF_WEEK"] = df["OCCURRED_ON_DATE"].dt.day_name()
df["DAY_OF_WEEK"] = df["my_DAY_OF_WEEK"]
df["YearMonth"] = df["OCCURRED_ON_DATE"].dt.to_period("M").astype(str)

df = df.dropna(
    subset=["OCCURRED_ON_DATE", "OFFENSE_DESCRIPTION", "DISTRICT"]
).copy()

print(f"After datetime basic cleaning: {df.shape}")


# =============================================================================
# 3. CLEANING specific to Hawkes/Point Process model
# =============================================================================

crime_mapping = {
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

df["hawkes_category"] = df["OFFENSE_DESCRIPTION"].map(crime_mapping)

df = df.dropna(subset=["hawkes_category"]).copy()

print(f"After mapping to Hawkes categories: {df.shape}")


# =============================================================================
# 4. Duplicates before city/district construction
# =============================================================================
#
# These are global administrative duplicates.
# Exact timestamp ties are not removed here anymore.
# They are removed later inside each model unit: full city or each district.

before = len(df)

df = df.drop_duplicates(
    subset="INCIDENT_NUMBER",
    keep="first"
).copy()

print(f"After dropping repeated INCIDENT_NUMBER rows: {df.shape}")
print(f"Rows removed in this step: {before - len(df)}")


### Remove repeated offense-time records --------------------------------------
# These are rows with the same offense code, offense description, timestamp,
# year, month, day of week, and hour. These rows are likely repeated database 
# records for the same offense-time information. We keep the first occurrence.

offense_time_duplicate_cols = [
    "OFFENSE_CODE",
    "OFFENSE_DESCRIPTION",
    "OCCURRED_ON_DATE", 
    "YEAR",
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR",
]

before = len(df)

df = df.drop_duplicates(
    subset=offense_time_duplicate_cols,
    keep="first"
).copy()

print(f"After dropping offense-time duplicate rows: {df.shape}")
print(f"Rows removed in this step: {before - len(df)}")


### Remove exact event-location duplicates ------------------------------------
# These are rows with the same timestamp, category, district, reporting area,
# street, latitude, and longitude. These records are indistinguishable as 
# event-location observations. We keep the first occurrence.

event_duplicate_cols = [
    "OCCURRED_ON_DATE",
    "hawkes_category",
    "DISTRICT",
    "REPORTING_AREA",
    "STREET",
    "Lat",
    "Long",
]

before = len(df)

df = df.drop_duplicates(
    subset=event_duplicate_cols,
    keep="first"
).copy()

print(f"After exact event-location duplicate cleaning: {df.shape}")
print(f"Rows removed in this step: {before - len(df)}")


# =============================================================================
# 5. Prepare model units: full city and valid BPD districts
# =============================================================================
#
# We keep only districts with a letter-number code, such as A1, B2, C11.
# This removes entries such as nan, External, and Outside of.

df["DISTRICT"] = df["DISTRICT"].astype(str).str.upper()

district_mask = df["DISTRICT"].str.fullmatch(r"[A-Z]\d+")
valid_districts = sorted(df.loc[district_mask, "DISTRICT"].dropna().unique())

print("\nValid districts used:")
print(valid_districts)

model_units = [("full_city", df.copy())]

for district in valid_districts:
    model_units.append((
        f"district_{district}",
        df[df["DISTRICT"] == district].copy()
    ))


# =============================================================================
# 6. Build train/test Hawkes datasets for each model unit
# =============================================================================

summary_rows = []

for unit_name, df_unit in model_units:

    print("\n" + "=" * 80)
    print(f"Model unit: {unit_name}")
    print("=" * 80)

    unit_dir = os.path.join(output_dir, unit_name)
    os.makedirs(unit_dir, exist_ok=True)

    ### Study Window ----------------------------------------------------------
    # Train: 2020-01-01 to 2024-12-31.
    # Test:  2025-01-01 to end of data availability.
    
    df_model = df_unit[
        (df_unit["OCCURRED_ON_DATE"] >= train_start) &
        (df_unit["OCCURRED_ON_DATE"] < test_end)
    ].copy()

    df_model = df_model.sort_values("OCCURRED_ON_DATE").reset_index(drop=True)

    print(f"After applying train/test analysis window: {df_model.shape}")


    ### Remove all exact timestamp ties in the final model data ---------------
    # The Hawkes model is continuous-time and purely temporal.
    # Events with exactly the same timestamp cannot be ordered.
    # Therefore, all events sharing the same exact timestamp are removed
    # within the current model unit, regardless of category.
    #
    # Important:
    # A timestamp tie across districts matters for the full-city process,
    # but it does not matter for a district-specific process. For this reason,
    # this step is done separately for each model unit.

    same_time_rows = (
        df_model[df_model.duplicated(subset=["OCCURRED_ON_DATE"], keep=False)]
        .sort_values("OCCURRED_ON_DATE")
        .copy()
    )

    n_tied_events = len(same_time_rows)
    n_tied_groups = same_time_rows["OCCURRED_ON_DATE"].nunique()

    before = len(df_model)

    df_model = df_model.drop_duplicates(
        subset=["OCCURRED_ON_DATE"],
        keep=False
    ).copy()

    df_model = df_model.sort_values("OCCURRED_ON_DATE").reset_index(drop=True)

    print(f"Number of exact timestamp tie groups: {n_tied_groups}")
    print(f"Number of events involved in exact timestamp ties: {n_tied_events}")
    print(f"After dropping all exact timestamp ties: {df_model.shape}")
    print(f"Rows removed in this step: {before - len(df_model)}")


    ### Final checks ----------------------------------------------------------

    same_time_any_category = df_model.duplicated(
        subset=["OCCURRED_ON_DATE"], 
        keep=False
    ).sum()

    same_time_same_category = df_model.duplicated(
        subset=["OCCURRED_ON_DATE", "hawkes_category"],
        keep=False
    ).sum()

    tmp = df_model.sort_values("OCCURRED_ON_DATE").copy()
    time_gaps = tmp["OCCURRED_ON_DATE"].diff().dt.total_seconds()
    non_positive_time_gaps = int((time_gaps <= 0).sum())

    print("\nFinal timestamp checks:")
    print(f"Same timestamp, any category:       {same_time_any_category}")
    print(f"Same timestamp, same category:      {same_time_same_category}")
    print(f"Non-positive time gaps after sort:  {non_positive_time_gaps}")


    ### Final mapping to Hawkes events and category ids -----------------------
    # The category ids are fixed across the full city and all districts.
    # This makes the outputs directly comparable across model units.

    df_model = df_model[df_model["hawkes_category"].isin(category_order)].copy()

    df_model["hawkes_id"] = (
        df_model["hawkes_category"]
        .map(category_to_id)
        .astype(int)
    )

    print("\nFinal category mapping for Multivariate Process:")
    for cat, idx in category_to_id.items():
        print(f"  {idx}: {cat}")


    ### Continuous time in days -----------------------------------------------
    # Time is measured in days from the beginning of the training window.
    # This preserves the exact timestamp information through fractional days.
    # For example, an event 12 hours after train_start has time 0.5.

    df_model["event_time_days"] = (
        (df_model["OCCURRED_ON_DATE"] - train_start)
        .dt.total_seconds()
        / (24 * 3600)
    )


    ### Train/test split ------------------------------------------------------

    df_model["sample_split"] = np.where(
        df_model["OCCURRED_ON_DATE"] < test_start,
        "train",
        "test"
    )

    df_train = df_model[df_model["sample_split"] == "train"].copy()
    df_test = df_model[df_model["sample_split"] == "test"].copy()


    ### Save outputs ----------------------------------------------------------

    all_path = os.path.join(unit_dir, f"{unit_name}_all_2020_end.csv")
    train_path = os.path.join(unit_dir, f"{unit_name}_train_2020_2024.csv")
    test_path = os.path.join(unit_dir, f"{unit_name}_test_2025_end.csv")
    ties_path = os.path.join(unit_dir, f"{unit_name}_removed_timestamp_ties.csv")

    df_model.to_csv(all_path, index=False)
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    same_time_rows.to_csv(ties_path, index=False)


    ### Final Hawkes data printout --------------------------------------------

    if len(df_model) > 0:
        print("\nFinal Hawkes data:")
        print(f"  N_events all:   {len(df_model)}")
        print(f"  N_events train: {len(df_train)}")
        print(f"  N_events test:  {len(df_test)}")
        print(f"  D_dims:         {len(category_order)}")
        print(f"  t0:             {train_start}")
        print(f"  train end:      {test_start}")
        print(f"  max time:       {df_model['event_time_days'].max():.4f} days")
    else:
        print("\nFinal Hawkes data:")
        print("  No events remain after filtering.")


    ### Summary row -----------------------------------------------------------

    summary_rows.append({
        "unit_name": unit_name,
        "n_before_timestamp_tie_removal": before,
        "n_timestamp_tied_groups": n_tied_groups,
        "n_timestamp_tied_events": n_tied_events,
        "n_all": len(df_model),
        "n_train": len(df_train),
        "n_test": len(df_test),
        "same_time_any_category": int(same_time_any_category),
        "same_time_same_category": int(same_time_same_category),
        "non_positive_time_gaps": int(non_positive_time_gaps),
        "first_event_all": df_model["OCCURRED_ON_DATE"].min() if len(df_model) else pd.NaT,
        "last_event_all": df_model["OCCURRED_ON_DATE"].max() if len(df_model) else pd.NaT,
        "first_event_train": df_train["OCCURRED_ON_DATE"].min() if len(df_train) else pd.NaT,
        "last_event_train": df_train["OCCURRED_ON_DATE"].max() if len(df_train) else pd.NaT,
        "first_event_test": df_test["OCCURRED_ON_DATE"].min() if len(df_test) else pd.NaT,
        "last_event_test": df_test["OCCURRED_ON_DATE"].max() if len(df_test) else pd.NaT,
        "all_path": all_path,
        "train_path": train_path,
        "test_path": test_path,
        "removed_ties_path": ties_path,
    })


# =============================================================================
# 7. Save global summary
# =============================================================================

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    os.path.join(output_dir, "train_test_dataset_summary.csv"),
    index=False
)

print("\nSaved all train/test datasets to:")
print(output_dir)

print("\nSummary:")
print(summary_df)


test_f = pd.read_csv(r'C:\Users\pess\Desktop\Boston_Project\hawkes_train_test_datasets\full_city\full_city_all_2020_end.csv')

print("\nTraining counts by category:")
print(test_f["hawkes_category"].value_counts().reindex(category_order, fill_value=0))



import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = r"C:\Users\pess\Desktop\Boston_Project\hawkes_train_test_datasets"

category_order = [
    "Vehicle_Theft",
    "Vandalism_Disorder",
    "Burglary",
    "Violent_Retaliatory",
]

rows = []

for folder in sorted(os.listdir(base_dir)):

    folder_path = os.path.join(base_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    all_file = os.path.join(folder_path, f"{folder}_all_2020_end.csv")

    if not os.path.exists(all_file):
        continue

    tmp = pd.read_csv(all_file, low_memory=False)

    counts = (
        tmp["hawkes_category"]
        .value_counts()
        .reindex(category_order, fill_value=0)
    )

    print("\n" + "=" * 60)
    print(folder)
    print("=" * 60)
    print(counts)

    rows.append({"unit": folder, **counts.to_dict()})

counts_df = pd.DataFrame(rows).set_index("unit")

print("\nFull summary table:")
print(counts_df)
