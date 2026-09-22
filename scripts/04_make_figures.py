# -*- coding: utf-8 -*-
"""
EDA, monthly counts, and posterior-parameter figures for the paper.

Ported from the first ~300 lines of the old `new_reader.py` (itself a
few notebook cells concatenated together: reading the cleaned event
data, monthly-count validation/export, and the full-city branching
matrix heatmap). The self-excitation / background district maps and
event-count map further down that file are ported separately below
(see `make_self_excitation_maps` etc., added in the next step).

Usage
-----
    python scripts/04_make_figures.py \
        --data-root results/train_test \
        --results-root results/mcmc/constant \
        --output-dir results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from pmhp.forecast import branching_prefix as _branching_prefix

# Map styling -- shared by the self-excitation and background maps below.
PLOT_MODE = "panel"  # "panel" (2x2) or "separate" (one figure per category)
SHOW_DISTRICT_CODES = True
TILE_ZOOM = 11
CONTEXT_MARGIN = 0.10
POLYGON_ALPHA = 0.62
DISTRICT_LINE_COLOR = "black"
DISTRICT_LINE_WIDTH = 0.8
CITY_LINE_COLOR = "0.25"
CITY_LINE_WIDTH = 1.2
PANEL_FIGSIZE = (23.5, 20.5)
SEPARATE_FIGSIZE = (13.8, 12.2)
PANEL_H_PAD = 0.02
PANEL_W_PAD = 0.03
PANEL_HSPACE = 0.03
PANEL_WSPACE = 0.03
CATEGORY_TITLE_SIZE = 24
CATEGORY_TITLE_Y = 1.015

# Event-count raster map (fine-grained incident density, not district
# deviation) -- separate styling from the deviation maps above.
EVENT_COUNT_CATEGORIES = {1: "Vehicle theft", 2: "Vandalism", 3: "Burglary", 4: "Violent"}
EVENT_COUNT_CELL_METRES = 150
EVENT_COUNT_CRS_METRIC = "EPSG:26986"
EVENT_COUNT_GRID_COLORS = [
    "#F7D9A1",  # 1
    "#F1B66F",  # 2--3
    "#E98949",  # 4--7
    "#D95734",  # 8--15
    "#A62D2D",  # 16--31
    "#591616",  # 32+
]
EVENT_COUNT_BACKGROUND_COLOR = "#F7F3EF"

CATEGORY_LABELS = {1: "Vehicle theft", 2: "Vandalism and disorder", 3: "Burglary", 4: "Violent crime"}
CATEGORY_ORDER = list(CATEGORY_LABELS.values())

CATEGORY_NAMES = {
    1: "vehicle_theft",
    2: "vandalism_disorder",
    3: "burglary",
    4: "violent_crime",
}

START_DATE = pd.Timestamp("2020-01-01")
END_DATE = pd.Timestamp("2026-03-31 23:59:59")

# Sanity-check totals from the reported 55,627-event final sample.
# Update these (or drop the check) if the underlying data changes.
EXPECTED_CATEGORY_COUNTS = {1: 19267, 2: 16736, 3: 5936, 4: 13688}


def read_event_data(data_root: Path) -> dict[str, pd.DataFrame]:
    """Read every unit's cleaned '*_all_*' event CSV under data_root
    (full_city first, then districts alphabetically).
    """
    event_data: dict[str, pd.DataFrame] = {}
    unit_dirs = sorted(
        (p for p in data_root.iterdir() if p.is_dir()),
        key=lambda p: (p.name != "full_city", p.name),
    )

    for unit_dir in unit_dirs:
        files = [
            p
            for p in unit_dir.glob("*.csv")
            if "_all_" in p.name.lower() and "removed" not in p.name.lower()
        ]
        if not files:
            continue

        df = pd.read_csv(sorted(files)[0], parse_dates=["OCCURRED_ON_DATE"], low_memory=False)
        required = {"OCCURRED_ON_DATE", "hawkes_category", "hawkes_id", "DISTRICT"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{unit_dir.name}: missing columns {sorted(missing)}")
        if df["OCCURRED_ON_DATE"].isna().any() or df["hawkes_id"].isna().any():
            raise ValueError(f"{unit_dir.name}: missing dates or Hawkes categories")

        df = df.sort_values("OCCURRED_ON_DATE").reset_index(drop=True)
        df["hawkes_id"] = df["hawkes_id"].astype(int)
        df["category"] = df["hawkes_id"].map(CATEGORY_LABELS)
        df["year_month"] = df["OCCURRED_ON_DATE"].dt.to_period("M").dt.to_timestamp()
        event_data[unit_dir.name] = df

        print(unit_dir.name, len(df), df["OCCURRED_ON_DATE"].min(), df["OCCURRED_ON_DATE"].max())
        print(df["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0))

    return event_data


def build_monthly_counts(full_city: pd.DataFrame) -> pd.DataFrame:
    """Monthly event counts by category for full_city, validated against
    EXPECTED_CATEGORY_COUNTS, with a 3-month centered moving average.
    """
    monthly_source = full_city[["OCCURRED_ON_DATE", "hawkes_id"]].copy()
    monthly_source["date"] = pd.to_datetime(monthly_source["OCCURRED_ON_DATE"], errors="coerce")
    monthly_source["hawkes_id"] = pd.to_numeric(monthly_source["hawkes_id"], errors="coerce")
    monthly_source = monthly_source.dropna(subset=["date", "hawkes_id"]).copy()
    monthly_source["hawkes_id"] = monthly_source["hawkes_id"].astype(int)
    monthly_source = monthly_source[
        (monthly_source["date"] >= START_DATE)
        & (monthly_source["date"] <= END_DATE)
        & (monthly_source["hawkes_id"].isin(CATEGORY_NAMES))
    ].copy()

    monthly_source["month"] = monthly_source["date"].dt.to_period("M").dt.to_timestamp()
    complete_months = pd.date_range(
        START_DATE.normalize(), END_DATE.normalize().replace(day=1), freq="MS"
    )

    monthly_counts = (
        monthly_source.groupby(["month", "hawkes_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=complete_months, fill_value=0)
    )
    for category_id in CATEGORY_NAMES:
        if category_id not in monthly_counts.columns:
            monthly_counts[category_id] = 0
    monthly_counts = monthly_counts[list(CATEGORY_NAMES)].rename(columns=CATEGORY_NAMES)

    monthly_counts["monthly_total"] = monthly_counts[list(CATEGORY_NAMES.values())].sum(axis=1)
    monthly_counts["ma_3m"] = (
        monthly_counts["monthly_total"].rolling(window=3, center=True, min_periods=2).mean()
    )
    monthly_counts = monthly_counts.rename_axis("date").reset_index()

    # Validation
    print("Final events used:", len(monthly_source))
    print("Counts by category:")
    print(monthly_source["hawkes_id"].value_counts().sort_index().rename(index=CATEGORY_NAMES))
    print("Monthly total sum:", monthly_counts["monthly_total"].sum())
    assert monthly_counts["monthly_total"].sum() == len(monthly_source)

    actual_counts = (
        monthly_source["hawkes_id"]
        .value_counts()
        .reindex(EXPECTED_CATEGORY_COUNTS, fill_value=0)
        .to_dict()
    )
    print("Expected category counts:", EXPECTED_CATEGORY_COUNTS)
    print("Actual category counts:  ", actual_counts)
    if actual_counts != EXPECTED_CATEGORY_COUNTS:
        print(
            "\nWARNING: the category totals do not match the reported "
            "55,627-event final sample. If this is a different dataset, "
            "update EXPECTED_CATEGORY_COUNTS."
        )

    return monthly_counts


def read_posterior_draws(results_root: Path) -> dict[str, pd.DataFrame]:
    """Read every unit's chain CSVs under results_root into one
    concatenated draws DataFrame per unit, tagged with chain id.
    """
    posterior_draws: dict[str, pd.DataFrame] = {}
    unit_dirs = sorted(
        (p for p in results_root.iterdir() if p.is_dir()),
        key=lambda p: (p.name != "full_city", p.name),
    )

    for unit_dir in unit_dirs:
        chain_files = sorted(
            p for p in unit_dir.glob("*.csv") if p.name.lower().startswith("hawkes")
        )
        if not chain_files:
            continue

        chains = []
        for chain_number, file_path in enumerate(chain_files, start=1):
            chain = pd.read_csv(file_path, comment="#", low_memory=False)
            chain["chain"] = chain_number
            chain["draw_within_chain"] = np.arange(len(chain))
            chains.append(chain)

        posterior_draws[unit_dir.name] = pd.concat(chains, ignore_index=True)
        print(unit_dir.name, len(posterior_draws[unit_dir.name]), "posterior draws")

    return posterior_draws


def plot_branching_matrix(full_draws: pd.DataFrame, output_path: Path) -> None:
    """Full-city posterior-mean branching-ratio heatmap."""
    import re

    branch_prefix = (
        "branching_ratio"
        if any(c.startswith("branching_ratio.") for c in full_draws.columns)
        else "alpha"
    )
    branch_pattern = re.compile(rf"^{branch_prefix}\.(\d+)\.(\d+)$")
    branch_columns = [c for c in full_draws.columns if branch_pattern.match(c)]

    branching_matrix = np.full((4, 4), np.nan)
    for column in branch_columns:
        child, parent = map(int, branch_pattern.match(column).groups())
        branching_matrix[child - 1, parent - 1] = full_draws[column].mean()

    fig, ax = plt.subplots(figsize=(6.8, 6))
    image = ax.imshow(branching_matrix, cmap="YlOrRd")
    ax.set_xticks(range(4), CATEGORY_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(4), CATEGORY_ORDER)
    ax.set_xlabel("Parent category")
    ax.set_ylabel("Child category")

    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{branching_matrix[i, j]:.3f}", ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Posterior mean branching ratio")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def branching_prefix(full_draws: pd.DataFrame) -> str:
    return _branching_prefix(full_draws.columns)


def load_district_geometries(
    geojson_path: Path,
    district_col: str,
    district_codes: list[str],
    context_margin: float = CONTEXT_MARGIN,
) -> tuple["gpd.GeoDataFrame", list[float]]:
    """District polygons + label points + a common padded map extent."""
    district_map = gpd.read_file(geojson_path)
    district_map["district_code"] = district_map[district_col].astype(str).str.upper().str.strip()
    district_map = district_map[district_map["district_code"].isin(district_codes)].copy()
    district_map = district_map.to_crs(epsg=4326)

    label_points = district_map.to_crs(epsg=26986).geometry.representative_point()
    label_points = gpd.GeoSeries(label_points, crs="EPSG:26986").to_crs(epsg=4326)
    district_map["label_longitude"] = label_points.x.to_numpy()
    district_map["label_latitude"] = label_points.y.to_numpy()

    min_lon, min_lat, max_lon, max_lat = district_map.total_bounds
    lon_pad = (max_lon - min_lon) * context_margin
    lat_pad = (max_lat - min_lat) * context_margin
    map_extent = [min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad]
    return district_map, map_extent


def district_deviation_percentages(
    full_draws: pd.DataFrame,
    posterior_draws: dict[str, pd.DataFrame],
    category_labels: dict[int, str],
    parameter_of,
) -> pd.DataFrame:
    """Percentage deviation of each district's posterior mean from the
    full-city posterior mean, for whichever parameter `parameter_of`
    (a function of category_id -> column name) points at. Used for both
    the self-excitation map (branching_ratio/alpha.i.i) and the
    background-intensity map (mu.i) -- the two were previously
    ~300-line copy-pasted blocks differing only in this parameter.
    """
    rows = []
    for category_id, category_name in category_labels.items():
        parameter = parameter_of(category_id)
        city_mean = full_draws[parameter].mean()
        if abs(city_mean) < 1e-12:
            raise ValueError(f"The full-city estimate for {category_name} is too close to zero.")

        for unit, draws in posterior_draws.items():
            if not unit.startswith("district_") or parameter not in draws.columns:
                continue
            district_mean = draws[parameter].mean()
            percentage_change = 100 * (district_mean - city_mean) / city_mean
            rows.append(
                {
                    "district_code": unit.replace("district_", "").upper(),
                    "category_id": category_id,
                    "category": category_name,
                    "city_mean": city_mean,
                    "district_mean": district_mean,
                    "percentage_change": percentage_change,
                }
            )
    return pd.DataFrame(rows)


def plot_district_deviation_map(
    district_map: "gpd.GeoDataFrame",
    map_extent: list[float],
    percentages: pd.DataFrame,
    category_labels: dict[int, str],
    colorbar_label: str,
    output_path: Path,
) -> None:
    """2x2 (or 4 separate) basemap panels shading each district by its
    percentage deviation from the full-city posterior mean, one panel
    per category. Shared by the self-excitation and background maps.
    """
    maximum_absolute_change = percentages["percentage_change"].abs().max()
    color_limit = max(10, np.ceil(maximum_absolute_change / 10) * 10)
    color_norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit)
    color_map = plt.get_cmap("RdBu_r")
    colorbar_object = ScalarMappable(norm=color_norm, cmap=color_map)
    colorbar_object.set_array([])

    background = cimgt.GoogleTiles(
        url="https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png", cache=True
    )
    data_crs = ccrs.PlateCarree()
    longitude_ticks = np.linspace(map_extent[0], map_extent[1], 4)
    latitude_ticks = np.linspace(map_extent[2], map_extent[3], 4)

    if PLOT_MODE == "panel":
        fig, axes = plt.subplots(
            2, 2, figsize=PANEL_FIGSIZE, dpi=150,
            subplot_kw={"projection": background.crs}, constrained_layout=True,
        )
        fig.set_constrained_layout_pads(
            h_pad=PANEL_H_PAD, w_pad=PANEL_W_PAD, hspace=PANEL_HSPACE, wspace=PANEL_WSPACE
        )
        axes = axes.ravel()
    else:
        fig, axes = None, [None] * 4

    for plot_index, (category_id, category_name) in enumerate(category_labels.items()):
        category_values = percentages[percentages["category_id"] == category_id].copy()
        plot_map = district_map.merge(category_values, on="district_code", how="left")
        city_outline = plot_map.geometry.union_all()

        if PLOT_MODE == "panel":
            ax = axes[plot_index]
        else:
            fig, ax = plt.subplots(
                figsize=SEPARATE_FIGSIZE, dpi=150, subplot_kw={"projection": background.crs}
            )

        ax.set_extent(map_extent, crs=data_crs)
        ax.add_image(background, TILE_ZOOM, zorder=0)

        gridlines = ax.gridlines(
            crs=data_crs, draw_labels=True, linewidth=0.4, color="0.30",
            alpha=0.45, linestyle=":", zorder=1,
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        if PLOT_MODE == "panel":
            gridlines.bottom_labels = plot_index >= 2
            gridlines.left_labels = plot_index % 2 == 0
        else:
            gridlines.bottom_labels = True
            gridlines.left_labels = True
        gridlines.xlocator = mticker.FixedLocator(longitude_ticks)
        gridlines.ylocator = mticker.FixedLocator(latitude_ticks)
        gridlines.xformatter = LONGITUDE_FORMATTER
        gridlines.yformatter = LATITUDE_FORMATTER
        gridlines.xlabel_style = {"size": 22, "color": "0.25"}
        gridlines.ylabel_style = {
            "size": 22, "color": "0.25", "rotation": 90,
            "rotation_mode": "anchor", "ha": "center", "va": "center",
        }

        for _, row in plot_map.iterrows():
            value = row["percentage_change"]
            district_color = color_map(color_norm(value)) if pd.notna(value) else "0.80"
            ax.add_geometries(
                [row.geometry], crs=data_crs, facecolor=district_color,
                edgecolor=DISTRICT_LINE_COLOR, linewidth=DISTRICT_LINE_WIDTH,
                alpha=POLYGON_ALPHA, zorder=2,
            )
            if SHOW_DISTRICT_CODES:
                ax.text(
                    row["label_longitude"], row["label_latitude"], row["district_code"],
                    transform=data_crs, ha="center", va="center",
                    fontsize=16.5 if PLOT_MODE == "panel" else 9.0,
                    fontweight="semibold", color="0.15", zorder=4,
                )

        ax.add_geometries(
            [city_outline], crs=data_crs, facecolor="none",
            edgecolor=CITY_LINE_COLOR, linewidth=CITY_LINE_WIDTH, zorder=3,
        )

        panel_letter = chr(ord("a") + plot_index)
        ax.text(
            0.5, CATEGORY_TITLE_Y, f"({panel_letter}) {category_name}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=CATEGORY_TITLE_SIZE if PLOT_MODE == "panel" else 18,
            fontweight="semibold", color="0.15", zorder=10, clip_on=False,
        )

        if PLOT_MODE != "panel":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            single_path = output_path.with_stem(f"{output_path.stem}_{category_name.replace(' ', '_')}")
            fig.savefig(single_path, dpi=150)
            plt.close(fig)
            print(f"Saved: {single_path}")

    if PLOT_MODE == "panel":
        colorbar = fig.colorbar(
            colorbar_object, ax=axes.tolist(), orientation="vertical",
            fraction=0.025, pad=0.04, shrink=0.96, aspect=38,
        )
        colorbar.ax.tick_params(labelsize=18, length=6, width=1.1)
        colorbar.set_label(colorbar_label, fontsize=28, labelpad=14)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {output_path}")


def plot_seasonal_multiplier(
    full_draws: pd.DataFrame, full_city: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """Estimated monthly seasonal multiplier (with 95% credible band)
    against the observed monthly profile for each complete calendar
    year in the data. Also returns/saves the monthly posterior summary
    table so 05_make_tables.py can build the LaTeX table from it
    without re-deriving the seasonal draws.
    """
    season_cols = sorted(
        (c for c in full_draws.columns if c.startswith("season_factor_grid.")),
        key=lambda c: int(c.split(".")[-1]),
    )
    season_draws = full_draws[season_cols].to_numpy()
    calendar_month = pd.date_range("2021-01-01", periods=365).month.to_numpy()

    model_monthly = np.column_stack(
        [season_draws[:, calendar_month == month].mean(axis=1) for month in range(1, 13)]
    )
    model_mean = model_monthly.mean(axis=0)
    model_lower = np.quantile(model_monthly, 0.025, axis=0)
    model_upper = np.quantile(model_monthly, 0.975, axis=0)

    dates = pd.to_datetime(full_city["OCCURRED_ON_DATE"])
    daily_counts = full_city.assign(date=dates.dt.normalize()).groupby("date").size()
    all_days = pd.date_range(dates.min().normalize(), dates.max().normalize(), freq="D")
    daily_counts = daily_counts.reindex(all_days, fill_value=0)

    daily_df = daily_counts.rename("count").reset_index().rename(columns={"index": "date"})
    daily_df["year"] = daily_df["date"].dt.year
    daily_df["month"] = daily_df["date"].dt.month

    full_years = (
        daily_df.groupby("year")["month"].nunique().loc[lambda x: x == 12].index.tolist()
    )
    years_to_plot = full_years[:6]

    observed_profiles = {}
    for year in years_to_plot:
        tmp = daily_df[daily_df["year"] == year]
        monthly_profile = tmp.groupby("month")["count"].mean()
        observed_profiles[year] = monthly_profile / monthly_profile.mean()

    months = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), dpi=300, sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, year in zip(axes, years_to_plot):
        ax.fill_between(months, model_lower, model_upper, color="0.75", alpha=0.35, linewidth=0)
        ax.plot(months, model_mean, color="0.10", linewidth=2.0)
        ax.plot(
            months, observed_profiles[year].values, color="0.45", linestyle="--",
            marker="o", markersize=4, linewidth=1.6,
        )
        ax.axhline(1, color="0.60", linestyle=":", linewidth=1.0)
        ax.tick_params(axis="x", labelsize=16, rotation=45)
        ax.tick_params(axis="y", labelsize=16)
        ax.text(0.03, 0.92, str(year), transform=ax.transAxes, ha="left", va="top", fontsize=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(years_to_plot):]:
        ax.axis("off")
    for ax in axes[-2:]:
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)

    fig.supylabel("Relative level (annual mean = 1)", fontsize=18)
    fig.supxlabel("Month", fontsize=18)

    legend_handles = [
        Line2D([0], [0], color="0.10", lw=2.0, label="Estimated seasonal component"),
        Patch(facecolor="0.75", alpha=0.35, edgecolor="none", label="95% credible band"),
        Line2D(
            [0], [0], color="0.45", lw=1.6, ls="--", marker="o", markersize=4,
            label="Observed seasonal component",
        ),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.985), fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "seasonal_multiplier_monthly.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    summary = pd.DataFrame(
        {"month": month_names, "posterior_mean": model_mean, "lower": model_lower, "upper": model_upper}
    )
    summary_path = output_dir / "seasonal_multiplier_monthly_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    return summary


def plot_event_count_map(
    full_city: pd.DataFrame,
    geojson_path: Path,
    geo_district_col: str,
    output_path: Path,
) -> None:
    """2x2 raster map of recorded-incident density per 150m grid cell,
    one panel per category. Distinct from `plot_district_deviation_map`
    above: this shows raw spatial density, not a district-level
    deviation from the city-wide posterior mean.
    """
    districts = gpd.read_file(geojson_path).to_crs(EVENT_COUNT_CRS_METRIC)
    city_geometry = districts.geometry.union_all()

    events = full_city[["hawkes_id", "Lat", "Long"]].copy()
    events["hawkes_id"] = pd.to_numeric(events["hawkes_id"], errors="coerce")
    events["Lat"] = pd.to_numeric(events["Lat"], errors="coerce")
    events["Long"] = pd.to_numeric(events["Long"], errors="coerce")
    events = events.dropna(subset=["hawkes_id", "Lat", "Long"]).copy()
    events["hawkes_id"] = events["hawkes_id"].astype(int)
    events = events[
        events["hawkes_id"].isin(EVENT_COUNT_CATEGORIES)
        & events["Lat"].between(42.20, 42.45)
        & events["Long"].between(-71.25, -70.90)
    ].copy()

    points = gpd.GeoDataFrame(
        events, geometry=gpd.points_from_xy(events["Long"], events["Lat"]), crs="EPSG:4326"
    ).to_crs(EVENT_COUNT_CRS_METRIC)
    points = points[points.geometry.within(city_geometry)].copy()
    print(f"Geocoded events represented: {len(points):,}")
    print(points["hawkes_id"].value_counts().sort_index())

    cell = EVENT_COUNT_CELL_METRES
    xmin, ymin, xmax, ymax = districts.total_bounds
    xmin = np.floor(xmin / cell) * cell
    ymin = np.floor(ymin / cell) * cell
    xmax = np.ceil(xmax / cell) * cell
    ymax = np.ceil(ymax / cell) * cell
    x_edges = np.arange(xmin, xmax + cell, cell)
    y_edges = np.arange(ymin, ymax + cell, cell)

    rasters = {}
    for category_id in EVENT_COUNT_CATEGORIES:
        category_points = points[points["hawkes_id"] == category_id]
        counts, _, _ = np.histogram2d(
            category_points.geometry.x, category_points.geometry.y, bins=[x_edges, y_edges]
        )
        rasters[category_id] = counts.T  # imshow expects rows = y

    global_max = int(max(raster.max() for raster in rasters.values()))
    bounds = [1, 2, 4, 8, 16, 32, max(33, global_max + 1)]
    count_labels = ["1", "2\u20133", "4\u20137", "8\u201315", "16\u201331", "32+"]
    colorbar_ticks = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)]
    cmap = mcolors.ListedColormap(EVENT_COUNT_GRID_COLORS)
    cmap.set_bad((0, 0, 0, 0))
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=True)

    fig, axes = plt.subplots(2, 2, figsize=(19.5, 18), dpi=150, constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.05, w_pad=0.02, hspace=0.05, wspace=0.02)
    axes = axes.ravel()

    for ax, (category_id, title) in zip(axes, EVENT_COUNT_CATEGORIES.items()):
        raster_masked = np.ma.masked_where(rasters[category_id] == 0, rasters[category_id])
        ax.set_facecolor(EVENT_COUNT_BACKGROUND_COLOR)
        ax.imshow(
            raster_masked, extent=[xmin, xmax, ymin, ymax], origin="lower",
            interpolation="nearest", cmap=cmap, norm=norm, zorder=1,
        )
        districts.boundary.plot(ax=ax, color="black", linewidth=0.75, alpha=0.75, zorder=2)
        ax.set_title(f"{title}\n", fontsize=26, pad=4)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_axis_off()

    scale = ScalarMappable(norm=norm, cmap=cmap)
    scale.set_array([])
    cbar = fig.colorbar(
        scale, ax=axes.tolist(), orientation="vertical", boundaries=bounds,
        ticks=colorbar_ticks, spacing="uniform", fraction=0.025, pad=0.035,
        shrink=0.92, aspect=38,
    )
    cbar.ax.set_yticklabels(count_labels)
    cbar.ax.tick_params(axis="y", labelsize=18, length=6, width=1.1)
    cbar.set_label(f"Recorded incidents per {cell} m \u00d7 {cell} m cell", fontsize=26, labelpad=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--geojson-path", type=Path, default=None,
        help="Police district boundaries GeoJSON; omit to skip the two district maps.",
    )
    p.add_argument("--geo-district-col", default="DISTRICT")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    event_data = read_event_data(args.data_root)
    full_city = event_data["full_city"]

    monthly_counts = build_monthly_counts(full_city)
    monthly_csv = args.output_dir / "monthly_crime_counts.csv"
    monthly_counts.to_csv(monthly_csv, index=False, date_format="%Y-%m-%d", float_format="%.3f")
    print(f"Saved: {monthly_csv}")
    print(monthly_counts.head())
    print(monthly_counts.tail())

    posterior_draws = read_posterior_draws(args.results_root)
    full_draws = posterior_draws["full_city"]
    plot_branching_matrix(full_draws, args.output_dir / "branching_matrix.png")

    plot_seasonal_multiplier(full_draws, full_city, args.output_dir)

    if args.geojson_path is None:
        print("No --geojson-path given; skipping all map figures.")
        return

    plot_event_count_map(
        full_city, args.geojson_path, args.geo_district_col,
        output_path=args.output_dir / "event_count_map.png",
    )

    district_codes = [
        unit.replace("district_", "").upper()
        for unit in posterior_draws
        if unit.startswith("district_")
    ]
    if not district_codes:
        print("No district-level results found; skipping the self-excitation and background maps.")
        return

    district_map, map_extent = load_district_geometries(
        args.geojson_path, args.geo_district_col, district_codes
    )

    prefix = branching_prefix(full_draws)
    self_excitation_percentages = district_deviation_percentages(
        full_draws, posterior_draws, CATEGORY_LABELS,
        parameter_of=lambda cid: f"{prefix}.{cid}.{cid}",
    )
    plot_district_deviation_map(
        district_map, map_extent, self_excitation_percentages, CATEGORY_LABELS,
        colorbar_label="District deviation ratios for the self-excitation estimate (%)",
        output_path=args.output_dir / "self_excitation_map.png",
    )

    background_percentages = district_deviation_percentages(
        full_draws, posterior_draws, CATEGORY_LABELS,
        parameter_of=lambda cid: f"mu.{cid}",
    )
    plot_district_deviation_map(
        district_map, map_extent, background_percentages, CATEGORY_LABELS,
        colorbar_label="District deviation ratios for the background estimate (%)",
        output_path=args.output_dir / "background_map.png",
    )


if __name__ == "__main__":
    main()
