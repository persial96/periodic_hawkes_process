# -*- coding: utf-8 -*-
"""
Paper appendix LaTeX tables: district posterior estimates of the
background intensities, branching ratios, decay rates, excitation
half-lives, and (if the seasonal model was fit) the annual seasonal
coefficients, plus a master .tex file that \\input{}s all of them.

Ported from `new_reader.py` lines ~1400-2196. The four "one table per
triggered category" loops (branching / beta / half-life) were kept as
one parameterized loop (`parameter_tables`) rather than three
hand-copied blocks, as they already were in the original.

Usage
-----
    python scripts/05_make_tables.py \
        --results-root results/mcmc/seasonal \
        --output-dir results/tables/appendix_district
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pmhp.forecast import branching_prefix

LOWER_Q = 0.025
UPPER_Q = 0.975

CATEGORY_LABELS = {
    1: "Vehicle theft",
    2: "Vandalism and disorder",
    3: "Burglary",
    4: "Violent crime",
}
CATEGORY_HEADERS = {
    1: r"\makecell{Vehicle\\theft}",
    2: r"\makecell{Vandalism\\and disorder}",
    3: r"Burglary",
    4: r"\makecell{Violent\\crime}",
}
DISTRICT_ORDER = ["A1", "A15", "A7", "B2", "B3", "C11", "C6", "D14", "D4", "E13", "E18", "E5"]

PARAMETER_TABLES = {
    "branching": {"caption": "branching ratios", "symbol": r"\alpha", "digits": 3},
    "beta": {"caption": "decay rates", "symbol": r"\beta", "digits": 3},
    "half_life": {"caption": "excitation half-lives", "symbol": r"h", "digits": 3},
}


def posterior_summary(values: pd.Series, digits: int = 3) -> tuple[str, str]:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return "--", "--"
    estimate = values.mean()
    lower = values.quantile(LOWER_Q)
    upper = values.quantile(UPPER_Q)
    return f"{estimate:.{digits}f}", f"[{lower:.{digits}f}, {upper:.{digits}f}]"


def posterior_cell(values: pd.Series, digits: int = 3) -> str:
    estimate, interval = posterior_summary(values, digits=digits)
    if estimate == "--":
        return "--"
    return rf"\makecell{{{estimate}\\{{\scriptsize {interval}}}}}"


def category_slug(category_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category_name.lower()).strip("_")


def order_units(posterior_draws: dict[str, pd.DataFrame], include_full_city: bool) -> list[tuple[str, str]]:
    """(display label, unit key) pairs, full city first then DISTRICT_ORDER."""
    unit_lookup: dict[str, str] = {}
    for unit_name in posterior_draws:
        if unit_name == "full_city":
            unit_lookup["Full city"] = unit_name
        elif unit_name.startswith("district_"):
            unit_lookup[unit_name.replace("district_", "").upper()] = unit_name

    unit_rows = []
    if include_full_city and "Full city" in unit_lookup:
        unit_rows.append(("Full city", unit_lookup["Full city"]))
    for district_code in DISTRICT_ORDER:
        if district_code in unit_lookup:
            unit_rows.append((district_code, unit_lookup[district_code]))
    return unit_rows


def write_background_tables(
    posterior_draws: dict[str, pd.DataFrame], unit_rows: list[tuple[str, str]], output_dir: Path
) -> list[Path]:
    """One table per category: district posterior mean background intensity mu_i."""
    generated = []
    for child_id, child_name in CATEGORY_LABELS.items():
        slug = category_slug(child_name)
        output_file = output_dir / f"background_{slug}.tex"

        lines = [
            r"\begin{table}[!htbp]",
            r"\centering",
            rf"\caption{{District posterior estimates of the average "
            rf"background intensity for {child_name.lower()}.}}",
            rf"\label{{tab:appendix-background-{slug}}}",
            r"\small",
            r"\renewcommand{\arraystretch}{1.20}",
            r"\setlength{\tabcolsep}{8pt}",
            r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X"
            r">{\centering\arraybackslash}X>{\centering\arraybackslash}X@{}}",
            r"\toprule",
            r"Geographical unit & Posterior mean & 95\% CrI \\",
            r"\midrule",
        ]

        parameter = f"mu.{child_id}"
        for geographical_label, unit_name in unit_rows:
            draws = posterior_draws[unit_name]
            if parameter in draws.columns:
                estimate, interval = posterior_summary(draws[parameter], digits=3)
            else:
                estimate, interval = "--", "--"
            lines.append(f"{geographical_label} & {estimate} & {interval} \\\\")
            if geographical_label == "Full city":
                lines.append(r"\addlinespace[2pt]")

        lines += [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\vspace{1mm}",
            r"\begin{minipage}{\textwidth}",
            r"\raggedright",
            r"\footnotesize",
            r"\textit{Notes:} Entries report posterior means and "
            r"95\% credible intervals for the average background "
            rf"intensity \(\mu_{{{child_id}}}^{{(r)}}\), measured in "
            r"expected events per day. The seasonal multiplier in each "
            r"geographical model is normalized to have an annual mean of one.",
            r"\end{minipage}",
            r"\end{table}",
        ]

        output_file.write_text("\n".join(lines), encoding="utf-8")
        generated.append(output_file)
    return generated


def write_parameter_tables(
    posterior_draws: dict[str, pd.DataFrame],
    unit_rows: list[tuple[str, str]],
    branch_prefix: str,
    output_dir: Path,
) -> list[Path]:
    """One table per (parameter_type, triggered category): branching
    ratio, decay rate (beta), and excitation half-life. These three
    were three hand-copied ~200-line loops in the original; here they
    share one loop keyed by PARAMETER_TABLES.
    """
    generated = []

    for parameter_type, settings in PARAMETER_TABLES.items():
        for child_id, child_name in CATEGORY_LABELS.items():
            slug = category_slug(child_name)
            output_file = output_dir / f"{parameter_type}_{slug}.tex"

            if parameter_type == "branching":
                caption = (
                    "District posterior estimates of branching ratios "
                    f"for {child_name.lower()} as the triggered category."
                )
            elif parameter_type == "beta":
                caption = (
                    "District posterior estimates of excitation decay rates "
                    f"for {child_name.lower()} as the triggered category."
                )
            else:
                caption = (
                    "District posterior estimates of excitation half-lives "
                    f"for {child_name.lower()} as the triggered category."
                )

            lines = [
                r"\begin{table}[!htbp]",
                r"\centering",
                rf"\caption{{{caption}}}",
                rf"\label{{tab:appendix-{parameter_type}-{slug}}}",
                r"\scriptsize",
                r"\renewcommand{\arraystretch}{1.35}",
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{2.2cm}"
                r"*{4}{>{\centering\arraybackslash}X}@{}}",
                r"\toprule",
                "Geographical unit & "
                + " & ".join(CATEGORY_HEADERS[pid] for pid in CATEGORY_LABELS)
                + r" \\",
                r"\midrule",
            ]

            for geographical_label, unit_name in unit_rows:
                draws = posterior_draws[unit_name]
                cells = []
                for parent_id in CATEGORY_LABELS:
                    if parameter_type == "branching":
                        parameter = f"{branch_prefix}.{child_id}.{parent_id}"
                        values = draws[parameter] if parameter in draws.columns else pd.Series(dtype=float)
                    elif parameter_type == "beta":
                        parameter = f"beta.{child_id}.{parent_id}"
                        values = draws[parameter] if parameter in draws.columns else pd.Series(dtype=float)
                    else:
                        stored_half_life = f"half_life_days.{child_id}.{parent_id}"
                        beta_parameter = f"beta.{child_id}.{parent_id}"
                        if stored_half_life in draws.columns:
                            values = draws[stored_half_life]
                        elif beta_parameter in draws.columns:
                            beta_draws = pd.to_numeric(draws[beta_parameter], errors="coerce")
                            values = np.log(2) / beta_draws
                        else:
                            values = pd.Series(dtype=float)
                    cells.append(posterior_cell(values, digits=settings["digits"]))

                lines.append(geographical_label + " & " + " & ".join(cells) + r" \\")
                if geographical_label == "Full city":
                    lines.append(r"\addlinespace[2pt]")

            lines += [r"\bottomrule", r"\end{tabularx}", r"\vspace{1mm}",
                      r"\begin{minipage}{\textwidth}", r"\raggedright", r"\footnotesize"]

            if parameter_type == "branching":
                lines.append(
                    r"\textit{Notes:} Cells report posterior means with "
                    r"95\% credible intervals underneath. Columns identify "
                    rf"the parent category \(j\), while {child_name.lower()} "
                    rf"is the triggered category \(i={child_id}\). "
                    r"Diagonal entries therefore represent self-excitation."
                )
            elif parameter_type == "beta":
                lines.append(
                    r"\textit{Notes:} Cells report posterior means with "
                    r"95\% credible intervals underneath. Decay rates are "
                    r"measured in inverse days. Columns identify the parent "
                    rf"category \(j\), while {child_name.lower()} is the "
                    rf"triggered category \(i={child_id}\)."
                )
            else:
                lines.append(
                    r"\textit{Notes:} Cells report posterior means with "
                    r"95\% credible intervals underneath. Half-lives are "
                    r"measured in days and are calculated separately for "
                    r"every posterior draw as "
                    r"\(\log(2)/\beta_{ij}^{(r)}\). Columns identify the "
                    rf"parent category \(j\), while {child_name.lower()} "
                    rf"is the triggered category \(i={child_id}\)."
                )

            lines += [r"\end{minipage}", r"\end{table}"]
            output_file.write_text("\n".join(lines), encoding="utf-8")
            generated.append(output_file)

    return generated


def write_seasonal_coefficient_table(
    posterior_draws: dict[str, pd.DataFrame], unit_rows: list[tuple[str, str]], output_dir: Path
) -> Path:
    """One table: annual seasonal Fourier coefficients (gamma_c, gamma_s),
    shared across categories within each geographical model.
    """
    output_file = output_dir / "seasonal_coefficients.tex"

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{District posterior estimates of the annual seasonal coefficients.}",
        r"\label{tab:appendix-seasonal-coefficients}",
        r"\small",
        r"\renewcommand{\arraystretch}{1.20}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X"
        r">{\centering\arraybackslash}X>{\centering\arraybackslash}X"
        r">{\centering\arraybackslash}X>{\centering\arraybackslash}X@{}}",
        r"\toprule",
        r"Geographical unit & \(\gamma_c\) & 95\% CrI & \(\gamma_s\) & 95\% CrI \\",
        r"\midrule",
    ]

    cosine_candidates = ["season_cos_coef", "gamma_c", "gamma_cos", "season_cos"]
    sine_candidates = ["season_sin_coef", "gamma_s", "gamma_sin", "season_sin"]

    for geographical_label, unit_name in unit_rows:
        draws = posterior_draws[unit_name]
        cosine_parameter = next((p for p in cosine_candidates if p in draws.columns), None)
        sine_parameter = next((p for p in sine_candidates if p in draws.columns), None)

        cosine_mean, cosine_interval = (
            posterior_summary(draws[cosine_parameter], digits=3) if cosine_parameter else ("--", "--")
        )
        sine_mean, sine_interval = (
            posterior_summary(draws[sine_parameter], digits=3) if sine_parameter else ("--", "--")
        )

        lines.append(
            f"{geographical_label} & {cosine_mean} & {cosine_interval} "
            f"& {sine_mean} & {sine_interval} \\\\"
        )
        if geographical_label == "Full city":
            lines.append(r"\addlinespace[2pt]")

    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\vspace{1mm}",
        r"\begin{minipage}{\textwidth}",
        r"\raggedright",
        r"\footnotesize",
        r"\textit{Notes:} Entries report posterior means and "
        r"95\% credible intervals for the cosine coefficient "
        r"\(\gamma_c\) and sine coefficient \(\gamma_s\) of the "
        r"normalized annual Fourier component. The seasonal component "
        r"is shared across crime categories within each geographical model.",
        r"\end{minipage}",
        r"\end{table}",
    ]

    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file


def write_master_file(output_dir: Path, write_seasonal: bool) -> Path:
    master_file = output_dir / "district_parameter_tables.tex"
    slugs = [category_slug(name) for name in CATEGORY_LABELS.values()]

    sections = [
        (r"\subsection{District background intensities}", "background"),
        (r"\subsection{District branching ratios}", "branching"),
        (r"\subsection{District excitation decay rates}", "beta"),
        (r"\subsection{District excitation half-lives}", "half_life"),
    ]

    master_lines = []
    for heading, stem in sections:
        master_lines.append(heading)
        master_lines.extend(rf"\input{{{stem}_{slug}}}" for slug in slugs)

    if write_seasonal:
        master_lines.append(r"\subsection{District seasonal coefficients}")
        master_lines.append(r"\input{seasonal_coefficients}")

    master_file.write_text("\n\n".join(master_lines), encoding="utf-8")
    return master_file


def full_city_decay_summary(full_draws: pd.DataFrame) -> pd.DataFrame:
    """Full-city decay-rate / half-life summary across all category
    pairs, as a small DataFrame -- handy as a quick sanity check
    alongside the per-category appendix tables above.
    """
    rows = []
    for parent_id, parent_name in CATEGORY_LABELS.items():
        for child_id, child_name in CATEGORY_LABELS.items():
            beta_draws = full_draws[f"beta.{child_id}.{parent_id}"]
            half_life_draws = full_draws[f"half_life_days.{child_id}.{parent_id}"]
            rows.append(
                {
                    "relationship": f"{parent_name} \u2192 {child_name}",
                    "beta_mean": beta_draws.mean(),
                    "beta_lower": beta_draws.quantile(LOWER_Q),
                    "beta_upper": beta_draws.quantile(UPPER_Q),
                    "half_life_mean": half_life_draws.mean(),
                    "half_life_lower": half_life_draws.quantile(LOWER_Q),
                    "half_life_upper": half_life_draws.quantile(UPPER_Q),
                }
            )
    return pd.DataFrame(rows)


def read_posterior_draws(results_root: Path) -> dict[str, pd.DataFrame]:
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
        chains = [pd.read_csv(f, comment="#", low_memory=False) for f in chain_files]
        posterior_draws[unit_dir.name] = pd.concat(chains, ignore_index=True)
    return posterior_draws


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--no-full-city", action="store_true", help="Exclude the full-city row.")
    p.add_argument(
        "--no-seasonal-table", action="store_true",
        help="Skip the seasonal-coefficient table (e.g. for a constant-baseline fit).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    posterior_draws = read_posterior_draws(args.results_root)
    if not posterior_draws:
        raise FileNotFoundError(f"No unit results found under {args.results_root}")
    full_draws = posterior_draws["full_city"]

    unit_rows = order_units(posterior_draws, include_full_city=not args.no_full_city)
    print("Units included:", [label for label, _ in unit_rows])

    generated = []
    generated += write_background_tables(posterior_draws, unit_rows, args.output_dir)
    generated += write_parameter_tables(
        posterior_draws, unit_rows, branching_prefix(full_draws.columns), args.output_dir
    )
    if not args.no_seasonal_table:
        generated.append(write_seasonal_coefficient_table(posterior_draws, unit_rows, args.output_dir))
    generated.append(write_master_file(args.output_dir, write_seasonal=not args.no_seasonal_table))

    print(f"\nGenerated {len(generated)} LaTeX files:")
    for file_path in generated:
        print(file_path)

    decay_summary = full_city_decay_summary(full_draws)
    decay_summary_path = args.output_dir / "full_city_decay_summary.csv"
    decay_summary.to_csv(decay_summary_path, index=False)
    print(f"\nSaved: {decay_summary_path}")
    print(decay_summary)


if __name__ == "__main__":
    main()
