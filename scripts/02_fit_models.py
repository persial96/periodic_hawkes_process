# -*- coding: utf-8 -*-
"""
Fit the periodic MHP model to one cleaned train dataset.

Author: Persia Luca (2026), Università della Svizzera italiana, Lugano, Switzerland
Notes: additional revision used Claude Code (model Sonnet 5 and Opus 4.8) to improve code clarity and maintainability.
"""

from __future__ import annotations
import argparse, pandas as pd
from pathlib import Path
from categories import CATEGORY_ORDER
from pmhp.fit import PMHPModel, Priors, prepare_stan_data

DATE_COL = "OCCURRED_ON_DATE"
CATEGORY_COL = "hawkes_category"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--unit", required=True, help="e.g. full_city, district_A1")
    p.add_argument("--baseline", choices=["seasonal", "constant"], default="seasonal")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--stan-file", type=Path, default=None, help="defaults to the bundled model")
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--train-end", default="2025-01-01")
    p.add_argument("--chains", type=int, default=1)
    p.add_argument("--chain-id", type=int, default=1)
    p.add_argument("--threads-per-chain", type=int, default=4)
    p.add_argument("--iter-warmup", type=int, default=1000)
    p.add_argument("--iter-sampling", type=int, default=500)
    p.add_argument("--adapt-delta", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)

    # loading single train CSV (already cleaned and filtered to the desired unit)
    print(f"Loading {args.train_csv} ...")
    df = pd.read_csv(args.train_csv, low_memory=False, na_values=["", " "])
    print(f"Initial shape: {df.shape}")

    # prepare data for Stan
    prepared = prepare_stan_data(
        df,
        date_col=DATE_COL,
        category_col=CATEGORY_COL,
        category_order=CATEGORY_ORDER,
        observation_start=train_start,
        observation_end=train_end,
        baseline=args.baseline,
        priors=Priors(),
    )
    print(
        f"Prepared {prepared.N_events} events across {prepared.D_dims} categories "
        f"({args.baseline} baseline)."
    )

    # fit the model
    model = PMHPModel(stan_file=args.stan_file)
    model.sample(
        prepared,
        chains=args.chains,
        chain_ids=[args.chain_id],
        threads_per_chain=args.threads_per_chain,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        adapt_delta=args.adapt_delta,
        output_dir=str(args.output_dir),
        show_progress=True,
    )
    print(f"Done. CmdStan output written to {args.output_dir}")

    # saving category mapping and run metadata
    category_mapping_df = pd.DataFrame(
        {
            "hawkes_category": list(prepared.category_to_id.keys()),
            "stan_type_id": list(prepared.category_to_id.values()),
        }
    )
    category_mapping_path = (
        args.output_dir / f"category_mapping_{args.baseline}_{args.unit}.csv"
    )
    category_mapping_df.to_csv(category_mapping_path, index=False)
    print(f"Saved category mapping to: {category_mapping_path}")

    # saving run metadata for summary statistics and reproducibility
    metadata_df = pd.DataFrame(
        [
            {
                "model_unit": args.unit,
                "baseline_spec": args.baseline,
                "chain_id": args.chain_id,
                "train_csv": str(args.train_csv),
                "N_events": prepared.N_events,
                "D_dims": prepared.D_dims,
                "observation_start": prepared.observation_start,
                "observation_end": prepared.observation_end,
                "T_observation_days": prepared.T_observation,
            }
        ]
    )
    metadata_path = args.output_dir / f"run_metadata_{args.baseline}_{args.unit}.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Saved run metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
