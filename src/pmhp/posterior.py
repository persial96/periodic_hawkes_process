# -*- coding: utf-8 -*-
"""
Shared utilities for reading a fitted unit's CmdStan posterior draws.

Author: Persia Luca (2026), Università della Svizzera italiana, Lugano, Switzerland
Notes: additional revision used Claude (model Sonnet 5) to improve code clarity and maintainability.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np, pandas as pd


def branching_prefix(columns: Iterable[str]) -> str:
    """Some chains store the branching matrix as alpha.i.j, others as branching_ratio.i.j -- detect which one a set of draw columns uses.
    """
    columns = list(columns)
    if any(c.startswith("branching_ratio.") for c in columns):
        return "branching_ratio"
    if any(c.startswith("alpha.") for c in columns):
        return "alpha"
    raise KeyError("Could not find alpha.i.j or branching_ratio.i.j columns.")


def resolve_result_directory(results_root: Path, unit: str) -> Path:
    """Locate a unit's CmdStan output folder, allowing for the 'district_A1' -> 'A1' naming variant some result trees use.
    """
    candidates = [results_root / unit]
    if unit.startswith("district_"):
        candidates.append(results_root / unit.replace("district_", ""))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find {unit} under:\n{results_root}")


def read_posterior_draws(results_root: Path, unit: str) -> pd.DataFrame:
    """All posterior draws (every chain, concatenated) for one unit tagged with `chain` and `draw_within_chain`.
    """
    unit_dir = resolve_result_directory(results_root, unit)
    chain_files = sorted(
        p for p in unit_dir.glob("*.csv") if p.name.lower().startswith("hawkes")
    )
    if not chain_files:
        raise FileNotFoundError(f"No Stan chain CSVs found in:\n{unit_dir}")

    chains = []
    for chain_number, file_path in enumerate(chain_files, start=1):
        chain = pd.read_csv(file_path, comment="#", low_memory=False)
        chain["chain"] = chain_number
        chain["draw_within_chain"] = np.arange(len(chain))
        chains.append(chain)

    return pd.concat(chains, ignore_index=True)
