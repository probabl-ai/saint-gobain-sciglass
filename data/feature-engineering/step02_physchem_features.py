"""Step 2 & 3: 77 atomic mole fractions -> 627 raw GlassNet features.

Cassar 2023 (arXiv:2303.15538 §2.2, second step):
For each of the 55 elemental property columns S_i, aggregate over the
full 77-element vector including zeros:
  - Weighted: w = f(C ⊙ S_i) — Eq. 1
  - Absolute: a = f(⌈C⌉ ⊙ S_i) — Eq. 2
Aggregators f: {sum, min, max, mean, std}.

Output feature vector:
  [C | weighted | absolute] -> exactly 627 features:
  77 atomic fractions + 275 weighted (55 x 5) + 275 absolute (55 x 5).

Outputs:
  data/feature-engineering/train_physchem_raw.csv
  data/feature-engineering/test_physchem_raw.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from property_table import get_property_table
from step01_atomic_fractions import ELEMENTS_77

logger = logging.getLogger(__name__)

AGGREGATORS = ["sum", "min", "max", "mean", "std"]


def atomic_to_physchem_features(
    atomic_df: pd.DataFrame,
    target_col: str | None = "Tg",
    property_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute 627 raw GlassNet features from 77 atomic mole fractions.

    Parameters
    ----------
    atomic_df : pd.DataFrame
        DataFrame containing the 77 element columns (and optional target_col).
    target_col : str or None, default="Tg"
        If present, retained as the last column of the output DataFrame.
    property_table : pd.DataFrame or None, default=None
        Pre-loaded (77, 55) property table S. If None, loaded via get_property_table().

    Returns
    -------
    pd.DataFrame
        DataFrame with 627 feature columns [C | weighted | absolute] + optional target.
    """
    if property_table is None:
        property_table = get_property_table()

    # Verify elements match canonical order
    S = property_table.loc[ELEMENTS_77].values.astype(np.float64)  # shape (77, 55)
    prop_names = list(property_table.columns)
    n_props = len(prop_names)

    has_target = target_col is not None and target_col in atomic_df.columns
    if has_target:
        C = atomic_df[ELEMENTS_77].values.astype(np.float64)
        targets = atomic_df[target_col].reset_index(drop=True)
    else:
        C = atomic_df[ELEMENTS_77].values.astype(np.float64)
        targets = None

    n_samples = len(C)

    # 1. Elemental fractions C (shape: n_samples, 77)
    # 2. Weighted: v_w = C ⊙ S_i (3D broadcast: (N, 77, 1) * (1, 77, 55) -> (N, 77, 55))
    v_w = C[:, :, None] * S[None, :, :]
    w_sum = v_w.sum(axis=1)
    w_min = v_w.min(axis=1)
    w_max = v_w.max(axis=1)
    w_mean = v_w.mean(axis=1)
    w_std = v_w.std(axis=1, ddof=0)

    # 3. Absolute: v_a = ⌈C⌉ ⊙ S_i (ceil operator keeps 0 as 0, and non-zero as 1)
    C_ceil = (C > 0.0).astype(np.float64)
    v_a = C_ceil[:, :, None] * S[None, :, :]
    a_sum = v_a.sum(axis=1)
    a_min = v_a.min(axis=1)
    a_max = v_a.max(axis=1)
    a_mean = v_a.mean(axis=1)
    a_std = v_a.std(axis=1, ddof=0)

    # Build feature column names and matrices in deterministic order
    col_names: list[str] = list(ELEMENTS_77)
    blocks: list[np.ndarray] = [C]

    # Weighted columns: W|<property>|<agg>
    w_dict = {
        "sum": w_sum,
        "min": w_min,
        "max": w_max,
        "mean": w_mean,
        "std": w_std,
    }
    for prop_idx, prop in enumerate(prop_names):
        for agg in AGGREGATORS:
            col_names.append(f"W|{prop}|{agg}")
            blocks.append(w_dict[agg][:, prop_idx : prop_idx + 1])

    # Absolute columns: A|<property>|<agg>
    a_dict = {
        "sum": a_sum,
        "min": a_min,
        "max": a_max,
        "mean": a_mean,
        "std": a_std,
    }
    for prop_idx, prop in enumerate(prop_names):
        for agg in AGGREGATORS:
            col_names.append(f"A|{prop}|{agg}")
            blocks.append(a_dict[agg][:, prop_idx : prop_idx + 1])

    feat_matrix = np.hstack(blocks)
    expected_cols = 77 + 2 * n_props * len(AGGREGATORS)
    assert feat_matrix.shape == (n_samples, expected_cols), (
        f"Expected {expected_cols} columns, got {feat_matrix.shape[1]}"
    )

    out_df = pd.DataFrame(feat_matrix, columns=col_names)
    if targets is not None:
        out_df[target_col] = targets.values

    return out_df


def main() -> None:
    """Execute Step 2 & 3 on atomic fraction CSVs, saving 627 raw features."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fe_dir = Path(__file__).resolve().parent

    train_atomic_path = fe_dir / "train_atomic.csv"
    test_atomic_path = fe_dir / "test_atomic.csv"

    if not train_atomic_path.exists() or not test_atomic_path.exists():
        msg = "Atomic fraction CSVs not found. Run step01_atomic_fractions.py first."
        raise FileNotFoundError(msg)

    logger.info("Loading atomic fraction tables...")
    train_atomic = pd.read_csv(train_atomic_path)
    test_atomic = pd.read_csv(test_atomic_path)

    property_table = get_property_table()
    logger.info(
        "Computing 627 raw features using property table with %d properties...",
        len(property_table.columns),
    )

    train_raw = atomic_to_physchem_features(
        train_atomic, target_col="Tg", property_table=property_table
    )
    test_raw = atomic_to_physchem_features(
        test_atomic, target_col="Tg", property_table=property_table
    )

    train_out = fe_dir / "train_physchem_raw.csv"
    test_out = fe_dir / "test_physchem_raw.csv"

    logger.info("Saving %s (shape %s)...", train_out, train_raw.shape)
    train_raw.to_csv(train_out, index=False)
    logger.info("Saving %s (shape %s)...", test_out, test_raw.shape)
    test_raw.to_csv(test_out, index=False)
    logger.info("Done! Raw features: %d columns + target", train_raw.shape[1] - 1)


if __name__ == "__main__":
    main()
