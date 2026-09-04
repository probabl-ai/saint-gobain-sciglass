"""Step 1: Oxide compositions -> 77 atomic mole fractions.

Cassar 2023 (arXiv:2303.15538 §2.1 steps 3-4):
1. Decompose each oxide into elemental contributions (e.g. SiO2 -> Si + 2 O).
2. Sum elemental contributions; renormalize each row to sum to 1.
3. Output a fixed-length vector C over Z = 1-83 (H-Bi), excluding Pm and
   noble gases (He, Ne, Ar, Kr, Xe) -> exactly 77 elements.
4. Absent elements: x_e = 0.
5. Glasses with a non-zero amount of any excluded element: drop the glass.

Outputs:
  data/feature-engineering/train_atomic.csv
  data/feature-engineering/test_atomic.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from glasspy.chemistry import to_element_array

logger = logging.getLogger(__name__)

# Elements Z = 1 to 83, excluding Pm and noble gases (He, Ne, Ar, Kr, Xe)
ELEMENTS_77: list[str] = [
    "H",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
]

ELEMENTS_77_SET = set(ELEMENTS_77)


def oxides_to_atomic_fractions(
    df: pd.DataFrame,
    target_col: str | None = "Tg",
) -> pd.DataFrame:
    """Convert an oxide DataFrame into 77-element atomic mole fractions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with oxide columns (and optional target_col).
    target_col : str or None, default="Tg"
        If present in df, it will be retained and placed as the last column.

    Returns
    -------
    pd.DataFrame
        77 atomic mole fractions [C], normalized to sum to 1.0, plus target_col.
        Rows containing excluded elements are dropped.
    """
    has_target = target_col is not None and target_col in df.columns
    if has_target:
        oxides = df.drop(columns=[target_col])
        targets = df[target_col].reset_index(drop=True)
    else:
        oxides = df.copy()
        targets = None

    # Decompose oxides into element fractions
    elem_arr = to_element_array(oxides, rescale_to_sum=1)
    elem_cols = list(elem_arr.cols)
    elem_mat = np.asarray(elem_arr, dtype=np.float64)

    # Check for excluded / out-of-scope elements
    excluded_mask = np.zeros(len(df), dtype=bool)
    for col_idx, col_name in enumerate(elem_cols):
        if col_name not in ELEMENTS_77_SET:
            # Non-zero amount of an excluded element -> drop the glass
            non_zero = elem_mat[:, col_idx] > 0.0
            excluded_mask |= non_zero

    if np.any(excluded_mask):
        n_dropped = int(np.sum(excluded_mask))
        logger.warning(
            "Dropping %d glasses containing excluded / out-of-scope elements.",
            n_dropped,
        )
        keep_mask = ~excluded_mask
        elem_mat = elem_mat[keep_mask]
        if targets is not None:
            targets = targets[keep_mask].reset_index(drop=True)
    else:
        keep_mask = np.ones(len(df), dtype=bool)

    # Build the 77-element matrix in canonical order
    n_samples = len(elem_mat)
    out_mat = np.zeros((n_samples, len(ELEMENTS_77)), dtype=np.float64)

    col_to_idx = {name: i for i, name in enumerate(elem_cols)}
    for target_idx, el in enumerate(ELEMENTS_77):
        if el in col_to_idx:
            out_mat[:, target_idx] = elem_mat[:, col_to_idx[el]]

    # Renormalize rows to sum to 1.0
    row_sums = out_mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    out_mat = out_mat / row_sums

    res_df = pd.DataFrame(out_mat, columns=ELEMENTS_77)
    if targets is not None:
        res_df[target_col] = targets.values

    return res_df


def main() -> None:
    """Execute Step 1 on train.csv and test.csv, saving atomic fraction tables."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    fe_dir = data_dir / "feature-engineering"
    fe_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    logger.info("Loading %s and %s...", train_path, test_path)
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    logger.info("Converting train oxides to 77 atomic fractions...")
    train_atomic = oxides_to_atomic_fractions(train_raw, target_col="Tg")
    logger.info("Converting test oxides to 77 atomic fractions...")
    test_atomic = oxides_to_atomic_fractions(test_raw, target_col="Tg")

    train_out = fe_dir / "train_atomic.csv"
    test_out = fe_dir / "test_atomic.csv"

    train_atomic.to_csv(train_out, index=False)
    test_atomic.to_csv(test_out, index=False)

    logger.info("Saved %s (shape %s)", train_out, train_atomic.shape)
    logger.info("Saved %s (shape %s)", test_out, test_atomic.shape)


if __name__ == "__main__":
    main()
