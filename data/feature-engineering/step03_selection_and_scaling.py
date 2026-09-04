"""Step 4 & 5: Feature selection (variance + iterative VIF) and MinMaxScaler.

Cassar 2023 (arXiv:2303.15538 §2.2, third step & final paragraph):
1. Drop features with standard deviation < 10^-3:
   VarianceThreshold(threshold=(1e-3) ** 2)
2. Iterative VIF: compute VIF for all remaining features; while any VIF > 5.0,
   drop the feature with the highest VIF and repeat.
3. MinMaxScaler on selected features (fit on train only, transform train & test).
4. Targets preserved in original units (Kelvin) for comparable RMSE reporting.

Outputs:
  data/train_glassnet_features.csv
  data/test_glassnet_features.csv
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def vif_filter(X: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Drop features with VIF above threshold iteratively.

    Uses vectorized correlation matrix inversion: VIF_j = 1 / (1 - R^2_j),
    equal to the j-th diagonal element of the inverse correlation matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (fit on train only).
    threshold : float, default=5.0
        Maximum allowed VIF.

    Returns
    -------
    mask : np.ndarray of bool, shape (n_features,)
        True for features that survive.
    """
    n_features = X.shape[1]
    mask = np.ones(n_features, dtype=bool)

    while True:
        idx = np.where(mask)[0]
        if len(idx) <= 1:
            break

        X_sub = X[:, idx]
        corr = np.corrcoef(X_sub, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        try:
            inv = np.linalg.inv(corr)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(corr)

        vif = np.diag(inv)
        vif = np.maximum(vif, 1.0)

        max_vif = vif.max()
        if max_vif <= threshold:
            break

        drop_local = vif.argmax()
        mask[idx[drop_local]] = False

    return mask


def select_and_scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Tg",
    variance_threshold: float = (1e-3) ** 2,
    vif_threshold: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run variance filter, VIF filter, and MinMaxScaler on train and test.

    All selections and scalers are fit strictly on the train split and
    applied to test. Target values are preserved in original units.

    Parameters
    ----------
    train_df : pd.DataFrame
        Raw features + target for train split.
    test_df : pd.DataFrame
        Raw features + target for test split.
    target_col : str, default="Tg"
        Name of target column.
    variance_threshold : float, default=(1e-3)**2
        Threshold for VarianceThreshold (variance = std^2).
    vif_threshold : float, default=5.0
        VIF cutoff for iterative multicollinearity removal.

    Returns
    -------
    train_scaled : pd.DataFrame
        Selected & scaled features + target for train.
    test_scaled : pd.DataFrame
        Selected & scaled features + target for test.
    """
    has_target_train = target_col in train_df.columns
    has_target_test = target_col in test_df.columns

    X_train = (
        train_df.drop(columns=[target_col]).values
        if has_target_train
        else train_df.values
    )
    X_test = (
        test_df.drop(columns=[target_col]).values if has_target_test else test_df.values
    )
    feature_names = [c for c in train_df.columns if c != target_col]

    n_init = X_train.shape[1]
    logger.info("Starting selection on %d features...", n_init)

    # Step 4a: VarianceThreshold (threshold = std^2 = (1e-3)^2 = 1e-6)
    t0 = time.time()
    vt = VarianceThreshold(threshold=variance_threshold)
    X_train_vt = vt.fit_transform(X_train)
    X_test_vt = vt.transform(X_test)
    vt_mask = vt.get_support()
    feature_names = [f for f, keep in zip(feature_names, vt_mask, strict=True) if keep]
    logger.info(
        "Step 4a (VarianceThreshold): %d -> %d features (%.2fs)",
        n_init,
        X_train_vt.shape[1],
        time.time() - t0,
    )

    # Step 4b: Iterative VIF
    t0 = time.time()
    vif_mask = vif_filter(X_train_vt, threshold=vif_threshold)
    X_train_vif = X_train_vt[:, vif_mask]
    X_test_vif = X_test_vt[:, vif_mask]
    feature_names = [f for f, keep in zip(feature_names, vif_mask, strict=True) if keep]
    logger.info(
        "Step 4b (VIF filter <= %.1f): %d -> %d features (%.2fs)",
        vif_threshold,
        X_train_vt.shape[1],
        X_train_vif.shape[1],
        time.time() - t0,
    )

    # Step 5: MinMaxScaler (fit on train only)
    t0 = time.time()
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_vif)
    X_test_scaled = scaler.transform(X_test_vif)
    logger.info(
        "Step 5 (MinMaxScaler fit on train): scaled %d features (%.2fs)",
        len(feature_names),
        time.time() - t0,
    )

    train_out = pd.DataFrame(X_train_scaled, columns=feature_names)
    test_out = pd.DataFrame(X_test_scaled, columns=feature_names)

    if has_target_train:
        train_out[target_col] = train_df[target_col].values
    if has_target_test:
        test_out[target_col] = test_df[target_col].values

    return train_out, test_out


def main() -> None:
    """Execute Step 4 & 5 on raw physchem CSVs, saving final GlassNet features."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    fe_dir = data_dir / "feature-engineering"

    train_raw_path = fe_dir / "train_physchem_raw.csv"
    test_raw_path = fe_dir / "test_physchem_raw.csv"

    if not train_raw_path.exists() or not test_raw_path.exists():
        msg = "Physchem CSVs not found. Run step02_physchem_features.py first."
        raise FileNotFoundError(msg)

    logger.info("Loading %s and %s...", train_raw_path, test_raw_path)
    train_raw = pd.read_csv(train_raw_path)
    test_raw = pd.read_csv(test_raw_path)

    train_glassnet, test_glassnet = select_and_scale_features(
        train_raw, test_raw, target_col="Tg"
    )

    final_train_path = data_dir / "train_glassnet_features.csv"
    final_test_path = data_dir / "test_glassnet_features.csv"

    logger.info("Saving final GlassNet features to %s...", final_train_path)
    train_glassnet.to_csv(final_train_path, index=False)
    logger.info("Saving final GlassNet features to %s...", final_test_path)
    test_glassnet.to_csv(final_test_path, index=False)

    logger.info(
        "\nDone! Final GlassNet features: %d columns (+ %s)",
        train_glassnet.shape[1] - 1,
        "Tg",
    )
    logger.info(
        "Train shape: %s | Test shape: %s", train_glassnet.shape, test_glassnet.shape
    )


if __name__ == "__main__":
    main()
