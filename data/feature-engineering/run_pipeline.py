"""End-to-end runner for all feature engineering steps.

Executes:
  1. step01_atomic_fractions -> train_atomic.csv, test_atomic.csv
  2. step02_physchem_features -> train_physchem_raw.csv, test_physchem_raw.csv
  3. step03_selection_and_scaling -> final train & test glassnet features
"""

from __future__ import annotations

import logging

import step01_atomic_fractions
import step02_physchem_features
import step03_selection_and_scaling

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run all feature engineering steps sequentially."""
    logger.info("=== Step 1: Oxide -> 77 atomic mole fractions ===")
    step01_atomic_fractions.main()

    logger.info("\n=== Step 2 & 3: 77 fractions -> 627 raw physchem features ===")
    step02_physchem_features.main()

    logger.info("\n=== Step 4 & 5: Variance filter + VIF + MinMaxScaler ===")
    step03_selection_and_scaling.main()

    logger.info("\n=== All feature engineering steps completed successfully! ===")


if __name__ == "__main__":
    main()
