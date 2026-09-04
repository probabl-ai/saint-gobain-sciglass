# GlassNet feature helpers

`Optional pipeline that turns frozen oxide tables into GlassNet features.
Do not fold this into `data/prepare.py`. Method details: [glassnet.md](glassnet.md).

## Run

Scripts import each other as local modules, so run them from this folder:

```bash
cd data/feature-engineering
python run_pipeline.py
```

Or one step at a time (same order):

```bash
python step01_atomic_fractions.py
python step02_physchem_features.py
python step03_selection_and_scaling.py
```

Fit selection and scaling **once on train**. Never train on `data/test.csv`.
Never overwrite `data/train.csv` or `data/test.csv`.

## Outputs

| File | Written by | Notes |
|---|---|---|
| `element_properties_55.csv` | `property_table.py` | Cached 77×55 table **S** (gitignored) |
| `train_atomic.csv`, `test_atomic.csv` | `step01_atomic_fractions.py` | 77 atomic mole fractions + `Tg` |
| `train_physchem_raw.csv`, `test_physchem_raw.csv` | `step02_physchem_features.py` | 627 raw features + `Tg` |
| `data/train_glassnet_features.csv`, `data/test_glassnet_features.csv` | `step03_selection_and_scaling.py` | Selected, MinMax-scaled features; `Tg` stays in Kelvin |

Hub / CV work that uses GlassNet must load those two `*_glassnet_features.csv` files, not the oxide tables.

## Helpers

Import from this directory (or add it to `PYTHONPATH`):

```python
from step01_atomic_fractions import oxides_to_atomic_fractions
from property_table import get_property_table
from step02_physchem_features import atomic_to_physchem_features
from step03_selection_and_scaling import select_and_scale_features
```

- `oxides_to_atomic_fractions(df, target_col="Tg")` — oxide columns → 77-vector **C**. Drops rows with excluded elements.
- `get_property_table()` — loads/caches **S**. Rebuild from GlassPy with `get_property_table(force_reload=True)`.
- `atomic_to_physchem_features(atomic_df, target_col="Tg")` — **C** → 627 columns `[C | weighted | absolute]`.
- `select_and_scale_features(train_df, test_df, target_col="Tg")` — variance filter, iterative VIF, MinMax on **features only**; fit on train, apply to test.
