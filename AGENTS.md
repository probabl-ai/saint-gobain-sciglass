# Agents instructions

Only reports on the test data should be sent to the hub. Not reports on train
data. If you run tests, it should also always run on a carefully crafted subset
of data so that tests are fast. Also please ask if I want to limit the computer
resource that you take when running heavy stuff, (like throttle to 2 CPU max).

# Train / test protocol

- **Fit and cross-validate only on `data/train.csv`.**
- **Pass `data/test.csv` manually as the test set** into skore evaluation (do
  not use a random holdout sliced from train as the report’s test set).
- **Skore Hub must use that same `test.csv` as the test set** for the pushed
  report (train = `train.csv`, test = `test.csv`).
- **Always push a fitted estimator alongside every CrossValidationReport**
  submitted to Hub (same train/test split). Use an `EstimatorReport` (or
  equivalent skore report type) for the fitted estimator — `Project.put` does
  **not** accept a raw sklearn estimator object.

# G_SKORE_MODE and project/workspace

G_SKORE_MODE=hub

When pushing a CV report to Skore Hub, use:

- **workspace:** `saint-gobain-testing`
- **project:** `saint-gobain-cv`

One part that fits a learner on the whole train set of our data, and then
evaluate the learner on the test set with skore, and push the resulting report.

- **workspace:** `saint-gobain-testing`
- **project:** `saint-gobain-leaderboard`

## Skore Hub login (required)

`SKORE_HUB_API_KEY` must be present in local `.env` (user-created; not shipped
in the repo). See [`README.md`](README.md) → **Skore Hub API key**.

**Do not start modelling, Hub pushes, or other workshop work until the key is
set.** If `.env` is missing `SKORE_HUB_API_KEY` (or it is empty), **stop and
ask the user** to create a key in Skore Hub workspace settings and add it to
`.env`. Do not fall back to interactive / browser login.

Non-interactive Hub auth only:

1. Load `.env` (`set -a && source .env && set +a`, or equivalent).
2. Assert `SKORE_HUB_API_KEY` is non-empty.
3. Call `skore.login(mode="hub")` — with the env var set, this uses the API-key
   path (`X-API-Key`) and must not open a browser.

```python
import os
from skore import login
import skore

assert os.environ.get("SKORE_HUB_API_KEY"), "SKORE_HUB_API_KEY missing"
login(mode="hub")

project = skore.Project(
    name="saint-gobain",
    mode="hub",
    workspace="saint-gobain-testing",
)
```

`skore.CrossValidationReport` takes `splitter=`, not `cv=`.

## Data (prepare once — not part of the ML pipeline)

Workshop modelling **must not** re-run SciGlass loading or the silicate filter
inside `sklearn.pipeline.Pipeline`, `cross_validate`, or `skore.evaluate`.
Prepare **once**, write `data/train.csv` and `data/test.csv`, then train on
that frozen table.

If `data/train.csv` and `data/test.csv` already exist and are non-empty,
**do not regenerate them**. Use them as-is so all workshop reports share the
same rows and split.

### What the frozen tables are (bootstrap / baseline)

The one-time write is **filtered oxide compositions + target**.

1. Load SciGlass, keep **SiO2 > 60 mol%**, keep oxides present in >1% of
   silicates, drop rows with NaN target, leak-free composition split.
2. Write those **raw oxide mole fractions** plus the target column(s).
3. The **baseline** trains on that table.

**Do not implement GlassNet (atomic fractions, mendeleev aggregations, VIF,
MinMax) at bootstrap or as part of the first baseline.** After a baseline
exists on the oxide table, you may **suggest** GlassNet as a later experiment.
If the user agrees, follow § GlassNet below and the pitfalls in
**Known difficulties**.

### 1. Load SciGlass (GlassPy)

Source: [GlassPy](https://github.com/drcassar/glasspy) / SciGlass.

```python
from glasspy.data import SciGlass

source = SciGlass()
df = source.data
```

The first load downloads and parses; later loads use the local GlassPy cache
(`~/Library/Application Support/GlassPy/` on macOS). See `data/README.md` if
a shared cache archive is available.

`df` is a **MultiIndex** DataFrame. Level 0 names are `elements`, `compounds`,
`property`, `metadata` — not `properties`. Example: `df[("property", "Tg")]`,
`df["compounds"]`.

Composition is **oxide mole fractions** (rows sum to ~1), not wt%. Do **not**
convert wt% → mol%. Confirm on a few rows with `X.sum(axis=1)`.

### 2. Filter to silicate glasses

Keep glasses with **SiO2 > 60 mol%** (`df["compounds"]["SiO2"] > 0.60`).
Do not change this filter.

Use oxide columns present in >1% of silicates (typically 29 oxides, including
halides such as `F` and `Cl`).

Targets (drop rows where the chosen target is NaN): `Tg`, `Tliquidus`, `T3`,
`T4`.

### 3. Leak-free train/test split (before any later feature fitting)

Use **`GroupShuffleSplit`** grouped by **exact oxide composition** (identical
composition vector → same group). Never put the same composition in both
train and test.

- `test_size=0.2`, `random_state=42`, `n_splits=1`
- Split **raw oxide** tables first. If you later add GlassNet, fit that
  pipeline on **train only**, then `transform` train and test.

### 4. Write the shared tables (baseline)

After the filter and split, write:

- `data/train.csv` — oxide columns + target column(s)
- `data/test.csv` — same columns, test rows only

Modelling then follows **Train / test protocol**: fit/CV on `train.csv`;
evaluate and push hub reports with `test.csv` as the test set.

## GlassNet feature engineering (optional, after baseline)

Cassar 2023, arXiv:2303.15538 §2.2. **Suggestion only** once a baseline on
raw oxides exists. Do not fold this into the one-time silicate prep above.
Do not re-fit variance/VIF/scalers inside later modelling CV folds: if you
run GlassNet, fit selection and scaling **once on the frozen train split**.
If you implement this, **fit once on train** (never on test, never
on the full matrix). Save results as `data/train_glassnet_features.csv` and
`data/test_glassnet_features.csv` — **do not overwrite** `data/train.csv` /
`data/test.csv`.

### Step 1 — oxide → atomic mole fractions

Decompose each oxide (e.g. SiO2 → Si + 2 O; Na2O → 2 Na + O). Sum elemental
contributions; renormalize each row to sum to 1. Output a fixed-length vector
**C** over Z = 1–83 (H–Bi), **excluding Pm and noble gases** (He, Ne, Ar, Kr,
Xe, Rn) → ~77 elements. Absent elements: `x_e = 0`. Elements in the data
**outside** this scope: **raise**. Store the element order on the transformer.

**GlassPy 0.6 has no `CompositionToAtomicFractions` class.** Use
`glasspy.chemistry.to_element_array` (also exported from
`glasspy.chemistry.featurizer`).

### Step 2 — elemental property table S (mendeleev; cache; not per-row)

Numeric properties for every element in scope. The paper used 55 (mendeleev +
matminer); pull what mendeleev has: radii, volume, weight, electron affinity,
ionization energy, electronegativities, boiling/melting, fusion/evaporation
heat, heat capacity / thermal conductivity, polarizability, density, oxidation
states, period, group, block, valence electrons.

Parse electronic configuration for s/p/d/f electron counts; unit-test Fe, O,
Na (and similar). **Drop** any property with missing values on any of the ~77
elements. Record kept properties. Cache S.

### Step 3 — weighted + absolute features

For each property column `S_i`, aggregate over the **full** ~77-vector
**including zeros** (do not drop absent elements):

- Weighted: `agg(C ⊙ S_i)` — composition-sensitive
- Absolute: `agg(⌈C⌉ ⊙ S_i)` — presence (`x_e > 0`) only

Aggregators: `{sum, min, max, mean, std}`.

Concatenate **[C | weighted | absolute]**. Redundant/constant features are
expected; pruning is Step 4.

**GlassPy 0.6 has no `PhysicoChemicalFeaturizer` class.** Use
`glasspy.chemistry.physchem_featurizer`. `all_features` is a **list of
`(property, aggregator)` tuples**, not a callable. Passing the full
`all_features` list will raise `ValueError: Invalid features` — see
**Known difficulties**.

### Step 4 — feature selection (fit on train only, once)

1. `VarianceThreshold(threshold=(1e-3) ** 2)` (sklearn’s threshold is
   **variance**, not std).
2. Iterative VIF: while any VIF > 5, drop the highest-VIF feature. Custom
   `VIFSelector`; **vectorize** (do not loop `statsmodels`
   `variance_inflation_factor` on hundreds of columns — that can run for
   tens of minutes with no useful output). Log runtime and before/after
   counts.

Paper reference (counts will differ): 627 → 98 features. A tiny hand-picked
property subset plus VIF ≤ 5 can collapse to **two** features; that is not a
faithful GlassNet table.

### Step 5 — scaling (fit on train only, once)

`MinMaxScaler` on **selected features**. Fit on train, transform train and
test. Min-max (not `StandardScaler`) preserves exact zeros and is robust to
tiny standard deviations on sparse compositional features.

If you also scale **targets**, fit a separate scaler on **train y only** and
persist it; inverse-transform predictions before reporting Kelvin RMSE on
the hub so metrics stay comparable.

Write GlassNet outputs as **new files**. Never overwrite `data/train.csv` or
`data/test.csv`:

- `data/train_glassnet_features.csv`
- `data/test_glassnet_features.csv`

Fit/CV on the GlassNet train file; Hub test reports must use the matching
GlassNet test file. Same composition split as the oxide tables.

## Known difficulties (skore-agent / GlassPy 0.6)

These are real failure modes from implementing the above against current
GlassPy and skore. Read them before touching GlassNet or Hub pushes.

### SciGlass frame layout

- MultiIndex level 0 is `property`, not `properties`.
- Oxides live under `df["compounds"]`. Selecting
  `df[("compounds", list_of_names)]` is invalid; slice the `compounds` block,
  then subset columns.
- Silicate filter: `df[df[("compounds", "SiO2")] > 0.60]` (or equivalent on
  the compounds block).

### GlassPy chemistry API ≠ the paper class names

- `from glasspy.chemistry import CompositionToAtomicFractions` **fails**.
- `from glasspy.chemistry import PhysicoChemicalFeaturizer` **fails**.
- Use `to_element_array` and `physchem_featurizer` from
  `glasspy.chemistry` / `glasspy.chemistry.featurizer`.
- `inspect.signature(all_features)` fails: `all_features` is not a function.

### mendeleev / `physchem_featurizer` invalid properties

Many names in `all_features` are missing for elements that appear in the
29-oxide silicate table. `physchem_featurizer` then raises
`ValueError: Invalid features: {...}`.

A property that works on a 3-oxide toy row can still fail on the full
silicate set. Probe validity on a **small slice of the real oxide columns**,
collect the invalid set, and pass only valid `(property, agg)` pairs for
**both** `weighted_features` and `absolute_features`.

Do not “fix” this by shrinking to a handful of properties before VIF..

### VIF cost

Iterative VIF with per-feature OLS (`statsmodels.variance_inflation_factor`)
on ~650 columns × ~10k rows is extremely slow and looks hung if stdout is
buffered. Vectorize (e.g. VIF from the inverse of the correlation matrix),
print with `flush=True`, and ask before using all CPUs.
