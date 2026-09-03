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