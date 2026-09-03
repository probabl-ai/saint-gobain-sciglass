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
