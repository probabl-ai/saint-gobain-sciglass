## GlassNet feature engineering (optional, after baseline)

How to run the helpers: [README.md](README.md).

Cassar 2023, arXiv:2303.15538: oxide → atomic mole fractions in **§2.1**
(steps 3–4); feature extraction, selection, and scaling in **§2.2**.
**Suggestion only** once a baseline on raw oxides exists. Do not fold this
into the one-time silicate prep. Do not re-fit variance/VIF/scalers inside
later modelling CV folds: if you run GlassNet, fit selection and scaling
**once on the frozen train split** (never on test, never on the full
matrix). Save results as `data/train_glassnet_features.csv` and
`data/test_glassnet_features.csv` — **do not overwrite** `data/train.csv` /
`data/test.csv`.

### Step 1 — oxide → atomic mole fractions (§2.1 steps 3–4)

Decompose each oxide (e.g. SiO2 → Si + 2 O; Na2O → 2 Na + O). Sum
elemental contributions; renormalize each row to sum to 1. Output a
fixed-length vector **C** over Z = 1–83 (H–Bi), **excluding Pm and the
noble gases** He, Ne, Ar, Kr, Xe → **77** elements. Absent elements:
`x_e = 0`. Glasses with a non-zero amount of any **excluded** element:
**drop the glass** (paper rule; do not add extra columns). Store the
element order on the transformer.

On this project’s frozen silicate tables those elements should already be
absent. If a row still has one, drop that row from both GlassNet files
rather than inventing a feature.

**GlassPy 0.6 has no `CompositionToAtomicFractions` class.** Use
`glasspy.chemistry.to_element_array` (also exported from
`glasspy.chemistry.featurizer`) with `rescale_to_sum=1`. See
[`TROUBLESHOOTING.md`](../../TROUBLESHOOTING.md).

### Step 2 — elemental property table S (§2.2, first step)

Build one numeric table **S** for every in-scope element (cache; not
per-row). The paper collected **55** properties with **mendeleev** and
**matminer**. The only collection filter: the property must be available
for **all** 77 elements (that is why Pm and noble gases are out of
scope). **Drop** any property with a missing value on any of the 77;
record what you kept.

Do **not** substitute a generic mendeleev dump (period, group, block,
heat capacity, thermal conductivity, evaporation heat, parsed electron
shells, …). Those are not the GlassNet table. Filled/unfilled s/p/d/f
counts are matminer Magpie **valence-orbital** descriptors (Ward et al.
2016, paper ref. [26]), not electronic-configuration parsing.

Table 1 of the paper lists the **25** of these 55 that later survived
selection. The other **30** are in the Supplementary Material
(“Physicochemical features considered, but not selected”). Collect
**all 55** as input to Step 3:

**Survivors (Table 1) — still collect all of these before selection**

| Symbol | Property | Source |
|---|---|---|
| rW | Van der Waals radius (UFF) | [56] |
| rR | Atomic radius (Rahm) | [57] |
| Vat | Atomic volume | mendeleev [30] |
| Eea | Electron affinity | [58], [59] |
| Eg | DFT bandgap, T = 0 K ground state | matminer [26] |
| Eat | DFT energy per atom, T = 0 K ground state | matminer [26] |
| mm | DFT magnetic moment, T = 0 K ground state | matminer [26] |
| FCClp | FCC lattice parameter from OQMD DFT volume | matminer [26] |
| Zeff | Effective nuclear charge | mendeleev [30] |
| χS | Electronegativity, Sanderson | [60], [61] |
| χTO | Electronegativity, Tardini–Oganov | [62] |
| Tb | Boiling point | [58] |
| ΔHm | Melting (fusion) enthalpy | matminer [26] |
| C6 | C6 coefficient | [63], [64] |
| Nv | Number of valence electrons | mendeleev [30] |
| Nox | Number of oxidation states | mendeleev [30] |
| Nu | Number of unfilled valence orbitals | matminer [26] |
| Nu,s / Nu,p / Nu,d / Nu,f | Unfilled s/p/d/f valence orbitals | matminer [26] |
| Nf,s / Nf,p / Nf,d / Nf,f | Filled s/p/d/f valence orbitals | matminer [26] |

**Considered but not selected (Supplementary) — still collect these**

- Atomic number; atomic weight; mass number of the most abundant
  isotope; maximum ionization energy; number of electrons / neutrons /
  protons (mendeleev [30])
- Atomic radius (Slater) [65]
- Covalent radius (Cordero) [66]; single-bond covalent radius (Pyykkö)
  [67]
- Density at 295 K; heat of formation; melting point; van der Waals
  radius (CRC) [58]
- Dipole polarizability [68]
- Electronegativities: Allred–Rochow [69], Cottrell–Sutton [70], Gordy
  [71], Ghosh [72], Martynov–Batsanov [73], Nagle [74]
- Energy to remove the first electron; number of valence electrons
  (matminer copy, distinct from Nv); BCC lattice parameter from OQMD
  DFT volume; DFT volume per atom, T = 0 K ground state [26]
- Glawe’s number [75]; Mendeleev’s number [76], [77]; Pettifor’s number
  [76]
- Van der Waals radius (Alvarez) [78]; van der Waals radius (Allinger)
  [79]

A tiny hand-picked mendeleev subset is **not** a faithful GlassNet table.

### Step 3 — weighted + absolute features (§2.2, second step)

For each of the 55 property columns `S_i`, aggregate over the **full**
77-vector **including zeros** (do not drop absent elements):

- Weighted: `w = f(C ⊙ S_i)` — composition-sensitive (paper Eq. 1)
- Absolute: `a = f(⌈C⌉ ⊙ S_i)` — presence (`x_e > 0`) only (paper Eq. 2)

Aggregators `f`: `{sum, min, max, mean, std}`.

Concatenate **[C | weighted | absolute]** → **627** features: 77 atomic
fractions + 275 weighted (55 × 5) + 275 absolute (55 × 5).
Redundant/constant features are expected; pruning is Step 4.

**GlassPy 0.6 has no `PhysicoChemicalFeaturizer` class.**
`glasspy.chemistry.physchem_featurizer` is a local convenience, not the
paper’s 55-property catalog. `all_features` is a **list of
`(property, aggregator)` tuples**, not a callable. Passing the full
`all_features` list will raise `ValueError: Invalid features` — see
[`TROUBLESHOOTING.md`](../../TROUBLESHOOTING.md). Do not “fix” that by shrinking to a handful of
properties before VIF.

### Step 4 — feature selection (fit on train only, once) (§2.2, third step)

1. Drop features with **standard deviation < 10⁻³**. In sklearn that is
   `VarianceThreshold(threshold=(1e-3) ** 2)` (the kwarg is **variance**,
   not std).
2. Iterative VIF: compute VIF for all remaining features; if every VIF
   is below 5, stop; otherwise drop the highest-VIF feature and repeat.
   Custom `VIFSelector`; **vectorize** (do not loop `statsmodels`
   `variance_inflation_factor` on hundreds of columns — that can run for
   tens of minutes with no useful output). Log runtime and before/after
   counts.

Paper outcome on their dataset (yours will differ): **627 → 98**
features (64 elemental mole fractions, 12 weighted, 22 absolute; 25 of
the 55 properties appear in the 98; the `mean` aggregator did not
survive).

### Step 5 — scaling (fit on train only, once) (§2.2, final paragraph)

`MinMaxScaler` on **selected features and targets** (paper scaled both
to [0, 1]). Fit on train, transform train and test. Min-max (not
`StandardScaler`) preserves exact zeros and is robust to tiny standard
deviations on sparse compositional features.

If you scale **targets**, fit a **separate** scaler on **train y only**
and persist it; inverse-transform predictions before reporting Kelvin
RMSE on the hub so metrics stay comparable.

Write GlassNet outputs as **new files**. Never overwrite `data/train.csv`
or `data/test.csv`:

- `data/train_glassnet_features.csv`
- `data/test_glassnet_features.csv`

Fit/CV on the GlassNet train file; Hub test reports must use the matching
GlassNet test file. Same composition split as the oxide tables.
