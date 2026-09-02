# Saint-Gobain SciGlass

A workspace for predicting physical properties of silicate glasses with
[GlassPy](https://github.com/drcassar/glasspy) / SciGlass, sklearn, and the
skore agent. Reports are pushed to a shared Skore Hub and ranked against other
participants.

**Goal:** predict `Tg`, `Tliquidus`, `T3`, and `T4` from **oxide mole
fractions** (not wt%) for glasses with **SiO2 > 60 mol%**.

## What you train on (frozen tables)

Data preparation is **not** part of modelling. It is done **once**:

1. Load SciGlass via GlassPy.
2. Keep glasses with **SiO2 > 60 mol%**.
3. Keep oxide columns present in >1% of those silicates (~29 oxides).
4. Drop rows where the chosen target is NaN.
5. Split **raw oxide tables** with `GroupShuffleSplit` grouped by exact
   composition (`test_size=0.2`, `random_state=42`) so the same composition
   never appears in both train and test.

Write `data/train.csv` and `data/test.csv` (oxide features + target column).
If those files already exist and are non-empty, **do not regenerate them**.

**Do not** run GlassNet-style physiochemical featurization, VIF pruning, or
MinMax scaling as part of this first write. The baseline should train on the
filtered oxide table. GlassNet is an optional later idea (see
[`AGENTS.md`](AGENTS.md)), not the default input. If you add it later, write
`data/train_glassnet_features.csv` and `data/test_glassnet_features.csv`;
never overwrite the oxide baseline files.

## Prerequisites

You need a terminal with uv installed. Please use WSL (Windows Subsystem for
Linux).

## Setup

### Dependencies

All dependencies are installed in the virtual environment. To reinstall or
update:

```bash
pip install -r requirements.txt
```

### GlassPy

GlassPy is installed and ready to use. See the
[GlassPy documentation](https://github.com/drcassar/glasspy) for usage
examples.

### Skore Hub API key

Create a local `.env` (not committed) with:

```
SKORE_HUB_API_KEY=<your key>
```

Get the key from Skore Hub → workspace settings → API keys (workspace
`saint-gobain-testing`). Modelling and Hub pushes must not start until this
is set. Do not use interactive / browser login.

Hub targets:

- CV reports: workspace `saint-gobain-testing`, project `saint-gobain-cv`
- Fitted-model test evaluation: workspace `saint-gobain-testing`, project
  `saint-gobain-leaderboard`

Only **test-set** reports go to the Hub, never train-only reports. Fit and
cross-validate on `data/train.csv`; pass `data/test.csv` as the test set.

## PyTorch

PyTorch is installed for CPU. If you need GPU support, reinstall PyTorch with:

```bash
# For CUDA (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with your CUDA version (e.g., `cu121` for CUDA 12.1).
