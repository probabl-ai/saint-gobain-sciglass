# Saint-Gobain SciGlass

A workspace for predicting physical properties of silicate glasses with
[GlassPy](https://github.com/drcassar/glasspy) / SciGlass, sklearn, and the
skore agent. Reports are pushed to a shared Skore Hub and ranked against other
participants.

## System requirements

Install skore: `pip install skore-cli`.
Then, to start skore-agent:
- in cli: `skore-copilot cli`
- in desktop: `skore-copilot desktop` and follow the instructions.
- in vscode: `skore-copilot` and follow the instructions.

You need a UNIX terminal with either `pip`, `pixi`, `uv` or `conda` installed. If you are on windows, please use WSL (Windows Subsystem for Linux). Permissions to run commands on Windows Powershell are messy, so it is recommended to use WSL.

## Goal

Predict `Tg`, `Tliquidus`, `T3`, and `T4` from **oxide mole fractions** (not wt%) for glasses with **SiO2 > 60 mol%**.

## Data

You are training the model on frozen tables.
Data preparation is **not** part of modelling. It is done **once** by running `data/prepare.py`, which does the following:
1. Loads SciGlass via GlassPy.
2. Keeps only glasses with **SiO2 > 60 mol%**.
3. Keeps only oxide columns present in >1% of those silicates (~29 oxides).
4. Drops rows where the chosen target is NaN.
5. Splits **raw oxide tables** with `GroupShuffleSplit` grouped by exact
   composition (`test_size=0.2`, `random_state=42`) so the same composition
   never appears in both train and test.
6. Writes `data/train.csv` and `data/test.csv` (oxide features + target column).