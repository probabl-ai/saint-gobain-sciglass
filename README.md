# Saint-Gobain SciGlass

A workspace for predicting physical properties of silicate glasses with
[GlassPy](https://github.com/drcassar/glasspy) / SciGlass, sklearn, and the
skore agent. Reports are pushed to a shared Skore Hub and ranked against other
participants.

## System requirements

Install skore: `pip install skore-cli`.

Then, to start skore-agent on windows powershell:
- in cli: `.\skore-copilot.ps1 cli`
- in desktop: `.\skore-copilot.ps1 desktop` and follow the instructions.
- in vscode: `.\skore-copilot.ps1 vscode` and follow the instructions.

To start skore-agent on unix terminal:
- in cli: `./skore-copilot cli`
- in desktop: `./skore-copilot desktop` and follow the instructions.
- in vscode: `./skore-copilot vscode` and follow the instructions.

Join the workspace on skore hub: [sciglass](https://saint-gobain.api.skore.probabl.ai/identity/invitations/65c3b3c7-6289-4fcb-8398-0c5d01d6841f?success_uri=https://saint-gobain.skore.probabl.ai/login/success)

## Goals

1. Predict `Tliquidus`  from **oxide mole fractions** (not wt%) for glasses with **SiO2 > 60 mol%**; use project `Tliquidus` on skore hub.
2. Predict `T4` from **oxide mole fractions** (not wt%) for glasses with **SiO2 > 60 mol%**; use project `T4` on skore hub.

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