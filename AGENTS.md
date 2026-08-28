# Agents instructions

Only reports on the test data should be sent to the hub. Not reports on train data. If you run tests, it should also always run on a carefully crafted subset of data so that tests are fast. Also please ask if I want to limit the computer resource that you take when running heavy stuff, (like throttle to 2 CPU max).

# Train / test protocol

- **Fit and cross-validate only on `data/train.csv`.**
- **Pass `data/test.csv` manually as the test set** into skore evaluation (do
  not use a random holdout sliced from train as the report’s test set).
- **Skore Hub must use that same `test.csv` as the test set** for the pushed
  report (train = `train.csv`, test = `test.csv`).
- **Always push a fitted estimator alongside every CrossValidationReport**
  submitted to Hub (same train/test split).

# G_SKORE_MODE and project/workspace

G_SKORE_MODE=hub

When pushing an estimator or a CV report to Skore Hub, use:

- **workspace:** `saint-gobain-testing`
- **project:** `saint-gobain`

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