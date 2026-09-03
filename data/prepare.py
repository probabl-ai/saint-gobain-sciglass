import pandas as pd
from glasspy.data import SciGlass
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

print("Loading SciGlass data...")
source = SciGlass()
df = source.data

print(f"Loaded {len(df)} rows")
print(f"MultiIndex levels: {df.index.names}")
print(f"Level 0 name: {df.index.get_level_values(0).name}")

compounds = df["compounds"]
print(f"Compounds shape: {compounds.shape}")
print(f"Oxide columns: {list(compounds.columns[:10])}...")

print("\nFiltering to SiO2 > 60 mol%...")
sio2_mask = compounds["SiO2"] > 0.60
df_silicates = df[sio2_mask]
print(f"After SiO2 filter: {len(df_silicates)} rows")

print("\nFinding oxides present in >1% of silicates...")
compounds_sil = df_silicates["compounds"]
oxide_prevalence = (compounds_sil > 0).sum() / len(compounds_sil)
oxides_kept = oxide_prevalence[oxide_prevalence > 0.01].index.tolist()
print(f"Kept {len(oxides_kept)} oxides: {oxides_kept}")

X = compounds_sil[oxides_kept].copy()
print(f"\nComposition table shape: {X.shape}")
print(f"Composition sums: min={X.sum(axis=1).min():.4f}, max={X.sum(axis=1).max():.4f}, mean={X.sum(axis=1).mean():.4f}")

target_col = "Tg"
y = df_silicates[("property", target_col)].copy()
print(f"\nTarget ({target_col}) shape: {y.shape}")
print(f"Target NaN count before drop: {y.isna().sum()}")

valid_idx = ~y.isna()
X = X[valid_idx]
y = y[valid_idx]
print(f"After dropping NaN targets: X shape {X.shape}, y shape {y.shape}")

print("\nCreating leak-free split (GroupShuffleSplit by exact composition)...")
composition_groups = pd.util.hash_pandas_object(X, index=False).values
groups = pd.Series(composition_groups).rank(method="dense").astype(int).values - 1
print(f"Composition groups: {len(np.unique(groups))} unique groups")

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

print(f"Train set: {len(X_train)} rows")
print(f"Test set: {len(X_test)} rows")

train_df = pd.concat([X_train, y_train.rename(target_col)], axis=1)
test_df = pd.concat([X_test, y_test.rename(target_col)], axis=1)

train_path = "data/train.csv"
test_path = "data/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"\nWrote {train_path} ({len(train_df)} rows, {train_df.shape[1]} columns)")
print(f"Wrote {test_path} ({len(test_df)} rows, {test_df.shape[1]} columns)")
print(f"Train target mean: {train_df[target_col].mean():.2f} K")
print(f"Test target mean: {test_df[target_col].mean():.2f} K")
