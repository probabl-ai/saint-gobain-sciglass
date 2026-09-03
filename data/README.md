# Dataset

Frozen oxide compositions and targets for silicate glass property prediction.

## Preparation (one-time)

```bash
pixi run python data/prepare.py
```

Data is filtered to:
- SiO2 > 60 mol% (silicate glasses only)
- Oxides present in >1% of silicates (29 oxides)
- Rows with non-NaN Tg values
- Grouped by exact oxide composition, split 80/20 (train/test)

## Reference

- Source: GlassPy / SciGlass
- Target: Glass transition temperature (`Tg`)
- Alternative targets available: `Tliquidus`, `T3`, `T4` (swap in `prepare.py` line 35)
