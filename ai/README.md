# AI, labeling, and journaling tools

This directory keeps offline data-prep and model-training utilities separate from the intraday scripts in `scripts/`. Day-to-day users can ignore everything here and focus on the bots under `scripts/`.

## Utilities
- `build_feature_universe.py` – Assemble the daily feature universe from AVWAP scans and other sources.
- `prepare_label_file.py` – Generate a label-ready CSV for manual review based on the latest universe data.
- `append_labels_to_master.py` – Append manually labeled setups back into the master dataset.
- `merge_trade_outcomes.py` – Merge realized trade outcomes into the master setups dataset.
- `build_ai_snapshot.py` – Consolidate AVWAP signals and intraday bounce logs into a single daily snapshot.
- `train_trade_quality_model.py` – Train a trade quality classification model on labeled trades.

Run any of these tools directly, for example:

```bash
python ai/build_feature_universe.py
```
