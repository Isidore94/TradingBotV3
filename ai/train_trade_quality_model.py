"""Train a trade quality classification model."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

NUMERIC_FEATURES = [
    "avwap_price",
    "band_price",
    "stdev",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ema_8",
    "ema_15",
    "ema_21",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",
    "has_intraday_bounce_today",
    "prev_lower_bounce_last_5d",
    "prev_lower_cross_last_5d",
    "conviction_score",
]

CATEGORICAL_FEATURES = ["side", "anchor_type", "signal_type"]


def _coerce_binary(series: pd.Series) -> pd.Series:
    """Convert various truthy/falsy representations to 1.0/0.0."""
    truthy = {"1", "true", "yes", "y", "t"}
    falsy = {"0", "false", "no", "n", "f"}

    def map_value(value: object) -> float:
        if pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return 1.0 if float(value) > 0 else 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        text = str(value).strip().lower()
        if text in truthy:
            return 1.0
        if text in falsy:
            return 0.0
        try:
            return 1.0 if float(text) > 0 else 0.0
        except ValueError:
            return 0.0

    return series.apply(map_value).astype(float)


def load_data() -> pd.DataFrame:
    data_path = DATA_DIR / "master_setups.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows from {data_path}")
    return df


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    filtered = df.copy()
    filtered = filtered.loc[filtered["taken"].astype(str).str.strip() == "1"].copy()
    filtered = filtered.loc[filtered["realized_rr"].notna()].copy()
    print(f"Filtered down to {len(filtered):,} candidate rows after applying criteria.")

    if filtered.empty:
        raise ValueError("No rows available after filtering. Cannot train model.")

    filtered["trade_date"] = pd.to_datetime(filtered["trade_date"], errors="coerce")
    filtered = filtered.sort_values("trade_date", kind="mergesort")

    y = (pd.to_numeric(filtered["realized_rr"], errors="coerce") > 0.0).astype(int)

    missing_numeric = [col for col in NUMERIC_FEATURES if col not in filtered.columns]
    missing_categorical = [col for col in CATEGORICAL_FEATURES if col not in filtered.columns]
    if missing_numeric or missing_categorical:
        missing_cols = ", ".join(missing_numeric + missing_categorical)
        raise KeyError(f"Missing required feature columns: {missing_cols}")

    X = filtered[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    bool_like_cols = [
        "has_intraday_bounce_today",
        "prev_lower_bounce_last_5d",
        "prev_lower_cross_last_5d",
    ]
    for col in bool_like_cols:
        X[col] = _coerce_binary(X[col])

    for col in NUMERIC_FEATURES:
        if col not in bool_like_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(0.0).astype(float)

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("unknown").astype(str)

    return X, y


def time_based_split(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n_samples = len(X)
    if n_samples < 2:
        raise ValueError("Need at least two samples for a train/validation split.")

    split_idx = int(n_samples * 0.8)
    split_idx = max(1, min(split_idx, n_samples - 1))

    X_train = X.iloc[:split_idx]
    X_valid = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_valid = y.iloc[split_idx:]

    print(
        "Time-based split: {train:,} training rows, {valid:,} validation rows.".format(
            train=len(X_train), valid=len(X_valid)
        )
    )

    return X_train, X_valid, y_train, y_valid


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )

    return clf


def evaluate_model(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> tuple[float, float | None]:
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    roc_auc = None
    try:
        y_proba = model.predict_proba(X_valid)[:, 1]
        roc_auc = roc_auc_score(y_valid, y_proba)
    except ValueError:
        print("Warning: Unable to compute ROC AUC (only one class present in validation set).")

    return accuracy, roc_auc


def save_artifacts(model: Pipeline, metadata: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "trade_quality_model.pkl"
    meta_path = MODELS_DIR / "trade_quality_model_meta.json"

    joblib.dump(model, model_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model pipeline to {model_path}")
    print(f"Saved metadata to {meta_path}")


def main() -> None:
    df = load_data()
    X, y = prepare_dataset(df)
    X_train, X_valid, y_train, y_valid = time_based_split(X, y)

    model = build_pipeline()
    model.fit(X_train, y_train)

    accuracy, roc_auc = evaluate_model(model, X_valid, y_valid)
    print(f"Validation accuracy: {accuracy:.4f}")
    if roc_auc is not None:
        print(f"Validation ROC AUC: {roc_auc:.4f}")

    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_samples_train": int(len(X_train)),
        "n_samples_valid": int(len(X_valid)),
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
        "target_definition": "win = realized_rr > 0",
    }

    save_artifacts(model, metadata)


if __name__ == "__main__":
    main()
