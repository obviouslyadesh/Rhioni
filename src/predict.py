"""
src/predict.py
Reusable inference utilities shared between notebook and app.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


# ──────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "HOUR", "MINUTE", "HOUR_SIN", "HOUR_COS",
    "DAY_OF_WEEK", "LIGHT_CONDITION", "SPEED_ZONE",
    "IS_PEAK_HOUR", "IS_WEEKEND",
]

CATEGORICAL_FEATURES = [
    "ROAD_GEOMETRY_DESC",
    "HIGHWAY",
    "TIME_OF_DAY",
    "SPEED_RISK",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SEVERITY_LABELS = {
    1: "Fatal",
    2: "Serious Injury",
    3: "Other Injury",
    4: "Non-Injury",
}


# ──────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Works on a DataFrame of any size (1 row or millions).
    Input must have:  ACCIDENT_TIME (HH:MM:SS), DAY_OF_WEEK, LIGHT_CONDITION,
                      ROAD_GEOMETRY_DESC, SPEED_ZONE, HIGHWAY
    """
    df = df.copy()

    # Time parsing
    if "ACCIDENT_TIME" in df.columns:
        df["ACCIDENT_TIME"] = pd.to_datetime(
            df["ACCIDENT_TIME"], format="%H:%M:%S", errors="coerce"
        )
        df["HOUR"]   = df["ACCIDENT_TIME"].dt.hour.fillna(0).astype(int)
        df["MINUTE"] = df["ACCIDENT_TIME"].dt.minute.fillna(0).astype(int)
    elif "HOUR" not in df.columns:
        raise ValueError("DataFrame must contain either 'ACCIDENT_TIME' or 'HOUR'.")

    # Cyclical hour
    df["HOUR_SIN"] = np.sin(2 * np.pi * df["HOUR"] / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * df["HOUR"] / 24)

    # Time-of-day bucket
    bins   = [-1, 5, 11, 16, 20, 23]
    labels = ["NIGHT", "MORNING", "AFTERNOON", "EVENING", "LATE_NIGHT"]
    df["TIME_OF_DAY"] = pd.cut(df["HOUR"], bins=bins, labels=labels).astype(str)

    # Peak / weekend flags
    df["IS_PEAK_HOUR"] = df["HOUR"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["IS_WEEKEND"]   = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)

    # Speed risk bucket
    df["SPEED_RISK"] = pd.cut(
        df["SPEED_ZONE"],
        bins=[0, 50, 70, 90, 999],
        labels=["LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
    ).astype(str)

    return df


# ──────────────────────────────────────────────────────────────────
# SINGLE-ROW PREDICTION
# ──────────────────────────────────────────────────────────────────
def predict_single(
    hour: int,
    minute: int,
    day_of_week: int,
    light_condition: int,
    road_geometry: str,
    speed_zone: int,
    highway: str,
    preprocessor,
    model,
) -> Tuple[int, Dict[str, float]]:
    """
    Real-time single prediction.

    Parameters
    ----------
    hour, minute        : time of day
    day_of_week         : 1 (Mon) … 7 (Sun)
    light_condition     : integer code 1–9
    road_geometry       : string e.g. 'T intersection'
    speed_zone          : integer km/h
    highway             : string segment name
    preprocessor        : fitted sklearn ColumnTransformer
    model               : fitted LGBMClassifier

    Returns
    -------
    severity  : int 1–4
    probs     : dict {'Severity 1': 34.2, 'Severity 2': ...}
    """
    row = pd.DataFrame([{
        "HOUR": hour, "MINUTE": minute,
        "DAY_OF_WEEK": day_of_week,
        "LIGHT_CONDITION": light_condition,
        "SPEED_ZONE": speed_zone,
        "ROAD_GEOMETRY_DESC": road_geometry,
        "HIGHWAY": highway,
    }])
    row = engineer_features(row)

    X_proc = preprocessor.transform(row[ALL_FEATURES])
    pred   = model.predict(X_proc)[0]
    proba  = model.predict_proba(X_proc)[0]

    severity = int(pred) + 1
    probs    = {f"Severity {i+1}": round(float(p) * 100, 2) for i, p in enumerate(proba)}
    return severity, probs


# ──────────────────────────────────────────────────────────────────
# BATCH PREDICTION
# ──────────────────────────────────────────────────────────────────
def batch_predict(df: pd.DataFrame, preprocessor, model) -> pd.DataFrame:
    """
    Batch prediction.
    Input df must have the raw columns (ACCIDENT_TIME, DAY_OF_WEEK, …).
    Returns df with additional columns: PREDICTED_SEVERITY, PROB_SEV_1..4.
    """
    df = df.copy()
    df_feat = engineer_features(df)
    X_proc  = preprocessor.transform(df_feat[ALL_FEATURES])

    preds  = model.predict(X_proc) + 1
    probas = model.predict_proba(X_proc)

    df["PREDICTED_SEVERITY"] = preds
    df["PREDICTED_LABEL"]    = [SEVERITY_LABELS[s] for s in preds]
    for i in range(4):
        df[f"PROB_SEV_{i+1}"] = (probas[:, i] * 100).round(2)

    return df
