"""Entrenamiento de los modelos de ComplicaSense.

Ejecutar desde la raíz del proyecto:
    python train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import MODEL_FEATURES, prepare_training_frame


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "MASTER_CHART_F1000Research.xlsx"
ARTIFACTS = ROOT / "artifacts"


def make_pipeline(classifier):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def metrics_for(model, X_test, y_test):
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "precision": round(float(precision_score(y_test, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, prediction, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, prediction, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "confusion_matrix": matrix.tolist(),
    }


def metadata(features: pd.DataFrame):
    """Metadatos consumidos por el frontend para mostrar rangos sensatos."""
    labels = {
        "hb": ("Hemoglobina", "g/dL"),
        "pcv": ("Hematocrito", "%"),
        "lowest_wbc": ("WBC mínimo", "células/µL"),
        "highest_wbc": ("WBC máximo", "células/µL"),
        "platelet_count": ("Plaquetas", "células/µL"),
        "alt": ("ALT", "U/L"),
        "creatinine": ("Creatinina", "mg/dL"),
        "ferritin": ("Ferritina", "ng/mL"),
    }
    result = {}
    for column, (label, unit) in labels.items():
        series = features[column].dropna()
        result[column] = {
            "label": label,
            "unit": unit,
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }
    return result


def main():
    source = pd.read_excel(DATASET)
    features, target, constant_columns = prepare_training_frame(source)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    logistic = make_pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    )
    neural = make_pipeline(
        MLPClassifier(
            hidden_layer_sizes=(24, 12),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.003,
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=35,
            random_state=42,
        )
    )

    logistic.fit(X_train, y_train)
    neural.fit(X_train, y_train)

    model_metrics = {
        "logistic": metrics_for(logistic, X_test, y_test),
        "neural_network": metrics_for(neural, X_test, y_test),
    }

    ARTIFACTS.mkdir(exist_ok=True)
    bundle = {
        "logistic": logistic,
        "neural_network": neural,
        "feature_names": MODEL_FEATURES,
        "metrics": model_metrics,
        "metadata": metadata(features),
        "dataset": {
            "rows": int(len(source)),
            "raw_columns": int(source.shape[1]),
            "target_distribution": {
                "uncomplicated": int((target == 0).sum()),
                "complicated": int((target == 1).sum()),
            },
            "constant_columns_removed": constant_columns,
            "missing_values": {
                column: int(source[column].isna().sum())
                for column in source.columns
                if source[column].isna().any()
            },
        },
    }
    joblib.dump(bundle, ARTIFACTS / "models.joblib")

    # Archivo legible para la sustentación y para revisar el entrenamiento.
    report = {
        "project": "ComplicaSense",
        "target": "Dengue complicado (1) vs. no complicado (0)",
        "features": MODEL_FEATURES,
        "metrics": model_metrics,
        "dataset": bundle["dataset"],
    }
    (ARTIFACTS / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Modelos entrenados y guardados en artifacts/models.joblib")
    print(json.dumps(model_metrics, indent=2))


if __name__ == "__main__":
    main()
