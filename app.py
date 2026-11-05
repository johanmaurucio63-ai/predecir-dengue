"""API web de ComplicaSense."""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from data_pipeline import BASE_FEATURES, prepare_batch_frame, prepare_user_record


ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "models.joblib"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

try:
    BUNDLE = joblib.load(ARTIFACT_PATH)
    LOAD_ERROR = None
except Exception as error:  # pragma: no cover - solo aparece antes del entrenamiento
    BUNDLE = None
    LOAD_ERROR = str(error)


def risk_level(probability: float) -> tuple[str, str]:
    if probability < 0.35:
        return "bajo", "Vigilancia estándar"
    if probability < 0.65:
        return "intermedio", "Revisar señales"
    return "alto", "Requiere valoración"


def prediction_payload(model, features):
    probability = float(model.predict_proba(features)[0][1])
    level, message = risk_level(probability)
    return {
        "probability": round(probability, 4),
        "percentage": round(probability * 100, 1),
        "class": "complicated" if probability >= 0.5 else "uncomplicated",
        "label": "Riesgo elevado" if probability >= 0.5 else "Riesgo controlado",
        "level": level,
        "message": message,
    }


def batch_metrics(actual, predicted, probabilities):
    """Calcula métricas solo cuando el archivo trae la columna objetivo."""
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    result = {
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "precision": round(float(precision_score(actual, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(actual, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(actual, predicted, zero_division=0)), 4),
        "confusion_matrix": matrix.tolist(),
    }
    result["roc_auc"] = round(float(roc_auc_score(actual, probabilities)), 4) if len(set(actual)) > 1 else None
    return result


def read_uploaded_table(upload):
    filename = (upload.filename or "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(upload)
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)
    raise ValueError("El archivo debe estar en formato CSV o Excel (.xlsx)")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok" if BUNDLE else "error", "models_loaded": bool(BUNDLE)})


@app.get("/api/meta")
def meta():
    if not BUNDLE:
        return jsonify({"error": "Los modelos no están disponibles", "detail": LOAD_ERROR}), 503
    return jsonify(
        {
            "project": "ComplicaSense",
            "features": BUNDLE["metadata"],
            "metrics": BUNDLE["metrics"],
            "dataset": BUNDLE["dataset"],
        }
    )


@app.get("/api/batch-template")
def batch_template():
    """Entrega una plantilla pequeña con las columnas aceptadas por el lote."""
    columns = BASE_FEATURES + ["target"]
    example = {
        "age_over_6": 1, "gender_female": 0, "fever_days": 3,
        "headache": 0, "myalgia": 0, "abdominal_pain": 0, "rash": 0,
        "vomiting": 0, "breathlessness": 0, "bleeding": 0,
        "low_blood_pressure": 0, "organomegaly": 0, "hb": 12.5,
        "pcv": 38, "lowest_wbc": 4000, "highest_wbc": 6000,
        "platelet_count": 100000, "alt": 45, "creatinine": 0.5,
        "ferritin": np.nan, "iv_fluids": 0, "antibiotics": 0,
        "antipyretics": 0, "steroids": 0, "target": np.nan,
    }
    content = pd.DataFrame([example], columns=columns).to_csv(index=False)
    return send_file(io.BytesIO(content.encode("utf-8")), mimetype="text/csv", as_attachment=True, download_name="plantilla_complicasense.csv")


@app.post("/api/predict")
def predict():
    if not BUNDLE:
        return jsonify({"error": "Los modelos no están disponibles", "detail": LOAD_ERROR}), 503

    payload = request.get_json(silent=True) or {}
    record = payload.get("features", payload)
    if not isinstance(record, dict):
        return jsonify({"error": "El cuerpo debe contener un objeto features"}), 400

    try:
        features = prepare_user_record(record)
        result = {
            "logistic": prediction_payload(BUNDLE["logistic"], features),
            "neural_network": prediction_payload(BUNDLE["neural_network"], features),
        }
        consensus = (result["logistic"]["probability"] + result["neural_network"]["probability"]) / 2
        level, message = risk_level(consensus)
        return jsonify(
            {
                "models": result,
                "consensus": {
                    "probability": round(consensus, 4),
                    "percentage": round(consensus * 100, 1),
                    "level": level,
                    "message": message,
                },
            }
        )
    except (TypeError, ValueError, KeyError) as error:
        return jsonify({"error": "No fue posible procesar los datos", "detail": str(error)}), 400


@app.post("/api/predict-batch")
def predict_batch():
    """Predice todas las filas de un archivo CSV/XLSX y devuelve métricas opcionales."""
    if not BUNDLE:
        return jsonify({"error": "Los modelos no están disponibles", "detail": LOAD_ERROR}), 503

    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Selecciona un archivo CSV o Excel"}), 400

    try:
        source = read_uploaded_table(upload)
        if source.empty:
            raise ValueError("El archivo no contiene registros")
        if len(source) > 5000:
            raise ValueError("El lote no puede superar 5.000 registros")

        features, actual = prepare_batch_frame(source)
        logistic_probability = BUNDLE["logistic"].predict_proba(features)[:, 1]
        neural_probability = BUNDLE["neural_network"].predict_proba(features)[:, 1]
        logistic_prediction = (logistic_probability >= 0.5).astype(int)
        neural_prediction = (neural_probability >= 0.5).astype(int)
        consensus_probability = (logistic_probability + neural_probability) / 2
        has_target = bool(actual is not None and actual.notna().any())
        valid_target = actual.notna().to_numpy() if actual is not None else None
        rows = []
        for index, (log_prob, ann_prob, consensus) in enumerate(zip(logistic_probability, neural_probability, consensus_probability), start=1):
            row = {
                "registro": index,
                "logistic_percentage": round(float(log_prob * 100), 1),
                "logistic_class": "Complicado" if log_prob >= 0.5 else "No complicado",
                "neural_percentage": round(float(ann_prob * 100), 1),
                "neural_class": "Complicado" if ann_prob >= 0.5 else "No complicado",
                "consensus_percentage": round(float(consensus * 100), 1),
                "consensus_class": "Complicado" if consensus >= 0.5 else "No complicado",
            }
            if has_target and valid_target[index - 1]:
                row["actual_class"] = "Complicado" if actual.iloc[index - 1] == 1 else "No complicado"
            elif has_target:
                row["actual_class"] = "Sin dato"
            rows.append(row)

        metrics = None
        if has_target:
            valid = actual.notna()
            actual_values = actual[valid].astype(int)
            metrics = {
                "logistic": batch_metrics(actual_values, logistic_prediction[valid_target], logistic_probability[valid_target]),
                "neural_network": batch_metrics(actual_values, neural_prediction[valid_target], neural_probability[valid_target]),
            }
        return jsonify({"rows": rows, "row_count": len(rows), "has_target": has_target, "metrics": metrics})
    except (TypeError, ValueError, KeyError, pd.errors.ParserError) as error:
        return jsonify({"error": "No fue posible procesar el lote", "detail": str(error)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
