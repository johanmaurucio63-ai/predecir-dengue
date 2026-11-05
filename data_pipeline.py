"""Preparación reproducible de los datos para ComplicaSense.

El archivo Excel usa códigos 1/2. En el modelo todos los indicadores binarios
se expresan como 0/1, donde 1 significa presencia de la característica.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "age_over_6",
    "gender_female",
    "fever_days",
    "headache",
    "myalgia",
    "abdominal_pain",
    "rash",
    "vomiting",
    "breathlessness",
    "bleeding",
    "low_blood_pressure",
    "organomegaly",
    "hb",
    "pcv",
    "lowest_wbc",
    "highest_wbc",
    "platelet_count",
    "alt",
    "creatinine",
    "ferritin",
    "iv_fluids",
    "antibiotics",
    "antipyretics",
    "steroids",
]

ENGINEERED_FEATURES = [
    "ferritin_missing",
    "platelet_wbc_ratio",
    "hb_pcv_ratio",
    "symptom_count",
    "treatment_count",
]

MODEL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

BINARY_FEATURES = [
    "headache",
    "myalgia",
    "abdominal_pain",
    "rash",
    "vomiting",
    "breathlessness",
    "bleeding",
    "low_blood_pressure",
    "organomegaly",
    "iv_fluids",
    "antibiotics",
    "antipyretics",
    "steroids",
]

SOURCE_ALIASES = {
    "groups dengue fever or complicated dengue": "target",
    "groups: dengue fever or complicated dengue": "target",
    "age": "age_code",
    "gender": "gender_code",
    "duration of fever": "fever_days",
    "headache": "headache",
    "myalgia": "myalgia",
    "abdominal pain": "abdominal_pain",
    "rash": "rash",
    "vomiting": "vomiting",
    "breathlessness": "breathlessness",
    "bleeding": "bleeding",
    "bp at admission": "low_blood_pressure",
    "organomegaly": "organomegaly",
    "hb": "hb",
    "pcv": "pcv",
    "lowest wbc": "lowest_wbc",
    "highest wbc": "highest_wbc",
    "platelet count": "platelet_count",
    "alt": "alt",
    "creatinine": "creatinine",
    "ferritin": "ferritin",
    "iv fluids": "iv_fluids",
    "antibiotics": "antibiotics",
    "antipyretics": "antipyretics",
    "steroids": "steroids",
}


def _clean_name(value: Any) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def canonicalize_source(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas del Excel a nombres estables para código y API."""
    rename = {}
    for column in df.columns:
        cleaned = _clean_name(column)
        if cleaned in SOURCE_ALIASES:
            rename[column] = SOURCE_ALIASES[cleaned]
    result = df.rename(columns=rename).copy()
    missing = {"target", *BASE_FEATURES[:0], "fever_days"} - set(result.columns)
    if "target" not in result.columns:
        raise ValueError("No se encontró la columna objetivo GROUPS en el Excel")
    return result


def _binary_code(series: pd.Series) -> pd.Series:
    """Convierte los códigos del dataset: 1=presente, 2=ausente."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map({1: 1, 2: 0}).fillna(numeric).astype(float)


def prepare_training_frame(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Limpia el dataset y crea las variables derivadas del proyecto."""
    df = canonicalize_source(source)

    # Eliminar columnas constantes de forma programática (ninguna en este dataset).
    constant_columns = [
        column
        for column in df.columns
        if column != "target" and df[column].nunique(dropna=False) <= 1
    ]
    df = df.drop(columns=constant_columns)

    df["target"] = pd.to_numeric(df["target"], errors="coerce").map({1: 0, 2: 1})
    df["age_over_6"] = pd.to_numeric(df.pop("age_code"), errors="coerce").map({1: 0, 2: 1})
    df["gender_female"] = pd.to_numeric(df.pop("gender_code"), errors="coerce").map({1: 0, 2: 1})

    for column in BINARY_FEATURES:
        if column in df:
            df[column] = _binary_code(df[column])

    # Las mediciones pueden contener celdas vacías; el imputador del pipeline
    # ajustará la mediana usando exclusivamente el conjunto de entrenamiento.
    for column in BASE_FEATURES:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    features = build_model_features(df)
    target = df["target"].astype(int)
    return features, target, constant_columns


def build_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Genera las columnas que consumen ambos modelos.

    Acepta tanto filas limpias del dataset como registros enviados por la API.
    """
    df = frame.copy()
    for column in BASE_FEATURES:
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["ferritin_missing"] = df["ferritin"].isna().astype(float)
    df["platelet_wbc_ratio"] = df["platelet_count"] / df["lowest_wbc"].replace(0, np.nan)
    df["hb_pcv_ratio"] = df["hb"] / df["pcv"].replace(0, np.nan)
    df["symptom_count"] = df[BINARY_FEATURES[:9]].sum(axis=1, skipna=False)
    df["treatment_count"] = df[BINARY_FEATURES[9:]].sum(axis=1, skipna=False)
    return df[MODEL_FEATURES]


def prepare_user_record(record: dict[str, Any]) -> pd.DataFrame:
    """Normaliza un registro de la interfaz y devuelve una fila de features."""
    frame = pd.DataFrame([record])
    return build_model_features(frame)


def prepare_batch_frame(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Prepara un CSV/XLSX para predicción masiva.

    Acepta tanto las columnas originales del Excel como las columnas limpias
    usadas por la interfaz. La columna objetivo es opcional: si está presente
    se utiliza para calcular métricas de evaluación del lote.
    """
    df = canonicalize_source(source)
    target = None

    if "target" in df:
        raw_target = pd.to_numeric(df.pop("target"), errors="coerce")
        # La plantilla usa 0/1; el Excel original usa 1/2.
        if set(raw_target.dropna().unique()).issubset({0, 1}):
            target = raw_target
        else:
            target = raw_target.map({1: 0, 2: 1})

    # El Excel original usa códigos; la plantilla web ya usa 0/1. Para ambos
    # casos el mapeo conserva 0 y convierte 2 en ausencia.
    if "age_code" in df:
        df["age_over_6"] = pd.to_numeric(df.pop("age_code"), errors="coerce").map({1: 0, 2: 1})
    if "gender_code" in df:
        df["gender_female"] = pd.to_numeric(df.pop("gender_code"), errors="coerce").map({1: 0, 2: 1})
    for column in BINARY_FEATURES:
        if column in df:
            numeric = pd.to_numeric(df[column], errors="coerce")
            df[column] = numeric.map({1: 1, 2: 0}).fillna(numeric)

    return build_model_features(df), target
