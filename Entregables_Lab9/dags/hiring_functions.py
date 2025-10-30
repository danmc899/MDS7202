"""
Funciones auxiliares para el pipeline de Airflow.
Este contiene funciones reutilizables.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os

from airflow.models import Variable

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def create_folders(base_dir: str | None = None, experiment_type: str = "linear", **kwargs):
    """Crea carpetas raw, splits, models para la fecha de ejecución"""
    ds = kwargs.get("ds")
    if not ds:
        ds = datetime.now().strftime("%Y-%m-%d")
    
    if base_dir is None:
        base_dir = Variable.get("BASE_DATA_DIR", "/opt/airflow/data")

    run_dir = Path(base_dir) / experiment_type / ds
    subdirs = ["raw", "splits", "models"]
    
    for subdir in subdirs:
        dir_path = run_dir / subdir
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Carpetas creadas en: {run_dir}")
    
    ti = kwargs.get("ti")
    if ti:
        ti.xcom_push(key="run_dir", value=str(run_dir))


def split_data(
    base_dir: str | None = None,
    experiment_type: str = "linear",
    target_col: str = "HiringDecision",
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs):
    """Aplica hold-out 80/20 estratificado"""
    
    ds = kwargs.get("ds")
    if not ds:
        ds = datetime.now().strftime("%Y-%m-%d")
    
    if base_dir is None:
        base_dir = Variable.get("BASE_DATA_DIR", "/opt/airflow/data")
    
    run_dir = Path(base_dir) / experiment_type / ds
    raw_path = run_dir / "raw"
    splits_path = run_dir / "splits"
    splits_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = raw_path / "data_1.csv"
    df = pd.read_csv(csv_path)
    
    print(f"📊 Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    train_out = splits_path / "train.csv"
    test_out = splits_path / "test.csv"
    
    pd.concat([X_train, y_train], axis=1).to_csv(train_out, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_out, index=False)
    
    print(f"Train: {len(X_train)} muestras")
    print(f"Test: {len(X_test)} muestras")
    
    ti = kwargs.get("ti")
    if ti:
        ti.xcom_push(key="train_path", value=str(train_out))
        ti.xcom_push(key="test_path", value=str(test_out))


def preprocess_and_train(
    base_dir: str | None = None,
    experiment_type: str = "linear",
    target_col: str = "HiringDecision",
    positive_label=1,
    rf_params: dict | None = None,
    **kwargs,
):
    """Preprocesa y entrena RandomForest, guarda pipeline en models/"""
    
    ds = kwargs.get("ds") or datetime.now().strftime("%Y-%m-%d")

    if base_dir is None:
        base_dir = Variable.get("BASE_DATA_DIR", default_var="/opt/airflow/data")

    run_dir = Path(base_dir) / experiment_type / ds
    splits_dir = run_dir / "splits"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    test_path  = splits_dir / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Faltan splits:\n- {train_path}\n- {test_path}")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    y_test  = test_df[target_col]
    X_test  = test_df.drop(columns=[target_col])

    cat_ordinal = [c for c in ["PreviousCompanies", "EducationLevel"] if c in X_train.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             cat_ordinal),
        ],
        remainder="passthrough"
    )

    rf_params = rf_params or {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1
    }
    clf = RandomForestClassifier(**rf_params)

    pipe = Pipeline([
        ("preprocesador", preprocessor),
        ("rf", clf)
    ])

    print("🔧 Entrenando RandomForest...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="binary", pos_label=positive_label)
    
    print(f"[{ds}] Accuracy (test): {acc:.4f}")
    print(f"[{ds}] F1-Score (clase positiva): {f1:.4f}")

    model_path = models_dir / "model_rf_pipeline.joblib"
    joblib.dump(pipe, model_path)
    print(f"Modelo guardado en: {model_path}")

    ti = kwargs.get("ti")
    if ti:
        ti.xcom_push(key="model_path", value=str(model_path))
        ti.xcom_push(key="metrics", value={"accuracy": float(acc), "f1": float(f1)})