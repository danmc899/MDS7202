from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os

from airflow.models import Variable
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from glob import glob


def create_folders(base_dir: str | None = None, experiment_type: str = "dynamic", **kwargs):
    """
    Crea una carpeta con nombre igual a la fecha de ejecución (ds) y,
    dentro de ella, crea las carpetas raw, preprocessed, splits y models.
    
    Params
    ------
    base_dir: str | None
        Directorio base donde crear la estructura. Si es None, intenta
        leer Variable ("BASE_DATA_DIR"). Si no, usa "/opt/airflow/data".
    experiment_type: str
        Tipo de experimento: "linear" o "dynamic"
    """
    
    # Obtener la fecha de ejecución
    ds = kwargs.get("ds")
    if not ds:
        ds = datetime.now().strftime("%Y-%m-%d")
    
    # Resolver el directorio base
    if base_dir is None:
        base_dir = Variable.get("BASE_DATA_DIR", "/opt/airflow/data")

    # Construir paths con experiment_type
    run_dir = Path(base_dir) / experiment_type / ds
    subdirs = ["raw", "preprocessed", "splits", "models"]
    
    # Crear carpetas
    for subdir in subdirs:
        dir_path = run_dir / subdir
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Carpetas creadas en: {run_dir}")
    
    # Publicar por XCom
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="run_dir", value=str(run_dir))


def load_and_merge(base_dir: str | None = None, experiment_type: str = "dynamic", **kwargs):
    """
    Lee data_1.csv y data_2.csv (si existe) desde la carpeta raw,
    los concatena y guarda el resultado en preprocessed/merged_data.csv.
    
    Params
    ------
    base_dir: str | None
        Directorio base que contiene la carpeta de la corrida (YYYY-MM-DD).
    experiment_type: str
        Tipo de experimento: "linear" o "dynamic"
    """
    
    # Obtener fecha de ejecución
    ds = kwargs.get("ds")
    if not ds:
        ds = datetime.now().strftime("%Y-%m-%d")
    
    # Resolver el directorio base
    if base_dir is None:
        base_dir = Variable.get("BASE_DATA_DIR", "/opt/airflow/data")
    
    run_dir = Path(base_dir) / experiment_type / ds
    raw_path = run_dir / "raw"
    preprocessed_path = run_dir / "preprocessed"
    preprocessed_path.mkdir(parents=True, exist_ok=True)
    
    # Buscar archivos disponibles
    data_1_path = raw_path / "data_1.csv"
    data_2_path = raw_path / "data_2.csv"
    
    dfs = []
    
    if data_1_path.exists():
        print(f"Leyendo {data_1_path}")
        dfs.append(pd.read_csv(data_1_path))
    
    if data_2_path.exists():
        print(f"Leyendo {data_2_path}")
        dfs.append(pd.read_csv(data_2_path))
    
    if not dfs:
        raise FileNotFoundError(f"No se encontraron archivos en {raw_path}")
    
    # Concatenar datasets
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Guardar en preprocessed
    output_path = preprocessed_path / "merged_data.csv"
    merged_df.to_csv(output_path, index=False)
    
    print(f"Datos concatenados guardados en: {output_path}")
    print(f"Forma del dataset: {merged_df.shape}")
    
    # Publicar por XCom
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="merged_path", value=str(output_path))


def split_data(
    base_dir: str | None = None,
    experiment_type: str = "dynamic",
    target_col: str = "HiringDecision",
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
):
    """
    Lee merged_data.csv desde preprocessed y aplica un hold-out 80/20,
    estratificado por la variable objetivo. Guarda train/test en splits/.
    
    Params
    ------
    base_dir : str | None
        Directorio base que contiene la carpeta de la corrida.
    experiment_type: str
        Tipo de experimento: "linear" o "dynamic"
    target_col : str
        Nombre de la variable objetivo para estratificar.
    test_size : float
        Proporción de test.
    random_state : int
        Semilla para replicabilidad.
    """
    
    # Obtener fecha de ejecución
    ds = kwargs.get("ds")
    if not ds:
        ds = datetime.now().strftime("%Y-%m-%d")
    
    # Resolver el directorio base
    if base_dir is None:
        base_dir = Variable.get("BASE_DATA_DIR", "/opt/airflow/data")
    
    run_dir = Path(base_dir) / experiment_type / ds
    preprocessed_path = run_dir / "preprocessed"
    splits_path = run_dir / "splits"
    splits_path.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos procesados
    merged_path = preprocessed_path / "merged_data.csv"
    if not merged_path.exists():
        raise FileNotFoundError(f"No se encontró {merged_path}")
    
    df = pd.read_csv(merged_path)
    
    # Separar características y objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Hold out
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # Guardar los datasets
    train_out = splits_path / "train.csv"
    test_out = splits_path / "test.csv"
    
    pd.concat([X_train, y_train], axis=1).to_csv(train_out, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_out, index=False)
    
    print(f"Train guardado en: {train_out} - Shape: {X_train.shape}")
    print(f"Test guardado en: {test_out} - Shape: {X_test.shape}")
    
    # Publicar por XCom
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="train_path", value=str(train_out))
        kwargs["ti"].xcom_push(key="test_path", value=str(test_out))


def train_model(
    model,
    model_name: str,
    base_dir: str | None = None,
    experiment_type: str = "dynamic",
    target_col: str = "HiringDecision",
    **kwargs
):
    """
    Entrena un modelo de clasificación con preprocesamiento y lo guarda.
    
    Params
    ------
    model : estimator
        Modelo de sklearn a entrenar.
    model_name : str
        Nombre identificador del modelo para guardar el archivo.
    base_dir : str | None
        Directorio base de la corrida.
    experiment_type: str
        Tipo de experimento: "linear" o "dynamic"
    target_col : str
        Nombre de la columna objetivo.
    """
    
    # Obtener fecha de ejecución
    ds = kwargs.get("ds") or datetime.now().strftime("%Y-%m-%d")
    
    # Resolver el directorio base
    if base_dir is None:
        base_dir = Variable.get("BASE_DATA_DIR", default_var="/opt/airflow/data")
    
    run_dir = Path(base_dir) / experiment_type / ds
    splits_dir = run_dir / "splits"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos de entrenamiento
    train_path = splits_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"No se encontró {train_path}")
    
    train_df = pd.read_csv(train_path)
    
    # Separar características y objetivo
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    
    # Preprocesamiento
    cat_ordinal = [c for c in ["PreviousCompanies", "EducationLevel"] 
                   if c in X_train.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             cat_ordinal),
        ],
        remainder="passthrough"
    )
    
    # Pipeline completo
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Entrenar
    print(f"Entrenando modelo: {model_name}")
    pipe.fit(X_train, y_train)
    
    # Guardar modelo
    model_path = models_dir / f"{model_name}.joblib"
    joblib.dump(pipe, model_path)
    
    print(f"Modelo {model_name} guardado en: {model_path}")
    
    # Publicar por XCom
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key=f"model_{model_name}", value=str(model_path))


def evaluate_models(
    base_dir: str | None = None,
    experiment_type: str = "dynamic",
    target_col: str = "HiringDecision",
    **kwargs
):
    """
    Evalúa todos los modelos guardados en la carpeta models/ usando el conjunto
    de prueba, selecciona el mejor por accuracy y lo guarda como best_model.joblib.
    
    Params
    ------
    base_dir : str | None
        Directorio base de la corrida.
    experiment_type: str
        Tipo de experimento: "linear" o "dynamic"
    target_col : str
        Nombre de la columna objetivo.
    """
    
    # Obtener fecha de ejecución
    ds = kwargs.get("ds") or datetime.now().strftime("%Y-%m-%d")
    
    # Resolver el directorio base
    if base_dir is None:
        base_dir = Variable.get("BASE_DATA_DIR", default_var="/opt/airflow/data")
    
    run_dir = Path(base_dir) / experiment_type / ds
    splits_dir = run_dir / "splits"
    models_dir = run_dir / "models"
    
    # Cargar datos de prueba
    test_path = splits_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"No se encontró {test_path}")
    
    test_df = pd.read_csv(test_path)
    y_test = test_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    
    # Buscar todos los modelos
    model_files = list(models_dir.glob("*.joblib"))
    # Excluir best_model si ya existe
    model_files = [f for f in model_files if f.stem != "best_model"]
    
    if not model_files:
        raise FileNotFoundError(f"No se encontraron modelos en {models_dir}")
    
    print(f"Evaluando {len(model_files)} modelos en experimento '{experiment_type}'...")
    
    results = []
    
    for model_path in model_files:
        model_name = model_path.stem
        pipe = joblib.load(model_path)
        
        # Predecir
        y_pred = pipe.predict(X_test)
        
        # Calcular accuracy
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            "model_name": model_name,
            "accuracy": acc,
            "model_path": model_path
        })
        
        print(f"  {model_name}: Accuracy = {acc:.4f}")
    
    # Seleccionar el mejor modelo
    best_result = max(results, key=lambda x: x["accuracy"])
    
    print(f"\n✅ Mejor modelo ({experiment_type}): {best_result['model_name']}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    
    # Guardar el mejor modelo como best_model.joblib
    best_model_path = models_dir / "best_model.joblib"
    best_pipe = joblib.load(best_result["model_path"])
    joblib.dump(best_pipe, best_model_path)
    
    print(f"   Guardado en: {best_model_path}")
    
    # Publicar por XCom
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="best_model_name", value=best_result["model_name"])
        kwargs["ti"].xcom_push(key="best_model_accuracy", value=best_result["accuracy"])
        kwargs["ti"].xcom_push(key="best_model_path", value=str(best_model_path))