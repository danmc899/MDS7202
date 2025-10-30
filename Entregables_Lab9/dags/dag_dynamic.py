from __future__ import annotations
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Importar funciones del archivo de funciones dinámicas
from hiring_dynamic_functions import (
    create_folders,
    load_and_merge,
    split_data,
    train_model,
    evaluate_models
)

# Configuración
DEFAULT_BASE_DIR = Variable.get("BASE_DATA_DIR", default_var="/opt/airflow/data")
DATA_1_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
DATA_2_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"


# Función para decidir qué datos descargar según la fecha
def decide_download_branch(**kwargs):
    """
    Decide qué rama tomar según la fecha de ejecución:
    - Antes del 1 de noviembre 2024: solo data_1
    - Desde el 1 de noviembre 2024: data_1 y data_2
    """
    ds = kwargs.get("ds")
    execution_date = datetime.strptime(ds, "%Y-%m-%d")
    cutoff_date = datetime(2024, 11, 1)
    
    if execution_date < cutoff_date:
        print(f"Fecha {ds} < 2024-11-01: descargando solo data_1")
        return "download_data_1_only"
    else:
        print(f"Fecha {ds} >= 2024-11-01: descargando data_1 y data_2")
        return ["download_data_1", "download_data_2"]

# Definición del DAG
with DAG(
    dag_id="hiring_dynamic",
    start_date=datetime(2024, 10, 1),
    schedule="0 15 5 * *",  # Día 5 de cada mes a las 15:00 UTC
    catchup=True,  # Habilitar backfill
    tags=["hiring", "dynamic", "parallel"],
) as dag:

    # Marcador de inicio
    start = EmptyOperator(task_id="start_pipeline")

    # Crear carpetas
    make_dirs = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"base_dir": DEFAULT_BASE_DIR, "experiment_type": "dynamic"},
    )

    # Branch: decidir qué datos descargar
    branch_download = BranchPythonOperator(
        task_id="decide_download",
        python_callable=decide_download_branch,
    )

    # a. Descargar solo data_1
    download_data_1_only = BashOperator(
    task_id="download_data_1_only",
    bash_command="""
        RAW_DIR="{{ params.base_dir }}/dynamic/{{ ds }}/raw"
        mkdir -p "$RAW_DIR"
        curl -L -f -o "$RAW_DIR/data_1.csv" {{ params.url }}
        echo '✅ Archivo data_1.csv descargado exitosamente'
        ls -lh "$RAW_DIR/data_1.csv"
    """,
    params={
        "base_dir": DEFAULT_BASE_DIR,
        "url": DATA_1_URL,
    },
    )

    # b. Descargar data_1 y data_2
    download_data_1 = BashOperator(
    task_id="download_data_1",
    bash_command="""
        RAW_DIR="{{ params.base_dir }}/dynamic/{{ ds }}/raw"
        mkdir -p "$RAW_DIR"
        curl -L -f -o "$RAW_DIR/data_1.csv" {{ params.url }}
        echo '✅ Archivo data_1.csv descargado exitosamente'
        ls -lh "$RAW_DIR/data_1.csv"
    """,
    params={
        "base_dir": DEFAULT_BASE_DIR,
        "url": DATA_1_URL,
    },
    )

    download_data_2 = BashOperator(
    task_id="download_data_2",
    bash_command="""
        RAW_DIR="{{ params.base_dir }}/dynamic/{{ ds }}/raw"
        mkdir -p "$RAW_DIR"
        curl -L -f -o "$RAW_DIR/data_2.csv" {{ params.url }}
        echo '✅ Archivo data_2.csv descargado exitosamente'
        ls -lh "$RAW_DIR/data_2.csv"
    """,
    params={
        "base_dir": DEFAULT_BASE_DIR,
        "url": DATA_2_URL,
    },
    )

    # Tarea dummy para unir las ramas
    download_complete = EmptyOperator(
        task_id="download_complete",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Concatenar datasets disponibles
    merge_data = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_and_merge,
        op_kwargs={"base_dir": DEFAULT_BASE_DIR, "experiment_type": "dynamic"},
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Split train/test
    split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={
            "base_dir": DEFAULT_BASE_DIR,
            "experiment_type": "dynamic",
            "target_col": "HiringDecision",
            "test_size": 0.2,
            "random_state": 42,
        },
    )

    # Entrenamientos en paralelo
    # Modelo 1: Random Forest
    train_rf = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_model,
        op_kwargs={
            "model": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            "model_name": "random_forest",
            "base_dir": DEFAULT_BASE_DIR,
            "experiment_type": "dynamic",
            "target_col": "HiringDecision",
        },
    )

    # Modelo 2: Gradient Boosting
    train_gb = PythonOperator(
        task_id="train_gradient_boosting",
        python_callable=train_model,
        op_kwargs={
            "model": GradientBoostingClassifier(n_estimators=200, random_state=42),
            "model_name": "gradient_boosting",
            "base_dir": DEFAULT_BASE_DIR,
            "experiment_type": "dynamic",
            "target_col": "HiringDecision",
        },
    )

    # Modelo 3: Logistic Regression
    train_lr = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=train_model,
        op_kwargs={
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "model_name": "logistic_regression",
            "base_dir": DEFAULT_BASE_DIR,
            "experiment_type": "dynamic",
            "target_col": "HiringDecision",
        },
    )

    # Evaluación y selección del mejor modelo
    evaluate = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        op_kwargs={
            "base_dir": DEFAULT_BASE_DIR,
            "experiment_type": "dynamic",
            "target_col": "HiringDecision",
        },
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Solo si todos los modelos se entrenaron
    )

    # Marcador de fin
    end = EmptyOperator(task_id="end_pipeline")

    # Definir dependencias
    start >> make_dirs >> branch_download

    # Rama 1: Solo data_1
    branch_download >> download_data_1_only >> download_complete

    # Rama 2: data_1 y data_2
    branch_download >> download_data_1 >> download_complete
    branch_download >> download_data_2 >> download_complete

    # Flujo principal
    download_complete >> merge_data >> split

    # Entrenamientos en paralelo
    split >> [train_rf, train_gb, train_lr]

    # Evaluación después de todos los entrenamientos
    [train_rf, train_gb, train_lr] >> evaluate >> end