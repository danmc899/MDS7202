from __future__ import annotations
from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable

from hiring_functions import create_folders, split_data, preprocess_and_train

# Configuración
DEFAULT_BASE_DIR = Variable.get("BASE_DATA_DIR", default_var="/opt/airflow/data")
DATA_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"  # ✅ Agregado .csv

with DAG(
    dag_id="hiring_lineal",
    start_date=datetime(2024, 10, 1),
    schedule=None,
    catchup=False,
    tags=["hiring", "pipeline", "rf"],
) as dag:

    start = EmptyOperator(task_id="start_pipeline")

    make_dirs = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"base_dir": DEFAULT_BASE_DIR, "experiment_type": "linear"},
    )

    download_data = BashOperator(
    task_id="download_data",
    bash_command="""
        mkdir -p {{ params.base_dir }}/linear/{{ ds }}/raw &&
        curl -L -f -o {{ params.base_dir }}/linear/{{ ds }}/raw/data_1.csv {{ params.url }} &&
        echo '✅ Archivo descargado exitosamente' &&
        ls -lh {{ params.base_dir }}/linear/{{ ds }}/raw/data_1.csv
    """,
    params={
        "base_dir": DEFAULT_BASE_DIR,
        "url": DATA_URL,
    },
    )
    split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={
            "base_dir": DEFAULT_BASE_DIR,
            "experiment_type": "linear",
            "target_col": "HiringDecision",
            "test_size": 0.2,
            "random_state": 42,
        },
    )

    train = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
        op_kwargs={
            "base_dir": DEFAULT_BASE_DIR,
            "experiment_type": "linear",
            "target_col": "HiringDecision",
            "positive_label": 1,
        },
    )

    end = EmptyOperator(task_id="pipeline_completed")

    start >> make_dirs >> download_data >> split >> train >> end