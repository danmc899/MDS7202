"""
DAG de Pipeline Productivo para SodAI Drinks
Orquesta el flujo completo: extracción, preprocessing, detección de drift, 
entrenamiento y generación de predicciones
"""
import warnings
# Suprimir warnings de Pydantic sobre namespaces protegidos
warnings.filterwarnings('ignore', message='.*model_.*protected namespace.*')

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Importar tasks
from tasks.data_extraction import extract_data
from tasks.data_preprocessing import preprocess_data
from tasks.drift_detection import detect_drift
from tasks.model_training import train_model
from tasks.prediction_generation import generate_predictions

logger = logging.getLogger(__name__)

# Argumentos por defecto del DAG
default_args = {
    'owner': 'sodai_drinks_team',
    'depends_on_past': False,
    'email': ['team@sodaidrinks.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definir DAG
dag = DAG(
    'sodai_drinks_production_pipeline',
    default_args=default_args,
    description='Pipeline productivo completo para predicción de compras',
    schedule_interval='@weekly',  # Ejecutar semanalmente
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['production', 'ml', 'sodai_drinks'],
)


def decide_retraining(**context):
    """
    Decide si se debe reentrenar el modelo basándose en el drift detectado
    """
    ti = context['ti']
    has_drift = ti.xcom_pull(key='has_drift', task_ids='detect_drift')
    drift_reason = ti.xcom_pull(key='drift_reason', task_ids='detect_drift')
    
    logger.info(f"Evaluando necesidad de reentrenamiento...")
    logger.info(f"Drift detectado: {has_drift}, Razón: {drift_reason}")
    
    # Verificar si existe modelo previo
    models_dir = Path("/opt/airflow/models")
    model_exists = (models_dir / "best_model.pkl").exists()
    
    if not model_exists:
        logger.warning("⚠️ No existe modelo previo - Forzando entrenamiento inicial")
        return 'train_model'
    
    # Si es la primera ejecución, siempre entrenar
    if drift_reason == 'first_run':
        logger.info("Primera ejecución - Se procederá con el entrenamiento")
        return 'train_model'
    
    # Si hay drift, reentrenar
    if has_drift:
        logger.warning("Drift detectado - Se procederá con el reentrenamiento")
        return 'train_model'
    
    # TODO: Aquí se puede agregar lógica para reentrenamiento periódico
    # Por ejemplo, si han pasado más de N semanas desde el último entrenamiento
    
    logger.info("No se detectó drift - Se omitirá el reentrenamiento")
    return 'skip_training'


# Definir tareas del DAG

# 1. Extracción de datos
extract_data_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

# 2. Preprocessing
preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

# 3. Detección de drift (BONUS)
detect_drift_task = PythonOperator(
    task_id='detect_drift',
    python_callable=detect_drift,
    dag=dag,
)

# 4. Decisión de reentrenamiento
decide_retraining_task = BranchPythonOperator(
    task_id='decide_retraining',
    python_callable=decide_retraining,
    dag=dag,
)

# 5. Entrenamiento del modelo
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# 6. Skip training (dummy task)
skip_training_task = EmptyOperator(
    task_id='skip_training',
    dag=dag,
)

# 7. Join después del branching
join_task = EmptyOperator(
    task_id='join_after_training',
    trigger_rule='none_failed_min_one_success',  # Continuar si al menos una rama tuvo éxito
    dag=dag,
)

# 8. Generación de predicciones
generate_predictions_task = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_predictions,
    dag=dag,
)

# 9. Tarea final
end_task = EmptyOperator(
    task_id='pipeline_completed',
    dag=dag,
)

# Definir dependencias del DAG
extract_data_task >> preprocess_data_task >> detect_drift_task >> decide_retraining_task

# Branching para reentrenamiento
decide_retraining_task >> [train_model_task, skip_training_task]

# Join después del branching
[train_model_task, skip_training_task] >> join_task

# Continuar con predicciones
join_task >> generate_predictions_task >> end_task
