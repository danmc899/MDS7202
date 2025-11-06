"""
Prediction Generation Task
Genera predicciones para la próxima semana usando el mejor modelo disponible
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_predictions(**context):
    """
    Genera predicciones para la próxima semana
    
    Args:
        **context: Contexto de Airflow
        
    Returns:
        dict: Estadísticas de las predicciones generadas
    """
    logger.info("Iniciando generación de predicciones...")
    
    ti = context['ti']
    processed_data_path = ti.xcom_pull(key='processed_data_path', task_ids='preprocess_data')
    model_path_xcom = ti.xcom_pull(key='model_path', task_ids='train_model')
    prediction_week = ti.xcom_pull(key='prediction_week', task_ids='preprocess_data')
    
    # Si no hay model_path del XCom (porque se saltó el entrenamiento),
    # usar el modelo existente en disco
    models_dir = Path("/opt/airflow/models")
    if model_path_xcom is None:
        model_path = models_dir / "best_model.pkl"
        logger.info(f"No hay modelo nuevo en XCom. Usando modelo existente: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(
                "No se encontró ningún modelo. Debe entrenar un modelo primero o "
                "verificar que existe best_model.pkl en /opt/airflow/models/"
            )
    else:
        model_path = Path(model_path_xcom)
        logger.info(f"Usando modelo del XCom: {model_path}")
    
    # Cargar datos procesados
    df = pd.read_parquet(processed_data_path)
    
    # Cargar modelo
    logger.info(f"Cargando modelo desde: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Cargar metadata del modelo
    metadata_path = models_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No se encontró model_metadata.json en {models_dir}. "
            "Debe entrenar un modelo primero."
        )
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    
    # Preparar features para predicción
    X_pred = df[feature_names]
    
    logger.info(f"Generando predicciones para {len(X_pred)} combinaciones cliente-producto...")
    
    # Generar predicciones
    y_pred = model.predict(X_pred)
    y_proba = model.predict_proba(X_pred)[:, 1]
    
    # Crear DataFrame de predicciones
    df_predictions = pd.DataFrame({
        'customer_id': df['customer_id'],
        'product_id': df['product_id'],
        'prediction': y_pred,
        'probability': y_proba,
        'prediction_week': prediction_week
    })
    
    # Filtrar solo predicciones positivas (clientes que probablemente comprarán)
    df_predictions_positive = df_predictions[df_predictions['prediction'] == 1].copy()
    
    # Ordenar por probabilidad descendente
    df_predictions_positive = df_predictions_positive.sort_values('probability', ascending=False)
    
    logger.info(f"Predicciones positivas: {len(df_predictions_positive)} de {len(df_predictions)}")
    logger.info(f"Tasa de predicciones positivas: {len(df_predictions_positive)/len(df_predictions)*100:.2f}%")
    
    # Guardar predicciones
    predictions_dir = Path("/opt/airflow/data/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    predictions_path = predictions_dir / f"predictions_{timestamp}.parquet"
    df_predictions.to_parquet(predictions_path, index=False)
    
    # Guardar solo predicciones positivas en formato CSV para revisión
    df_predictions_positive.to_csv(
        predictions_dir / f"predictions_positive_{timestamp}.csv",
        index=False
    )
    
    # Estadísticas por cliente (top clientes con más compras predichas)
    top_customers = df_predictions_positive.groupby('customer_id').agg({
        'product_id': 'count',
        'probability': 'mean'
    }).reset_index()
    top_customers.columns = ['customer_id', 'predicted_purchases', 'avg_probability']
    top_customers = top_customers.sort_values('predicted_purchases', ascending=False).head(20)
    
    logger.info(f"Top 20 clientes con más compras predichas:")
    logger.info(f"\n{top_customers.to_string()}")
    
    # Estadísticas por producto (top productos más demandados)
    top_products = df_predictions_positive.groupby('product_id').agg({
        'customer_id': 'count',
        'probability': 'mean'
    }).reset_index()
    top_products.columns = ['product_id', 'predicted_buyers', 'avg_probability']
    top_products = top_products.sort_values('predicted_buyers', ascending=False).head(20)
    
    logger.info(f"Top 20 productos con más compradores predichos:")
    logger.info(f"\n{top_products.to_string()}")
    
    # Guardar estadísticas
    stats = {
        'total_predictions': len(df_predictions),
        'positive_predictions': len(df_predictions_positive),
        'positive_rate': len(df_predictions_positive) / len(df_predictions),
        'avg_probability': float(df_predictions['probability'].mean()),
        'prediction_week': prediction_week,
        'timestamp': timestamp
    }
    
    with open(predictions_dir / f"prediction_stats_{timestamp}.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Pushear información
    ti.xcom_push(key='predictions_path', value=str(predictions_path))
    ti.xcom_push(key='num_positive_predictions', value=len(df_predictions_positive))
    
    logger.info("Generación de predicciones completada exitosamente")
    
    return {
        'status': 'success',
        'total_predictions': len(df_predictions),
        'positive_predictions': len(df_predictions_positive),
        'predictions_file': str(predictions_path)
    }
