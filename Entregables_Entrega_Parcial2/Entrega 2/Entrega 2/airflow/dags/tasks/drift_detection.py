"""
Drift Detection Task (BONUS)
Detecta drift en los datos utilizando técnicas estadísticas
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import logging
import pickle
import json

logger = logging.getLogger(__name__)


def kolmogorov_smirnov_test(reference_data, current_data, threshold=0.05):
    """
    Aplica el test de Kolmogorov-Smirnov para detectar drift
    
    Args:
        reference_data: Datos de referencia (entrenamiento original)
        current_data: Datos actuales (nuevos datos)
        threshold: Umbral de p-value para rechazar H0
        
    Returns:
        bool: True si hay drift, False en caso contrario
    """
    statistic, p_value = stats.ks_2samp(reference_data, current_data)
    
    logger.info(f"KS Test - Statistic: {statistic:.4f}, P-value: {p_value:.4f}")
    
    # Si p-value < threshold, rechazamos H0 (las distribuciones son diferentes)
    has_drift = p_value < threshold
    
    return has_drift, statistic, p_value


def population_stability_index(reference_data, current_data, bins=10):
    """
    Calcula el Population Stability Index (PSI) para detectar drift
    
    Args:
        reference_data: Datos de referencia
        current_data: Datos actuales
        bins: Número de bins para discretizar
        
    Returns:
        float: Valor PSI (>0.1 indica drift moderado, >0.25 drift severo)
    """
    # Crear bins basados en datos de referencia
    breakpoints = np.percentile(reference_data, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Eliminar duplicados
    
    # Discretizar ambas distribuciones
    ref_counts = np.histogram(reference_data, bins=breakpoints)[0]
    curr_counts = np.histogram(current_data, bins=breakpoints)[0]
    
    # Calcular proporciones (evitar divisiones por 0)
    ref_props = (ref_counts + 1e-10) / (len(reference_data) + 1e-10 * len(breakpoints))
    curr_props = (curr_counts + 1e-10) / (len(current_data) + 1e-10 * len(breakpoints))
    
    # Calcular PSI
    psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    
    logger.info(f"PSI: {psi:.4f}")
    
    return psi


def detect_drift(**context):
    """
    Tarea principal de detección de drift
    
    Args:
        **context: Contexto de Airflow
        
    Returns:
        dict: Resultados de la detección de drift
    """
    logger.info("Iniciando detección de drift...")
    
    ti = context['ti']
    processed_data_path = ti.xcom_pull(key='processed_data_path', task_ids='preprocess_data')
    
    # Cargar datos procesados actuales
    df_current = pd.read_parquet(processed_data_path)
    
    # Intentar cargar datos de referencia (del entrenamiento anterior)
    reference_dir = Path("/opt/airflow/data/processed")
    reference_path = reference_dir / "reference_data.parquet"
    
    if not reference_path.exists():
        logger.warning("No hay datos de referencia. Primera ejecución del pipeline.")
        logger.info("Guardando datos actuales como referencia para futuras comparaciones...")
        df_current.to_parquet(reference_path, index=False)
        
        # No hay drift en la primera ejecución
        ti.xcom_push(key='has_drift', value=False)
        ti.xcom_push(key='drift_reason', value='first_run')
        
        return {
            'status': 'success',
            'has_drift': False,
            'reason': 'Primera ejecución - datos guardados como referencia'
        }
    
    # Cargar datos de referencia
    df_reference = pd.read_parquet(reference_path)
    
    logger.info(f"Datos de referencia: {len(df_reference)} registros")
    logger.info(f"Datos actuales: {len(df_current)} registros")
    
    # Seleccionar features numéricas para análisis de drift
    numeric_features = df_current.select_dtypes(include=[np.number]).columns.tolist()
    # Excluir target y IDs
    exclude_cols = ['customer_id', 'product_id', 'compra_semanal']
    numeric_features = [col for col in numeric_features if col not in exclude_cols]
    
    logger.info(f"Analizando drift en {len(numeric_features)} features numéricas")
    
    # Resultados de drift por feature
    drift_results = {}
    drift_detected = False
    
    for feature in numeric_features[:5]:  # Analizar top 5 features más importantes
        if feature not in df_reference.columns:
            continue
            
        # Test de Kolmogorov-Smirnov
        has_drift_ks, ks_stat, p_value = kolmogorov_smirnov_test(
            df_reference[feature].dropna(),
            df_current[feature].dropna(),
            threshold=0.05
        )
        
        # Population Stability Index
        psi = population_stability_index(
            df_reference[feature].dropna(),
            df_current[feature].dropna()
        )
        
        drift_results[feature] = {
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(p_value),
            'ks_drift': bool(has_drift_ks),
            'psi': float(psi),
            'psi_drift': bool(psi > 0.1)  # Umbral moderado
        }
        
        # Si algún feature muestra drift, marcamos drift global
        if has_drift_ks or psi > 0.1:
            drift_detected = True
            logger.warning(f"Drift detectado en feature '{feature}' - PSI: {psi:.4f}, KS p-value: {p_value:.4f}")
    
    # Guardar resultados
    results_dir = Path("/opt/airflow/data/processed")
    with open(results_dir / "drift_report.json", 'w') as f:
        json.dump(drift_results, f, indent=2)
    
    # Pushear resultado
    ti.xcom_push(key='has_drift', value=drift_detected)
    ti.xcom_push(key='drift_results', value=drift_results)
    
    if drift_detected:
        logger.warning("⚠️ DRIFT DETECTADO - Se recomienda reentrenamiento del modelo")
        ti.xcom_push(key='drift_reason', value='statistical_drift')
    else:
        logger.info("✅ No se detectó drift significativo en los datos")
        ti.xcom_push(key='drift_reason', value='no_drift')
    
    return {
        'status': 'success',
        'has_drift': drift_detected,
        'features_analyzed': len(numeric_features),
        'drift_results': drift_results
    }
