"""
Data Extraction Task
Extrae los datos de las fuentes (parquet files) y los prepara para procesamiento.

Lógica de semanas (2 semanas):
- Si NO hay best_model.pkl → usa penúltimas 2 semanas para entrenar modelo base
- Si YA hay best_model.pkl → usa últimas 2 semanas para evaluar drift y predecir
"""
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Configuración de ventanas de tiempo
SEMANAS_ENTRENAMIENTO = 2  # Usar 2 semanas para entrenar


def extract_data(**context):
    """
    Extrae los datos de los archivos parquet.
    
    - Si NO hay best_model.pkl → usa 2 semanas anteriores a la última para entrenar
    - Si YA hay best_model.pkl → usa últimas 2 semanas para evaluar drift y predecir
    
    Args:
        **context: Contexto de Airflow con información del DAG
        
    Returns:
        dict: Información sobre los datos extraídos
    """
    logger.info("Iniciando extracción de datos...")
    logger.info(f"📅 Configuración: {SEMANAS_ENTRENAMIENTO} semanas para entrenamiento")
    
    # Directorios
    base_dir = Path("/opt/airflow/data")
    raw_dir = base_dir / "raw"
    models_dir = Path("/opt/airflow/models")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Rutas de archivos
    transacciones_full_path = raw_dir / "transacciones.parquet"
    clientes_path = raw_dir / "clientes.parquet"
    productos_path = raw_dir / "productos.parquet"
    
    # Verificar que existen los archivos base
    if not transacciones_full_path.exists():
        logger.error(f"No se encontró el archivo de transacciones en {transacciones_full_path}")
        raise FileNotFoundError(f"Archivo no encontrado: {transacciones_full_path}")
    
    # Cargar datos completos para obtener fechas
    df_full = pd.read_parquet(transacciones_full_path)
    df_full['purchase_date'] = pd.to_datetime(df_full['purchase_date'])
    
    # Calcular límites de semanas
    fecha_maxima = df_full['purchase_date'].max()
    fecha_inicio_ultima_semana = fecha_maxima - timedelta(days=7)
    fecha_inicio_ventana_train = fecha_maxima - timedelta(days=7 * (SEMANAS_ENTRENAMIENTO + 1))  # +1 para excluir última
    fecha_inicio_ventana_predict = fecha_maxima - timedelta(days=7 * SEMANAS_ENTRENAMIENTO)
    
    logger.info(f"📅 Fechas en dataset completo:")
    logger.info(f"   Fecha máxima: {fecha_maxima.date()}")
    logger.info(f"   Ventana entrenamiento ({SEMANAS_ENTRENAMIENTO} sem): {fecha_inicio_ventana_train.date()} → {fecha_inicio_ultima_semana.date()}")
    logger.info(f"   Ventana predicción ({SEMANAS_ENTRENAMIENTO} sem): {fecha_inicio_ventana_predict.date()} → {fecha_maxima.date()}")
    
    # Verificar si existe un modelo entrenado
    model_path = models_dir / "best_model.pkl"
    model_exists = model_path.exists()
    
    if not model_exists:
        # ═══════════════════════════════════════════════════════════════════
        # PRIMERA EJECUCIÓN: Usar 2 semanas anteriores a la última para entrenar
        # ═══════════════════════════════════════════════════════════════════
        logger.info("🔵 PRIMERA EJECUCIÓN: No hay best_model.pkl")
        logger.info(f"   → Usando {SEMANAS_ENTRENAMIENTO} semanas (excluyendo última) para entrenar modelo base")
        
        df_transacciones = df_full[
            (df_full['purchase_date'] > fecha_inicio_ventana_train) &
            (df_full['purchase_date'] <= fecha_inicio_ultima_semana)
        ].copy()
        
        semana_inicio = fecha_inicio_ventana_train + timedelta(days=1)
        semana_fin = fecha_inicio_ultima_semana
        is_first_training = True
        periodo_desc = f"{SEMANAS_ENTRENAMIENTO}sem_train"
        
    else:
        # ═══════════════════════════════════════════════════════════════════
        # SEGUNDA EJECUCIÓN: Usar últimas 2 semanas para evaluar drift y predecir
        # ═══════════════════════════════════════════════════════════════════
        logger.info("�� EJECUCIÓN CON MODELO: best_model.pkl detectado")
        logger.info(f"   → Usando últimas {SEMANAS_ENTRENAMIENTO} semanas para evaluar drift y predecir")
        
        df_transacciones = df_full[
            df_full['purchase_date'] > fecha_inicio_ventana_predict
        ].copy()
        
        semana_inicio = fecha_inicio_ventana_predict + timedelta(days=1)
        semana_fin = fecha_maxima
        is_first_training = False
        periodo_desc = f"{SEMANAS_ENTRENAMIENTO}sem_predict"
    
    # Calcular identificador de semana (año-semana ISO de la fecha final)
    semana_iso = semana_fin.isocalendar()
    week_id = f"{semana_iso.year}-W{semana_iso.week:02d}_{periodo_desc}"
    
    # Guardar subsample con timestamp (único archivo de subsample)
    subsample_path = raw_dir / f"transacciones_subsample_{week_id}.parquet"
    df_transacciones.to_parquet(subsample_path, index=False)
    
    # Leer clientes y productos
    df_clientes = pd.read_parquet(clientes_path)
    df_productos = pd.read_parquet(productos_path)
    
    # Estadísticas del subsample
    num_transacciones = len(df_transacciones)
    num_clientes = df_transacciones['customer_id'].nunique()
    num_productos = df_transacciones['product_id'].nunique()
    pares_unicos = df_transacciones.groupby(['customer_id', 'product_id']).size().reset_index()
    num_pares = len(pares_unicos)
    
    logger.info(f"📊 Datos del subsample ({week_id}):")
    logger.info(f"   Rango: {semana_inicio.date()} → {semana_fin.date()}")
    logger.info(f"   Días incluidos: {(semana_fin - semana_inicio).days}")
    logger.info(f"   Transacciones: {num_transacciones:,}")
    logger.info(f"   Clientes únicos: {num_clientes:,}")
    logger.info(f"   Productos únicos: {num_productos:,}")
    logger.info(f"   Pares cliente-producto: {num_pares:,}")
    
    # Validar estructura de datos
    required_cols = ['customer_id', 'product_id', 'order_id', 'purchase_date', 'items']
    if not all(col in df_transacciones.columns for col in required_cols):
        raise ValueError(f"Columnas requeridas: {required_cols}")
    
    # Pushear información al XCom
    context['ti'].xcom_push(key='transacciones_path', value=str(subsample_path))
    context['ti'].xcom_push(key='clientes_path', value=str(clientes_path))
    context['ti'].xcom_push(key='productos_path', value=str(productos_path))
    context['ti'].xcom_push(key='num_transacciones', value=num_transacciones)
    context['ti'].xcom_push(key='num_pares', value=num_pares)
    context['ti'].xcom_push(key='is_first_training', value=is_first_training)
    context['ti'].xcom_push(key='week_id', value=week_id)
    context['ti'].xcom_push(key='max_date', value=semana_fin.isoformat())
    
    logger.info(f"✅ Extracción completada - Periodo: {week_id}")
    
    return {
        'status': 'success',
        'week_id': week_id,
        'transacciones_count': num_transacciones,
        'pares_count': num_pares,
        'clientes_count': num_clientes,
        'productos_count': num_productos,
        'is_first_training': is_first_training,
        'fecha_inicio': semana_inicio.isoformat(),
        'fecha_fin': semana_fin.isoformat(),
        'semanas_usadas': SEMANAS_ENTRENAMIENTO
    }
