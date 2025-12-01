"""
Data Extraction Task
Extrae los datos de las fuentes (parquet files) y los prepara para procesamiento
"""
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def extract_data(**context):
    """
    Extrae los datos de los archivos parquet y crea un subsample de la primera semana
    para simular la llegada de datos nuevos.
    
    Args:
        **context: Contexto de Airflow con información del DAG
        
    Returns:
        dict: Rutas de los archivos extraídos
    """
    logger.info("Iniciando extracción de datos...")
    
    # Directorios
    base_dir = Path("/opt/airflow/data")
    raw_dir = base_dir / "raw"
    models_dir = Path("/opt/airflow/models")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Rutas de archivos
    transacciones_full_path = raw_dir / "transacciones.parquet"
    transacciones_subsample_path = raw_dir / "transacciones_subsample.parquet"
    clientes_path = raw_dir / "clientes.parquet"
    productos_path = raw_dir / "productos.parquet"
    
    # Verificar que existen los archivos base
    if not transacciones_full_path.exists():
        logger.error(f"No se encontró el archivo de transacciones en {transacciones_full_path}")
        raise FileNotFoundError(f"Archivo no encontrado: {transacciones_full_path}")
    
    # Verificar si existe un modelo entrenado
    model_path = models_dir / "best_model.pkl"
    model_exists = model_path.exists()
    
    # LÓGICA DE SELECCIÓN DE DATOS:
    # - Si NO hay modelo → usar transacciones.parquet (datos completos, primer entrenamiento)
    # - Si YA hay modelo → usar transacciones_subsample.parquet (datos incrementales)
    
    if not model_exists:
        logger.info("🔵 PRIMER ENTRENAMIENTO: No se detectó modelo previo")
        logger.info("   Usando datos completos (transacciones.parquet)")
        transacciones_path = transacciones_full_path
        is_subsample = False
        
        # Leer datos completos
        df_transacciones = pd.read_parquet(transacciones_path)
        
    else:
        logger.info("🟢 REENTRENAMIENTO: Modelo previo detectado")
        logger.info("   Usando datos incrementales (transacciones_subsample.parquet)")
        
        # Si no existe el subsample, crearlo a partir de los datos completos
        if not transacciones_subsample_path.exists():
            logger.info("   Creando subsample de primera semana...")
            df_full = pd.read_parquet(transacciones_full_path)
            df_full['purchase_date'] = pd.to_datetime(df_full['purchase_date'])
            
            fecha_minima = df_full['purchase_date'].min()
            fecha_primera_semana = fecha_minima + timedelta(days=7)
            
            df_transacciones = df_full[
                df_full['purchase_date'] < fecha_primera_semana
            ].copy()
            
            df_transacciones.to_parquet(transacciones_subsample_path, index=False)
            logger.info(f"   Subsample creado: {len(df_transacciones)} transacciones")
        else:
            df_transacciones = pd.read_parquet(transacciones_subsample_path)
        
        transacciones_path = transacciones_subsample_path
        is_subsample = True
    
    # Leer clientes y productos
    df_clientes = pd.read_parquet(clientes_path)
    df_productos = pd.read_parquet(productos_path)
    
    logger.info(f"📊 Datos cargados:")
    logger.info(f"   Transacciones: {len(df_transacciones)} registros")
    logger.info(f"   Clientes: {len(df_clientes)} registros")
    logger.info(f"   Productos: {len(df_productos)} registros")
    logger.info(f"   Tipo: {'Subsample semanal' if is_subsample else 'Dataset completo'}")
    
    # Validar estructura de datos
    required_cols_transacciones = ['customer_id', 'product_id', 'order_id', 'purchase_date', 'items']
    if not all(col in df_transacciones.columns for col in required_cols_transacciones):
        raise ValueError(f"El archivo de transacciones no tiene la estructura esperada. Columnas requeridas: {required_cols_transacciones}")
    
    # Pushear información al XCom para siguientes tareas
    context['ti'].xcom_push(key='transacciones_path', value=str(transacciones_path))
    context['ti'].xcom_push(key='clientes_path', value=str(clientes_path))
    context['ti'].xcom_push(key='productos_path', value=str(productos_path))
    context['ti'].xcom_push(key='num_transacciones', value=len(df_transacciones))
    context['ti'].xcom_push(key='is_first_training', value=not model_exists)
    
    # Obtener fecha máxima en los datos
    df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])
    max_date = df_transacciones['purchase_date'].max()
    logger.info(f"   Fecha máxima: {max_date}")
    context['ti'].xcom_push(key='max_date', value=max_date.isoformat())
    
    logger.info("✅ Extracción de datos completada exitosamente")
    
    return {
        'status': 'success',
        'transacciones_count': len(df_transacciones),
        'clientes_count': len(df_clientes),
        'productos_count': len(df_productos),
        'max_date': max_date.isoformat(),
        'is_subsample': is_subsample,
        'is_first_training': not model_exists
    }
