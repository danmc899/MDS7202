"""
Data Extraction Task
Extrae los datos de las fuentes (parquet files) y los prepara para procesamiento
"""
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def extract_data(**context):
    """
    Extrae los datos de los archivos parquet y los carga en el directorio de trabajo.
    
    Args:
        **context: Contexto de Airflow con información del DAG
        
    Returns:
        dict: Rutas de los archivos extraídos
    """
    logger.info("Iniciando extracción de datos...")
    
    # Directorios
    base_dir = Path("/opt/airflow/data")
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar si existen archivos nuevos de transacciones
    # En producción, estos archivos "aparecen mágicamente" en raw_dir
    transacciones_path = raw_dir / "transacciones.parquet"
    clientes_path = raw_dir / "clientes.parquet"
    productos_path = raw_dir / "productos.parquet"
    
    # Verificar que existen los archivos
    if not transacciones_path.exists():
        logger.error(f"No se encontró el archivo de transacciones en {transacciones_path}")
        raise FileNotFoundError(f"Archivo no encontrado: {transacciones_path}")
    
    # Leer los datos para validar estructura
    df_transacciones = pd.read_parquet(transacciones_path)
    df_clientes = pd.read_parquet(clientes_path)
    df_productos = pd.read_parquet(productos_path)
    
    logger.info(f"Transacciones cargadas: {len(df_transacciones)} registros")
    logger.info(f"Clientes cargados: {len(df_clientes)} registros")
    logger.info(f"Productos cargados: {len(df_productos)} registros")
    
    # Validar estructura de datos
    required_cols_transacciones = ['customer_id', 'product_id', 'order_id', 'purchase_date', 'items']
    if not all(col in df_transacciones.columns for col in required_cols_transacciones):
        raise ValueError(f"El archivo de transacciones no tiene la estructura esperada. Columnas requeridas: {required_cols_transacciones}")
    
    # Pushear información al XCom para siguientes tareas
    context['ti'].xcom_push(key='transacciones_path', value=str(transacciones_path))
    context['ti'].xcom_push(key='clientes_path', value=str(clientes_path))
    context['ti'].xcom_push(key='productos_path', value=str(productos_path))
    context['ti'].xcom_push(key='num_transacciones', value=len(df_transacciones))
    
    # Obtener la semana más reciente en los datos
    df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])
    max_date = df_transacciones['purchase_date'].max()
    logger.info(f"Fecha máxima en datos: {max_date}")
    context['ti'].xcom_push(key='max_date', value=max_date.isoformat())
    
    logger.info("Extracción de datos completada exitosamente")
    
    return {
        'status': 'success',
        'transacciones_count': len(df_transacciones),
        'clientes_count': len(df_clientes),
        'productos_count': len(df_productos),
        'max_date': max_date.isoformat()
    }
