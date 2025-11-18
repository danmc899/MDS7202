"""
Helper functions for the Airflow DAG
"""
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


def check_data_availability(data_dir="/opt/airflow/data/raw"):
    """
    Verifica si hay datos disponibles para procesar
    
    Args:
        data_dir: Directorio donde aparecen los datos
        
    Returns:
        bool: True si hay datos disponibles
    """
    data_path = Path(data_dir)
    required_files = ['transacciones.parquet', 'clientes.parquet', 'productos.parquet']
    
    for file in required_files:
        if not (data_path / file).exists():
            logger.warning(f"Archivo requerido no encontrado: {file}")
            return False
    
    logger.info("Todos los archivos requeridos están disponibles")
    return True


def cleanup_old_predictions(predictions_dir="/opt/airflow/data/predictions", keep_last_n=5):
    """
    Limpia predicciones antiguas, manteniendo solo las últimas N
    
    Args:
        predictions_dir: Directorio de predicciones
        keep_last_n: Número de predicciones a mantener
    """
    predictions_path = Path(predictions_dir)
    
    if not predictions_path.exists():
        return
    
    # Obtener todos los archivos de predicciones
    prediction_files = sorted(predictions_path.glob("predictions_*.parquet"))
    
    # Si hay más de keep_last_n, eliminar los más antiguos
    if len(prediction_files) > keep_last_n:
        files_to_delete = prediction_files[:-keep_last_n]
        for file in files_to_delete:
            logger.info(f"Eliminando predicción antigua: {file.name}")
            file.unlink()


def archive_current_data(source_dir="/opt/airflow/data/raw", archive_dir="/opt/airflow/data/archive"):
    """
    Archiva los datos actuales después de procesarlos
    
    Args:
        source_dir: Directorio de datos actuales
        archive_dir: Directorio de archivo
    """
    source_path = Path(source_dir)
    archive_path = Path(archive_dir)
    archive_path.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for file in source_path.glob("*.parquet"):
        dest = archive_path / f"{file.stem}_{timestamp}{file.suffix}"
        shutil.copy2(file, dest)
        logger.info(f"Archivado: {file.name} -> {dest.name}")
