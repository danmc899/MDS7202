"""
RecSys Data Sync Task
Sincroniza los datos procesados con el backend de RecSys y llama al endpoint de reentrenamiento
"""
import pandas as pd
import shutil
from pathlib import Path
import logging
import requests
import time

logger = logging.getLogger(__name__)


def sync_recsys_data(**context):
    """
    Copia los datos al directorio del RecSys backend y activa el reentrenamiento.
    
    El RecSys espera encontrar los archivos en:
    - /app/data/transacciones.parquet
    - /app/data/clientes.parquet  
    - /app/data/productos.parquet
    
    Esta tarea copia los datos actualizados y llama al endpoint /api/v1/retrain
    """
    logger.info("🔄 Iniciando sincronización con RecSys backend...")
    
    ti = context['ti']
    
    # Obtener rutas de los datos extraídos
    transacciones_path = ti.xcom_pull(key='transacciones_path', task_ids='extract_data')
    clientes_path = ti.xcom_pull(key='clientes_path', task_ids='extract_data')
    productos_path = ti.xcom_pull(key='productos_path', task_ids='extract_data')
    
    # Directorios destino (montados en el contenedor de RecSys)
    # Nota: Este directorio debe estar montado como volumen compartido entre Airflow y RecSys
    recsys_data_dir = Path("/opt/airflow/shared_data/recsys")
    recsys_data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📂 Directorio destino RecSys: {recsys_data_dir}")
    
    # Copiar archivos
    files_to_copy = {
        'transacciones': transacciones_path,
        'clientes': clientes_path,
        'productos': productos_path
    }
    
    for name, source_path in files_to_copy.items():
        if source_path and Path(source_path).exists():
            dest_file = recsys_data_dir / f"{Path(source_path).name}"
            shutil.copy2(source_path, dest_file)
            logger.info(f"✅ Copiado {name}: {source_path} → {dest_file}")
            
            # Verificar tamaño del archivo copiado
            df = pd.read_parquet(dest_file)
            logger.info(f"   📊 Registros en {name}: {len(df)}")
        else:
            logger.warning(f"⚠️ Archivo no encontrado: {source_path}")
    
    # Llamar al endpoint de reentrenamiento del RecSys
    recsys_url = "http://sodai-recsys-backend:8001/api/v1/retrain"
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Llamando a RecSys retrain endpoint (intento {attempt + 1}/{max_retries})...")
            response = requests.post(recsys_url, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ RecSys reentrenado exitosamente!")
                logger.info(f"   📊 Clientes: {result.get('num_customers', 'N/A')}")
                logger.info(f"   📊 Productos: {result.get('num_products', 'N/A')}")
                logger.info(f"   📊 Ratings: {result.get('num_ratings', 'N/A')}")
                
                # Pushear resultado al XCom
                ti.xcom_push(key='recsys_retrain_status', value='success')
                ti.xcom_push(key='recsys_num_ratings', value=result.get('num_ratings', 0))
                
                return {
                    'status': 'success',
                    'recsys_retrained': True,
                    'num_customers': result.get('num_customers', 0),
                    'num_products': result.get('num_products', 0),
                    'num_ratings': result.get('num_ratings', 0)
                }
            else:
                logger.warning(f"⚠️ RecSys retrain falló con status {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception(f"RecSys retrain falló después de {max_retries} intentos")
                    
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"⚠️ No se pudo conectar con RecSys backend: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Reintentando en {retry_delay} segundos...")
                time.sleep(retry_delay)
            else:
                logger.error("❌ RecSys backend no disponible después de múltiples intentos")
                # No fallar el DAG, solo marcar como warning
                ti.xcom_push(key='recsys_retrain_status', value='unavailable')
                return {
                    'status': 'warning',
                    'recsys_retrained': False,
                    'message': 'RecSys backend no disponible'
                }
        except Exception as e:
            logger.error(f"❌ Error al llamar RecSys: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
    
    logger.info("✅ Sincronización con RecSys completada")
