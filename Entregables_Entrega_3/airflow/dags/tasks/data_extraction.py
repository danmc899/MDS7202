"""
Data Extraction Task
Lee los archivos de batch semanales pre-generados de data/raw/.

Archivos esperados en data/raw/:
- batch_YYYY-WXX.parquet  (batches semanales de Condabench)
- clientes.parquet
- productos.parquet

Lógica:
- Detecta todos los archivos batch_YYYY-WXX.parquet disponibles
- Ordena por semana ISO (cronológicamente)
- Busca un archivo "processed_weeks.txt" para saber cuáles ya se procesaron
- Procesa el siguiente batch no procesado (del más antiguo al más nuevo)
- El primer batch genera best_model.pkl
- Los siguientes evalúan drift y predicen
"""
import pandas as pd
import os
import re
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def get_available_batches(raw_dir: Path) -> list:
    """
    Detecta todos los archivos batch disponibles y extrae sus semanas ISO.
    
    Returns:
        Lista de tuplas (week_id, filepath) ordenadas cronológicamente (más antiguo primero)
    """
    pattern = re.compile(r'batch_(\d{4}-W\d{2})\.parquet')
    batches = []
    
    for f in raw_dir.glob("batch_*.parquet"):
        match = pattern.match(f.name)
        if match:
            week_id = match.group(1)  # Ej: "2024-W51"
            batches.append((week_id, f))
    
    # Ordenar cronológicamente (año, semana)
    def sort_key(x):
        year, week = x[0].split('-W')
        return (int(year), int(week))
    
    batches.sort(key=sort_key)
    return batches


def get_processed_weeks(data_dir: Path) -> set:
    """
    Lee el archivo de semanas ya procesadas.
    """
    processed_file = data_dir / "processed_weeks.txt"
    if processed_file.exists():
        with open(processed_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def mark_week_processed(data_dir: Path, week_id: str):
    """
    Marca una semana como procesada.
    """
    processed_file = data_dir / "processed_weeks.txt"
    with open(processed_file, 'a') as f:
        f.write(f"{week_id}\n")


def extract_data(**context):
    """
    Lee el siguiente batch semanal no procesado.
    
    Lógica:
    - Itera cronológicamente por los batches disponibles
    - Encuentra el primero que no ha sido procesado
    - Si NO hay best_model.pkl → es entrenamiento inicial
    - Si YA hay best_model.pkl → evalúa drift y predice
    
    Args:
        **context: Contexto de Airflow
        
    Returns:
        dict: Información sobre los datos extraídos
    """
    logger.info("=" * 60)
    logger.info("📂 EXTRACCIÓN DE DATOS - Buscando batch a procesar")
    logger.info("=" * 60)
    
    # Directorios
    base_dir = Path("/opt/airflow/data")
    raw_dir = base_dir / "raw"
    models_dir = Path("/opt/airflow/models")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Rutas de archivos auxiliares
    clientes_path = raw_dir / "clientes.parquet"
    productos_path = raw_dir / "productos.parquet"
    
    # Detectar batches disponibles (ordenados cronológicamente)
    available_batches = get_available_batches(raw_dir)
    
    if not available_batches:
        raise FileNotFoundError(
            f"No se encontraron archivos batch_YYYY-WXX.parquet en {raw_dir}"
        )
    
    logger.info(f"📅 Batches disponibles: {[b[0] for b in available_batches]}")
    
    # Ver cuáles ya se procesaron
    processed_weeks = get_processed_weeks(base_dir)
    logger.info(f"✅ Ya procesados: {sorted(processed_weeks) if processed_weeks else 'ninguno'}")
    
    # Encontrar el siguiente batch a procesar
    batch_to_process = None
    for week_id, filepath in available_batches:
        if week_id not in processed_weeks:
            batch_to_process = (week_id, filepath)
            break
    
    if batch_to_process is None:
        logger.info("🎉 Todos los batches ya fueron procesados!")
        # Retornar el último procesado para no romper el pipeline
        batch_to_process = available_batches[-1]
        context['ti'].xcom_push(key='all_processed', value=True)
    else:
        context['ti'].xcom_push(key='all_processed', value=False)
    
    week_id, filepath = batch_to_process
    logger.info(f"📦 Procesando batch: {week_id} ({filepath.name})")
    
    # Verificar si existe un modelo entrenado
    model_path = models_dir / "best_model.pkl"
    model_exists = model_path.exists()
    
    if not model_exists:
        # ═══════════════════════════════════════════════════════════════════
        # PRIMERA EJECUCIÓN: Entrenar modelo base con este batch
        # ═══════════════════════════════════════════════════════════════════
        logger.info("🔵 MODO: Entrenamiento inicial (no hay best_model.pkl)")
        is_first_training = True
        periodo_desc = "train"
    else:
        # ═══════════════════════════════════════════════════════════════════
        # EJECUCIÓN CON MODELO: Evaluar drift y predecir
        # ═══════════════════════════════════════════════════════════════════
        logger.info("🟢 MODO: Evaluación de drift + Predicción (best_model.pkl existe)")
        is_first_training = False
        periodo_desc = "predict"
    
    # Leer el batch
    logger.info(f"   📖 Leyendo: {filepath.name}")
    df_transacciones = pd.read_parquet(filepath)
    
    # Asegurar tipos de datos correctos
    df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])
    
    # Estadísticas
    fecha_min = df_transacciones['purchase_date'].min()
    fecha_max = df_transacciones['purchase_date'].max()
    num_transacciones = len(df_transacciones)
    num_clientes = df_transacciones['customer_id'].nunique()
    num_productos = df_transacciones['product_id'].nunique()
    pares_unicos = df_transacciones.groupby(['customer_id', 'product_id']).size().reset_index()
    num_pares = len(pares_unicos)
    
    logger.info(f"📊 Datos del batch {week_id}:")
    logger.info(f"   Rango fechas: {fecha_min.date()} → {fecha_max.date()}")
    logger.info(f"   Transacciones: {num_transacciones:,}")
    logger.info(f"   Clientes únicos: {num_clientes:,}")
    logger.info(f"   Productos únicos: {num_productos:,}")
    logger.info(f"   Pares cliente-producto: {num_pares:,}")
    
    # Construir week_id completo
    full_week_id = f"{week_id}_{periodo_desc}"
    
    # Guardar archivo para procesamiento posterior
    output_path = raw_dir / f"current_batch_{full_week_id}.parquet"
    df_transacciones.to_parquet(output_path, index=False)
    logger.info(f"   💾 Guardado: {output_path.name}")
    
    # Validar estructura de datos
    required_cols = ['customer_id', 'product_id', 'order_id', 'purchase_date', 'items']
    missing_cols = [col for col in required_cols if col not in df_transacciones.columns]
    if missing_cols:
        logger.warning(f"⚠️ Columnas faltantes: {missing_cols}")
    
    # Marcar como procesado
    mark_week_processed(base_dir, week_id)
    logger.info(f"   ✅ Marcado como procesado: {week_id}")
    
    # Pushear información al XCom
    context['ti'].xcom_push(key='transacciones_path', value=str(output_path))
    context['ti'].xcom_push(key='clientes_path', value=str(clientes_path))
    context['ti'].xcom_push(key='productos_path', value=str(productos_path))
    context['ti'].xcom_push(key='num_transacciones', value=num_transacciones)
    context['ti'].xcom_push(key='num_pares', value=num_pares)
    context['ti'].xcom_push(key='is_first_training', value=is_first_training)
    context['ti'].xcom_push(key='week_id', value=full_week_id)
    context['ti'].xcom_push(key='batch_week', value=week_id)
    context['ti'].xcom_push(key='max_date', value=fecha_max.isoformat())
    
    logger.info(f"✅ Extracción completada - Week ID: {full_week_id}")
    logger.info("=" * 60)
    
    return {
        'status': 'success',
        'week_id': full_week_id,
        'batch_week': week_id,
        'transacciones_count': num_transacciones,
        'pares_count': num_pares,
        'clientes_count': num_clientes,
        'productos_count': num_productos,
        'is_first_training': is_first_training,
        'fecha_inicio': fecha_min.isoformat(),
        'fecha_fin': fecha_max.isoformat(),
        'batches_disponibles': [b[0] for b in available_batches],
        'batches_procesados': list(processed_weeks | {week_id})
    }
