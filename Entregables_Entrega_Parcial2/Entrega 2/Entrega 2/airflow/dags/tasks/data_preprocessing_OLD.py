"""
Data Preprocessing Task
Limpia y transforma los datos para el modelo predictivo
Aplica IQR, StandardScaler, MinMaxScaler y transformaciones categóricas
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
import logging
import pickle

logger = logging.getLogger(__name__)


def create_target_variable(df_transacciones, prediction_week, sample_negatives=True, negative_ratio=3.0):
    """
    Crea la variable target: si el cliente compró el producto en la semana objetivo
    Genera ejemplos negativos de combinaciones cliente-producto que NO compraron
    
    Args:
        df_transacciones: DataFrame con transacciones
        prediction_week: Fecha de la semana a predecir
        sample_negatives: Si True, genera ejemplos negativos
        negative_ratio: Ratio de negativos por cada positivo (default 3:1)
        
    Returns:
        DataFrame con la variable target balanceado
    """
    df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])
    df_transacciones['week'] = df_transacciones['purchase_date'].dt.to_period('W')
    
    # Crear combinaciones cliente-producto POSITIVAS (compraron)
    df_target = df_transacciones.groupby(['customer_id', 'product_id', 'week']).agg({
        'items': 'sum',
        'order_id': 'count'
    }).reset_index()
    
    df_target.rename(columns={'order_id': 'num_orders'}, inplace=True)
    df_target['purchased'] = 1
    
    if sample_negatives:
        # Obtener todos los clientes y productos únicos
        all_customers = df_transacciones['customer_id'].unique()
        all_products = df_transacciones['product_id'].unique()
        
        # Combinaciones positivas existentes
        positive_pairs = set(zip(df_target['customer_id'], df_target['product_id']))
        
        # Generar ejemplos negativos
        num_negatives = int(len(positive_pairs) * negative_ratio)
        negative_samples = []
        
        logger.info(f"Generando {num_negatives} ejemplos negativos...")
        
        attempts = 0
        max_attempts = num_negatives * 10  # Límite de intentos
        
        while len(negative_samples) < num_negatives and attempts < max_attempts:
            customer = np.random.choice(all_customers)
            product = np.random.choice(all_products)
            
            if (customer, product) not in positive_pairs:
                negative_samples.append({
                    'customer_id': customer,
                    'product_id': product,
                    'purchased': 0
                })
                positive_pairs.add((customer, product))  # Evitar duplicados
            
            attempts += 1
        
        # Crear DataFrame de negativos
        df_negatives = pd.DataFrame(negative_samples)
        
        # Combinar positivos y negativos
        df_result = pd.concat([
            df_target[['customer_id', 'product_id', 'purchased']],
            df_negatives
        ], ignore_index=True)
        
        logger.info(f"Target balanceado - Positivos: {len(df_target)}, Negativos: {len(df_negatives)}")
        
        return df_result
    
    return df_target[['customer_id', 'product_id', 'purchased']]


def apply_iqr_filter(df, column, multiplier=1.5):
    """
    Aplica filtro IQR para eliminar outliers
    
    Args:
        df: DataFrame
        column: Columna a filtrar
        multiplier: Multiplicador para el rango IQR
        
    Returns:
        DataFrame filtrado
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    initial_count = len(df)
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = initial_count - len(df_filtered)
    
    logger.info(f"IQR filter en {column}: Removidos {removed_count} registros ({removed_count/initial_count*100:.2f}%)")
    
    return df_filtered


def engineer_features(df_transacciones, df_clientes, df_productos):
    """
    Genera features para el modelo basado en la Entrega 1
    
    Args:
        df_transacciones: DataFrame de transacciones
        df_clientes: DataFrame de clientes
        df_productos: DataFrame de productos
        
    Returns:
        DataFrame con features engineered
    """
    logger.info("Iniciando feature engineering...")
    
    # Merge de datos
    df_merged = df_transacciones.merge(df_clientes, on='customer_id', how='left')
    df_merged = df_merged.merge(df_productos, on='product_id', how='left')
    
    # Features temporales
    df_merged['purchase_date'] = pd.to_datetime(df_merged['purchase_date'])
    df_merged['day_of_week'] = df_merged['purchase_date'].dt.dayofweek
    df_merged['week_of_year'] = df_merged['purchase_date'].dt.isocalendar().week
    df_merged['month'] = df_merged['purchase_date'].dt.month
    
    # Features agregadas por cliente
    customer_features = df_merged.groupby('customer_id').agg({
        'items': ['sum', 'mean', 'std', 'count'],
        'order_id': 'nunique'
    }).reset_index()
    customer_features.columns = ['customer_id', 'total_items', 'avg_items', 'std_items', 
                                   'num_transactions', 'unique_orders']
    customer_features['std_items'].fillna(0, inplace=True)
    
    # Features agregadas por producto
    product_features = df_merged.groupby('product_id').agg({
        'items': ['sum', 'mean', 'count'],
        'customer_id': 'nunique'
    }).reset_index()
    product_features.columns = ['product_id', 'product_total_items', 'product_avg_items',
                                  'product_num_transactions', 'unique_customers']
    
    # Features cliente-producto
    customer_product_features = df_merged.groupby(['customer_id', 'product_id']).agg({
        'items': ['sum', 'mean', 'count'],
        'order_id': 'nunique'
    }).reset_index()
    customer_product_features.columns = ['customer_id', 'product_id', 'cp_total_items',
                                           'cp_avg_items', 'cp_num_transactions', 'cp_unique_orders']
    
    # Merge de features
    df_features = customer_product_features.merge(customer_features, on='customer_id', how='left')
    df_features = df_features.merge(product_features, on='product_id', how='left')
    
    # Aplicar IQR a features numéricas
    numeric_cols = ['cp_total_items', 'cp_avg_items', 'total_items', 'avg_items']
    for col in numeric_cols:
        df_features = apply_iqr_filter(df_features, col, multiplier=1.5)
    
    logger.info(f"Features generados: {df_features.shape[1]} columnas, {df_features.shape[0]} filas")
    
    return df_features


def preprocess_data(**context):
    """
    Tarea principal de preprocessing
    
    Args:
        **context: Contexto de Airflow
        
    Returns:
        dict: Estadísticas del preprocessing
    """
    logger.info("Iniciando preprocessing de datos...")
    
    # Obtener rutas de XCom
    ti = context['ti']
    transacciones_path = ti.xcom_pull(key='transacciones_path', task_ids='extract_data')
    clientes_path = ti.xcom_pull(key='clientes_path', task_ids='extract_data')
    productos_path = ti.xcom_pull(key='productos_path', task_ids='extract_data')
    max_date_str = ti.xcom_pull(key='max_date', task_ids='extract_data')
    
    # Cargar datos
    df_transacciones = pd.read_parquet(transacciones_path)
    df_clientes = pd.read_parquet(clientes_path)
    df_productos = pd.read_parquet(productos_path)
    
    # Feature engineering
    df_features = engineer_features(df_transacciones, df_clientes, df_productos)
    
    # Crear variable target (última semana de datos + 1)
    max_date = pd.to_datetime(max_date_str)
    prediction_week = max_date + timedelta(days=7)
    logger.info(f"Predicción para semana: {prediction_week}")
    
    df_target = create_target_variable(df_transacciones, prediction_week)
    
    # Merge features con target (ahora balanceado)
    df_final = df_features.merge(df_target, 
                                   on=['customer_id', 'product_id'], how='inner')
    
    # Log distribución de clases
    class_distribution = df_final['purchased'].value_counts()
    logger.info(f"Distribución de clases después del merge:")
    logger.info(f"  Clase 0 (no compró): {class_distribution.get(0, 0)}")
    logger.info(f"  Clase 1 (compró): {class_distribution.get(1, 0)}")
    
    # Aplicar scalers
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    
    # Columnas para StandardScaler
    standard_cols = ['cp_total_items', 'total_items', 'product_total_items']
    # Columnas para MinMaxScaler
    minmax_cols = ['cp_avg_items', 'avg_items', 'product_avg_items']
    
    for col in standard_cols:
        if col in df_final.columns:
            df_final[f'{col}_scaled'] = scaler_standard.fit_transform(df_final[[col]])
    
    for col in minmax_cols:
        if col in df_final.columns:
            df_final[f'{col}_normalized'] = scaler_minmax.fit_transform(df_final[[col]])
    
    # Guardar datos procesados
    processed_dir = Path("/opt/airflow/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    processed_path = processed_dir / "processed_data.parquet"
    df_final.to_parquet(processed_path, index=False)
    
    # Guardar scalers
    models_dir = Path("/opt/airflow/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    with open(models_dir / "standard_scaler.pkl", 'wb') as f:
        pickle.dump(scaler_standard, f)
    with open(models_dir / "minmax_scaler.pkl", 'wb') as f:
        pickle.dump(scaler_minmax, f)
    
    # Pushear información
    ti.xcom_push(key='processed_data_path', value=str(processed_path))
    ti.xcom_push(key='num_features', value=df_final.shape[1])
    ti.xcom_push(key='prediction_week', value=prediction_week.isoformat())
    
    logger.info("Preprocessing completado exitosamente")
    
    return {
        'status': 'success',
        'processed_records': len(df_final),
        'num_features': df_final.shape[1],
        'prediction_week': prediction_week.isoformat()
    }
