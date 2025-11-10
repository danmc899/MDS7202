"""
Data Preprocessing Task - Compatible con Entrega 1
Replica el preprocessing de la Entrega 1 con ColumnTransformer
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import logging
import pickle
from sklearn.utils import resample

logger = logging.getLogger(__name__)


class IQR(BaseEstimator, TransformerMixin):
    """
    Transformer personalizado para eliminar outliers usando IQR
    Compatible con ColumnTransformer
    """
    def __init__(self, columns=None, lam=1.5):
        self.columns = columns
        self.lam = lam

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        if self.columns is None:
            self.columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.columns_ = list(self.columns)

        # Calcular Q1, Q3 y IQR
        q1 = X[self.columns_].quantile(0.25)
        q3 = X[self.columns_].quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.lam * iqr).to_dict()
        self.upper_ = (q3 + self.lam * iqr).to_dict()
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame(X).copy()
        for c in self.columns_:
            lower = self.lower_[c]
            upper = self.upper_[c]
            X_transformed[c] = X_transformed[c].clip(lower, upper)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.columns_
        return input_features

    def set_output(self, transform='default'):
        return self


def create_target_variable(df_transacciones, balance_method="undersample", random_state=42):
    """
    Crea y balancea la variable target según la lógica de la Entrega 1.

    Args:
        df_transacciones: DataFrame con transacciones
        balance_method: "undersample" o "oversample"
    Returns:
        DataFrame con variable target balanceada
    """
    # Preparar variable target 
    df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])
    df_transacciones['purchase_week'] = df_transacciones['purchase_date'].dt.to_period('W').astype(str)
    
    df_target = df_transacciones.groupby(['customer_id', 'product_id', 'purchase_week']).agg({
        'items': 'sum'
    }).reset_index()
    df_target['compra_semanal'] = (df_target['items'] > 0).astype(int)

    # Balancear clases 
    df_pos = df_target[df_target['compra_semanal'] == 1]
    df_neg = df_target[df_target['compra_semanal'] == 0]

    if balance_method == "undersample":
        df_neg_bal = resample(df_neg, replace=False, n_samples=len(df_pos), random_state=random_state)
        df_bal = pd.concat([df_pos, df_neg_bal])
    elif balance_method == "oversample":
        df_pos_bal = resample(df_pos, replace=True, n_samples=len(df_neg), random_state=random_state)
        df_bal = pd.concat([df_pos_bal, df_neg])
    else:
        raise ValueError("balance_method debe ser 'undersample' o 'oversample'")

    df_bal = df_bal.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Clases balanceadas: {df_bal['compra_semanal'].value_counts().to_dict()}")
    return df_bal


def preprocess_data(**context):
    """
    Tarea principal de preprocessing siguiendo la Entrega 1
    """
    logger.info("Iniciando preprocessing de datos (Entrega 1 compatible)...")
    
    ti = context['ti']
    transacciones_path = ti.xcom_pull(key='transacciones_path', task_ids='extract_data')
    clientes_path = ti.xcom_pull(key='clientes_path', task_ids='extract_data')
    productos_path = ti.xcom_pull(key='productos_path', task_ids='extract_data')
    
    # Cargar datos
    df_transacciones = pd.read_parquet(transacciones_path)
    df_clientes = pd.read_parquet(clientes_path)
    df_productos = pd.read_parquet(productos_path)
    
    logger.info(f"Datos cargados - Transacciones: {len(df_transacciones)}, "
                f"Clientes: {len(df_clientes)}, Productos: {len(df_productos)}")
    
    # Calcular la semana de predicción (última semana en datos + 1)
    df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])
    max_date = df_transacciones['purchase_date'].max()
    prediction_week = max_date + timedelta(days=7)
    logger.info(f"Predicción para semana: {prediction_week.strftime('%Y-%W')}")
    
    # Crear variable target
    df_target = create_target_variable(df_transacciones, sample_negatives=True, negative_ratio=2.0)
    
    # Merge con datos de clientes y productos
    df_merged = df_target.merge(df_clientes, on='customer_id', how='left')
    df_merged = df_merged.merge(df_productos, on='product_id', how='left')
    
    logger.info(f"Dataset merged: {df_merged.shape}")
    
    # Columnas a excluir (siguiendo Entrega 1)
    cols_to_exclude = ['customer_id', 'product_id', 'region_id', 'zone_id', 
                       'purchase_week', 'compra_semanal', 'num_visit_per_week']
    
    # Separar features y target
    feature_cols = [col for col in df_merged.columns if col not in cols_to_exclude]
    X = df_merged[feature_cols].copy()
    y = df_merged['compra_semanal'].copy()
    
    logger.info(f"Features: {X.columns.tolist()}")
    logger.info(f"Distribución de clases - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
    
    # Procesar columna brand
    if 'brand' in X.columns:
        # Extraer número de marca y convertir a float
        X['brand_num'] = X['brand'].str.extract(r'(\d+)').astype(float)
        # Eliminar columna brand original (string)
        X = X.drop(columns=['brand'])
        logger.info("Columna 'brand' convertida a 'brand_num' (numérica)")
    
    # Definir columnas según Entrega 1
    coords_cols = ['Y', 'X']
    positive_cols = ['num_deliver_per_week', 'size']
    ordinal_cols = ['segment']
    nominal_cols = ['customer_type', 'category', 'sub_category', 'package']
    outlier_cols = ['size', 'Y', 'X']
    # brand_num se dejará en remainder='passthrough' como numérica
    
    # Orden jerárquico para segment
    segment_order = [['POPULAR', 'MAINSTREAM', 'PREMIUM']]
    
    # Crear el ColumnTransformer
    # IMPORTANTE: Las columnas procesadas por IQR son transformadas in-place,
    # luego otros transformers las procesan. Para evitar duplicados, IQR no
    # debe estar en el ColumnTransformer, sino aplicarse antes.
    
    # Aplicar IQR primero (fuera del ColumnTransformer)
    iqr_transformer = IQR(columns=outlier_cols, lam=1.5)
    X_iqr = iqr_transformer.fit_transform(X)
    X = pd.DataFrame(X_iqr, columns=X.columns, index=X.index)
    
    # Ahora crear ColumnTransformer sin IQR
    preprocessor = ColumnTransformer(
        transformers=[
            # 1. Escalado para coordenadas (después de IQR)
            ('coords_scaler', StandardScaler(), coords_cols),
            
            # 2. Escalado para size (después de IQR)
            ('size_scaler', MinMaxScaler(), ['size']),
            
            # 3. Escalado para num_deliver_per_week
            ('positive_scaler', MinMaxScaler(), ['num_deliver_per_week']),
            
            # 4. Encoder ordinal
            ('ordinal', OrdinalEncoder(
                categories=segment_order, 
                handle_unknown='use_encoded_value', 
                unknown_value=-1
            ), ordinal_cols),
            
            # 5. OneHotEncoder para categóricas
            ('nominal', OneHotEncoder(
                drop='first', 
                handle_unknown='ignore'
            ), nominal_cols)
        ],
        remainder='passthrough'
    )
    
    logger.info("Ajustando preprocessor...")
    X_transformed = preprocessor.fit_transform(X)
    
    # Obtener nombres de features después de la transformación
    feature_names = []
    processed_cols = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'nominal':
            # OneHotEncoder genera múltiples columnas
            feature_names.extend(transformer.get_feature_names_out(columns))
            processed_cols.extend(columns)
        elif name == 'remainder':
            # Remainder agrega columnas que no fueron procesadas
            if columns != 'drop':
                remaining_cols = [col for col in X.columns if col not in processed_cols]
                feature_names.extend(remaining_cols)
        else:
            # Otros transformers mantienen los nombres originales
            cols = columns if isinstance(columns, list) else [columns]
            feature_names.extend(cols)
            processed_cols.extend(cols)
    
    # Crear DataFrame final
    if hasattr(X_transformed, 'toarray'):
        X_transformed = X_transformed.toarray()
    
    df_final = pd.DataFrame(X_transformed, columns=feature_names)
    df_final['compra_semanal'] = y.values
    df_final['customer_id'] = df_merged['customer_id'].values
    df_final['product_id'] = df_merged['product_id'].values
    
    logger.info(f"Datos transformados: {df_final.shape}")
    logger.info(f"Verificación de clases - 0: {(df_final['compra_semanal']==0).sum()}, "
                f"1: {(df_final['compra_semanal']==1).sum()}")
    
    # Guardar datos procesados
    processed_dir = Path("/opt/airflow/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    processed_path = processed_dir / "processed_data.parquet"
    df_final.to_parquet(processed_path, index=False)
    
    # Guardar preprocessor
    models_dir = Path("/opt/airflow/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar el preprocessor completo (ColumnTransformer)
    with open(models_dir / "preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Guardar el IQR transformer por separado (se aplica antes del preprocessor)
    with open(models_dir / "iqr_transformer.pkl", 'wb') as f:
        pickle.dump(iqr_transformer, f)
    
    # Pushear información
    ti.xcom_push(key='processed_data_path', value=str(processed_path))
    ti.xcom_push(key='num_features', value=df_final.shape[1] - 3)  # Excluye target y IDs
    ti.xcom_push(key='prediction_week', value=prediction_week.isoformat())
    
    logger.info("Preprocessing completado exitosamente")
    
    return {
        'status': 'success',
        'processed_records': len(df_final),
        'num_features': df_final.shape[1] - 3,
        'positive_class': int((y == 1).sum()),
        'negative_class': int((y == 0).sum()),
        'prediction_week': prediction_week.isoformat()
    }
