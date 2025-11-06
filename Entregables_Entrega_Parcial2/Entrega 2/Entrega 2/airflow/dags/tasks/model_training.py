"""
Model Training Task
Entrena el modelo con optimización de hiperparámetros usando Optuna
Tracking con MLflow (BONUS)
Genera gráficos de interpretabilidad con SHAP
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# Optuna para optimización
import optuna

# MLflow para tracking (BONUS)
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow no está disponible. Tracking deshabilitado.")

# SHAP para interpretabilidad
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Optimiza hiperparámetros usando Optuna
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        n_trials: Número de trials para Optuna
        
    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    logger.info(f"Iniciando optimización de hiperparámetros con {n_trials} trials...")
    
    def objective(trial):
        # Sugerir modelo
        model_name = trial.suggest_categorical('model', ['random_forest', 'xgboost', 'gradient_boosting'])
        
        # Verificar que tenemos ambas clases en y_train
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            logger.warning(f"Solo se encontró una clase en y_train: {unique_classes}. Prueba omitida.")
            raise optuna.exceptions.TrialPruned()
        
        if model_name == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
            
        elif model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            model = xgb.XGBClassifier(**params)
            
        else:  # gradient_boosting
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': 42
            }
            model = GradientBoostingClassifier(**params)
        
        # Entrenar y evaluar
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
        except Exception as e:
            logger.warning(f"Error al entrenar modelo {model_name}: {e}")
            raise optuna.exceptions.TrialPruned()
        
        return f1
    
    # Crear estudio de Optuna
    study = optuna.create_study(direction='maximize', study_name='model_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Mejor F1-Score encontrado: {study.best_value:.4f}")
    logger.info(f"Mejores hiperparámetros: {study.best_params}")
    
    return study.best_params, study


def generate_shap_plots(model, X_sample, feature_names, output_dir):
    """
    Genera gráficos de interpretabilidad con SHAP
    
    Args:
        model: Modelo entrenado
        X_sample: Muestra de datos para SHAP
        feature_names: Nombres de features
        output_dir: Directorio para guardar gráficos
    """
    logger.info("Generando gráficos de interpretabilidad con SHAP...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Si shap_values es una lista (clasificación binaria), tomar valores de clase positiva
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Gráficos SHAP generados exitosamente")


def train_model(**context):
    """
    Tarea principal de entrenamiento
    
    Args:
        **context: Contexto de Airflow
        
    Returns:
        dict: Métricas del modelo entrenado
    """
    logger.info("Iniciando entrenamiento del modelo...")
    
    ti = context['ti']
    processed_data_path = ti.xcom_pull(key='processed_data_path', task_ids='preprocess_data')
    
    # Cargar datos procesados
    df = pd.read_parquet(processed_data_path)
    
    # Preparar features y target
    exclude_cols = ['customer_id', 'product_id', 'compra_semanal']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['compra_semanal']
    
    logger.info(f"Features: {X.shape[1]}, Registros: {X.shape[0]}")
    logger.info(f"Distribución de clases - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
    
    # Split train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Inicializar MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("file:///opt/airflow/mlruns")
        mlflow.set_experiment("sodai_drinks_prediction")
        mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Optimizar hiperparámetros
    best_params, study = optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=30)
    
    # Entrenar modelo final con mejores hiperparámetros
    logger.info("Entrenando modelo final con mejores hiperparámetros...")
    
    model_type = best_params.pop('model')
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(**best_params)
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(**best_params)
    else:
        model = GradientBoostingClassifier(**best_params)
    
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'train_f1': f1_score(y_train, y_pred_train),
        'val_accuracy': accuracy_score(y_val, y_pred_val),
        'val_f1': f1_score(y_val, y_pred_val),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test),
        'test_recall': recall_score(y_test, y_pred_test),
        'test_f1': f1_score(y_test, y_pred_test),
        'test_roc_auc': roc_auc_score(y_test, y_proba_test)
    }
    
    logger.info(f"Métricas del modelo - Test F1: {metrics['test_f1']:.4f}, Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    
    # Generar gráficos SHAP
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    generate_shap_plots(model, X_sample, feature_cols, "/opt/airflow/diagrams")
    
    # Guardar modelo
    models_dir = Path("/opt/airflow/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "best_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Guardar metadata
    metadata = {
        'model_type': model_type,
        'best_params': best_params,
        'metrics': metrics,
        'feature_names': feature_cols,
        'training_date': datetime.now().isoformat()
    }
    
    with open(models_dir / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # MLflow logging
    if MLFLOW_AVAILABLE:
        mlflow.log_params({f"model_{k}": v for k, v in best_params.items()})
        mlflow.log_params({"model_type": model_type})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("/opt/airflow/diagrams/shap_summary.png")
        mlflow.log_artifact("/opt/airflow/diagrams/shap_feature_importance.png")
        
        # Registrar modelo
        if model_type == 'xgboost':
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        mlflow.end_run()
    
    # Pushear información
    ti.xcom_push(key='model_path', value=str(model_path))
    ti.xcom_push(key='model_metrics', value=metrics)
    ti.xcom_push(key='model_type', value=model_type)
    
    logger.info("Entrenamiento completado exitosamente")
    
    return {
        'status': 'success',
        'model_type': model_type,
        'metrics': metrics
    }
