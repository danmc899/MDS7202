"""
Model Training Task
Entrena únicamente XGBoost con optimización de hiperparámetros usando Optuna
Tracking completo con MLflow incluyendo SHAP artifacts
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import xgboost as xgb

# Optuna para optimización
import optuna
from optuna.samplers import TPESampler

# MLflow para tracking
import mlflow
import mlflow.xgboost

# SHAP para interpretabilidad
import shap
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def optimize_xgboost_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Optimiza hiperparámetros de XGBoost usando Optuna
    
    Hiperparámetros optimizados:
    - max_depth: Profundidad máxima de los árboles
    - learning_rate: Tasa de aprendizaje
    - subsample: Proporción de muestras por árbol
    - colsample_bytree: Proporción de features por árbol
    - min_child_weight: Suma mínima de pesos en nodo hijo
    - gamma: Reducción mínima de pérdida para hacer partición
    - n_estimators: Número de árboles (rondas de boosting)
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        n_trials: Número de trials para Optuna
        
    Returns:
        tuple: (mejores_parametros, estudio_optuna)
    """
    logger.info(f"🔍 Iniciando optimización de hiperparámetros con {n_trials} trials...")
    
    def objective(trial):
        # Sugerir hiperparámetros
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'tree_method': 'hist'
        }
        
        # Entrenar modelo
        try:
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluar en validación
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
            
        except Exception as e:
            logger.warning(f"Error al entrenar modelo: {e}")
            raise optuna.exceptions.TrialPruned()
        
        return f1
    
    # Crear estudio de Optuna con sampler TPE
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_optimization',
        sampler=TPESampler(seed=42)
    )
    
    # Optimizar
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"✅ Mejor F1-Score (validación): {study.best_value:.4f}")
    logger.info(f"📊 Mejores hiperparámetros encontrados:")
    for param, value in study.best_params.items():
        logger.info(f"   - {param}: {value}")
    
    return study.best_params, study


def generate_shap_plots(model, X_sample, feature_names, output_dir):
    """
    Genera gráficos de interpretabilidad con SHAP
    
    Args:
        model: Modelo XGBoost entrenado
        X_sample: Muestra de datos para SHAP
        feature_names: Nombres de features
        output_dir: Directorio para guardar gráficos
        
    Returns:
        list: Rutas de los gráficos generados
    """
    logger.info("📈 Generando gráficos de interpretabilidad con SHAP...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = []
    
    try:
        # Crear explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Si shap_values es una lista (clasificación binaria), tomar valores de clase positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # 1. Summary plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=20)
        plt.title('SHAP Summary Plot - Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        summary_path = output_dir / "shap_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(summary_path))
        logger.info(f"   ✓ Summary plot guardado: {summary_path}")
        
        # 2. Feature importance plot (bar)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        importance_path = output_dir / "shap_feature_importance.png"
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(importance_path))
        logger.info(f"   ✓ Feature importance plot guardado: {importance_path}")
        
        # 3. Waterfall plot para primera predicción
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_sample.iloc[0],
            feature_names=feature_names
        ), show=False, max_display=15)
        plt.title('SHAP Waterfall Plot - Sample Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        waterfall_path = output_dir / "shap_waterfall.png"
        plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(waterfall_path))
        logger.info(f"   ✓ Waterfall plot guardado: {waterfall_path}")
        
        logger.info("✅ Gráficos SHAP generados exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error al generar gráficos SHAP: {e}")
        raise
    
    return plot_paths


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """
    Genera matriz de confusión
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✓ Confusion matrix guardada: {cm_path}")
    return str(cm_path)


def train_model(**context):
    """
    Tarea principal de entrenamiento con XGBoost
    
    Args:
        **context: Contexto de Airflow
        
    Returns:
        dict: Métricas del modelo entrenado
    """
    logger.info("🚀 Iniciando entrenamiento del modelo XGBoost...")
    
    ti = context['ti']
    processed_data_path = ti.xcom_pull(key='processed_data_path', task_ids='process_data')
    
    # Cargar datos procesados
    df = pd.read_parquet(processed_data_path)
    
    # Preparar features y target
    # IMPORTANTE: Excluir columnas que podrían causar data leakage
    exclude_cols = [
        'customer_id', 'product_id', 'compra_semanal',
        'items',
        'purchase_week',
        'order_id',
        'purchase_date'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['compra_semanal']
    
    logger.info(f"📊 Dataset cargado:")
    logger.info(f"   - Features: {X.shape[1]}")
    logger.info(f"   - Registros: {X.shape[0]}")
    logger.info(f"   - Distribución de clases - Clase 0: {(y==0).sum()}, Clase 1: {(y==1).sum()}")
    
    # Split train/val/test (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"📂 Splits creados:")
    logger.info(f"   - Train: {X_train.shape[0]} muestras")
    logger.info(f"   - Validation: {X_val.shape[0]} muestras")
    logger.info(f"   - Test: {X_test.shape[0]} muestras")
    
    # Configurar MLflow (usar file-based tracking)
    mlflow.set_tracking_uri("file:///opt/airflow/mlruns")
    mlflow.set_experiment("sodai_drinks_xgboost")
    
    with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Registrar datasets en MLflow
        logger.info("📊 Registrando datasets en MLflow...")
        
        # Crear datasets temporales para logging
        datasets_dir = Path("/opt/airflow/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar datasets como parquet
        train_X_path = datasets_dir / "X_train.parquet"
        train_y_path = datasets_dir / "y_train.parquet"
        val_X_path = datasets_dir / "X_val.parquet"
        val_y_path = datasets_dir / "y_val.parquet"
        test_X_path = datasets_dir / "X_test.parquet"
        test_y_path = datasets_dir / "y_test.parquet"
        
        X_train.to_parquet(train_X_path)
        pd.DataFrame(y_train).to_parquet(train_y_path)
        X_val.to_parquet(val_X_path)
        pd.DataFrame(y_val).to_parquet(val_y_path)
        X_test.to_parquet(test_X_path)
        pd.DataFrame(y_test).to_parquet(test_y_path)
        
        # Registrar datasets en MLflow
        from mlflow.data.pandas_dataset import from_pandas
        
        # Dataset de entrenamiento (X + y combinados)
        train_dataset = pd.concat([X_train, pd.Series(y_train, name='compra_semanal')], axis=1)
        mlflow_train_dataset = from_pandas(
            train_dataset,
            source=str(train_X_path),
            name="training_data",
            targets="compra_semanal"
        )
        mlflow.log_input(mlflow_train_dataset, context="training")
        
        # Dataset de validación
        val_dataset = pd.concat([X_val, pd.Series(y_val, name='compra_semanal')], axis=1)
        mlflow_val_dataset = from_pandas(
            val_dataset,
            source=str(val_X_path),
            name="validation_data",
            targets="compra_semanal"
        )
        mlflow.log_input(mlflow_val_dataset, context="validation")
        
        # Dataset de test
        test_dataset = pd.concat([X_test, pd.Series(y_test, name='compra_semanal')], axis=1)
        mlflow_test_dataset = from_pandas(
            test_dataset,
            source=str(test_X_path),
            name="test_data",
            targets="compra_semanal"
        )
        mlflow.log_input(mlflow_test_dataset, context="testing")
        
        logger.info("✅ Datasets registrados en MLflow")
        
        # Optimizar hiperparámetros
        best_params, study = optimize_xgboost_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=50
        )
        
        # Entrenar modelo final con mejores hiperparámetros
        logger.info("🎯 Entrenando modelo final con mejores hiperparámetros...")
        
        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
            verbose=True
        )
        
        # Evaluar solo en TEST
        logger.info("📊 Evaluando modelo en conjunto de TEST...")
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas de TEST
        test_metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, y_proba_test)
        }
        
        logger.info(f"🎯 Métricas de TEST:")
        logger.info(f"   - Accuracy: {test_metrics['test_accuracy']:.4f}")
        logger.info(f"   - Precision: {test_metrics['test_precision']:.4f}")
        logger.info(f"   - Recall: {test_metrics['test_recall']:.4f}")
        logger.info(f"   - F1-Score: {test_metrics['test_f1']:.4f}")
        logger.info(f"   - ROC-AUC: {test_metrics['test_roc_auc']:.4f}")
        
        # Generar gráficos de interpretabilidad
        diagrams_dir = Path("/opt/airflow/diagrams")
        diagrams_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample para SHAP (100 muestras aleatorias del test)
        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
        shap_plots = generate_shap_plots(model, X_sample, feature_cols, diagrams_dir)
        
        # Generar confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred_test, diagrams_dir)
        
        # Registrar artifacts en MLflow
        logger.info("📦 Registrando artifacts en MLflow...")
        for plot_path_str in shap_plots:
            plot_path = Path(plot_path_str)
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), artifact_path="shap_plots")
                logger.info(f"   ✅ {plot_path.name} registrado")
        
        if cm_path:
            cm_path_obj = Path(cm_path)
            if cm_path_obj.exists():
                mlflow.log_artifact(str(cm_path_obj), artifact_path="plots")
                logger.info(f"   ✅ Confusion matrix registrada")
        
        # Guardar modelo con lógica de comparación
        models_dir = Path("/opt/airflow/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "best_model.pkl"
        metadata_path = models_dir / "model_metadata.json"
        
        # Verificar si existe modelo previo
        existing_model_exists = model_path.exists() and metadata_path.exists()
        should_save_model = True
        comparison_result = "first_model"
        
        if existing_model_exists:
            logger.info("🔍 Modelo existente detectado. Comparando rendimiento...")
            
            # Cargar metadata del modelo anterior
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            
            existing_f1 = existing_metadata['test_metrics']['f1']
            new_f1 = test_metrics['f1']
            
            logger.info(f"   📊 F1 Score Existente: {existing_f1:.4f}")
            logger.info(f"   📊 F1 Score Nuevo: {new_f1:.4f}")
            
            # Umbral de mejora mínima (1%)
            improvement_threshold = 0.01
            
            if new_f1 > existing_f1 + improvement_threshold:
                logger.info(f"   ✅ Nuevo modelo ES MEJOR (mejora > {improvement_threshold:.2%})")
                should_save_model = True
                comparison_result = "new_model_better"
            elif new_f1 >= existing_f1 - improvement_threshold:
                logger.info(f"   ⚖️  Nuevo modelo SIMILAR (diferencia < {improvement_threshold:.2%})")
                should_save_model = True
                comparison_result = "new_model_similar"
            else:
                logger.warning(f"   ❌ Nuevo modelo PEOR (degradación > {improvement_threshold:.2%})")
                logger.warning("   ⏭️  Manteniendo modelo existente.")
                should_save_model = False
                comparison_result = "keep_existing"
        
        if should_save_model:
            logger.info("💾 Guardando modelo y metadata...")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Guardar metadata
            metadata = {
                'model_type': 'xgboost',
                'best_params': best_params,
                'test_metrics': test_metrics,
                'feature_names': feature_cols,
                'training_date': datetime.now().isoformat(),
                'n_features': len(feature_cols),
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'comparison_result': comparison_result
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("   ✅ Modelo y metadata guardados exitosamente")
        else:
            logger.info("   ⏭️  Saltando guardado de modelo (modelo existente es superior)")
        
        # Logging en MLflow (solo si se guardó el modelo)
        if should_save_model:
            logger.info("📤 Registrando resultados en MLflow...")
            
            # Log de hiperparámetros optimizados
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("optimization_trials", 50)
            mlflow.log_param("comparison_result", comparison_result)
            
            # Log de métricas de TEST únicamente
            mlflow.log_metrics(test_metrics)
            
            # Log del modelo
            mlflow.xgboost.log_model(model, "model")
            
            # Log de artifacts (gráficos SHAP y confusion matrix)
            for plot_path in shap_plots:
                mlflow.log_artifact(plot_path, artifact_path="shap_plots")
            mlflow.log_artifact(cm_path, artifact_path="evaluation")
            mlflow.log_artifact(str(metadata_path), artifact_path="metadata")
            
            # Log de importancia de features
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_path = diagrams_dir / "feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path), artifact_path="metadata")
            
            logger.info("✅ Resultados registrados en MLflow exitosamente")
        else:
            logger.info("⏭️  Saltando registro en MLflow (modelo existente conservado)")
        
        # Pushear información (siempre, para tracking)
        ti.xcom_push(key='model_path', value=str(model_path))
        ti.xcom_push(key='model_metrics', value=test_metrics)
        ti.xcom_push(key='model_type', value='xgboost')
        ti.xcom_push(key='best_params', value=best_params)
        ti.xcom_push(key='model_saved', value=should_save_model)
        ti.xcom_push(key='comparison_result', value=comparison_result)
    
    logger.info("✅ Entrenamiento completado exitosamente")
    
    return {
        'status': 'success',
        'model_type': 'xgboost',
        'test_metrics': test_metrics,
        'best_params': best_params
    }
