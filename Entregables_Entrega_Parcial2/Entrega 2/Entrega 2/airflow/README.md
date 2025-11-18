# Pipeline Airflow - MLOps

## Descripción

Pipeline de MLOps que orquesta el flujo completo desde ingesta de datos hasta entrenamiento de modelos XGBoost, incluyendo detección de drift, evaluación con SHAP, y registro en MLflow.

## Arquitectura

El DAG ejecuta las siguientes tareas en secuencia:

1. **extract_data**: Lee archivos parquet de transacciones, clientes y productos
2. **process_data**: Transforma datos crudos en features engineered, eliminando columnas con data leakage (items, purchase_week, order_id, purchase_date)
3. **detect_drift**: Aplica Kolmogorov-Smirnov test y PSI para detectar cambios en distribuciones de features
4. **train_model**: Optimiza hiperparámetros con Optuna (50 trials) y entrena XGBoost
5. **sync_to_recsys**: Notifica al sistema de recomendaciones sobre datos actualizados

## Features Clave

### Prevención de Data Leakage
Columnas que solo existen para compra_semanal=1 se eliminan antes del merge para evitar que el modelo aprenda patrones triviales.

### Detección de Drift
- **KS Test**: Compara distribuciones de features numéricas (threshold: p-value < 0.05)
- **PSI**: Mide cambios en features categóricas (threshold: PSI > 0.2)
- Primer run establece baseline de referencia

### MLflow Integration
Cada run registra:
- Hiperparámetros optimizados
- Métricas de test (accuracy, precision, recall, F1, ROC-AUC)
- Modelo serializado (pickle)
- Artifacts: SHAP plots (summary, waterfall, feature importance), confusion matrix

Se logro un recall de 0.82 en el conjunto de test, indicando buena capacidad del modelo para identificar compras.

### Interpretabilidad SHAP
Tres tipos de visualizaciones:
- **Summary Plot**: Importancia global de features
- **Waterfall Plot**: Explicación de predicción individual
- **Feature Importance**: Ranking de features por impacto

Para la figura obtenida, se tiene que el tamaño del producto y el número de marca son las características más influyentes en la predicción del modelo.

## Configuración

### Reintentos
Cada tarea tiene `retry=3` con `retry_delay=5min` para manejar fallos transitorios.

### MLflow
```python
mlflow.set_tracking_uri("sqlite:///opt/airflow/mlruns/mlflow.db")
mlflow.set_experiment("sodai_drinks_xgboost")
```

## Monitoreo

- **Airflow UI**: Estado de tareas, logs, Gantt chart de duración
- **MLflow UI** (puerto 5001): Histórico de experimentos, comparación de métricas, artifacts

## Mejoras Futuras

- Migrar a CeleryExecutor para paralelizar tareas independientes
- Data quality checks con Great Expectations
- Modelo de fallback para servir predicciones si entrenamiento falla
