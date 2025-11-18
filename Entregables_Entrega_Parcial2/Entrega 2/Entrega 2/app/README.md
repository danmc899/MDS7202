# Aplicación Web - Gradio

## Descripción

Interfaz web interactiva con Gradio para obtener predicciones de probabilidad de compra semanal usando el modelo XGBoost entrenado en Airflow.

## Arquitectura

- **Backend FastAPI**: Sirve el modelo y expone endpoints REST
- **Frontend Gradio**: Interfaz de usuario para cargar datos y visualizar predicciones

## Flujo de Predicción

1. Usuario carga archivo CSV con datos de clientes y productos
2. Backend valida formato y genera features (mismo feature engineering que entrenamiento)
3. Modelo XGBoost predice probabilidad de compra para cada fila
4. Interfaz muestra tabla con predicciones y permite descargar resultados

## Consistencia Train-Serving

El procesamiento de features en `app/model.py` replica exactamente el código de `airflow/dags/tasks/data_processing.py`, garantizando que el modelo reciba features idénticas en inferencia y entrenamiento. Esto previene train-serving skew.

## Configuración

### Carga del Modelo
```python
model = joblib.load("models/xgboost_model.pkl")
```

El modelo se carga desde el directorio `models/`, que debe contener el pickle generado por Airflow.

### Dependencias
- `gradio>=4.0.0`: Interfaz web
- `fastapi`: Backend REST
- `xgboost`: Inferencia
- `pandas`, `numpy`: Procesamiento de datos

## Uso

1. Iniciar aplicación: `docker-compose up -d gradio-app`
2. Abrir navegador en `http://localhost:7860`
3. Cargar archivo CSV con columnas: customer_id, customer_region, producto_id, categoria_producto, precio_producto
4. Click en "Predecir" para obtener probabilidades
5. Descargar resultados como CSV

## Mejoras Futuras

- Validación de esquema con Pydantic
- Explicaciones SHAP para predicciones individuales
- A/B testing entre versiones del modelo
- Cache de features para clientes frecuentes
