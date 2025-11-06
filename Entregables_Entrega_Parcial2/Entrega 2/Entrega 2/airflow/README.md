# 🚀 Pipeline Productivo con Apache Airflow - SodAI Drinks

Este directorio contiene el pipeline productivo completo para el sistema de predicción de compras de **SodAI Drinks**, implementado con Apache Airflow.

## 📋 Descripción

El pipeline orquesta todo el flujo de trabajo del sistema predictivo, desde la extracción de datos hasta la generación de predicciones, incluyendo:

- ✅ Extracción automática de datos
- ✅ Limpieza y transformación de datos (IQR, StandardScaler, MinMaxScaler)
- ✅ Detección de drift en los datos (BONUS)
- ✅ Reentrenamiento inteligente del modelo
- ✅ Optimización de hiperparámetros con Optuna
- ✅ Tracking con MLflow (BONUS)
- ✅ Interpretabilidad con SHAP
- ✅ Generación de predicciones para la próxima semana

## 🏗️ Arquitectura del DAG

```
extract_data → preprocess_data → detect_drift → decide_retraining
                                                      ↓
                                                   [Branch]
                                                  ↙         ↘
                                           train_model    skip_training
                                                  ↘         ↙
                                              join_after_training
                                                      ↓
                                            generate_predictions
                                                      ↓
                                             pipeline_completed
```

### Flujo de Decisión

1. **Extracción de datos**: Lee archivos parquet (transacciones, clientes, productos)
2. **Preprocessing**: Aplica feature engineering, IQR filtering y scaling
3. **Detección de drift**: Usa KS-test y PSI para detectar cambios en distribuciones
4. **Decisión de reentrenamiento**: 
   - Si hay drift → Reentrenar
   - Si es primera ejecución → Entrenar
   - Si no hay drift → Usar modelo existente
5. **Generación de predicciones**: Predice compras para la próxima semana

## 📁 Estructura de Archivos

```
airflow/
├── dags/
│   ├── pipeline_productivo.py      # DAG principal
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── data_extraction.py      # Extracción de datos
│   │   ├── data_preprocessing.py   # Preprocessing y feature engineering
│   │   ├── drift_detection.py      # Detección de drift (BONUS)
│   │   ├── model_training.py       # Entrenamiento con Optuna + MLflow
│   │   └── prediction_generation.py # Generación de predicciones
│   └── utils/
│       ├── __init__.py
│       └── helpers.py               # Funciones auxiliares
├── data/
│   ├── raw/                         # Datos originales (entrada)
│   ├── processed/                   # Datos procesados
│   └── predictions/                 # Predicciones generadas
├── models/                          # Modelos entrenados
├── mlruns/                          # Experimentos de MLflow
├── diagrams/                        # Gráficos SHAP y visualizaciones
├── logs/                            # Logs de Airflow
├── config/                          # Configuraciones
├── plugins/                         # Plugins personalizados
├── requirements.txt                 # Dependencias de Python
├── Dockerfile                       # Imagen de Docker
├── docker-compose.yml               # Orquestación de servicios
└── README.md                        # Este archivo
```

## 🚀 Instalación y Ejecución

### Prerequisitos

- Docker y Docker Compose instalados
- Al menos 4GB de RAM disponible
- Archivos de datos en `airflow/data/raw/`:
  - `transacciones.parquet`
  - `clientes.parquet`
  - `productos.parquet`

### Pasos para Ejecutar

1. **Navegar al directorio de Airflow:**
   ```bash
   cd "Entrega 2/airflow"
   ```

2. **Inicializar Airflow:**
   ```bash
   echo -e "AIRFLOW_UID=$(id -u)" > .env
   ```

3. **Levantar los servicios:**
   ```bash
   docker-compose up -d
   ```

4. **Verificar que los servicios estén corriendo:**
   ```bash
   docker-compose ps
   ```

5. **Acceder a la interfaz web de Airflow:**
   - URL: http://localhost:8080
   - Usuario: `airflow`
   - Contraseña: `airflow`

6. **Acceder a MLflow UI (BONUS):**
   - URL: http://localhost:5001

7. **Activar el DAG:**
   - En la interfaz de Airflow, busca el DAG `sodai_drinks_production_pipeline`
   - Activa el toggle para habilitarlo
   - Puedes ejecutarlo manualmente con el botón "▶️" o esperar la ejecución programada (semanal)

### Colocar Datos de Entrada

Los datos deben estar en el directorio `airflow/data/raw/`. Si estás usando Docker, puedes copiarlos así:

```bash
# Copiar desde el directorio principal de "Entrega 2"
cp transacciones.parquet clientes.parquet productos.parquet airflow/data/raw/
```

### Detener los Servicios

```bash
docker-compose down
```

Para eliminar también los volúmenes (datos persistentes):
```bash
docker-compose down -v
```

## 📊 Monitoreo y Logs

### Ver Logs en Tiempo Real

```bash
# Todos los servicios
docker-compose logs -f

# Solo el scheduler
docker-compose logs -f airflow-scheduler

# Solo el webserver
docker-compose logs -f airflow-webserver
```

### Verificar Ejecución del DAG

1. En la interfaz web de Airflow (http://localhost:8080)
2. Click en el DAG `sodai_drinks_production_pipeline`
3. Ver el Graph View para visualizar el flujo
4. Ver el Tree View para ver ejecuciones históricas
5. Click en cada tarea para ver logs específicos

## 🔧 Configuración

### Variables de Entorno

Puedes modificar el archivo `.env` para configurar:

```bash
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
```

### Modificar Schedule

En `dags/pipeline_productivo.py`, línea del `schedule_interval`:

```python
schedule_interval='@weekly',  # Ejecutar semanalmente
# Otras opciones:
# '@daily'     - Diariamente
# '@hourly'    - Cada hora
# '0 0 * * 0'  - Cron expression (Domingos a medianoche)
```

## 📈 Métricas y Tracking (BONUS - MLflow)

### Visualizar Experimentos

1. Accede a http://localhost:5001
2. Navega al experimento `sodai_drinks_prediction`
3. Compara diferentes runs
4. Visualiza métricas, parámetros y artefactos

### Métricas Tracked

- Accuracy (train, val, test)
- Precision, Recall, F1-Score
- ROC-AUC
- Hiperparámetros del modelo
- Gráficos SHAP
- Metadata del modelo

## 🎯 Detección de Drift (BONUS)

El sistema detecta drift usando:

1. **Kolmogorov-Smirnov Test**: Compara distribuciones estadísticamente
2. **Population Stability Index (PSI)**: Mide cambios en proporciones

**Umbrales:**
- PSI > 0.1: Drift moderado
- PSI > 0.25: Drift severo
- KS p-value < 0.05: Distribuciones significativamente diferentes

**Acción:** Si se detecta drift, el modelo se reentrena automáticamente.

## 📤 Salidas del Pipeline

### Predicciones

Ubicación: `airflow/data/predictions/`

Archivos generados:
- `predictions_YYYYMMDD_HHMMSS.parquet`: Todas las predicciones
- `predictions_positive_YYYYMMDD_HHMMSS.csv`: Solo predicciones positivas
- `prediction_stats_YYYYMMDD_HHMMSS.json`: Estadísticas agregadas

### Modelos

Ubicación: `airflow/models/`

Archivos:
- `best_model.pkl`: Modelo entrenado
- `model_metadata.json`: Metadata (tipo, hiperparámetros, métricas)
- `standard_scaler.pkl`: Scaler para features
- `minmax_scaler.pkl`: Scaler MinMax

### Gráficos

Ubicación: `airflow/diagrams/`

- `shap_summary.png`: Resumen de importancia de features
- `shap_feature_importance.png`: Importancia agregada de features

## 🐛 Troubleshooting

### Error: "No module named 'airflow'"
- Asegúrate de estar usando Docker. El código está diseñado para ejecutarse en contenedores.

### Error: "Permission denied"
- Ejecuta: `chmod -R 777 airflow/logs airflow/dags`

### El DAG no aparece en la interfaz
- Verifica que el archivo esté en `airflow/dags/`
- Revisa los logs: `docker-compose logs airflow-scheduler`
- Espera 30 segundos, Airflow escanea DAGs periódicamente

### MLflow no funciona
- Verifica que el servicio esté corriendo: `docker-compose ps`
- Accede a http://localhost:5001
- Revisa logs: `docker-compose logs mlflow-server`

## 📝 Supuestos y Consideraciones

1. **Datos de entrada**: Se asume que los archivos parquet aparecen "mágicamente" en `data/raw/`
2. **Estructura de datos**: Transacciones deben tener columnas: `customer_id`, `product_id`, `order_id`, `purchase_date`, `items`
3. **Semana de predicción**: Se predice para la semana siguiente a la más reciente en los datos
4. **Primera ejecución**: En la primera ejecución siempre se entrena el modelo
5. **Reentrenamiento**: Solo ocurre si hay drift o es primera ejecución

## 👥 Autores

- **Javier Pinochet**
- **Daniel Muñoz**

**Curso**: MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos  
**Institución**: Universidad de Chile

---

Para más información, consulta la documentación oficial de [Apache Airflow](https://airflow.apache.org/docs/) y [MLflow](https://mlflow.org/docs/latest/index.html).
