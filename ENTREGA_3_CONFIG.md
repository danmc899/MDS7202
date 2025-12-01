# Configuración y Contexto - Entrega 3 MDS7202

## 📋 Información del Proyecto

### Objetivo
Sistema de predicción de compras para **Sodai Drinks** usando Apache Airflow + MLflow + XGBoost.
- **Competencia:** Condabench
- **Métrica objetivo:** Score ≥ 0.45
- **Último score obtenido:** 0.38 (con 1 semana de datos)

---

## 🏗️ Arquitectura del Sistema

### Componentes Docker
```
Entregables_Entrega_3/
├── airflow/           # Pipeline principal
│   ├── docker-compose.yml
│   ├── dags/          # DAGs de Airflow
│   ├── mlruns/        # Tracking de MLflow
│   ├── models/        # Modelos entrenados
│   └── data/          # Datos procesados y predicciones
├── app/               # Aplicación Gradio (puerto 7860)
└── bonus/             # Sistema de recomendación
```

### Puertos
- **Airflow:** http://localhost:8080 (user: airflow, pass: airflow)
- **MLflow:** http://localhost:5001
- **Gradio App:** http://localhost:7860

---

## 🔧 Configuración Actual del Pipeline

### Parámetros de Extracción (data_extraction.py)

```python
SEMANAS_ENTRENAMIENTO = 2  # Usa 2 semanas de datos (óptimo según pruebas)
```

### Lógica de Semanas
- **Primera ejecución (sin modelo):** 
  - Entrena con semanas 2 a (última-1)
  - Ejemplo: semanas 49-52 de 2024
  
- **Ejecuciones siguientes (con modelo):**
  - Usa las últimas 4 semanas para predicción
  - Detecta drift y re-entrena si es necesario

### Parámetros de Entrenamiento (model_training.py)
```python
OPTUNA_TIMEOUT_SECONDS = 300  # 5 minutos de optimización
OPTUNA_N_TRIALS = 50          # Máximo 50 trials
```

### Formato de Submission
- **Archivo:** `submission_{week_id}_{timestamp}.zip`
- **Contenido:** CSV sin header, customer_id y product_id como enteros
- **Orden:** Por probabilidad descendente

---

## 📊 Resultados de Ejecuciones

### Ejecución 1 - Entrenamiento Inicial (4 semanas)
- **Week ID:** `2024-W52_4sem_train`
- **Datos:** 27,502 transacciones
- **Pares únicos:** 11,419
- **Predicciones:** 11,419
- **Archivo:** `submission_2024-W52_4sem_train_20251201_210826.zip`

### Ejecución 2 - Predicción (4 semanas)
- **Week ID:** `2025-W01_4sem_predict`
- **Datos:** 29,149 transacciones
- **Predicciones positivas:** 24,058
- **Archivo:** `submission_2025-W01_4sem_predict_20251201_211254.zip`
- **Estado:** ✅ Copiado a `submission.zip` en raíz del proyecto

---

## 🔄 Comandos Frecuentes

### Iniciar servicios
```bash
cd /home/javi02/MDS7202/Entregables_Entrega_3/airflow
docker compose up -d
```

### Ejecutar DAG manualmente
```bash
docker compose exec airflow-webserver airflow dags trigger sodai_drinks_production_pipeline
```

### Ver logs del DAG
```bash
docker compose logs -f airflow-worker
```

### Lanzar MLflow UI
```bash
docker compose exec -d airflow-webserver mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri file:///opt/airflow/mlruns
```

### Copiar submission a raíz
```bash
cp airflow/data/predictions/submissions/submission_XXXX.zip ../submission.zip
```

---

## 📁 Estructura de Archivos de Datos

### Datos crudos
```
airflow/data/raw/transacciones.parquet  # Dataset completo
```

### Datos procesados
```
airflow/data/processed/
├── transacciones_subsample_{week_id}.parquet  # Subsample extraído
└── features_target_{week_id}.parquet          # Features procesadas
```

### Predicciones
```
airflow/data/predictions/
├── stats/
│   └── prediction_stats_{week_id}_{timestamp}.json
└── submissions/
    ├── submission_{week_id}_{timestamp}.csv
    └── submission_{week_id}_{timestamp}.zip
```

---

## 🎯 Para la Próxima Semana

### Si quieres simular nueva semana:
1. El sistema detecta que ya existe un modelo
2. Usa las últimas 4 semanas disponibles
3. Evalúa drift y decide si re-entrenar
4. Genera nuevas predicciones

### Si quieres re-entrenar desde cero:
```bash
# Eliminar modelo existente
sudo rm airflow/models/best_model.pkl
sudo rm airflow/models/model_metadata.json

# Ejecutar DAG - entrenará desde cero
docker compose exec airflow-webserver airflow dags trigger sodai_drinks_production_pipeline
```

### Si agregas nuevos datos:
1. Agregar al archivo `transacciones.parquet`
2. Ejecutar el DAG
3. El sistema detectará las nuevas semanas automáticamente

---

## 🐛 Troubleshooting

### Error de permisos
```bash
sudo chown -R 50000:0 airflow/dags/ airflow/models/ airflow/data/ airflow/mlruns/
```

### DAG no aparece en Airflow
```bash
docker compose exec airflow-webserver airflow dags list
docker compose restart airflow-scheduler
```

### MLflow no muestra experimentos
Verificar que el tracking URI apunte a `/opt/airflow/mlruns`

---

## 📝 Notas Importantes

1. **UID de Airflow:** Los archivos deben tener owner `50000:0` para que el container pueda escribir
2. **Week ID formato:** `YYYY-WXX_Nsem_{train|predict}` (ej: `2024-W52_4sem_train`)
3. **Modelo guardado en:** `airflow/models/best_model.pkl`
4. **App Gradio:** Monta `airflow/models/` - cualquier modelo nuevo estará disponible automáticamente

---

*Última actualización: 1 de Diciembre de 2025*
*Score Condabench: Pendiente de evaluar con 4 semanas*
