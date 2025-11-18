# 🚀 Backend FastAPI - SodAI Drinks

API REST para realizar predicciones de compra de productos usando el modelo entrenado.

## 📋 Descripción

Backend desarrollado con **FastAPI** que expone endpoints para:

- ✅ Predicciones individuales cliente-producto
- ✅ Predicciones en lote
- ✅ Recomendaciones de productos para clientes
- ✅ Información del modelo cargado
- ✅ Health checks
- ✅ Recarga del modelo en caliente

## 🏗️ Arquitectura

```
app/
├── __init__.py
├── main.py                    # Aplicación principal FastAPI
├── api/
│   ├── __init__.py
│   └── endpoints.py           # Endpoints de la API
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuración
│   └── model_loader.py        # Cargador de modelos (soporta MLflow)
└── schemas/
    ├── __init__.py
    └── prediction.py          # Schemas de Pydantic
```

## 🚀 Instalación y Ejecución

### Opción 1: Docker (Recomendado)

```bash
# Desde el directorio backend/
docker build -t sodai-backend .
docker run -p 8000:8000 -v $(pwd)/../../airflow/models:/app/models sodai-backend
```

### Opción 2: Local (Desarrollo)

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 Endpoints Disponibles

### Health Check

```http
GET /
GET /health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "random_forest"
}
```

### Predicción Individual

```http
POST /api/v1/predict
```

**Request Body:**
```json
{
  "customer_id": "C12345",
  "product_id": "P6789",
  "cp_total_items": 15.0,
  "cp_avg_items": 3.0,
  "total_items": 100.0,
  "avg_items": 5.0
}
```

**Respuesta:**
```json
{
  "customer_id": "C12345",
  "product_id": "P6789",
  "will_purchase": true,
  "probability": 0.85,
  "model_type": "random_forest"
}
```

### Predicción en Lote

```http
POST /api/v1/predict/batch
```

**Request Body:**
```json
{
  "predictions": [
    {
      "customer_id": "C12345",
      "product_id": "P6789"
    },
    {
      "customer_id": "C12345",
      "product_id": "P6790"
    }
  ],
  "only_positive": true,
  "sort_by_probability": true
}
```

### Recomendaciones para Cliente

```http
GET /api/v1/predict/customer/{customer_id}?top_n=10
```

### Información del Modelo

```http
GET /api/v1/model/info
```

### Recargar Modelo

```http
POST /api/v1/model/reload
```

## 📚 Documentación Interactiva

Una vez ejecutando, accede a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Configuración

### Variables de Entorno

Crea un archivo `.env` en el directorio backend:

```bash
# Información del proyecto
PROJECT_NAME="SodAI Drinks Prediction API"
VERSION="1.0.0"

# Rutas de modelos
MODEL_PATH="/app/models/best_model.pkl"
MODEL_METADATA_PATH="/app/models/model_metadata.json"
MLFLOW_TRACKING_URI="file:///app/mlruns"

# CORS
ALLOWED_ORIGINS='["http://localhost:7860", "http://frontend:7860"]'
```

## 🧪 Testing

```bash
# Instalar dependencias de testing
pip install pytest httpx

# Ejecutar tests
pytest tests/
```

### Ejemplo de test manual con curl

```bash
# Health check
curl http://localhost:8000/health

# Predicción
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C12345",
    "product_id": "P6789",
    "total_items": 100.0,
    "avg_items": 5.0
  }'
```

## 🔗 Integración con MLflow (BONUS)

El backend puede cargar modelos desde MLflow automáticamente:

1. El servicio intenta cargar el último modelo desde MLflow
2. Si falla, carga desde archivo pickle como fallback
3. Configura `MLFLOW_TRACKING_URI` para usar MLflow

## 📊 Monitoreo

### Logs

Los logs se muestran en stdout con formato:

```
2024-10-28 10:30:45 - app.core.model_loader - INFO - Modelo cargado exitosamente
2024-10-28 10:30:50 - app.api.endpoints - INFO - Predicción solicitada para customer_id=C12345
```

### Métricas

Considera agregar:
- Prometheus para métricas
- Sentry para error tracking
- Elasticsearch para logs centralizados

## 🐛 Troubleshooting

### Error: "Modelo no está cargado"
- Verifica que existe `/app/models/best_model.pkl`
- Verifica permisos de lectura del directorio
- Revisa logs para detalles del error

### Error: "Feature X no encontrada"
- El modelo espera ciertas features específicas
- Consulta `/api/v1/model/info` para ver features requeridas
- Asegúrate de enviar todas las features necesarias

### Error de CORS
- Agrega el origin del frontend a `ALLOWED_ORIGINS`
- Reinicia el servidor después de cambiar configuración

## 👥 Autores

- **Javier Pinochet**
- **Daniel Muñoz**

**Curso**: MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos

---

Para más información, consulta la documentación de [FastAPI](https://fastapi.tiangolo.com/).
