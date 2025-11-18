"""
FastAPI Backend para SodAI Drinks
Proporciona endpoints para realizar predicciones de compra
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# Importar routers
from app.api import endpoints
from app.core.config import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API para predicciones de compra de productos",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(endpoints.router, prefix="/api/v1", tags=["predictions"])


@app.get("/")
async def root():
    """
    Endpoint raíz - Health check
    """
    return {
        "message": "SodAI Drinks Prediction API",
        "status": "online",
        "version": settings.VERSION,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Verifica que el servicio esté funcionando
    NOTA: Retorna 200 incluso si el modelo no está cargado (modo degradado)
    """
    try:
        # Verificar que el modelo esté cargado
        from app.core.model_loader import model_service
        model_loaded = model_service.model is not None
        
        # SIEMPRE retorna 200 para pasar healthcheck de Docker
        # El frontend puede verificar model_loaded para mostrar warnings
        return {
            "status": "healthy" if model_loaded else "degraded",
            "model_loaded": model_loaded,
            "model_type": model_service.model_type if model_loaded else None,
            "message": "Modelo cargado correctamente" if model_loaded else "Esperando modelo de Airflow. Ejecuta el DAG primero."
        }
    except Exception as e:
        logger.warning(f"Health check con warning: {str(e)}")
        # Retornar 200 de todas formas para no fallar el healthcheck
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": str(e),
            "message": "Servicio iniciado pero modelo no disponible. Ejecuta el DAG de Airflow primero."
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
