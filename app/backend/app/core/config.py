"""
Configuración de la aplicación
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """
    Configuración de la aplicación usando variables de entorno
    """
    # Información general
    PROJECT_NAME: str = "SodAI Drinks Prediction API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API para predicción de compras de productos"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:7860",  # Gradio default port
        "http://localhost:3000",
        "http://frontend:7860",
        "*"  # En producción, especificar origins exactos
    ]
    
    # Rutas de modelos
    MODEL_PATH: str = "/app/models/best_model.pkl"
    MODEL_METADATA_PATH: str = "/app/models/model_metadata.json"
    MLFLOW_TRACKING_URI: str = "file:///app/mlruns"
    MLFLOW_MODEL_NAME: str = "sodai_drinks_model"
    
    # Configuración de la API
    API_V1_STR: str = "/api/v1"
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
