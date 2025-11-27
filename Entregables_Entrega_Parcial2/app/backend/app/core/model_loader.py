"""
Model Loader Service
Carga y gestiona el modelo de ML para predicciones
Soporta carga desde pickle o MLflow
"""
import pickle
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """
    Servicio para cargar y usar el modelo de predicción
    """
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.feature_names = []
        self.metadata = {}
        self.load_model()
    
    def load_model(self):
        """
        Carga el modelo desde disco o MLflow
        Si no existe, el servicio sigue funcionando en modo degradado
        """
        try:
            # Intentar cargar con MLflow primero (BONUS)
            if self._try_load_from_mlflow():
                logger.info("✅ Modelo cargado exitosamente desde MLflow")
                return
            
            # Fallback: cargar desde pickle
            self._load_from_pickle()
            logger.info("✅ Modelo cargado exitosamente desde pickle")
            
        except FileNotFoundError as e:
            logger.warning(f"⚠️ Modelo no encontrado: {str(e)}")
            logger.warning("El servicio funcionará en modo degradado hasta que el modelo esté disponible")
            logger.warning("Ejecuta el DAG de Airflow 'sodai_drinks_production_pipeline' para entrenar el modelo")
            # No levantamos excepción, solo dejamos model=None
        except Exception as e:
            logger.error(f"❌ Error al cargar modelo: {str(e)}")
            logger.warning("El servicio funcionará en modo degradado")
    
    def _try_load_from_mlflow(self) -> bool:
        """
        Intenta cargar el modelo desde MLflow
        
        Returns:
            True si se cargó exitosamente, False en caso contrario
        """
        try:
            import mlflow
            import mlflow.pyfunc
            
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            
            # Buscar el último modelo registrado
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            
            if not experiments:
                logger.warning("No se encontraron experimentos en MLflow")
                return False
            
            # Obtener el último run del experimento
            experiment_id = experiments[0].experiment_id
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not runs:
                logger.warning("No se encontraron runs en MLflow")
                return False
            
            run_id = runs[0].info.run_id
            
            # Cargar modelo
            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Cargar metadata
            self.model_type = runs[0].data.params.get("model_type", "unknown")
            self.metadata = {
                "run_id": run_id,
                "model_type": self.model_type,
                "metrics": dict(runs[0].data.metrics)
            }
            
            logger.info(f"Modelo cargado desde MLflow - Run ID: {run_id}")
            return True
            
        except ImportError:
            logger.warning("MLflow no está disponible, usando método de carga alternativo")
            return False
        except Exception as e:
            logger.warning(f"No se pudo cargar desde MLflow: {str(e)}")
            return False
    
    def _load_from_pickle(self):
        """
        Carga el modelo desde archivo pickle
        """
        model_path = Path(settings.MODEL_PATH)
        metadata_path = Path(settings.MODEL_METADATA_PATH)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        
        # Cargar modelo
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Cargar metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.model_type = self.metadata.get("model_type", "unknown")
                self.feature_names = self.metadata.get("feature_names", [])
        else:
            logger.warning("Metadata del modelo no encontrada")
            self.model_type = "unknown"
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza predicciones sobre los datos proporcionados
        
        Args:
            data: DataFrame con los features necesarios
            
        Returns:
            Tuple de (predicciones, probabilidades)
        """
        if self.model is None:
            raise ValueError("Modelo no está cargado")
        
        # Preparar features
        X = self._prepare_features(data)
        
        # Predecir
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]  # Probabilidad de clase positiva
        
        return predictions, probabilities
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara los features para el modelo
        
        Args:
            data: DataFrame con datos crudos
            
        Returns:
            DataFrame con features preparados
        """
        # Si tenemos feature_names definidos, usar solo esos
        if self.feature_names:
            # Verificar que todas las features necesarias estén presentes
            missing_features = set(self.feature_names) - set(data.columns)
            if missing_features:
                logger.warning(f"Features faltantes: {missing_features}")
                # Agregar features faltantes con valores por defecto
                for feature in missing_features:
                    data[feature] = 0
            
            # Seleccionar solo las features necesarias en el orden correcto
            X = data[self.feature_names]
        else:
            # Usar todas las columnas excepto IDs
            exclude_cols = ['customer_id', 'product_id', 'purchased']
            X = data[[col for col in data.columns if col not in exclude_cols]]
        
        return X
    
    def predict_for_customer(self, customer_id: str, top_n: int = 10) -> List[dict]:
        """
        Genera predicciones para un cliente específico
        
        Args:
            customer_id: ID del cliente
            top_n: Número de recomendaciones a retornar
            
        Returns:
            Lista de predicciones ordenadas por probabilidad
        """
        # Nota: En producción, esto debería consultar una base de datos
        # Por ahora, retornamos un ejemplo
        logger.warning("predict_for_customer requiere implementación con datos reales")
        
        return [
            {
                "product_id": f"prod_{i}",
                "probability": 0.9 - (i * 0.05),
                "will_purchase": True
            }
            for i in range(top_n)
        ]
    
    def get_model_info(self) -> dict:
        """
        Obtiene información sobre el modelo cargado
        
        Returns:
            Diccionario con información del modelo
        """
        info = {
            "model_loaded": self.model is not None,
            "model_type": self.model_type,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names[:10],  # Mostrar solo las primeras 10
        }
        
        # Agregar métricas si están disponibles
        if "metrics" in self.metadata:
            info["metrics"] = self.metadata["metrics"]
        
        return info


# Instancia global del servicio
model_service = ModelService()
