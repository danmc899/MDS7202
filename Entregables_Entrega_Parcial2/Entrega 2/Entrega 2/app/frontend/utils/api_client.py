"""
Cliente API para comunicación con el backend FastAPI
"""
import requests
import logging
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)


class APIClient:
    """
    Cliente para interactuar con el backend FastAPI
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Inicializa el cliente API
        
        Args:
            base_url: URL base del backend (default: desde variable de entorno)
        """
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://backend:8000")
        logger.info(f"API Client inicializado con URL: {self.base_url}")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Realiza una petición HTTP al backend
        
        Args:
            method: Método HTTP (GET, POST, etc.)
            endpoint: Endpoint de la API
            **kwargs: Argumentos adicionales para requests
            
        Returns:
            Respuesta JSON del backend
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.request(method, url, **kwargs, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en petición a {url}: {str(e)}")
            raise Exception(f"Error de comunicación con el backend: {str(e)}")
    
    def predict_single(
        self,
        customer_id: str,
        product_id: str,
        cp_total_items: Optional[float] = None,
        cp_avg_items: Optional[float] = None,
        total_items: Optional[float] = None,
        avg_items: Optional[float] = None
    ) -> Dict:
        """
        Realiza una predicción única
        
        Args:
            customer_id: ID del cliente
            product_id: ID del producto
            cp_total_items: Total items cliente-producto
            cp_avg_items: Promedio items cliente-producto
            total_items: Total items del cliente
            avg_items: Promedio items del cliente
            
        Returns:
            Resultado de la predicción
        """
        payload = {
            "customer_id": customer_id,
            "product_id": product_id
        }
        
        # Agregar features opcionales si se proporcionan
        if cp_total_items is not None:
            payload["cp_total_items"] = cp_total_items
        if cp_avg_items is not None:
            payload["cp_avg_items"] = cp_avg_items
        if total_items is not None:
            payload["total_items"] = total_items
        if avg_items is not None:
            payload["avg_items"] = avg_items
        
        logger.info(f"Predicción única: {customer_id} - {product_id}")
        
        return self._make_request(
            "POST",
            "/api/v1/predict",
            json=payload
        )
    
    def predict_batch(
        self,
        data: List[Dict],
        only_positive: bool = False,
        sort_by_probability: bool = True
    ) -> Dict:
        """
        Realiza predicciones en lote
        
        Args:
            data: Lista de diccionarios con datos de predicción
            only_positive: Solo retornar predicciones positivas
            sort_by_probability: Ordenar por probabilidad
            
        Returns:
            Resultado de las predicciones
        """
        payload = {
            "predictions": data,
            "only_positive": only_positive,
            "sort_by_probability": sort_by_probability
        }
        
        logger.info(f"Predicción en lote: {len(data)} items")
        
        return self._make_request(
            "POST",
            "/api/v1/predict/batch",
            json=payload
        )
    
    def get_recommendations(self, customer_id: str, top_n: int = 10) -> Dict:
        """
        Obtiene recomendaciones de productos para un cliente
        
        Args:
            customer_id: ID del cliente
            top_n: Número de recomendaciones
            
        Returns:
            Lista de productos recomendados
        """
        logger.info(f"Recomendaciones para {customer_id}: top {top_n}")
        
        return self._make_request(
            "GET",
            f"/api/v1/predict/customer/{customer_id}",
            params={"top_n": top_n}
        )
    
    def get_model_info(self) -> Dict:
        """
        Obtiene información del modelo cargado
        
        Returns:
            Información del modelo
        """
        return self._make_request("GET", "/api/v1/model/info")
    
    def reload_model(self) -> Dict:
        """
        Solicita recarga del modelo en el backend
        
        Returns:
            Confirmación de recarga
        """
        logger.info("Solicitando recarga de modelo")
        return self._make_request("POST", "/api/v1/model/reload")
    
    def health_check(self) -> Dict:
        """
        Verifica el estado del backend
        
        Returns:
            Estado de salud del backend
        """
        return self._make_request("GET", "/health")
    
    def get_data_ranges(self) -> Dict:
        """
        Obtiene los rangos de IDs de clientes y productos
        
        Returns:
            Diccionario con rangos de customer_id y product_id
        """
        return self._make_request("GET", "/api/v1/data/ranges")
