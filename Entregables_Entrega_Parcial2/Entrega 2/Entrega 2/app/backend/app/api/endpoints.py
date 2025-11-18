"""
Endpoints de la API
Maneja las peticiones de predicción
"""
from fastapi import APIRouter, HTTPException, status
from typing import List, Dict
import pandas as pd
import numpy as np
import logging

from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerProductPrediction
)
from app.core.model_loader import model_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """
    Realiza una predicción única para un par cliente-producto
    
    Args:
        request: Datos del cliente y producto
        
    Returns:
        Predicción y probabilidad de compra
    """
    try:
        logger.info(f"Predicción solicitada para customer_id={request.customer_id}, product_id={request.product_id}")
        
        # Preparar datos para predicción
        input_data = pd.DataFrame([request.dict()])
        
        # Realizar predicción
        prediction, probability = model_service.predict(input_data)
        
        response = PredictionResponse(
            customer_id=request.customer_id,
            product_id=request.product_id,
            will_purchase=bool(prediction[0]),
            probability=float(probability[0]),
            model_type=model_service.model_type
        )
        
        logger.info(f"Predicción completada: {response.will_purchase} (prob={response.probability:.4f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar predicción: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Realiza predicciones en lote para múltiples pares cliente-producto
    
    Args:
        request: Lista de datos de clientes y productos
        
    Returns:
        Lista de predicciones y probabilidades
    """
    try:
        logger.info(f"Predicción en lote solicitada para {len(request.predictions)} items")
        
        # Preparar datos para predicción
        input_data = pd.DataFrame([item.dict() for item in request.predictions])
        
        # Realizar predicciones
        predictions, probabilities = model_service.predict(input_data)
        
        # Construir respuesta
        results = []
        for i, item in enumerate(request.predictions):
            results.append(
                CustomerProductPrediction(
                    customer_id=item.customer_id,
                    product_id=item.product_id,
                    will_purchase=bool(predictions[i]),
                    probability=float(probabilities[i])
                )
            )
        
        # Filtrar solo predicciones positivas si se solicita
        if request.only_positive:
            results = [r for r in results if r.will_purchase]
        
        # Ordenar por probabilidad descendente si se solicita
        if request.sort_by_probability:
            results = sorted(results, key=lambda x: x.probability, reverse=True)
        
        response = BatchPredictionResponse(
            predictions=results,
            total_predictions=len(results),
            model_type=model_service.model_type
        )
        
        logger.info(f"Predicción en lote completada: {len(results)} resultados")
        
        return response
        
    except Exception as e:
        logger.error(f"Error en predicción en lote: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar predicciones: {str(e)}"
        )


@router.get("/predict/customer/{customer_id}")
async def predict_customer_products(customer_id: str, top_n: int = 10):
    """
    Obtiene las top N recomendaciones de productos para un cliente
    
    Args:
        customer_id: ID del cliente
        top_n: Número de productos a recomendar (default: 10)
        
    Returns:
        Lista de productos recomendados con probabilidades
    """
    try:
        logger.info(f"Recomendaciones solicitadas para customer_id={customer_id}, top_n={top_n}")
        
        # Obtener predicciones para todos los productos
        # Nota: En producción, esto debería consultar una lista de productos disponibles
        predictions = model_service.predict_for_customer(customer_id, top_n=top_n)
        
        return {
            "customer_id": customer_id,
            "recommendations": predictions,
            "count": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Error al obtener recomendaciones: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener recomendaciones: {str(e)}"
        )


@router.get("/model/info")
async def get_model_info():
    """
    Obtiene información sobre el modelo cargado
    
    Returns:
        Información del modelo (tipo, métricas, features)
    """
    try:
        info = model_service.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error al obtener información del modelo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener información del modelo: {str(e)}"
        )


@router.post("/model/reload")
async def reload_model():
    """
    Recarga el modelo desde disco (útil si se actualiza el modelo)
    
    Returns:
        Confirmación de recarga
    """
    try:
        logger.info("Recarga de modelo solicitada")
        model_service.load_model()
        
        return {
            "status": "success",
            "message": "Modelo recargado exitosamente",
            "model_type": model_service.model_type
        }
    except Exception as e:
        logger.error(f"Error al recargar modelo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al recargar modelo: {str(e)}"
        )


@router.get("/data/ranges")
async def get_data_ranges():
    """
    Obtiene los rangos de IDs de clientes y productos
    
    Returns:
        Rangos de customer_id y product_id
    """
    try:
        from pathlib import Path
        
        # Leer datos de clientes y productos
        data_dir = Path("/opt/airflow/data")
        clientes_path = data_dir / "clientes.parquet"
        productos_path = data_dir / "productos.parquet"
        
        if not clientes_path.exists() or not productos_path.exists():
            raise FileNotFoundError("Archivos de datos no encontrados")
        
        clientes_df = pd.read_parquet(clientes_path)
        productos_df = pd.read_parquet(productos_path)
        
        return {
            "customer_id": {
                "min": int(clientes_df['customer_id'].min()),
                "max": int(clientes_df['customer_id'].max()),
                "count": len(clientes_df)
            },
            "product_id": {
                "min": int(productos_df['product_id'].min()),
                "max": int(productos_df['product_id'].max()),
                "count": len(productos_df)
            }
        }
    except Exception as e:
        logger.error(f"Error al obtener rangos de datos: {str(e)}")
        # Retornar valores por defecto si falla
        return {
            "customer_id": {
                "min": 25734,
                "max": 2061063,
                "count": 1569
            },
            "product_id": {
                "min": 8,
                "max": 297994,
                "count": 971
            }
        }
