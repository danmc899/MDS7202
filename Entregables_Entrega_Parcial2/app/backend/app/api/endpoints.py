"""
Endpoints de la API
Maneja las peticiones de predicción
"""
from fastapi import APIRouter, HTTPException, status
from typing import List, Dict
import pandas as pd
import numpy as np
import logging
from pathlib import Path

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

# Cache para catálogos
_productos_df = None
_clientes_df = None
_reference_df = None

def get_productos_catalog():
    """Carga y cachea el catálogo de productos"""
    global _productos_df
    if _productos_df is None:
        productos_path = Path("/app/data/productos.parquet")
        if productos_path.exists():
            _productos_df = pd.read_parquet(productos_path)
    return _productos_df

def get_clientes_catalog():
    """Carga y cachea el catálogo de clientes"""
    global _clientes_df
    if _clientes_df is None:
        clientes_path = Path("/app/data/clientes.parquet")
        if clientes_path.exists():
            _clientes_df = pd.read_parquet(clientes_path)
    return _clientes_df

def get_reference_data():
    """Carga y cachea los datos de referencia procesados"""
    global _reference_df
    if _reference_df is None:
        reference_path = Path("/app/data/processed/reference_data.parquet")
        if reference_path.exists():
            _reference_df = pd.read_parquet(reference_path)
    return _reference_df

def encode_product_features(product_id, reference_df):
    """
    Codifica las features de un producto usando el catálogo y las columnas de referencia.
    
    Args:
        product_id: ID del producto
        reference_df: DataFrame de referencia con todas las columnas one-hot encoded
        
    Returns:
        dict: Features del producto codificadas
    """
    productos_df = get_productos_catalog()
    
    if productos_df is None:
        logger.warning("Catálogo de productos no disponible")
        return None
    
    # Buscar producto en catálogo
    product_data = productos_df[productos_df['product_id'] == int(product_id)]
    
    if product_data.empty:
        logger.warning(f"Producto {product_id} no encontrado en catálogo")
        return None
    
    product = product_data.iloc[0]
    features = {}
    
    # Features numéricas directas
    features['size'] = float(product['size'])
    
    # Segment (ordinal encoding: PREMIUM=1, MEDIUM=0, BASIC=-1)
    segment_map = {'PREMIUM': 1.0, 'MEDIUM': 0.0, 'BASIC': -1.0}
    features['segment'] = segment_map.get(product['segment'], 0.0)
    
    # One-hot encoding para categóricas
    # Obtener todas las columnas categóricas posibles de reference_df
    cat_prefixes = ['sub_category_', 'package_', 'brand_']
    
    for prefix in cat_prefixes:
        cat_cols = [col for col in reference_df.columns if col.startswith(prefix)]
        
        # Inicializar todas en 0
        for col in cat_cols:
            features[col] = 0
        
        # Activar la columna correspondiente al valor del producto
        if prefix == 'sub_category_':
            value = product['sub_category']
        elif prefix == 'package_':
            value = product['package']
        elif prefix == 'brand_':
            value = product['brand']
        
        # Buscar la columna correspondiente (formato: prefix + value)
        matching_col = f"{prefix}{value}"
        if matching_col in cat_cols:
            features[matching_col] = 1
    
    return features


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """
    Realiza una predicción única para un par cliente-producto HISTÓRICO
    
    Args:
        request: Datos del cliente y producto
        
    Returns:
        Predicción y probabilidad de compra
    """
    try:
        logger.info(f"Predicción solicitada para customer_id={request.customer_id}, product_id={request.product_id}")
        
        reference_df = get_reference_data()
        
        if reference_df is None:
            raise FileNotFoundError("Datos de referencia no encontrados. Ejecuta el pipeline de Airflow primero.")
        
        # Buscar la combinación cliente-producto en los datos de referencia
        match = reference_df[
            (reference_df['customer_id'] == int(request.customer_id)) &
            (reference_df['product_id'] == int(request.product_id))
        ]
        
        if match.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"El par cliente {request.customer_id} - producto {request.product_id} no existe en el histórico. Solo se permiten predicciones para combinaciones que el cliente ha comprado antes."
            )
        
        # Usar datos históricos directamente
        input_data = match.copy()
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar predicción: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Realiza predicciones en lote para múltiples pares cliente-producto HISTÓRICOS
    
    Args:
        request: Lista de datos de clientes y productos
        
    Returns:
        Lista de predicciones y probabilidades (solo para pares históricos)
    """
    try:
        logger.info(f"Predicción en lote solicitada para {len(request.predictions)} items")
        
        reference_df = get_reference_data()
        
        if reference_df is None:
            raise FileNotFoundError("Datos de referencia no encontrados. Ejecuta el pipeline de Airflow primero.")
        
        # Procesar cada par cliente-producto
        batch_input = []
        
        for item in request.predictions:
            customer_id = int(item.customer_id)
            product_id = int(item.product_id)
            
            # Buscar combinación en datos de referencia
            match = reference_df[
                (reference_df['customer_id'] == customer_id) &
                (reference_df['product_id'] == product_id)
            ]
            
            if not match.empty:
                # Solo procesar pares históricos
                batch_input.append(match.iloc[0].to_dict())
            else:
                # Omitir pares que no existen en histórico
                logger.debug(f"Par {customer_id}-{product_id} no encontrado en histórico, omitiendo")
                continue
        
        if not batch_input:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudieron procesar las predicciones solicitadas"
            )
        
        # Convertir a DataFrame
        input_data = pd.DataFrame(batch_input)
        
        # Realizar predicciones
        predictions, probabilities = model_service.predict(input_data)
        
        # Construir respuesta
        results = []
        for i in range(len(batch_input)):
            results.append(
                CustomerProductPrediction(
                    customer_id=str(batch_input[i]['customer_id']),
                    product_id=str(batch_input[i]['product_id']),
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
        
    except HTTPException:
        raise
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
        
        # Leer datos de clientes y productos (ruta correcta en el contenedor)
        data_dir = Path("/app/data")
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


@router.get("/data/available-ids")
async def get_available_ids():
    """
    Obtiene las listas de IDs disponibles de clientes y productos
    
    Returns:
        Listas de customer_ids y product_ids
    """
    try:
        from pathlib import Path
        
        # Leer datos de clientes y productos (archivos en /app/data/)
        clientes_path = Path("/app/data/clientes.parquet")
        productos_path = Path("/app/data/productos.parquet")
        
        if not clientes_path.exists() or not productos_path.exists():
            logger.warning(f"Archivos de datos no encontrados: {clientes_path}, {productos_path}")
            raise FileNotFoundError("Archivos de datos no encontrados")
        
        clientes_df = pd.read_parquet(clientes_path)
        productos_df = pd.read_parquet(productos_path)
        
        # Convertir a listas y ordenar
        customer_ids = sorted(clientes_df['customer_id'].unique().tolist())
        product_ids = sorted(productos_df['product_id'].unique().tolist())
        
        logger.info(f"IDs disponibles: {len(customer_ids)} clientes, {len(product_ids)} productos")
        
        return {
            "customer_ids": customer_ids,
            "product_ids": product_ids
        }
    except Exception as e:
        logger.error(f"Error al obtener IDs disponibles: {str(e)}")
        # Retornar listas vacías si falla
        return {
            "customer_ids": [],
            "product_ids": []
        }


@router.get("/data/customer-products/{customer_id}")
async def get_customer_products(customer_id: int):
    """
    Obtiene los productos que un cliente específico ha comprado históricamente
    
    Args:
        customer_id: ID del cliente
        
    Returns:
        Lista de product_ids que el cliente ha comprado
    """
    try:
        reference_df = get_reference_data()
        
        if reference_df is None:
            raise FileNotFoundError("Datos de referencia no encontrados")
        
        # Filtrar productos del cliente en datos procesados
        customer_products = reference_df[
            reference_df['customer_id'] == customer_id
        ]['product_id'].unique().tolist()
        
        # Ordenar
        customer_products = sorted(customer_products)
        
        logger.info(f"Cliente {customer_id}: {len(customer_products)} productos históricos")
        
        return {
            "customer_id": customer_id,
            "product_ids": customer_products,
            "total": len(customer_products)
        }
    except Exception as e:
        logger.error(f"Error al obtener productos del cliente {customer_id}: {str(e)}")
        return {
            "customer_id": customer_id,
            "product_ids": [],
            "total": 0,
            "error": str(e)
        }

