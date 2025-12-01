"""
Schemas de Pydantic para validación de datos
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """
    Schema para solicitud de predicción única
    """
    customer_id: str = Field(..., description="ID del cliente")
    product_id: str = Field(..., description="ID del producto")
    
    # Features opcionales (se pueden calcular en backend si no se proporcionan)
    cp_total_items: Optional[float] = Field(None, description="Total de items comprados de este producto por el cliente")
    cp_avg_items: Optional[float] = Field(None, description="Promedio de items por transacción")
    total_items: Optional[float] = Field(None, description="Total de items comprados por el cliente")
    avg_items: Optional[float] = Field(None, description="Promedio de items por transacción del cliente")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "C12345",
                "product_id": "P6789",
                "cp_total_items": 15.0,
                "cp_avg_items": 3.0,
                "total_items": 100.0,
                "avg_items": 5.0
            }
        }


class PredictionResponse(BaseModel):
    """
    Schema para respuesta de predicción única
    """
    customer_id: str
    product_id: str
    will_purchase: bool = Field(..., description="Si el cliente comprará el producto (True/False)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de compra (0-1)")
    model_type: str = Field(..., description="Tipo de modelo usado para la predicción")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "C12345",
                "product_id": "P6789",
                "will_purchase": True,
                "probability": 0.85,
                "model_type": "random_forest"
            }
        }


class CustomerProductPrediction(BaseModel):
    """
    Schema para una predicción en lote
    """
    customer_id: str
    product_id: str
    will_purchase: bool
    probability: float = Field(..., ge=0.0, le=1.0)


class BatchPredictionRequest(BaseModel):
    """
    Schema para solicitud de predicción en lote
    """
    predictions: List[PredictionRequest] = Field(..., description="Lista de predicciones a realizar")
    only_positive: bool = Field(False, description="Retornar solo predicciones positivas")
    sort_by_probability: bool = Field(True, description="Ordenar por probabilidad descendente")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "customer_id": "C12345",
                        "product_id": "P6789",
                        "total_items": 100.0,
                        "avg_items": 5.0
                    },
                    {
                        "customer_id": "C12345",
                        "product_id": "P6790",
                        "total_items": 100.0,
                        "avg_items": 5.0
                    }
                ],
                "only_positive": True,
                "sort_by_probability": True
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Schema para respuesta de predicción en lote
    """
    predictions: List[CustomerProductPrediction]
    total_predictions: int = Field(..., description="Número total de predicciones")
    model_type: str = Field(..., description="Tipo de modelo usado")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "customer_id": "C12345",
                        "product_id": "P6789",
                        "will_purchase": True,
                        "probability": 0.85
                    }
                ],
                "total_predictions": 1,
                "model_type": "random_forest"
            }
        }
