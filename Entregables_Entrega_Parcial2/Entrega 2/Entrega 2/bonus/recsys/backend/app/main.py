"""
Backend FastAPI para Sistema de Recomendación
Genera recomendaciones de productos para clientes usando collaborative filtering
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="SodAI Drinks - Sistema de Recomendación",
    description="API para generar recomendaciones de productos",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Schemas
class RecommendationRequest(BaseModel):
    customer_id: str = Field(..., description="ID del cliente")
    n_recommendations: int = Field(5, ge=1, le=20, description="Número de recomendaciones")


class ProductRecommendation(BaseModel):
    product_id: str
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[ProductRecommendation]
    total: int


# Sistema de recomendación simple
class RecommendationSystem:
    """
    Sistema de recomendación basado en collaborative filtering
    """
    
    def __init__(self):
        self.data_path = Path("/app/data")
        self.customer_product_matrix = None
        self.product_similarity = None
        self.load_data()
    
    def load_data(self):
        """
        Carga datos de transacciones y calcula matriz de similaridad
        """
        try:
            # Cargar datos de transacciones
            transacciones_path = self.data_path / "transacciones.parquet"
            
            if not transacciones_path.exists():
                logger.warning("Datos de transacciones no encontrados. Usando datos de ejemplo.")
                self._create_sample_data()
                return
            
            df = pd.read_parquet(transacciones_path)
            
            # Crear matriz cliente-producto
            self.customer_product_matrix = df.groupby(['customer_id', 'product_id'])['items'].sum().unstack(fill_value=0)
            
            # Calcular similaridad entre productos (cosine similarity)
            from sklearn.metrics.pairwise import cosine_similarity
            self.product_similarity = pd.DataFrame(
                cosine_similarity(self.customer_product_matrix.T),
                index=self.customer_product_matrix.columns,
                columns=self.customer_product_matrix.columns
            )
            
            logger.info(f"Datos cargados: {len(self.customer_product_matrix)} clientes, {len(self.customer_product_matrix.columns)} productos")
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """
        Crea datos de ejemplo para demostración
        """
        np.random.seed(42)
        n_customers = 100
        n_products = 50
        
        # Crear matriz aleatoria
        matrix = np.random.randint(0, 20, size=(n_customers, n_products))
        
        self.customer_product_matrix = pd.DataFrame(
            matrix,
            index=[f"C{i:04d}" for i in range(n_customers)],
            columns=[f"P{i:04d}" for i in range(n_products)]
        )
        
        # Calcular similaridad
        from sklearn.metrics.pairwise import cosine_similarity
        self.product_similarity = pd.DataFrame(
            cosine_similarity(self.customer_product_matrix.T),
            index=self.customer_product_matrix.columns,
            columns=self.customer_product_matrix.columns
        )
        
        logger.info("Datos de ejemplo creados")
    
    def get_recommendations(self, customer_id: str, n: int = 5) -> List[Dict]:
        """
        Genera recomendaciones para un cliente
        
        Args:
            customer_id: ID del cliente
            n: Número de recomendaciones
            
        Returns:
            Lista de recomendaciones
        """
        if customer_id not in self.customer_product_matrix.index:
            # Cliente nuevo, recomendar productos más populares
            logger.info(f"Cliente {customer_id} no encontrado, recomendando productos populares")
            return self._recommend_popular(n)
        
        # Obtener productos ya comprados por el cliente
        customer_products = self.customer_product_matrix.loc[customer_id]
        purchased_products = customer_products[customer_products > 0].index.tolist()
        
        # Calcular scores para productos no comprados
        recommendations = {}
        
        for product in purchased_products:
            if product not in self.product_similarity.index:
                continue
            
            # Obtener productos similares
            similar_products = self.product_similarity[product].sort_values(ascending=False)
            
            for similar_product, similarity in similar_products.items():
                if similar_product not in purchased_products:
                    if similar_product not in recommendations:
                        recommendations[similar_product] = 0
                    recommendations[similar_product] += similarity * customer_products[product]
        
        # Si no hay recomendaciones, usar productos populares
        if not recommendations:
            return self._recommend_popular(n)
        
        # Ordenar por score
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return [
            {
                "product_id": product,
                "score": float(score),
                "rank": i + 1
            }
            for i, (product, score) in enumerate(sorted_recommendations)
        ]
    
    def _recommend_popular(self, n: int = 5) -> List[Dict]:
        """
        Recomienda productos más populares
        """
        popular_products = self.customer_product_matrix.sum(axis=0).sort_values(ascending=False).head(n)
        
        return [
            {
                "product_id": product,
                "score": float(score),
                "rank": i + 1
            }
            for i, (product, score) in enumerate(popular_products.items())
        ]


# Instancia global del sistema de recomendación
recsys = RecommendationSystem()


@app.get("/")
async def root():
    return {
        "message": "Sistema de Recomendación - SodAI Drinks",
        "status": "online",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "data_loaded": recsys.customer_product_matrix is not None
    }


@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Genera recomendaciones de productos para un cliente
    """
    try:
        logger.info(f"Recomendaciones solicitadas para {request.customer_id}")
        
        recommendations = recsys.get_recommendations(
            customer_id=request.customer_id,
            n=request.n_recommendations
        )
        
        return RecommendationResponse(
            customer_id=request.customer_id,
            recommendations=[ProductRecommendation(**rec) for rec in recommendations],
            total=len(recommendations)
        )
        
    except Exception as e:
        logger.error(f"Error al generar recomendaciones: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/recommend/{customer_id}")
async def get_recommendations_get(customer_id: str, n: int = 5):
    """
    Genera recomendaciones (método GET alternativo)
    """
    try:
        recommendations = recsys.get_recommendations(customer_id=customer_id, n=n)
        
        return {
            "customer_id": customer_id,
            "recommendations": recommendations,
            "total": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
