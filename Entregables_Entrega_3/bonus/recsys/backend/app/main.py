"""
Backend FastAPI para Sistema de Recomendación
Genera recomendaciones de productos para clientes usando SVD (Singular Value Decomposition)
Implementación basada en Surprise library con collaborative filtering
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
from collections import defaultdict
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="SodAI Drinks - Sistema de Recomendación SVD",
    description="API para generar recomendaciones de productos usando SVD collaborative filtering",
    version="2.0.0"
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
    predicted_rating: float
    rank: int


class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[ProductRecommendation]
    total: int
    metrics: Dict = {}


# Sistema de recomendación con SVD
class SVDRecommendationSystem:
    """
    Sistema de recomendación basado en SVD (Singular Value Decomposition)
    Implementa collaborative filtering con Surprise library
    """
    
    def __init__(self):
        self.data_path = Path("/app/data/raw")
        self.model = None
        self.df_ratings = None
        self.all_customers = set()
        self.all_products = set()
        self.trainset = None
        self.testset = None
        self.load_and_train()
    
    def load_and_train(self):
        """
        Carga datos de transacciones, crea ratings implícitos y entrena modelo SVD
        con separación temporal para evitar data leakage
        """
        try:
            # Cargar datos de transacciones
            transacciones_path = self.data_path / "transacciones.parquet"
            
            if not transacciones_path.exists():
                logger.warning("Datos de transacciones no encontrados. Usando datos de ejemplo.")
                self._create_sample_data()
                return
            
            df = pd.read_parquet(transacciones_path)
            logger.info(f"Transacciones cargadas: {len(df)} registros")
            
            # Asegurar que purchase_date es datetime
            if 'purchase_date' in df.columns:
                df['purchase_date'] = pd.to_datetime(df['purchase_date'])
            else:
                logger.warning("No hay columna purchase_date, usando orden de registros")
                df['purchase_date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
            
            # Crear ratings implícitos basados en frecuencia de compra y cantidad
            # NUEVO ENFOQUE: Usar log-transform y percentiles para mejor distribución
            df_grouped = df.groupby(['customer_id', 'product_id']).agg({
                'items': 'sum',
                'order_id': 'count',  # frecuencia de compras
                'purchase_date': 'max'  # última compra para ordenamiento temporal
            }).reset_index()
            
            df_grouped.columns = ['customer_id', 'product_id', 'total_items', 'frequency', 'last_purchase']
            
            # Items negativos son DEVOLUCIONES (no eliminar, afectan el rating negativamente)
            # Separar compras de devoluciones
            df_grouped['purchases'] = df_grouped['total_items'].clip(lower=0)  # Solo positivos
            df_grouped['returns'] = (-df_grouped['total_items']).clip(lower=0)  # Solo negativos (absoluto)
            
            # Net items = compras - devoluciones (puede ser negativo si devolvió más de lo que compró)
            df_grouped['net_items'] = df_grouped['total_items']
            
            # Log-transform para reducir efecto de outliers y mejorar distribución
            # Usar net_items en lugar de total_items
            # Usar net_items (compras - devoluciones) en lugar de solo compras
            # Para log, necesitamos manejar valores negativos: log(1 + |x|) * sign(x)
            df_grouped['items_log'] = np.sign(df_grouped['net_items']) * np.log1p(np.abs(df_grouped['net_items']))
            df_grouped['freq_log'] = np.log1p(df_grouped['frequency'])
            
            # Normalizar usando percentiles (más robusto que min-max)
            items_percentiles = np.percentile(df_grouped['items_log'], [0, 25, 50, 75, 100])
            freq_percentiles = np.percentile(df_grouped['freq_log'], [0, 25, 50, 75, 100])
            
            logger.info(f"Percentiles items_log: P0={items_percentiles[0]:.2f}, P25={items_percentiles[1]:.2f}, P50={items_percentiles[2]:.2f}, P75={items_percentiles[3]:.2f}, P100={items_percentiles[4]:.2f}")
            logger.info(f"Percentiles freq_log: P0={freq_percentiles[0]:.2f}, P25={freq_percentiles[1]:.2f}, P50={freq_percentiles[2]:.2f}, P75={freq_percentiles[3]:.2f}, P100={freq_percentiles[4]:.2f}")
            
            # Crear rating con escala 1-5 usando combinación lineal
            # 70% peso en cantidad, 30% peso en frecuencia
            max_items_log = df_grouped['items_log'].max()
            max_freq_log = df_grouped['freq_log'].max()
            
            if max_items_log > 0:
                items_norm = df_grouped['items_log'] / max_items_log
            else:
                items_norm = 0
                
            if max_freq_log > 0:
                freq_norm = df_grouped['freq_log'] / max_freq_log
            else:
                freq_norm = 0
            
            # Escala 1-5 con mejor distribución
            df_grouped['rating'] = (items_norm * 0.7 + freq_norm * 0.3) * 4 + 1
            df_grouped['rating'] = df_grouped['rating'].clip(1, 5)
            
            logger.info(f"Ratings - Min: {df_grouped['rating'].min():.4f}, Max: {df_grouped['rating'].max():.4f}, Mean: {df_grouped['rating'].mean():.4f}, Std: {df_grouped['rating'].std():.4f}")
            
            # SEPARACIÓN TEMPORAL: Train = 80% más antiguo, Test = 20% más reciente
            df_grouped = df_grouped.sort_values('last_purchase')
            split_idx = int(len(df_grouped) * 0.8)
            
            train_data = df_grouped.iloc[:split_idx]
            test_data = df_grouped.iloc[split_idx:]
            
            logger.info(f"Split temporal: Train hasta {train_data['last_purchase'].max()}, Test desde {test_data['last_purchase'].min()}")
            
            self.df_ratings = df_grouped[['customer_id', 'product_id', 'rating']]
            # Convertir a strings para consistencia con API
            self.all_customers = set(self.df_ratings['customer_id'].astype(str).unique())
            self.all_products = set(self.df_ratings['product_id'].astype(str).unique())
            
            logger.info(f"Ratings creados: {len(self.df_ratings)} ratings para {len(self.all_customers)} clientes y {len(self.all_products)} productos")
            
            # Preparar datos para Surprise usando SOLO train_data
            reader = Reader(rating_scale=(1, 5))
            train_dataset = Dataset.load_from_df(
                train_data[['customer_id', 'product_id', 'rating']], 
                reader
            )
            test_dataset = Dataset.load_from_df(
                test_data[['customer_id', 'product_id', 'rating']], 
                reader
            )
            
            # Entrenar solo con datos temporales de train
            self.trainset = train_dataset.build_full_trainset()
            self.testset = test_dataset.build_full_trainset().build_testset()
            
            # Entrenar modelo SVD
            logger.info("Entrenando modelo SVD con separación temporal...")
            self.model = SVD(
                n_factors=50,
                n_epochs=20,
                lr_all=0.005,
                reg_all=0.02,
                random_state=42
            )
            self.model.fit(self.trainset)
            
            # Evaluar modelo
            predictions = self.model.test(self.testset)
            mae = np.mean([abs(pred.est - pred.r_ui) for pred in predictions])
            rmse = np.sqrt(np.mean([(pred.est - pred.r_ui)**2 for pred in predictions]))
            
            logger.info(f"Modelo SVD entrenado exitosamente - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            logger.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
            
        except Exception as e:
            logger.error(f"Error al cargar datos y entrenar modelo: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self._create_sample_data()
    
    def _create_sample_data(self):
        """
        Crea datos de ejemplo para demostración
        """
        np.random.seed(42)
        n_customers = 100
        n_products = 50
        n_ratings = 2000
        
        # Crear ratings aleatorios
        customers = [f"C{i:04d}" for i in range(n_customers)]
        products = [f"P{i:04d}" for i in range(n_products)]
        
        sample_data = []
        for _ in range(n_ratings):
            customer = np.random.choice(customers)
            product = np.random.choice(products)
            rating = np.random.uniform(1, 5)
            sample_data.append({
                'customer_id': customer,
                'product_id': product,
                'rating': rating
            })
        
        self.df_ratings = pd.DataFrame(sample_data)
        self.all_customers = set(customers)
        self.all_products = set(products)
        
        # Entrenar modelo con datos de ejemplo
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.df_ratings, reader)
        self.trainset, self.testset = train_test_split(data, test_size=0.2, random_state=42)
        
        self.model = SVD(n_factors=20, n_epochs=10, random_state=42)
        self.model.fit(self.trainset)
        
        logger.info("Modelo SVD entrenado con datos de ejemplo")
    
    def get_top_n(self, customer_id: str, n: int = 10) -> List[Dict]:
        """
        Obtiene las top N recomendaciones para un cliente usando SVD
        Basado en la función get_top_n() del Lab10
        
        Args:
            customer_id: ID del cliente (string)
            n: Número de recomendaciones
            
        Returns:
            Lista de recomendaciones ordenadas por rating predicho
        """
        if self.model is None:
            logger.error("Modelo SVD no entrenado")
            return []
        
        # IMPORTANTE: Convertir customer_id a int para que coincida con el tipo usado en entrenamiento
        try:
            customer_id_int = int(customer_id)
        except ValueError:
            logger.warning(f"Customer ID {customer_id} no es entero, usando como nuevo cliente")
            customer_id_int = customer_id  # Dejar como string si falla conversión
        
        # Obtener productos ya comprados por el cliente
        if customer_id in self.all_customers:
            purchased_products = set(
                self.df_ratings[self.df_ratings['customer_id'].astype(str) == customer_id]['product_id'].astype(str)
            )
        else:
            logger.info(f"Cliente nuevo {customer_id}, recomendando productos populares")
            purchased_products = set()
        
        # Predecir ratings para todos los productos no comprados
        predictions = []
        for product_str in self.all_products:
            if product_str not in purchased_products:
                # Convertir product_id también a int
                try:
                    product_int = int(product_str)
                except ValueError:
                    product_int = product_str
                
                # Predict con IDs originales (int)
                pred = self.model.predict(customer_id_int, product_int)
                predictions.append((product_str, pred.est))
        
        # Ordenar por rating predicho (descendente) y tomar top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n]
        
        logger.info(f"Top predicciones para cliente {customer_id}: {[(p, f'{r:.4f}') for p, r in top_n[:3]]}")
        
        return [
            {
                "product_id": str(product),
                "predicted_rating": float(rating),
                "rank": i + 1
            }
            for i, (product, rating) in enumerate(top_n)
        ]
    
    def precision_recall_at_k(self, k: int = 10, threshold: float = 3.5) -> Dict:
        """
        Calcula precision@k y recall@k para el modelo
        Basado en la función del Lab10
        
        Args:
            k: Número de recomendaciones a considerar
            threshold: Rating mínimo para considerar relevante
            
        Returns:
            Diccionario con métricas promedio
        """
        if self.model is None or self.testset is None:
            return {"precision": 0.0, "recall": 0.0}
        
        predictions = self.model.test(self.testset)
        
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))
        
        precisions = []
        recalls = []
        
        for uid, user_ratings in user_est_true.items():
            # Ordenar por estimación descendente
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            
            # Número de items relevantes
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            
            # Número de items recomendados en top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            
            # Número de items relevantes Y recomendados en top k
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )
            
            # Precision@k
            precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)
            
            # Recall@k
            recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)
        
        return {
            "precision@k": float(np.mean(precisions)),
            "recall@k": float(np.mean(recalls)),
            "k": k,
            "threshold": threshold
        }


# Instancia global del sistema de recomendación SVD
recsys = SVDRecommendationSystem()


@app.get("/")
async def root():
    return {
        "message": "Sistema de Recomendación SVD - SodAI Drinks",
        "status": "online",
        "version": "2.0.0",
        "model": "SVD (Singular Value Decomposition)",
        "num_customers": len(recsys.all_customers) if recsys.all_customers else 0,
        "num_products": len(recsys.all_products) if recsys.all_products else 0
    }


@app.get("/health")
async def health_check():
    model_trained = recsys.model is not None
    return {
        "status": "healthy" if model_trained else "model_not_trained",
        "model_trained": model_trained,
        "num_ratings": len(recsys.df_ratings) if recsys.df_ratings is not None else 0
    }


@app.get("/api/v1/customers")
async def get_customers():
    """
    Obtiene lista de todos los clientes disponibles
    """
    try:
        if recsys.all_customers is None or len(recsys.all_customers) == 0:
            return {"customers": [], "count": 0}
        
        # Convertir set a lista ordenada
        customers_list = sorted(list(recsys.all_customers), key=lambda x: int(x) if x.isdigit() else x)
        
        return {
            "customers": customers_list,
            "count": len(customers_list)
        }
    except Exception as e:
        logger.error(f"Error al obtener clientes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics")
async def get_metrics():
    """
    Obtiene métricas de precision@k y recall@k del modelo SVD
    """
    try:
        metrics_10 = recsys.precision_recall_at_k(k=10, threshold=3.5)
        metrics_5 = recsys.precision_recall_at_k(k=5, threshold=3.5)
        
        return {
            "metrics": {
                "precision@5": metrics_5["precision@k"],
                "recall@5": metrics_5["recall@k"],
                "precision@10": metrics_10["precision@k"],
                "recall@10": metrics_10["recall@k"]
            },
            "model_info": {
                "algorithm": "SVD",
                "num_customers": len(recsys.all_customers),
                "num_products": len(recsys.all_products),
                "num_ratings": len(recsys.df_ratings) if recsys.df_ratings is not None else 0
            }
        }
    except Exception as e:
        logger.error(f"Error al calcular métricas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Genera recomendaciones de productos para un cliente usando SVD
    """
    try:
        logger.info(f"Generando {request.n_recommendations} recomendaciones SVD para cliente {request.customer_id}")
        
        # Obtener recomendaciones usando SVD
        recommendations = recsys.get_top_n(
            customer_id=request.customer_id,
            n=request.n_recommendations
        )
        
        # Calcular métricas del modelo
        metrics = recsys.precision_recall_at_k(k=request.n_recommendations, threshold=3.5)
        
        return RecommendationResponse(
            customer_id=request.customer_id,
            recommendations=[ProductRecommendation(**rec) for rec in recommendations],
            total=len(recommendations),
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Error al generar recomendaciones: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/recommend/{customer_id}")
async def get_recommendations_get(customer_id: str, n: int = 5):
    """
    Genera recomendaciones usando método GET (para compatibilidad)
    """
    try:
        recommendations = recsys.get_top_n(customer_id=customer_id, n=n)
        
        return {
            "customer_id": customer_id,
            "recommendations": recommendations,
            "total": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrain")
async def retrain_model():
    """
    Reentrenar el modelo con datos actualizados
    Útil cuando se agregan nuevas semanas de datos
    """
    try:
        logger.info("Iniciando reentrenamiento del modelo...")
        recsys.load_and_train()
        
        return {
            "status": "success",
            "message": "Modelo reentrenado exitosamente",
            "num_customers": len(recsys.all_customers),
            "num_products": len(recsys.all_products),
            "num_ratings": len(recsys.df_ratings) if recsys.df_ratings is not None else 0
        }
    except Exception as e:
        logger.error(f"Error al reentrenar modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


@app.post("/api/v1/retrain")
async def retrain_model():
    """
    Reentrenar el modelo con datos actualizados
    Útil cuando se agregan nuevas semanas de datos
    """
    try:
        logger.info("Iniciando reentrenamiento del modelo...")
        recsys.load_and_train()
        
        return {
            "status": "success",
            "message": "Modelo reentrenado exitosamente",
            "num_customers": len(recsys.all_customers),
            "num_products": len(recsys.all_products),
            "num_ratings": len(recsys.df_ratings) if recsys.df_ratings is not None else 0
        }
    except Exception as e:
        logger.error(f"Error al reentrenar modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
