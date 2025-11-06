"""
Backend FastAPI para Chatbot Conversacional
Responde preguntas sobre los datos usando procesamiento de lenguaje natural
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="SodAI Drinks - Chatbot Conversacional",
    description="API para responder preguntas sobre los datos",
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
class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    data: Dict = {}


# Sistema de Question Answering
class DataQASystem:
    """
    Sistema simple de Question Answering sobre los datos
    """
    
    def __init__(self):
        self.data_path = Path("/app/data")
        self.df_transacciones = None
        self.df_clientes = None
        self.df_productos = None
        self.load_data()
    
    def load_data(self):
        """
        Carga los datos de parquet
        """
        try:
            transacciones_path = self.data_path / "transacciones.parquet"
            clientes_path = self.data_path / "clientes.parquet"
            productos_path = self.data_path / "productos.parquet"
            
            if transacciones_path.exists():
                self.df_transacciones = pd.read_parquet(transacciones_path)
                logger.info(f"Transacciones cargadas: {len(self.df_transacciones)} registros")
            
            if clientes_path.exists():
                self.df_clientes = pd.read_parquet(clientes_path)
                logger.info(f"Clientes cargados: {len(self.df_clientes)} registros")
            
            if productos_path.exists():
                self.df_productos = pd.read_parquet(productos_path)
                logger.info(f"Productos cargados: {len(self.df_productos)} registros")
            
            if self.df_transacciones is None:
                self._create_sample_data()
                
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """
        Crea datos de ejemplo para demostración
        """
        import numpy as np
        np.random.seed(42)
        
        n_transactions = 1000
        n_customers = 100
        n_products = 50
        
        self.df_transacciones = pd.DataFrame({
            'customer_id': [f"C{np.random.randint(0, n_customers):04d}" for _ in range(n_transactions)],
            'product_id': [f"P{np.random.randint(0, n_products):04d}" for _ in range(n_transactions)],
            'order_id': [f"O{i:06d}" for i in range(n_transactions)],
            'items': np.random.randint(1, 10, n_transactions),
            'purchase_date': pd.date_range('2024-01-01', periods=n_transactions, freq='H')
        })
        
        self.df_clientes = pd.DataFrame({
            'customer_id': [f"C{i:04d}" for i in range(n_customers)]
        })
        
        self.df_productos = pd.DataFrame({
            'product_id': [f"P{i:04d}" for i in range(n_products)]
        })
        
        logger.info("Datos de ejemplo creados")
    
    def answer_question(self, question: str) -> Dict:
        """
        Responde preguntas sobre los datos usando pattern matching
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Diccionario con respuesta y datos adicionales
        """
        question_lower = question.lower()
        
        # Patrones de preguntas
        
        # 1. ¿Cuántos clientes únicos?
        if re.search(r'(cuántos|cuantos|cantidad|número|numero).*clientes', question_lower):
            n_clientes = self.df_transacciones['customer_id'].nunique() if self.df_transacciones is not None else len(self.df_clientes) if self.df_clientes is not None else 0
            return {
                "answer": f"Hay **{n_clientes:,}** clientes únicos en el dataset.",
                "data": {"num_clientes": n_clientes}
            }
        
        # 2. ¿Cuántas transacciones de un cliente?
        customer_match = re.search(r'cliente\s+([A-Za-z0-9]+)', question_lower)
        if customer_match and re.search(r'transacci(ones|ón)', question_lower):
            customer_id = customer_match.group(1).upper()
            if self.df_transacciones is not None:
                # Buscar el customer_id (case insensitive)
                mask = self.df_transacciones['customer_id'].str.upper() == customer_id
                n_trans = mask.sum()
                
                if n_trans > 0:
                    total_items = self.df_transacciones[mask]['items'].sum()
                    return {
                        "answer": f"El cliente **{customer_id}** ha realizado **{n_trans:,}** transacciones, comprando un total de **{total_items:,}** items.",
                        "data": {
                            "customer_id": customer_id,
                            "num_transacciones": int(n_trans),
                            "total_items": int(total_items)
                        }
                    }
                else:
                    return {
                        "answer": f"No se encontraron transacciones para el cliente **{customer_id}**.",
                        "data": {"customer_id": customer_id, "num_transacciones": 0}
                    }
        
        # 3. ¿Cuántos productos únicos?
        if re.search(r'(cuántos|cuantos|cantidad|número|numero).*productos', question_lower):
            n_productos = self.df_transacciones['product_id'].nunique() if self.df_transacciones is not None else len(self.df_productos) if self.df_productos is not None else 0
            return {
                "answer": f"Hay **{n_productos:,}** productos únicos en el dataset.",
                "data": {"num_productos": n_productos}
            }
        
        # 4. ¿Cuántas transacciones totales?
        if re.search(r'(cuántas|cuantas|total).*transacci(ones|ón)', question_lower):
            n_trans = len(self.df_transacciones) if self.df_transacciones is not None else 0
            return {
                "answer": f"Se han registrado **{n_trans:,}** transacciones en total.",
                "data": {"num_transacciones": n_trans}
            }
        
        # 5. Producto más vendido
        if re.search(r'producto.*más.*vend(ido|idos)|más.*popular', question_lower):
            if self.df_transacciones is not None:
                top_product = self.df_transacciones.groupby('product_id')['items'].sum().idxmax()
                total_sales = self.df_transacciones.groupby('product_id')['items'].sum().max()
                return {
                    "answer": f"El producto más vendido es **{top_product}** con **{int(total_sales):,}** items vendidos.",
                    "data": {
                        "product_id": top_product,
                        "total_sales": int(total_sales)
                    }
                }
        
        # 6. Cliente con más compras
        if re.search(r'cliente.*más.*compr(as|a)|mejor.*cliente', question_lower):
            if self.df_transacciones is not None:
                top_customer = self.df_transacciones.groupby('customer_id')['items'].sum().idxmax()
                total_items = self.df_transacciones.groupby('customer_id')['items'].sum().max()
                n_trans = (self.df_transacciones['customer_id'] == top_customer).sum()
                return {
                    "answer": f"El cliente con más compras es **{top_customer}** con **{int(total_items):,}** items en **{n_trans}** transacciones.",
                    "data": {
                        "customer_id": top_customer,
                        "total_items": int(total_items),
                        "num_transacciones": int(n_trans)
                    }
                }
        
        # 7. Estadísticas generales
        if re.search(r'estadísticas|resumen|general|overview', question_lower):
            if self.df_transacciones is not None:
                stats = {
                    "num_clientes": self.df_transacciones['customer_id'].nunique(),
                    "num_productos": self.df_transacciones['product_id'].nunique(),
                    "num_transacciones": len(self.df_transacciones),
                    "total_items": int(self.df_transacciones['items'].sum()),
                    "avg_items_per_transaction": float(self.df_transacciones['items'].mean())
                }
                return {
                    "answer": f"""**Estadísticas Generales:**
                    
- **Clientes únicos**: {stats['num_clientes']:,}
- **Productos únicos**: {stats['num_productos']:,}
- **Transacciones totales**: {stats['num_transacciones']:,}
- **Items vendidos**: {stats['total_items']:,}
- **Promedio items/transacción**: {stats['avg_items_per_transaction']:.2f}
                    """,
                    "data": stats
                }
        
        # Pregunta no reconocida
        return {
            "answer": """No pude entender tu pregunta. Puedo ayudarte con preguntas como:

- ¿Cuántos clientes únicos hay en el dataset?
- ¿Cuántas transacciones ha realizado el cliente [ID]?
- ¿Cuántos productos únicos se encuentran en los datos?
- ¿Cuál es el producto más vendido?
- ¿Cuál es el cliente con más compras?
- Dame las estadísticas generales

Por favor, intenta reformular tu pregunta.""",
            "data": {}
        }


# Instancia global del sistema QA
qa_system = DataQASystem()


@app.get("/")
async def root():
    return {
        "message": "Chatbot Conversacional - SodAI Drinks",
        "status": "online",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "data_loaded": qa_system.df_transacciones is not None
    }


@app.post("/api/v1/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Responde preguntas sobre los datos
    """
    try:
        logger.info(f"Pregunta recibida: {request.question}")
        
        result = qa_system.answer_question(request.question)
        
        return QuestionResponse(
            question=request.question,
            answer=result["answer"],
            data=result.get("data", {})
        )
        
    except Exception as e:
        logger.error(f"Error al procesar pregunta: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
