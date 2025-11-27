"""
Backend FastAPI para Chatbot Conversacional con Google Gemini
Responde preguntas sobre los datos de SodAI Drinks usando RAG (Retrieval Augmented Generation)
Implementación basada en LangChain + Google Gemini API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import logging
from pathlib import Path
import os
from datetime import datetime

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="SodAI Drinks - Chatbot Conversacional con Google Gemini",
    description="API para responder preguntas sobre los datos usando RAG con Google Gemini",
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
class QuestionRequest(BaseModel):
    question: str
    include_sources: bool = False


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = []
    metadata: Dict = {}


# Sistema de Question Answering con RAG y Google Gemini
class GeminiRAGSystem:
    """
    Sistema de Question Answering usando Google Gemini con RAG
    sobre los datos de SodAI Drinks (transacciones, clientes, productos)
    """
    
    def __init__(self):
        self.data_path = Path("/app/data/raw")
        self.df_transacciones = None
        self.df_clientes = None
        self.df_productos = None
        self.vectorstore = None
        self.rag_chain = None
        self.llm = None
        
        # Verificar API key
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ GOOGLE_API_KEY no configurada. El sistema usará respuestas básicas.")
        
        self.load_data()
        self.setup_rag()
    
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
                logger.info(f"✓ Transacciones cargadas: {len(self.df_transacciones):,} registros")
            
            if clientes_path.exists():
                self.df_clientes = pd.read_parquet(clientes_path)
                logger.info(f"✓ Clientes cargados: {len(self.df_clientes):,} registros")
            
            if productos_path.exists():
                self.df_productos = pd.read_parquet(productos_path)
                logger.info(f"✓ Productos cargados: {len(self.df_productos):,} registros")
            
            if self.df_transacciones is None:
                logger.warning("No se encontraron datos. Creando datos de ejemplo...")
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
            'customer_id': [f"C{i:04d}" for i in range(n_customers)],
            'customer_name': [f"Cliente {i}" for i in range(n_customers)],
            'segment': np.random.choice(['Premium', 'Regular', 'Básico'], n_customers)
        })
        
        self.df_productos = pd.DataFrame({
            'product_id': [f"P{i:04d}" for i in range(n_products)],
            'product_name': [f"Producto {i}" for i in range(n_products)],
            'category': np.random.choice(['Bebidas', 'Snacks', 'Dulces'], n_products)
        })
        
        logger.info("✓ Datos de ejemplo creados")
    
    def create_documents_from_data(self) -> List[Document]:
        """
        Crea documentos para vectorización desde los DataFrames
        Genera resúmenes estadísticos y metadatos útiles
        """
        documents = []
        
        # 1. Documento de resumen general
        if self.df_transacciones is not None:
            stats = {
                "total_transacciones": len(self.df_transacciones),
                "total_clientes": self.df_transacciones['customer_id'].nunique(),
                "total_productos": self.df_transacciones['product_id'].nunique(),
                "total_items": int(self.df_transacciones['items'].sum()),
                "promedio_items": float(self.df_transacciones['items'].mean()),
                "fecha_inicio": str(self.df_transacciones['purchase_date'].min()),
                "fecha_fin": str(self.df_transacciones['purchase_date'].max())
            }
            
            doc_general = Document(
                page_content=f"""
                RESUMEN GENERAL DE SODAI DRINKS
                
                El dataset contiene {stats['total_transacciones']:,} transacciones realizadas por {stats['total_clientes']:,} clientes únicos.
                Se vendieron {stats['total_items']:,} items en total de {stats['total_productos']:,} productos diferentes.
                El promedio de items por transacción es {stats['promedio_items']:.2f}.
                Las transacciones abarcan desde {stats['fecha_inicio']} hasta {stats['fecha_fin']}.
                """,
                metadata={"type": "resumen_general", **stats}
            )
            documents.append(doc_general)
            
            # 2. Top 10 clientes por compras
            top_clientes = (
                self.df_transacciones
                .groupby('customer_id')['items']
                .agg(['sum', 'count'])
                .sort_values('sum', ascending=False)
                .head(10)
            )
            
            doc_top_clientes = Document(
                page_content=f"""
                TOP 10 CLIENTES CON MÁS COMPRAS
                
                Los 10 clientes que más han comprado son:
                {chr(10).join([f"- {idx}: {row['sum']:.0f} items en {row['count']} transacciones" for idx, row in top_clientes.iterrows()])}
                
                El cliente más activo es {top_clientes.index[0]} con {top_clientes.iloc[0]['sum']:.0f} items comprados.
                """,
                metadata={"type": "top_clientes", "top_customer": top_clientes.index[0]}
            )
            documents.append(doc_top_clientes)
            
            # Bottom 10 clientes con menos compras
            bottom_clientes = (
                self.df_transacciones
                .groupby('customer_id')['items']
                .agg(['sum', 'count'])
                .sort_values('sum', ascending=True)
                .head(10)
            )
            
            doc_bottom_clientes = Document(
                page_content=f"""
                BOTTOM 10 CLIENTES CON MENOS COMPRAS
                
                Los 10 clientes que menos han comprado son:
                {chr(10).join([f"- {idx}: {row['sum']:.0f} items en {row['count']} transacciones" for idx, row in bottom_clientes.iterrows()])}
                
                El cliente con menos compras es {bottom_clientes.index[0]} con {bottom_clientes.iloc[0]['sum']:.0f} items comprados.
                """,
                metadata={"type": "bottom_clientes", "bottom_customer": bottom_clientes.index[0]}
            )
            documents.append(doc_bottom_clientes)
            
            # 3. Top 10 productos más vendidos
            top_productos = (
                self.df_transacciones
                .groupby('product_id')['items']
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            
            doc_top_productos = Document(
                page_content=f"""
                TOP 10 PRODUCTOS MÁS VENDIDOS
                
                Los 10 productos más vendidos son:
                {chr(10).join([f"- {prod}: {items:.0f} items vendidos" for prod, items in top_productos.items()])}
                
                El producto más popular es {top_productos.index[0]} con {top_productos.iloc[0]:.0f} items vendidos.
                """,
                metadata={"type": "top_productos", "top_product": top_productos.index[0]}
            )
            documents.append(doc_top_productos)
            
            # Bottom 10 productos menos vendidos
            bottom_productos = (
                self.df_transacciones
                .groupby('product_id')['items']
                .sum()
                .sort_values(ascending=True)
                .head(10)
            )
            
            doc_bottom_productos = Document(
                page_content=f"""
                BOTTOM 10 PRODUCTOS MENOS VENDIDOS
                
                Los 10 productos menos vendidos son:
                {chr(10).join([f"- {prod}: {items:.0f} items vendidos" for prod, items in bottom_productos.items()])}
                
                El producto menos popular es {bottom_productos.index[0]} con {bottom_productos.iloc[0]:.0f} items vendidos.
                """,
                metadata={"type": "bottom_productos", "bottom_product": bottom_productos.index[0]}
            )
            documents.append(doc_bottom_productos)
            
            # 4. Información detallada de cada cliente (top 50)
            for customer_id in self.df_transacciones['customer_id'].value_counts().head(50).index:
                customer_trans = self.df_transacciones[self.df_transacciones['customer_id'] == customer_id]
                
                doc_cliente = Document(
                    page_content=f"""
                    INFORMACIÓN DEL CLIENTE {customer_id}
                    
                    - Total de transacciones: {len(customer_trans)}
                    - Total de items comprados: {customer_trans['items'].sum():.0f}
                    - Promedio de items por transacción: {customer_trans['items'].mean():.2f}
                    - Productos únicos comprados: {customer_trans['product_id'].nunique()}
                    - Producto favorito: {customer_trans.groupby('product_id')['items'].sum().idxmax()}
                    - Primera compra: {customer_trans['purchase_date'].min()}
                    - Última compra: {customer_trans['purchase_date'].max()}
                    """,
                    metadata={
                        "type": "detalle_cliente",
                        "customer_id": customer_id,
                        "num_transactions": len(customer_trans),
                        "total_items": int(customer_trans['items'].sum())
                    }
                )
                documents.append(doc_cliente)
        
        logger.info(f"✓ Creados {len(documents)} documentos para vectorización")
        return documents
    
    def setup_rag(self):
        """
        Configura el sistema RAG con Google Gemini
        """
        try:
            if not self.api_key:
                logger.warning("Sistema RAG no disponible sin GOOGLE_API_KEY")
                return
            
            # Inicializar LLM de Google Gemini
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.3,
                google_api_key=self.api_key
            )
            
            # Crear documentos desde los datos
            documents = self.create_documents_from_data()
            
            if not documents:
                logger.warning("No hay documentos para vectorizar")
                return
            
            # Split documentos en chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            splits = text_splitter.split_documents(documents)
            
            logger.info(f"✓ Creados {len(splits)} chunks de documentos")
            
            # Crear embeddings y vectorstore
            embedding = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=self.api_key
            )
            
            self.vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=embedding
            )
            
            logger.info("✓ Vectorstore FAISS creado exitosamente")
            
            # Crear retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Función para formatear documentos
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # Template RAG para SodAI Drinks
            rag_template = '''Eres un asistente experto en análisis de datos de SodAI Drinks.
Tu rol es responder preguntas del usuario usando ÚNICAMENTE la información relevante que se te proporciona.

REGLAS IMPORTANTES:
1. Responde SOLO con información de la fuente proporcionada
2. Si no tienes suficiente información, di "No tengo suficiente información para responder eso"
3. Usa números exactos cuando estén disponibles
4. Sé conciso pero completo
5. Usa formato markdown para mejor legibilidad

Información relevante: {context}

Pregunta del usuario: {question}

Respuesta útil y precisa:'''
            
            rag_prompt = PromptTemplate.from_template(rag_template)
            
            # Chain RAG
            self.rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | rag_prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("✓ Sistema RAG con Google Gemini configurado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al configurar RAG: {str(e)}")
            self.rag_chain = None
    
    def answer_question(self, question: str, include_sources: bool = False) -> Dict:
        """
        Responde preguntas usando RAG con Google Gemini
        
        Args:
            question: Pregunta del usuario
            include_sources: Si incluir documentos fuente
            
        Returns:
            Diccionario con respuesta y metadatos
        """
        # Intentar respuesta rápida con fallback PRIMERO para preguntas básicas
        question_lower = question.lower()
        
        # Patrones de preguntas básicas que se responden mejor con fallback
        basic_patterns = [
            "cuantos clientes", "cuántos clientes", "cantidad de clientes", "numero de clientes", "número de clientes",
            "cuantos productos", "cuántos productos", "cantidad de productos", "numero de productos", "número de productos",
            "cuantas transacciones", "cuántas transacciones", "cantidad de transacciones", "total de transacciones",
            "numero de transacciones", "número de transacciones", "estadisticas", "estadísticas", "resumen"
        ]
        
        if any(pattern in question_lower for pattern in basic_patterns):
            logger.info("Pregunta básica detectada, usando fallback directo")
            return self._fallback_answer(question)
        
        # Si no es pregunta básica y tenemos RAG, usar RAG
        if self.rag_chain is None:
            return self._fallback_answer(question)
        
        try:
            # Invocar chain RAG
            answer = self.rag_chain.invoke(question)
            
            # Obtener documentos fuente si se solicita
            sources = []
            if include_sources and self.vectorstore is not None:
                docs = self.vectorstore.similarity_search(question, k=3)
                sources = [doc.page_content[:200] + "..." for doc in docs]
            
            return {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "model": "gemini-2.0-flash-exp",
                    "method": "RAG",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error en RAG: {str(e)}")
            return self._fallback_answer(question)
    
    def _fallback_answer(self, question: str) -> Dict:
        """
        Respuestas básicas cuando RAG no está disponible
        """
        question_lower = question.lower()
        
        # Estadísticas básicas de clientes
        if ("cuantos clientes" in question_lower or "cuántos clientes" in question_lower or 
            "cantidad de clientes" in question_lower or "numero de clientes" in question_lower or 
            "número de clientes" in question_lower):
            n_clientes = self.df_transacciones['customer_id'].nunique() if self.df_transacciones is not None else 0
            return {
                "answer": f"Hay **{n_clientes:,}** clientes únicos en el dataset de SodAI Drinks.",
                "sources": [],
                "metadata": {"method": "fallback"}
            }
        
        # Estadísticas básicas de productos
        if ("cuantos productos" in question_lower or "cuántos productos" in question_lower or 
            "cantidad de productos" in question_lower or "numero de productos" in question_lower or 
            "número de productos" in question_lower):
            n_productos = self.df_transacciones['product_id'].nunique() if self.df_transacciones is not None else 0
            return {
                "answer": f"Hay **{n_productos:,}** productos únicos en el dataset.",
                "sources": [],
                "metadata": {"method": "fallback"}
            }
        
        # Estadísticas de transacciones
        if "cuántas transacciones" in question_lower or "cantidad de transacciones" in question_lower or "total de transacciones" in question_lower:
            n_trans = len(self.df_transacciones) if self.df_transacciones is not None else 0
            return {
                "answer": f"Se han registrado **{n_trans:,}** transacciones en total en SodAI Drinks.",
                "sources": [],
                "metadata": {"method": "fallback"}
            }
        
        # Estadísticas de transacciones
        if ("cuantas transacciones" in question_lower or "cuántas transacciones" in question_lower or 
            "cantidad de transacciones" in question_lower or "total de transacciones" in question_lower or 
            "numero de transacciones" in question_lower or "número de transacciones" in question_lower):
            n_trans = len(self.df_transacciones) if self.df_transacciones is not None else 0
            return {
                "answer": f"Se han registrado **{n_trans:,}** transacciones en total en SodAI Drinks.",
                "sources": [],
                "metadata": {"method": "fallback"}
            }
        
        # Producto más vendido
        if ("producto más vendido" in question_lower or "producto mas vendido" in question_lower or 
            ("qué producto" in question_lower and "vendido" in question_lower) or 
            ("que producto" in question_lower and "vendido" in question_lower)):
            if self.df_transacciones is not None:
                top_product = self.df_transacciones['product_id'].value_counts().head(1)
                if not top_product.empty:
                    product_id = int(top_product.index[0])
                    count = int(top_product.values[0])
                    return {
                        "answer": f"El producto más vendido es el **producto {product_id}** con **{count:,}** transacciones.",
                        "sources": [],
                        "metadata": {"method": "fallback"}
                    }
        
        # Cliente con más compras
        if (("cliente" in question_lower and ("más compras" in question_lower or "mas compras" in question_lower)) or
            ("cliente" in question_lower and ("mas compras" in question_lower))):
            if self.df_transacciones is not None:
                top_customer = self.df_transacciones.groupby('customer_id').size().sort_values(ascending=False).head(1)
                if not top_customer.empty:
                    customer_id = int(top_customer.index[0])
                    count = int(top_customer.values[0])
                    return {
                        "answer": f"El cliente con más compras es el **cliente {customer_id}** con **{count:,}** transacciones.",
                        "sources": [],
                        "metadata": {"method": "fallback"}
                    }
        
        # Estadísticas generales
        if "estadisticas" in question_lower or "estadísticas" in question_lower or "resumen" in question_lower:
            if self.df_transacciones is not None:
                n_clientes = self.df_transacciones['customer_id'].nunique()
                n_productos = self.df_transacciones['product_id'].nunique()
                n_trans = len(self.df_transacciones)
                avg_items = self.df_transacciones.groupby('customer_id').size().mean()
                
                return {
                    "answer": f"""📊 **Estadísticas Generales de SodAI Drinks:**

- 👥 Clientes únicos: **{n_clientes:,}**
- 🥤 Productos únicos: **{n_productos:,}**
- 🛒 Total de transacciones: **{n_trans:,}**
- 📈 Promedio de compras por cliente: **{avg_items:.1f}**

💡 *Para análisis más detallados, configura GOOGLE_API_KEY para habilitar el sistema RAG inteligente.*""",
                    "sources": [],
                    "metadata": {"method": "fallback"}
                }
        
        # Mensaje por defecto
        available_queries = """
📋 **Preguntas que puedo responder sin RAG:**
- ¿Cuántos clientes/productos/transacciones hay?
- ¿Cuál es el producto más vendido?
- ¿Cuál es el cliente con más compras?
- Dame estadísticas generales

💡 *Para respuestas más inteligentes y conversacionales, configura GOOGLE_API_KEY en el archivo .env*
"""
        
        return {
            "answer": available_queries,
            "sources": [],
            "metadata": {"method": "fallback", "info": "limited_mode"}
        }



# Instancia global del sistema RAG
qa_system = GeminiRAGSystem()


@app.get("/")
async def root():
    return {
        "message": "Chatbot Conversacional con Google Gemini - SodAI Drinks",
        "status": "online",
        "version": "2.0.0",
        "model": "gemini-2.0-flash-exp",
        "rag_enabled": qa_system.rag_chain is not None,
        "data_loaded": qa_system.df_transacciones is not None
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_enabled": qa_system.rag_chain is not None,
        "data_loaded": qa_system.df_transacciones is not None,
        "num_transactions": len(qa_system.df_transacciones) if qa_system.df_transacciones is not None else 0
    }


@app.post("/api/v1/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Responde preguntas sobre los datos usando RAG con Google Gemini
    """
    try:
        logger.info(f"Pregunta recibida: {request.question}")
        
        result = qa_system.answer_question(
            question=request.question,
            include_sources=request.include_sources
        )
        
        return QuestionResponse(
            question=request.question,
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Error al procesar pregunta: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
