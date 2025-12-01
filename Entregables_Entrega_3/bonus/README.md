# Módulo Bonus: RecSys y LLM con RAG

## Descripción

Dos sistemas avanzados que extienden el pipeline principal: sistema de recomendación colaborativo basado en SVD y chatbot con LLM y Retrieval-Augmented Generation.

## Sistema de Recomendación (RecSys)

### Arquitectura
Collaborative filtering mediante SVD (Singular Value Decomposition) que descompone la matriz de interacciones cliente-producto en factores latentes, permitiendo predecir ratings para combinaciones no observadas.

### Implementación
- **Algoritmo**: `surprise.SVD` con 100 factores latentes, regularización 0.02, 20 epochs
- **Datos**: Matriz dispersa de ratings cliente-producto desde transacciones
- **Salida**: Top-N recomendaciones ordenadas por rating predicho

### Endpoint
```python
POST /recommend
{
  "customer_id": "C123",
  "top_n": 5
}
```
Retorna lista de productos recomendados con ratings estimados.

### Métricas
- **RMSE**: Error en predicción de ratings
- **Coverage**: % de pares cliente-producto para los que puede recomendar

## Chatbot LLM con RAG

### Arquitectura
Combina búsqueda de información relevante (retrieval) con generación de texto (LLM) para responder preguntas sobre productos y clientes con contexto actualizado.

### Pipeline RAG
1. **Indexación**: Embeddings de descripciones de productos y perfiles de clientes con `sentence-transformers`
2. **Retrieval**: Búsqueda de top-k documentos más similares a la query usando FAISS
3. **Augmentation**: Inyecta documentos recuperados en el prompt del LLM
4. **Generation**: LLM genera respuesta condicionada en el contexto

### Ventajas sobre LLM puro
- **Actualización sin reentrenamiento**: Nueva información se agrega al índice
- **Reducción de alucinaciones**: LLM responde basado en documentos reales
- **Trazabilidad**: Cada respuesta incluye fuentes de información

### Endpoint
```python
POST /chat
{
  "query": "¿Qué productos compran clientes de la región Metropolitana?"
}
```
Retorna respuesta generada con contexto de documentos relevantes.

### Configuración
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dims)
- **LLM**: gemini-flash-2.0
- **Vector DB**: FAISS con índice IVF para escalabilidad

## Orquestación Docker

Ambos sistemas se ejecutan como servicios independientes en `docker-compose.yml`:

```yaml
services:
  recsys-backend:
    ports: ["8001:8000"]
    volumes: ["./data:/app/data"]
  
  llm-chatbot:
    ports: ["8002:8000"]
    volumes: ["./embeddings:/app/embeddings"]
```

## Integración con Pipeline Principal

- **RecSys**: Lee archivos parquet procesados por Airflow cuando detecta señal de sincronización
- **LLM**: Indexa datos de productos y clientes desde las mismas fuentes

Esta arquitectura loose-coupled permite evolución independiente de cada sistema sin romper el pipeline principal.

## Mejoras Futuras

### RecSys
- Hybrid filtering (combinar collaborative + content-based)
- Contextual bandits para exploración vs explotación
- Métricas de negocio (CTR, conversión) además de RMSE

### LLM
- Fine-tuning con datos específicos del dominio
- Multi-turn conversation con memoria de sesión
- Evaluación automática de calidad de respuestas (RAGAS)
