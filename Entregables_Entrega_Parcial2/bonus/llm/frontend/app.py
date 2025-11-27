"""
Frontend Gradio para Chatbot Conversacional con Google Gemini
Sistema RAG para consultas sobre datos de SodAI Drinks
"""
import gradio as gr
import requests
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL del backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8002")


def ask_question(question: str, history: list, include_sources: bool):
    """
    Envía pregunta al backend RAG y obtiene respuesta de Google Gemini
    """
    try:
        if not question or not question.strip():
            return history, ""
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/ask",
            json={
                "question": question,
                "include_sources": include_sources
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        # Formatear respuesta
        answer = data["answer"]
        
        # Agregar fuentes si están disponibles
        if data.get("sources") and include_sources:
            answer += "\n\n📚 **Fuentes:**\n"
            for i, source in enumerate(data["sources"], 1):
                answer += f"{i}. {source}\n"
        
        # Agregar metadata
        metadata = data.get("metadata", {})
        if metadata.get("model"):
            answer += f"\n\n🤖 *Modelo: {metadata['model']}*"
        
        # Agregar a historial
        history.append((question, answer))
        
        return history, ""
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión: {str(e)}")
        error_msg = f"❌ Error de conexión con el backend: {str(e)}"
        history.append((question, error_msg))
        return history, ""
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        error_msg = f"❌ Error: {str(e)}"
        history.append((question, error_msg))
        return history, ""


# Ejemplos de preguntas optimizadas para RAG
examples = [
    ["¿Cuántos clientes únicos hay en el dataset de SodAI Drinks?"],
    ["¿Cuál es el cliente con más compras y cuántos items ha comprado?"],
    ["¿Qué producto es el más vendido en SodAI Drinks?"],
    ["Dame estadísticas generales sobre las transacciones"],
    ["¿Cuántas transacciones se realizaron en total?"],
    ["¿Cuál es el promedio de items por transacción?"],
]


# Crear interfaz con tema moderno
with gr.Blocks(theme=gr.themes.Soft(), title="Chatbot RAG - SodAI Drinks", css="""
    .gradio-container {font-family: 'Inter', sans-serif;}
    .chat-message {border-radius: 10px; padding: 10px;}
""") as interface:
    
    gr.Markdown("""
    # 🤖 Chatbot Conversacional con Google Gemini
    ## SodAI Drinks - Asistente Inteligente RAG
    
    Este chatbot utiliza **Google Gemini** con **RAG (Retrieval Augmented Generation)** para responder
    preguntas precisas sobre los datos de SodAI Drinks. El sistema busca información relevante en la base 
    de datos vectorizada y genera respuestas contextualizadas.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 Conversación",
                height=500,
                show_label=True,
                avatar_images=(
                    "https://em-content.zobj.net/source/twitter/376/man-technologist_1f468-200d-1f4bb.png",
                    "https://em-content.zobj.net/source/twitter/376/robot_1f916.png"
                ),
                bubble_full_width=False
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuración")
            
            include_sources_check = gr.Checkbox(
                label="Incluir fuentes en las respuestas",
                value=False,
                info="Muestra los documentos fuente utilizados"
            )
            
            gr.Markdown("""
            ### 📊 Información del Sistema
            
            - **Modelo**: Google Gemini 2.0 Flash
            - **Método**: RAG con FAISS
            - **Embeddings**: text-embedding-004
            - **Datos**: Transacciones SodAI Drinks
            
            ### 🎯 Capacidades:
            
            - ✅ Estadísticas de clientes y productos
            - ✅ Rankings y top performers
            - ✅ Análisis de compras por cliente
            - ✅ Información detallada de transacciones
            - ✅ Respuestas contextualizadas con datos reales
            """)
    
    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Tu Pregunta",
                placeholder="Escribe tu pregunta aquí... (ej: ¿Cuál es el cliente con más compras?)",
                lines=2,
                show_label=False
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("📤 Enviar", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Limpiar Chat", variant="secondary")
    
    gr.Markdown("### 💡 Ejemplos de Preguntas:")
    gr.Examples(
        examples=examples,
        inputs=question_input,
        label=""
    )
    
    gr.Markdown("""
    ### 🔬 Tecnología Detrás del Chatbot:
    
    Este sistema implementa **Retrieval Augmented Generation (RAG)**, una técnica avanzada que combina:
    
    1. **Vectorización**: Los datos se convierten en embeddings usando `text-embedding-004`
    2. **Búsqueda Semántica**: FAISS encuentra la información más relevante
    3. **Generación Aumentada**: Google Gemini genera respuestas usando solo información verificada
    
    **Ventajas del RAG:**
    - ✅ Respuestas basadas en datos reales
    - ✅ No inventa información (sin alucinaciones)
    - ✅ Cita fuentes cuando se solicita
    - ✅ Actualizable sin reentrenar el modelo
    
    ---
    
    **Autores**: Javier Pinochet & Daniel Muñoz  
    **Curso**: MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos  
    **Universidad de Chile** - Primavera 2024
    """)
    
    # Event handlers
    submit_btn.click(
        fn=ask_question,
        inputs=[question_input, chatbot, include_sources_check],
        outputs=[chatbot, question_input]
    )
    
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, chatbot, include_sources_check],
        outputs=[chatbot, question_input]
    )
    
    clear_btn.click(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[chatbot, question_input]
    )


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False
    )
