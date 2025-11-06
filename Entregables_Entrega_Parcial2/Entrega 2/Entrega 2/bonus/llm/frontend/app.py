"""
Frontend Gradio para Chatbot Conversacional
"""
import gradio as gr
import requests
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL del backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8002")


def ask_question(question: str, history: list):
    """
    Envía pregunta al backend y obtiene respuesta
    """
    try:
        if not question or not question.strip():
            return history, ""
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/ask",
            json={"question": question},
            timeout=10
        )
        
        response.raise_for_status()
        data = response.json()
        
        # Agregar a historial
        history.append((question, data["answer"]))
        
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


# Ejemplos de preguntas
examples = [
    ["¿Cuántos clientes únicos hay en el dataset?"],
    ["¿Cuántas transacciones ha realizado el cliente C0001?"],
    ["¿Cuántos productos únicos se encuentran en los datos?"],
    ["¿Cuál es el producto más vendido?"],
    ["¿Quién es el cliente con más compras?"],
    ["Dame las estadísticas generales"],
]


# Crear interfaz
with gr.Blocks(theme=gr.themes.Soft(), title="Chatbot - SodAI Drinks") as interface:
    gr.Markdown("""
    # 💬 Chatbot Conversacional
    ## SodAI Drinks - Asistente Inteligente de Datos
    
    Pregunta cualquier cosa sobre nuestros datos de ventas. El chatbot analizará tu pregunta
    y te proporcionará respuestas basadas en datos reales.
    """)
    
    chatbot = gr.Chatbot(
        label="Conversación",
        height=400,
        show_label=True,
        avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/robot_1f916.png")
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Tu Pregunta",
                placeholder="Escribe tu pregunta aquí...",
                lines=2,
                show_label=False
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("📤 Enviar", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Limpiar", variant="secondary")
    
    gr.Markdown("### 💡 Ejemplos de Preguntas:")
    gr.Examples(
        examples=examples,
        inputs=question_input,
        label=""
    )
    
    gr.Markdown("""
    ### 🎯 Tipos de Preguntas que Puedo Responder:
    
    - 📊 **Estadísticas Generales**: Resúmenes y métricas del dataset
    - 👥 **Clientes**: Información sobre clientes específicos o totales
    - 📦 **Productos**: Datos sobre productos y ventas
    - 💰 **Transacciones**: Consultas sobre transacciones realizadas
    - 🏆 **Rankings**: Mejores clientes, productos más vendidos, etc.
    
    ### 🤖 Tecnología:
    
    Este chatbot utiliza **procesamiento de lenguaje natural** (NLP) con pattern matching
    para entender tus preguntas y consultar la base de datos en tiempo real.
    
    ---
    
    **Autores**: Javier Pinochet & Daniel Muñoz  
    **Curso**: MDS7202 - Universidad de Chile
    """)
    
    # Event handlers
    submit_btn.click(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input]
    )
    
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, chatbot],
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
