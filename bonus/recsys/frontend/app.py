"""
Frontend Gradio para Sistema de Recomendación
"""
import gradio as gr
import requests
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL del backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8001")


def get_available_customers():
    """
    Obtiene lista de clientes disponibles desde el backend
    """
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/customers", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("customers", [])
    except Exception as e:
        logger.error(f"Error obteniendo clientes: {e}")
        return []


def get_recommendations(customer_id: str, n_recommendations: int):
    """
    Obtiene recomendaciones desde el backend
    """
    try:
        if not customer_id:
            return pd.DataFrame({"error": ["Por favor ingresa un ID de cliente"]})
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/recommend",
            json={
                "customer_id": customer_id,
                "n_recommendations": int(n_recommendations)
            },
            timeout=10
        )
        
        response.raise_for_status()
        data = response.json()
        
        # Convertir a DataFrame
        if "recommendations" in data:
            df = pd.DataFrame(data["recommendations"])
            return df
        else:
            return pd.DataFrame({"error": ["No se encontraron recomendaciones"]})
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión: {str(e)}")
        return pd.DataFrame({"error": [f"Error de conexión con el backend: {str(e)}"]})
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return pd.DataFrame({"error": [str(e)]})


# Crear interfaz
with gr.Blocks(theme=gr.themes.Soft(), title="Sistema de Recomendación SVD - SodAI Drinks") as interface:
    gr.Markdown("""
    # 🎯 Sistema de Recomendación de Productos (SVD)
    ## SodAI Drinks
    
    Este sistema utiliza **SVD (Singular Value Decomposition)** con Matrix Factorization para recomendar productos basándose en:
    - Historial de compras del cliente
    - Factorización de la matriz de ratings implícitos
    - Patrones latentes descubiertos por SVD
    """)
    
    with gr.Row():
        with gr.Column():
            # Obtener lista de clientes disponibles
            available_customers = get_available_customers()
            
            if available_customers:
                customer_id = gr.Dropdown(
                    choices=available_customers,
                    label="ID del Cliente",
                    info="Selecciona un cliente de la lista",
                    allow_custom_value=True,
                    filterable=True
                )
            else:
                customer_id = gr.Textbox(
                    label="ID del Cliente",
                    placeholder="Ej: 181101",
                    info="Ingresa el identificador único del cliente"
                )
            
            n_recommendations = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Número de Recomendaciones",
                info="¿Cuántos productos quieres recomendar?"
            )
            
            recommend_btn = gr.Button("🎯 Obtener Recomendaciones", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 💡 Cómo funciona:
            
            1. **Análisis de Historial**: Revisamos qué ha comprado el cliente
            2. **Similaridad de Productos**: Encontramos productos similares
            3. **Scoring**: Calculamos puntuaciones basadas en preferencias
            4. **Ranking**: Ordenamos y mostramos las mejores recomendaciones
            
            ### 📊 Interpretación:
            
            - **Rank**: Posición en el ranking (1 = mejor recomendación)
            - **Score**: Puntuación de relevancia (mayor = más recomendado)
            - **Product ID**: Identificador del producto recomendado
            """)
        
        with gr.Column():
            output = gr.Dataframe(
                label="Productos Recomendados",
                headers=["rank", "product_id", "score"],
                interactive=False
            )
            
            gr.Markdown("""
            ### 🚀 Casos de Uso:
            
            - **Marketing Personalizado**: Enviar ofertas específicas
            - **Cross-Selling**: Sugerir productos complementarios
            - **Aumento de Ventas**: Incrementar ticket promedio
            - **Retención**: Mejorar experiencia del cliente
            """)
    
    recommend_btn.click(
        fn=get_recommendations,
        inputs=[customer_id, n_recommendations],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### 📞 Información
    
    **Sistema**: SVD (Singular Value Decomposition) con Matrix Factorization  
    **Autores**: Javier Pinochet & Daniel Muñoz  
    **Curso**: MDS7202 - Universidad de Chile
    """)


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
