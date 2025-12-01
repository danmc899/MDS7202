"""
Componentes de la interfaz Gradio
Define tabs y elementos visuales
"""
import gradio as gr
from utils.api_client import APIClient
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Inicializar cliente API
api_client = APIClient()


def create_single_prediction_tab():
    """
    Crea el tab para predicciones individuales con listas desplegables
    """
    with gr.Tab("🔮 Predicción Individual"):
        gr.Markdown("""
        ### Predicción Individual
        Predice si un cliente específico comprará un producto **que ya ha comprado antes** la próxima semana.
        
        **Instrucciones:**
        1. Selecciona el ID del cliente de la lista
        2. Selecciona el ID del producto (solo productos históricos del cliente)
        3. Click en "Predecir"
        
        ℹ️ **Nota:** Solo se muestran productos del histórico del cliente para evitar predicciones 
        sin sentido en combinaciones nunca vistas.
        """)
        
        with gr.Row():
            with gr.Column():
                # Obtener listas de clientes y productos disponibles
                try:
                    available_data = api_client.get_available_ids()
                    customer_choices = available_data.get('customer_ids', [])
                    product_choices = available_data.get('product_ids', [])
                except Exception as e:
                    logger.warning(f"No se pudieron cargar IDs disponibles: {e}")
                    customer_choices = []
                    product_choices = []
                
                customer_id = gr.Dropdown(
                    choices=customer_choices,
                    label="ID del Cliente",
                    info="Selecciona un cliente de la lista",
                    allow_custom_value=False,
                    filterable=True
                )
                
                product_id = gr.Dropdown(
                    choices=[],  # Se llenará dinámicamente
                    label="ID del Producto (históricos del cliente)",
                    info="Productos que este cliente ha comprado antes",
                    allow_custom_value=False,
                    filterable=True
                )
                
                predict_btn = gr.Button("🔮 Predecir", variant="primary", size="lg")
            
            with gr.Column():
                output = gr.JSON(label="Resultado de la Predicción")
                
                gr.Markdown("""
                ### 📊 Interpretación de Resultados
                
                - **will_purchase**: `true` = El cliente comprará, `false` = No comprará
                - **probability**: Probabilidad de compra (0-1). Valores más altos indican mayor confianza.
                - **model_type**: Tipo de modelo usado para la predicción
                
                #### Niveles de Probabilidad:
                - 🟢 **Alta (>0.7)**: Muy probable que compre
                - 🟡 **Media (0.4-0.7)**: Posibilidad moderada
                - 🔴 **Baja (<0.4)**: Poco probable que compre
                
                #### 💡 Casos de Uso:
                - **Recompra**: Predice si un cliente volverá a comprar un producto
                - **Campañas dirigidas**: Enfoca marketing en clientes con alta probabilidad
                - **Gestión de inventario**: Anticipa demanda de productos frecuentes
                """)
        
        # Función para actualizar productos según cliente seleccionado
        def update_products(customer_id_value):
            if not customer_id_value:
                return gr.Dropdown(choices=[], value=None)
            
            try:
                # Obtener productos históricos del cliente
                customer_products = api_client.get_customer_products(int(customer_id_value))
                if 'product_ids' in customer_products and customer_products['product_ids']:
                    product_choices = [str(pid) for pid in customer_products['product_ids']]
                    return gr.Dropdown(choices=product_choices, value=None)
            except Exception as e:
                logger.error(f"Error al obtener productos del cliente: {e}")
            
            return gr.Dropdown(choices=[], value=None)
        
        def predict_single(cust_id, prod_id):
            try:
                # Validar que se hayan seleccionado IDs
                if not cust_id or not prod_id:
                    return {"error": "Por favor selecciona un cliente y un producto"}
                
                result = api_client.predict_single(
                    customer_id=str(cust_id),
                    product_id=str(prod_id)
                )
                return result
            except Exception as e:
                return {"error": str(e)}
        
        # Conectar evento de cambio de cliente para actualizar productos
        customer_id.change(
            fn=update_products,
            inputs=[customer_id],
            outputs=[product_id]
        )
        
        predict_btn.click(
            fn=predict_single,
            inputs=[customer_id, product_id],
            outputs=output
        )



def create_batch_prediction_tab():
    """
    Crea el tab para predicciones en lote
    """
    with gr.Tab("📦 Predicción en Lote"):
        gr.Markdown("""
        ### Predicción en Lote
        Realiza múltiples predicciones simultáneas cargando un archivo CSV.
        
        **Formato del CSV:**
        El archivo debe contener las columnas: `customer_id`, `product_id`
        
        Ejemplo:
        ```csv
        customer_id,product_id
        25734,1370
        25734,50716
        25743,1304
        ```
        
        ⚠️ **Importante:** Solo se procesarán pares cliente-producto que existan en el histórico.
        Los pares no históricos serán omitidos automáticamente para evitar predicciones contaminadas.
        """)
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="📁 Cargar archivo CSV",
                    file_types=[".csv"],
                    type="filepath"
                )
                
                with gr.Row():
                    only_positive = gr.Checkbox(
                        label="Solo predicciones positivas",
                        value=True,
                        info="Mostrar solo clientes que comprarán"
                    )
                    sort_by_prob = gr.Checkbox(
                        label="Ordenar por probabilidad",
                        value=True,
                        info="Ordenar resultados de mayor a menor probabilidad"
                    )
                
                predict_batch_btn = gr.Button("📦 Predecir Lote", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 💡 Tips:
                - Archivos grandes pueden tardar más en procesarse
                - Se recomienda lotes de hasta 1000 registros
                - Puedes descargar los resultados como CSV
                """)
            
            with gr.Column():
                output_df = gr.Dataframe(
                    label="Resultados",
                    headers=["customer_id", "product_id", "will_purchase", "probability"],
                    interactive=False
                )
                
                stats_output = gr.JSON(label="Estadísticas del Lote")
        
        def predict_batch(file_path, only_pos, sort_prob):
            try:
                if file_path is None:
                    return None, {"error": "Por favor carga un archivo CSV"}
                
                # Leer CSV
                df = pd.read_csv(file_path)
                
                # Validar columnas requeridas
                required_cols = ['customer_id', 'product_id']
                if not all(col in df.columns for col in required_cols):
                    return None, {"error": f"El CSV debe contener las columnas: {required_cols}"}
                
                # Realizar predicción
                result = api_client.predict_batch(
                    data=df.to_dict('records'),
                    only_positive=only_pos,
                    sort_by_probability=sort_prob
                )
                
                # Convertir a DataFrame
                if 'predictions' in result:
                    predictions_df = pd.DataFrame(result['predictions'])
                    
                    stats = {
                        "total_predicciones": result.get('total_predictions', 0),
                        "tipo_modelo": result.get('model_type', 'unknown'),
                        "predicciones_positivas": len([p for p in result['predictions'] if p['will_purchase']]),
                        "probabilidad_promedio": sum(p['probability'] for p in result['predictions']) / len(result['predictions']) if result['predictions'] else 0
                    }
                    
                    return predictions_df, stats
                else:
                    return None, result
                    
            except Exception as e:
                logger.error(f"Error en predicción en lote: {str(e)}")
                return None, {"error": str(e)}
        
        predict_batch_btn.click(
            fn=predict_batch,
            inputs=[file_input, only_positive, sort_by_prob],
            outputs=[output_df, stats_output]
        )


# ELIMINADO: Sistema de recomendaciones existe como bonus separado
# def create_recommendations_tab():
#     """
#     Crea el tab para recomendaciones de productos
#     """
#     with gr.Tab("⭐ Recomendaciones"):
        gr.Markdown("""
        ### Recomendaciones de Productos
        Obtén los productos más recomendados para un cliente específico.
        
        **¿Cómo funciona?**
        El sistema analiza el historial del cliente y predice qué productos tiene más probabilidad de comprar.
        """)
        
        with gr.Row():
            with gr.Column():
                customer_id_rec = gr.Textbox(
                    label="ID del Cliente",
                    placeholder="Ej: C12345",
                    info="Identificador único del cliente"
                )
                
                top_n = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Número de Recomendaciones",
                    info="¿Cuántos productos recomendar?"
                )
                
                recommend_btn = gr.Button("⭐ Obtener Recomendaciones", variant="primary", size="lg")
            
            with gr.Column():
                recommendations_output = gr.Dataframe(
                    label="Top Productos Recomendados",
                    headers=["Ranking", "product_id", "probability", "will_purchase"],
                    interactive=False
                )
        
        def get_recommendations(cust_id, n):
            try:
                result = api_client.get_recommendations(customer_id=cust_id, top_n=n)
                
                if 'recommendations' in result:
                    recs = result['recommendations']
                    df = pd.DataFrame(recs)
                    df.insert(0, 'Ranking', range(1, len(df) + 1))
                    return df
                else:
                    return pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"Error al obtener recomendaciones: {str(e)}")
                return pd.DataFrame({"error": [str(e)]})
        
        recommend_btn.click(
            fn=get_recommendations,
            inputs=[customer_id_rec, top_n],
            outputs=recommendations_output
        )


def create_model_info_tab():
    """
    Crea el tab para información del modelo
    """
    with gr.Tab("ℹ️ Info del Modelo"):
        gr.Markdown("""
        ### Información del Modelo
        Detalles sobre el modelo de Machine Learning actualmente en uso.
        """)
        
        refresh_btn = gr.Button("🔄 Actualizar Información", variant="secondary")
        model_info_output = gr.JSON(label="Información del Modelo")
        
        gr.Markdown("""
        ### 📚 Sobre el Sistema
        
        Este sistema de predicción utiliza técnicas avanzadas de Machine Learning para:
        
        - ✅ Predecir comportamiento de compra de clientes
        - ✅ Optimizar estrategias de marketing
        - ✅ Mejorar gestión de inventario
        - ✅ Personalizar experiencia del cliente
        
        **Tecnologías utilizadas:**
        - Apache Airflow (Pipeline automatizado)
        - MLflow (Tracking de experimentos)
        - Optuna (Optimización de hiperparámetros)
        - SHAP (Interpretabilidad)
        - FastAPI (Backend)
        - Gradio (Frontend)
        
        **Autores:** Javier Pinochet & Daniel Muñoz  
        **Curso:** MDS7202 - Universidad de Chile
        """)
        
        def get_model_info():
            try:
                return api_client.get_model_info()
            except Exception as e:
                return {"error": str(e)}
        
        # Cargar info al inicio
        model_info_output.value = get_model_info()
        
        refresh_btn.click(
            fn=get_model_info,
            inputs=[],
            outputs=model_info_output
        )


def create_interface():
    """
    Crea la interfaz completa de Gradio
    """
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="SodAI Drinks - Predicción de Compras",
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🥤 SodAI Drinks - Sistema de Predicción de Compras
        
        ## Bienvenido al sistema inteligente de predicción de compras
        
        Este sistema utiliza Machine Learning para predecir qué productos comprarán los clientes la próxima semana,
        permitiendo optimizar inventario, marketing y experiencia del cliente.
        """)
        
        # Crear tabs
        create_single_prediction_tab()
        create_batch_prediction_tab()
        create_model_info_tab()
        
        gr.Markdown("""
        ---
        ### 📞 Soporte
        
        ¿Problemas o preguntas? Contacta al equipo de Deep Drinkers 🤖
        
        **Nota:** Este sistema está en producción y se actualiza automáticamente cada semana con nuevos datos.
        """)
    
    return interface
