"""
Aplicación Frontend con Gradio
Interfaz web para interactuar con el sistema de predicción
"""
import logging
from components.interface import create_interface

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """
    Función principal para ejecutar la aplicación Gradio
    """
    logger.info("Iniciando aplicación Gradio...")
    
    # Crear interfaz
    interface = create_interface()
    
    # Lanzar aplicación
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
