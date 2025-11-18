# 🎨 Frontend Gradio - SodAI Drinks

Interfaz web interactiva para realizar predicciones de compra de productos.

## 📋 Descripción

Frontend desarrollado con **Gradio** que proporciona una interfaz amigable para:

- ✅ Predicciones individuales cliente-producto
- ✅ Predicciones en lote mediante CSV
- ✅ Recomendaciones personalizadas de productos
- ✅ Visualización de información del modelo
- ✅ Estadísticas y análisis de resultados

## 🏗️ Estructura

```
frontend/
├── app.py                     # Aplicación principal
├── components/
│   ├── __init__.py
│   └── interface.py           # Componentes de la interfaz
├── utils/
│   ├── __init__.py
│   └── api_client.py          # Cliente para comunicación con backend
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Instalación y Ejecución

### Opción 1: Docker (Recomendado)

```bash
# Desde el directorio frontend/
docker build -t sodai-frontend .
docker run -p 7860:7860 -e BACKEND_URL=http://backend:8000 sodai-frontend
```

### Opción 2: Local (Desarrollo)

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar URL del backend
export BACKEND_URL=http://localhost:8000

# Ejecutar aplicación
python app.py
```

Accede a la aplicación en: http://localhost:7860

## 📱 Funcionalidades

### 1. 🔮 Predicción Individual

Predice si un cliente específico comprará un producto específico.

**Campos:**
- ID del Cliente (requerido)
- ID del Producto (requerido)
- Features opcionales (total items, promedio items, etc.)

**Resultado:**
- Predicción (comprará o no)
- Probabilidad de compra (0-1)
- Tipo de modelo utilizado

### 2. 📦 Predicción en Lote

Realiza múltiples predicciones cargando un archivo CSV.

**Formato CSV esperado:**
```csv
customer_id,product_id,total_items,avg_items
C12345,P6789,100,5.0
C12345,P6790,100,5.0
C67890,P6789,50,2.5
```

**Opciones:**
- Solo predicciones positivas
- Ordenar por probabilidad

**Resultado:**
- Tabla con todas las predicciones
- Estadísticas agregadas del lote

### 3. ⭐ Recomendaciones

Obtén los productos más recomendados para un cliente.

**Entrada:**
- ID del Cliente
- Número de recomendaciones (1-20)

**Resultado:**
- Ranking de productos
- Probabilidades de compra

### 4. ℹ️ Información del Modelo

Visualiza detalles sobre el modelo actualmente en uso:
- Tipo de modelo
- Métricas de desempeño
- Features utilizadas
- Metadata de entrenamiento

## 🎨 Personalización

### Cambiar Tema

En `components/interface.py`, modifica:

```python
with gr.Blocks(
    theme=gr.themes.Soft(),  # Opciones: Soft, Base, Monochrome, Glass
    ...
) as interface:
```

### Modificar URL del Backend

Variable de entorno:
```bash
export BACKEND_URL=http://tu-backend:8000
```

O en el código (`utils/api_client.py`):
```python
self.base_url = "http://tu-backend:8000"
```

## 🔧 Configuración

### Variables de Entorno

```bash
# URL del backend FastAPI
BACKEND_URL=http://backend:8000

# Puerto de Gradio (opcional, default: 7860)
GRADIO_SERVER_PORT=7860
```

## 📊 Capturas de Pantalla

### Predicción Individual
Interfaz simple e intuitiva para predicciones únicas.

### Predicción en Lote
Carga CSV y obtén resultados en segundos.

### Recomendaciones
Sistema de recomendación personalizado.

## 🧪 Testing

### Test Manual

1. Ejecutar aplicación localmente
2. Probar cada tab:
   - Predicción Individual: Ingresar IDs de prueba
   - Lote: Cargar CSV de ejemplo
   - Recomendaciones: Probar con diferentes clientes
   - Info del Modelo: Verificar que se muestra correctamente

### CSV de Ejemplo

Crear `test_data.csv`:
```csv
customer_id,product_id
C001,P001
C001,P002
C002,P001
```

## 🐛 Troubleshooting

### Error: "No se puede conectar al backend"
- Verifica que el backend esté corriendo
- Verifica `BACKEND_URL` apunta a la dirección correcta
- Prueba: `curl http://backend:8000/health`

### Error: "Module not found"
- Asegúrate de instalar todas las dependencias: `pip install -r requirements.txt`
- Verifica que estás en el directorio correcto

### La interfaz no carga
- Revisa los logs: busca mensajes de error
- Verifica que el puerto 7860 no esté en uso
- Prueba con otro puerto: modifica en `app.py`

### Predicciones no funcionan
- Verifica conectividad con backend
- Revisa formato de datos de entrada
- Consulta logs del backend para más detalles

## 💡 Tips de Uso

### Para Mejores Predicciones

1. Proporciona todas las features disponibles
2. Usa datos reales y actualizados
3. Revisa la probabilidad, no solo la predicción binaria

### Para Lotes Grandes

1. Divide en archivos más pequeños (<1000 registros)
2. Activa "Solo predicciones positivas" para reducir salida
3. Descarga resultados como CSV para análisis posterior

## 🔗 Integración

### Conectar con Backend

El frontend se comunica con el backend mediante HTTP REST:

```python
# En utils/api_client.py
response = requests.post(
    f"{self.base_url}/api/v1/predict",
    json=payload
)
```

### Agregar Nuevas Funcionalidades

1. Crea nueva función en `components/interface.py`
2. Agrega método correspondiente en `utils/api_client.py`
3. Integra en `create_interface()`

## 📚 Recursos

- [Documentación de Gradio](https://gradio.app/docs/)
- [Galería de Ejemplos](https://gradio.app/demos/)
- [Guía de Temas](https://gradio.app/theming-guide/)

## 👥 Autores

- **Javier Pinochet**
- **Daniel Muñoz**

**Curso**: MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos  
**Institución**: Universidad de Chile

---

Para reportar bugs o sugerir mejoras, contacta al equipo de desarrollo.
