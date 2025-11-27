FROM python:3.10-slim

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . /app

# Exponer puerto de Gradio
EXPOSE 7860

# Variables de entorno
ENV BACKEND_URL=http://backend:8000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
