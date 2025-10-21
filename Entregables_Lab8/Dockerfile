# 1️⃣ Imagen base oficial de Python
FROM python:3.10-slim

# 2️⃣ Directorio de trabajo dentro del contenedor
WORKDIR /app

# 3️⃣ Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copiar todos los archivos del proyecto (incluye main.py y models/)
COPY . .

# 5️⃣ Exponer el puerto 8000
EXPOSE 8000

# 6️⃣ Comando para ejecutar la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
