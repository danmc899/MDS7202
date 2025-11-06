FROM apache/airflow:2.7.3-python3.10

USER root

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copiar requirements
COPY requirements.txt /requirements.txt

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r /requirements.txt

# Crear directorios necesarios
RUN mkdir -p /opt/airflow/dags \
    /opt/airflow/logs \
    /opt/airflow/plugins \
    /opt/airflow/data/raw \
    /opt/airflow/data/processed \
    /opt/airflow/data/predictions \
    /opt/airflow/models \
    /opt/airflow/mlruns \
    /opt/airflow/diagrams

# Configurar permisos
USER root
RUN chown -R airflow: /opt/airflow
USER airflow
