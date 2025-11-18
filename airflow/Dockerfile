FROM apache/airflow:2.7.3-python3.10

USER root

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements file
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy umask wrapper script (must have +x permissions on host)
COPY entrypoint-wrapper.sh /entrypoint-wrapper.sh

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
# Configurar umask para que nuevos archivos/directorios sean grupo-escribibles
RUN echo "umask 002" >> /etc/profile && \
    echo "umask 002" >> /etc/bash.bashrc
USER airflow

# Configurar umask para el usuario airflow
RUN echo "umask 002" >> ~/.bashrc
