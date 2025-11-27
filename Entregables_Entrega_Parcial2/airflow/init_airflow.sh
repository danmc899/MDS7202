#!/bin/bash
# Script de inicialización robusto para Windows (WSL), Linux y macOS
# Detecta el sistema operativo y configura AIRFLOW_UID apropiadamente

set -e

echo "🔍 Detectando sistema operativo..."

# Detectar OS
OS=$(uname -s)
echo "Sistema detectado: $OS"

# Verificar si estamos en WSL
IS_WSL=false
if grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
    IS_WSL=true
    echo "🪟 Ejecutando en WSL (Windows Subsystem for Linux)"
fi

# Configurar AIRFLOW_UID según el OS
if [[ "$IS_WSL" == true ]]; then
    echo "🪟 Sistema WSL detectado"
    AIRFLOW_UID=1000
    echo "AIRFLOW_UID configurado a: $AIRFLOW_UID"
elif [[ "$OS" == "Linux" ]]; then
    echo "🐧 Sistema Linux detectado"
    AIRFLOW_UID=1000
    echo "AIRFLOW_UID configurado a: $AIRFLOW_UID"
elif [[ "$OS" == "Darwin" ]]; then
    echo "🍎 Sistema macOS detectado"
    # En macOS, usar UID actual del usuario (típicamente 501, 502, etc.)
    AIRFLOW_UID=$(id -u)
    echo "AIRFLOW_UID configurado a: $AIRFLOW_UID (UID del usuario actual)"
else
    echo "⚠️  Sistema no reconocido, usando UID por defecto"
    AIRFLOW_UID=50000
fi

# Escribir .env
echo "📝 Escribiendo archivo .env..."
cat > .env <<EOF
AIRFLOW_UID=$AIRFLOW_UID
EOF

echo "✅ Archivo .env creado con AIRFLOW_UID=$AIRFLOW_UID"

# Crear directorios necesarios si no existen
echo "📁 Creando directorios necesarios..."
mkdir -p dags logs plugins data models mlruns diagrams config

# Ajustar permisos
echo "🔐 Ajustando permisos de directorios..."
chmod -R 777 logs dags plugins data models mlruns diagrams 2>/dev/null || {
    echo "⚠️  No se pudieron ajustar todos los permisos (puede requerir sudo)"
}

echo ""
echo "✅ Inicialización completada exitosamente"
echo "📌 AIRFLOW_UID final: $AIRFLOW_UID"
echo ""
echo "Siguiente paso: ejecutar 'docker-compose up -d'"
