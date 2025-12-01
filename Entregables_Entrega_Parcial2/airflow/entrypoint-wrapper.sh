#!/bin/bash
# Script wrapper para configurar umask antes de ejecutar comandos de Airflow
# Esto asegura que todos los archivos/directorios creados tengan permisos de grupo-escritura

# Establecer umask para que los nuevos archivos sean grupo-escribibles
umask 002

# Ejecutar el comando original de Airflow
exec "$@"
