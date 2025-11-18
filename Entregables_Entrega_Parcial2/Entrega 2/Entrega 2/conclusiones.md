# Conclusiones del Proyecto: MLOps en Producción

## Impacto de las Herramientas de Tracking y Deployment

### MLflow: De Experimentos a Modelos Versionados

La incorporación de MLflow representó un cambio fundamental en la manera de trabajar con modelos de machine learning. Antes de implementar tracking sistemático, los experimentos vivían únicamente en notebooks individuales, con resultados registrados manualmente en comentarios o archivos CSV dispersos. Esta aproximación artesanal generaba problemas severos de reproducibilidad: semanas después, era imposible recordar qué hiperparámetros habían generado cierto resultado, o cómo comparar runs de diferentes días.

MLflow resolvió estos problemas al centralizar automáticamente toda la información relevante de cada experimento. Cada entrenamiento del modelo XGBoost ahora registra automáticamente todos los hiperparámetros (número de estimadores, profundidad máxima, learning rate), las métricas de evaluación (accuracy, precision, recall, F1), y los artefactos generados (diagramas SHAP, matrices de confusión, curvas ROC). Esta información se almacena de manera estructurada con timestamps, tags, y asociaciones a versiones específicas de código y datos.

El valor real de MLflow emerge cuando se comparan múltiples experimentos. La interfaz web permite filtrar runs por fecha, ordenar por métricas de performance, y visualizar cómo diferentes combinaciones de hiperparámetros afectan el resultado. Por ejemplo, descubrimos que aumentar la profundidad máxima de los árboles más allá de 6 niveles no mejoraba accuracy pero sí aumentaba significativamente el tiempo de entrenamiento. Este tipo de insights sería imposible de obtener sin visualización comparativa de decenas de experimentos.

### Gradio y FastAPI: Democratizando el Acceso a Modelos

El deployment de modelos mediante FastAPI y Gradio transformó experimentos técnicos en herramientas utilizables por stakeholders no técnicos. Antes de implementar estas interfaces, los modelos solo podían ser evaluados ejecutando código Python manualmente, lo que limitaba severamente el feedback de usuarios de negocio. Un vendedor o gerente regional no puede (ni debería) abrir Jupyter notebooks para hacer predicciones.

La arquitectura de dos capas (backend API + frontend web) demostró ser particularmente valiosa. FastAPI en el backend provee endpoints REST que exponen el modelo de manera programática, permitiendo integraciones con otros sistemas (CRM, ERPs, dashboards de BI). Gradio en el frontend ofrece una interfaz visual amigable para exploración manual. Esta separación de concerns permite que cada componente evolucione independientemente: podemos mejorar la UI sin tocar el modelo, o actualizar el modelo sin cambiar la interfaz.


### Desafíos en el Deployment

El proceso de deployment no estuvo exento de desafíos. El problema más complejo fue asegurar consistencia en el preprocesamiento de features entre train-time y inference-time. Durante el entrenamiento, las features se transforman usando ColumnTransformer con scalers y encoders que aprenden parámetros de los datos (medias, desviaciones standard, categorías válidas). Si estas transformaciones no se replican exactamente durante la inferencia, el modelo recibe inputs en un espacio diferente al que fue entrenado, degradando performance silenciosamente.

Resolvimos este problema guardando el pipeline de preprocesamiento completo como parte del artefacto de MLflow, no solo el modelo XGBoost. El backend FastAPI carga este pipeline y lo aplica a cada request de predicción, garantizando que las transformaciones sean idénticas. Sin embargo, este approach requirió diseño cuidadoso: el pipeline debe manejar  categorías nuevas no vistas durante el entrenamiento (usando handle_unknown='ignore' en OneHotEncoder) y valores faltantes (usando estrategias de imputación apropiadas).

Otro desafío fue el manejo de volúmenes compartidos entre Airflow y la aplicación de predicción. Ambos sistemas necesitan acceder al modelo entrenado, pero viven en contenedores Docker separados. La solución de compartir un volumen Docker funcionó, pero requirió implementar lógica de auto-reload en el backend para detectar cuando Airflow publica un modelo nuevo. Sin este mecanismo, la aplicación continuaría usando versiones obsoletas del modelo, generando predicciones subóptimas.

## Contribución de Airflow a la Robustez y Escalabilidad

### Orquestación de Pipelines Complejos

Apache Airflow transformó lo que antes era un conjunto de scripts Python desconectados en un pipeline cohesivo con dependencias explícitas y manejo robusto de errores. La estructura DAG (Directed Acyclic Graph) representa cada paso del pipeline como una tarea: extract data, process data, detect drift, train model, evaluate model, sync to RecSys. Las flechas entre tareas definen el orden de ejecución, asegurando que etapas dependientes nunca se ejecuten si sus prerequisitos fallaron.

Esta explicitación de dependencias elimina una clase completa de bugs comunes en pipelines ad-hoc. Por ejemplo, en una implementación con scripts independientes, es posible ejecutar accidentalmente el training antes de completar el procesamiento de datos, generando un modelo entrenado con features incompletas o corruptas. En Airflow, esto es imposible: la tarea model_training no iniciará hasta que process_data reporte éxito explícitamente.

El sistema de retry automático de Airflow mejoró significativamente la robustez del pipeline frente a errores transitorios. La conexión a bases de datos externos, lectura de archivos de red, o llamadas a APIs pueden fallar ocasionalmente por razones fuera de nuestro control (timeouts de red, carga del servidor, locks de archivos). Antes de Airflow, estos fallos requerían intervención manual para reintentar. Ahora, cada tarea está configurada para reintentar hasta 3 veces con backoff exponencial, recuperándose automáticamente de la mayoría de fallos temporales.

### Detección de Data Drift

La implementación de detección de drift mediante pruebas de Kolmogorov-Smirnov y Population Stability Index agregó una capa crítica de validación entre ingesta de datos y entrenamiento de modelos. El drift detection task compara la distribución estadística de cada feature en el batch nuevo contra una distribución de referencia histórica. Si detecta cambios significativos (p-value < 0.05 en KS test, o PSI > 0.2), levanta una alerta y opcionalmente detiene el pipeline.

La arquitectura de drift detection con almacenamiento de referencia en parquet permite evolucionar el concepto de "normalidad" con el tiempo. El primer run del pipeline no tiene baseline histórica, así que establece la distribución actual como referencia. Runs subsecuentes comparan contra esta baseline, pero si el dataset evoluciona gradualmente (más clientes en nuevas regiones, productos nuevos en el catálogo), podemos actualizar la baseline periódicamente para reflejar la "nueva normalidad". Este balance entre detectar cambios súbitos anómalos versus permitir evolución natural del negocio es fundamental en sistemas de producción.

### Escalabilidad mediante Paralelización

Airflow permite paralelizar tareas independientes, acelerando significativamente el pipeline. Por ejemplo, el procesamiento de features para clientes, productos, y transacciones puede ejecutarse en paralelo porque no tienen dependencias mutuas. En una ejecución secuencial, estas tres etapas tomarían tiempo_clientes + tiempo_productos + tiempo_transacciones. Con paralelización, el tiempo total es max(tiempo_clientes, tiempo_productos, tiempo_transacciones), efectivamente reduciendo la duración total del pipeline.

### Limitaciones de la Implementación Actual

A pesar de sus ventajas, nuestra implementación de Airflow tiene limitaciones importantes. La más significativa es que todo el código de tasks vive en archivos Python dentro del directorio `dags/tasks/`. Esto funciona para pipelines de complejidad moderada, pero no escala bien a equipos grandes donde múltiples personas modifican tareas simultáneamente. Conflictos de merge en Git y riesgo de introducir bugs en tareas no relacionadas son comunes.

Una arquitectura más escalable sería modularizar cada tarea como un paquete Python instalable independiente, versionado y testeado separadamente. El DAG de Airflow entonces importaría estos paquetes y los orquestaría sin contener lógica de negocio directamente. Esta separación permite que expertos en diferentes dominios (procesamiento de datos, training de modelos, deployment) trabajen en sus áreas sin interferir. Sin embargo, esta modularización agrega complejidad de gestión de dependencias y versionado que puede no justificarse para proyectos pequeños.

Otro problema es la falta de backfilling automático de datos faltantes. Si el pipeline falla un día y no se detecta inmediatamente, hay un gap en los datos históricos. Cuando finalmente reiniciamos el pipeline, procesa solo el día actual, dejando el gap sin llenar. Airflow soporta backfilling mediante el comando `airflow dags backfill`, pero requiere intervención manual. Un sistema más robusto debería detectar gaps automáticamente y backfill sin intervención humana.

## Direcciones Futuras y Mejoras Propuestas

### Automatización Completa del Deployment

Una mejora fundamental sería implementar continuous deployment donde cada run exitoso del pipeline de Airflow automáticamente deploya el nuevo modelo a producción. Esto requeriría: (1) tests automatizados que validen que el nuevo modelo supera umbrales mínimos de performance, (2) integración con el sistema de orquestación de contenedores (Docker Compose, Kubernetes) para actualizar el servicio FastAPI automáticamente, (3) capacidad de rollback instantáneo si el modelo nuevo genera predicciones anómalas en producción.

### Monitoreo Continuo con Dashboards en Tiempo Real

Una arquitectura deseable incluiría un dashboard en tiempo real (usando Grafana o Streamlit) que muestre: (1) volumen de requests de predicción por hora, (2) distribución de predicciones (qué porcentaje predice compra vs no-compra), (3) latencia promedio/percentil 95 de respuestas, (4) tasa de errores HTTP, (5) features con mayor drift comparado a la baseline histórica. Este dashboard sería visible para todo el equipo, creando transparencia sobre la salud del sistema.

Alertas automáticas via email o Slack cuando métricas críticas cruzan umbrales permitirían respuesta rápida a incidentes. Por ejemplo, si la latencia de predicción súbitamente se duplica, esto puede indicar que el servidor está sobrecargado o que un dataset nuevo es anormalmente grande. Detectar esto en minutos en lugar de horas minimiza impacto en usuarios.

### Experimentación A/B Integrada

Una limitación de nuestro sistema actual es que asumimos que el modelo nuevo es mejor que el anterior basándonos solo en métricas offline (evaluación en test set histórico). Sin embargo, métricas offline frecuentemente no correlacionan perfectamente con métricas de negocio en producción. Un modelo con accuracy 2% superior puede no traducirse en 2% más ventas si las predicciones incorrectas afectan desproporcionadamente clientes de alto valor.

Implementar experimentación A/B donde 50% de usuarios ven predicciones del modelo A y 50% del modelo B permitiría medir impacto real en KPIs de negocio: tasa de conversión, revenue por usuario, NPS. Comparar estos resultados provee validación definitiva de si el nuevo modelo mejora outcomes reales, no solo métricas técnicas.