from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# -------------------------------------------------------------------
# 1. Cargar el modelo entrenado
# -------------------------------------------------------------------
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------------------------
# 2. Crear la aplicación FastAPI
# -------------------------------------------------------------------
app = FastAPI(title="API de Predicción de Potabilidad del Agua")

# -------------------------------------------------------------------
# 3. Definir el modelo de datos de entrada
# -------------------------------------------------------------------
class WaterQuality(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# -------------------------------------------------------------------
# 4. Ruta principal (GET /)
# -------------------------------------------------------------------
@app.get("/")
def home():
    """
    Muestra información general de la API.
    Se accede escribiendo http://127.0.0.1:8000/ en el navegador.
    """
    return {
        "mensaje": "API de Predicción de Potabilidad del Agua",
        "descripcion": "Este modelo utiliza XGBoost optimizado con Optuna para predecir si una muestra de agua es potable o no.",
        "problema": "Clasificación binaria basada en 9 mediciones químicas.",
        "entrada": {
            "ph": "Nivel de pH del agua",
            "Hardness": "Dureza del agua",
            "Solids": "Sólidos totales disueltos",
            "Chloramines": "Nivel de cloraminas",
            "Sulfate": "Nivel de sulfatos",
            "Conductivity": "Conductividad eléctrica",
            "Organic_carbon": "Carbono orgánico",
            "Trihalomethanes": "Nivel de trihalometanos",
            "Turbidity": "Turbidez"
        },
        "salida": {
            "potabilidad": "0 (no potable) o 1 (potable)"
        }
    }

# -------------------------------------------------------------------
# 5. Ruta de predicción (POST /potabilidad/)
# -------------------------------------------------------------------
@app.post("/potabilidad/")
def predict_potability(data: WaterQuality):
    """
    Recibe los valores químicos del agua y devuelve si es potable o no.
    Se prueba desde http://127.0.0.1:8000/docs
    """
    # Convertir los datos de entrada a DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Realizar predicción
    prediction = int(model.predict(input_data)[0])

    # Probabilidad de la predicción
    try:
        probability = float(model.predict_proba(input_data)[0][prediction])
    except Exception:
        probability = None

    # Retornar resultado
    return {
        "prediccion": prediction,
        "probabilidad": round(probability, 4) if probability else None,
        "interpretacion": "Potable" if prediction == 1 else "No potable"
    }
