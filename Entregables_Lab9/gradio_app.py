from pathlib import Path
import os
import tempfile
import joblib
import pandas as pd
import gradio as gr
import io

BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", "/opt/airflow/data")
TARGET_COL = "HiringDecision"

# Leer desde la carpeta models, como lo indica la instrucción
def get_latest_model():
    base_path = Path(BASE_DATA_DIR) / "linear"
    run_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.count('-') == 2], reverse=True)
    
    for run_dir in run_dirs:
        model_path = run_dir / "models" / "model_rf_pipeline.joblib"
        if model_path.exists():
            print(f"Modelo encontrado: {model_path}")
            return model_path, run_dir.name
    
    raise FileNotFoundError("No modelo encontrado. Ejecuta el DAG primero.")

def predict_from_json(json_file):
    if json_file is None:
        return pd.DataFrame({"Error": ["⚠️ Sube un archivo JSON"]})
    
    try:
        model_path, run_date = get_latest_model()
        pipe = joblib.load(model_path)
        
        # Lee el contenido del archivo subido
        # json_file es un objeto de archivo temporal
        print(f"📂 Archivo recibido: {json_file.name}")
        
        # Leer JSON directamente desde el objeto de archivo
        df = pd.read_json(json_file.name, orient="records")
        print(f"Datos: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])
        
        # Predicción
        y_hat = pipe.predict(df)
        
        # Probabilidades
        prob_pos = None
        if hasattr(pipe, "predict_proba"):
            probas = pipe.predict_proba(df)
            if probas.shape[1] >= 2:
                prob_pos = probas[:, 1]
        
        # Resultados
        out = df.copy()
        out["Predicción"] = ["✅ Contratado" if p == 1 else "❌ No Contratado" for p in y_hat]
        if prob_pos is not None:
            out["Probabilidad_Contratación"] = [f"{p:.1%}" for p in prob_pos]
        
        print(f"✅ {len(out)} predicciones generadas (modelo: {run_date})")
        return out
    
    except Exception as e:
        import traceback
        error_msg = f"❌ {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return pd.DataFrame({"Error": [str(e)]})

iface = gr.Interface(
    fn=predict_from_json,
    inputs=gr.File(label="📁 Sube archivo JSON"),
    outputs=gr.Dataframe(label="📊 Predicciones"),
    title="🎓 Predictor de Contratación",
    description="""
    Sube un JSON con datos para predecir si serán contratados.
    """,
    allow_flagging="never"
)

if __name__ == "__main__":
    print("🚀 Gradio iniciando en puerto 7860...")
    print(f"📂 BASE_DATA_DIR: {BASE_DATA_DIR}")
    
    try:
        model_path, run_date = get_latest_model()
        print(f"✅ Modelo cargado: {run_date}")
    except:
        print("⚠️  Esperando modelo...")
    
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)