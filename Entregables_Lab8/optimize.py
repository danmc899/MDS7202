import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
#%pip install -U kaleido
#%pip install --upgrade plotly

# Configuración
RANDOM_STATE = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_best_model(experiment_id):
    """Obtiene el mejor modelo basado en valid_f1"""
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")
    return best_model

def optimize_model():
    """Función principal para optimizar el modelo con Optuna y registrar con MLflow"""
    
    # Cargar datos
    df = pd.read_csv(r"C:\Users\dlamu\Downloads\Lab8\water_potability.csv")
    df["Potability"] = df["Potability"].astype(float)
    
    # Aplicar IQR y clipear outliers
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols.remove("Potability")
    lam = 1.5
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR_val = Q3 - Q1
    lower_bound = Q1 - lam * IQR_val
    upper_bound = Q3 + lam * IQR_val
    
    # Aplicar clip
    df[num_cols] = df[num_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)
    
    # Separar variables
    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    
    # Preprocesamiento
    preprocessor = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", MinMaxScaler())
    ])
    
    # Crear o recuperar experimento en MLflow
    experiment_name = "Water_Potability_XGBoost_Optimization"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # Función objetivo para Optuna
    def objective(trial):
        # Hiperparámetros sugeridos por Optuna
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss"
        }
        
        # Crear pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(**params))
        ])
        
        # Nombre del run con información relevante
        run_name = f"XGBoost_lr_{params['learning_rate']:.4f}_depth_{params['max_depth']}"
        
        # Iniciar run de MLflow
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            # Registrar parámetros
            mlflow.log_params(params)
            
            # Entrenar modelo
            pipeline.fit(X_train, y_train)
            
            # Predecir
            y_pred = pipeline.predict(X_test)
            
            # Calcular f1-score
            f1 = f1_score(y_test, y_pred)
            
            # Registrar métrica como valid_f1
            mlflow.log_metric("valid_f1", f1)
            
            # Guardar modelo
            mlflow.sklearn.log_model(pipeline, "model", input_example=X_train.iloc[0:1])
            
        return f1
    
    # Configurar y ejecutar estudio de Optuna
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, timeout=60, show_progress_bar=True)
    
    # Imprimir resultados
    print(f"Número de trials completados: {len(study.trials)}")
    print(f"Mejor f1-score (test): {study.best_value}")
    print("Mejores hiperparámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    
    # Crear carpetas si no existen
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Guardar gráficos de Optuna
    fig1 = plot_optimization_history(study)
    fig1.write_image("plots/optimization_history.png")
    
    fig2 = plot_param_importances(study)
    fig2.write_image("plots/param_importances.png")
    
    # Registrar gráficos en MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name="Best_Model_Artifacts"):
        mlflow.log_artifact("plots/optimization_history.png", artifact_path="plots")
        mlflow.log_artifact("plots/param_importances.png", artifact_path="plots")
    
    # Obtener el mejor modelo
    best_model = get_best_model(experiment_id)
    
    # Serializar el mejor modelo con pickle
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    print("\nModelo guardado en: models/best_model.pkl")
    
    # Guardar importancia de features del mejor modelo
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        feature_importance = best_model.named_steps['model'].feature_importances_
        features = X.columns
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, feature_importance)
        plt.xlabel('Importancia')
        plt.ylabel('Features')
        plt.title('Importancia de Variables - Mejor Modelo XGBoost')
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png")
        plt.close()
        
        print("Gráfico de importancia guardado en: plots/feature_importance.png")
    
    # Guardar versiones de librerías
    with open("requirements.txt", "w") as f:
        f.write(f"pandas=={pd.__version__}\n")
        f.write(f"numpy=={np.__version__}\n")
        f.write(f"scikit-learn==1.7.2\n")
        f.write(f"xgboost==2.0.0\n")
        f.write(f"optuna==3.4.0\n")
        f.write(f"mlflow==3.5.0\n")
        f.write(f"matplotlib==3.8.0\n")
        f.write(f"plotly==5.17.0\n")
        f.write(f"kaleido==0.2.1\n")
    
    print("Requisitos guardados en: requirements.txt")
    
    return best_model

if __name__ == "__main__":
    optimize_model()