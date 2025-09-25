import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
import mlflow.sklearn

# Cargar dataset de cáncer de mama
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Iniciar un experimento de MLflow
with mlflow.start_run():
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # Modelo: Regresión logística
    model = LogisticRegression(max_iter=500, solver="liblinear")
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log MLflow
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 500)
    mlflow.log_param("solver", "liblinear")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Guardar modelo local
    joblib.dump(model, "logreg_model.pkl")

    # Registrar en MLflow
    mlflow.sklearn.log_model(model, "logreg-model")

    print(f"Modelo entrenado. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print("Experimento registrado en MLflow.")
