import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
import mlflow.sklearn

# Mismo dataset (Iris)
iris = datasets.load_iris()
X = iris.data
y = iris.target

with mlflow.start_run():
    # Split idéntico en espíritu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Modelo distinto: SVC con escalado
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42))
    ])
    model.fit(X_train, y_train)

    # Predicciones y métricas
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    # MLflow: params y métricas
    mlflow.log_param("model_type", "Pipeline(StandardScaler+SVC)")
    mlflow.log_param("svc_kernel", "rbf")
    mlflow.log_param("svc_C", 1.0)
    mlflow.log_param("svc_gamma", "scale")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1m)

    # Guardado local y registro
    joblib.dump(model, "svc_iris.pkl")
    mlflow.sklearn.log_model(model, "svc-iris-model")

    print(f"Accuracy: {acc:.4f} | F1-macro: {f1m:.4f}")
