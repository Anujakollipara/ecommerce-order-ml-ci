
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_model():
    print("Reading e-commerce dataset...")

    data = pd.read_csv("ecommerce.csv")

    print("Dataset shape:", data.shape)
    print("\nOrder status distribution:")
    print(data["Order_Status"].value_counts())

    # Remove identifiers and customer names.
    # They are not used as model input features.
    data = data.drop(
        columns=["Order_ID", "Customer_ID", "Customer_Name"],
        errors="ignore"
    )

    # Remove rows without the target label, if any.
    data = data.dropna(subset=["Order_Status"])

    X = data.drop(columns=["Order_Status"])
    y = data["Order_Status"]

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=model.named_steps["classifier"].classes_
    ))

    joblib.dump(model, "ecommerce_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "classes": [str(c) for c in model.classes_]
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel saved: ecommerce_model.pkl")
    print("Metrics saved: metrics.json")


if __name__ == "__main__":
    train_model()
