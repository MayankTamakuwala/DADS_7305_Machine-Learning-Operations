from pathlib import Path
from typing import Iterable, Tuple

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_and_evaluate(
    *,
    random_state: int = 7,
    test_size: float = 0.25,
    c: float = 1.0,
    max_iter: int = 500,
) -> Tuple[Pipeline, dict, Iterable[str], Iterable[str]]:
    """Train a logistic regression on the breast cancer dataset with simple tunable knobs."""
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Standardized logistic regression performs well on this dataset and remains interpretable.
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, solver="lbfgs", C=c)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report_text = classification_report(y_test, y_pred, target_names=data.target_names)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": report_text,
        "classification_report_dict": classification_report(
            y_test, y_pred, target_names=data.target_names, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return model, metrics, data.feature_names, data.target_names


def save_model(
    model,
    feature_names,
    target_names,
    model_path: str = "artifacts/breast_cancer_log_reg.pkl",
):
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": feature_names, "target_names": target_names},
        path,
    )
    return path


if __name__ == "__main__":
    model, metrics, feature_names, target_names = train_and_evaluate()
    artifact_path = save_model(model, feature_names, target_names)

    print(f"Model saved to {artifact_path}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("Classification report:")
    print(metrics["classification_report"])
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
