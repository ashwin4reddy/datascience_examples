import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve


def evaluate_model(model: callable, x_test: pd.DataFrame, y_test: pd.Series, threshold: float | None = None, detailed: bool = False) -> None:
    """Classification model evaluation and produce confusion matrix."""
    if threshold is None:
        y_pred = model.predict(x_test)
    else:
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    if detailed:
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        print(f"\nConfusion Matrix:\n{conf_matrix}")
        print(f"\nClassification Report:\n{class_report}")

    return y_pred


def get_column_importances(model: callable, preprocessor: callable, numeric_features: list, categorical_features: list) -> None:
    """Calculate feature importances."""
    feature_importances = model.named_steps["classifier"].feature_importances_

    feature_names = (
        preprocessor.named_transformers_["num"].get_feature_names_out(numeric_features).tolist()
        + preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features).tolist()
    )

    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})

    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
    feature_importance_df["Column"] = feature_importance_df["Feature"].apply(lambda x: x if x in numeric_features else x.rsplit("_", 1)[0])

    print("Column Feature Importances:")
    print(feature_importance_df.groupby("Column")["Importance"].sum().sort_values(ascending=False))


def roc_plot_threshold(model: callable, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Calculate optimum threshold to get best outcome based on ROC score."""
    y_prob = model.predict_proba(x_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal Threshold: {optimal_threshold}")
    print(f"ROC Score with y_prob: {roc_auc_score(y_test, y_prob)}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker="*", color="red")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    return optimal_threshold
