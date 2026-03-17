import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from text_utils import clean_text


REQUIRED_COLUMNS = {"ticket_text", "category", "priority"}


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def validate_data(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_display}")


def save_confusion_matrix(y_true, y_pred, labels, title: str, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train_and_evaluate(df: pd.DataFrame, target: str, reports_dir: Path):
    x = df["ticket_text"].astype(str)
    y = df[target].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    report_dict = classification_report(y_test, preds, zero_division=0, output_dict=True)

    print(f"\n===== {target.upper()} MODEL =====")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    labels = sorted(y.unique())
    cm_path = reports_dir / f"{target}_confusion_matrix.png"
    save_confusion_matrix(
        y_test,
        preds,
        labels,
        title=f"{target.title()} Confusion Matrix",
        output_path=cm_path,
    )
    print(f"Saved confusion matrix: {cm_path}")

    return model, {"accuracy": acc, "classification_report": report_dict}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train support ticket category and priority classifiers."
    )
    parser.add_argument(
        "--data",
        default="data/support_tickets.csv",
        help="Path to CSV dataset with ticket_text, category, priority columns.",
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="Directory where trained model files will be stored.",
    )
    parser.add_argument(
        "--reports_dir",
        default="reports",
        help="Directory where evaluation charts and metrics will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    reports_dir = Path(args.reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    validate_data(df)

    category_model, category_metrics = train_and_evaluate(df, "category", reports_dir)
    priority_model, priority_metrics = train_and_evaluate(df, "priority", reports_dir)

    category_model_path = model_dir / "ticket_category_model.joblib"
    priority_model_path = model_dir / "ticket_priority_model.joblib"

    joblib.dump(category_model, category_model_path)
    joblib.dump(priority_model, priority_model_path)

    metrics = {
        "category": category_metrics,
        "priority": priority_metrics,
    }
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nSaved models:")
    print(f"- {category_model_path}")
    print(f"- {priority_model_path}")
    print("Saved evaluation artifacts:")
    print(f"- {reports_dir / 'category_confusion_matrix.png'}")
    print(f"- {reports_dir / 'priority_confusion_matrix.png'}")
    print(f"- {metrics_path}")


if __name__ == "__main__":
    main()
