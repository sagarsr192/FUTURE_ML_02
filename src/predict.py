import argparse
from pathlib import Path

import joblib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict support ticket category and priority from text."
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Support ticket text to classify.",
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="Directory containing trained model files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    category_model = joblib.load(model_dir / "ticket_category_model.joblib")
    priority_model = joblib.load(model_dir / "ticket_priority_model.joblib")

    text = [args.text]
    pred_category = category_model.predict(text)[0]
    pred_priority = priority_model.predict(text)[0]

    print("Prediction")
    print(f"Ticket Text : {args.text}")
    print(f"Category    : {pred_category}")
    print(f"Priority    : {pred_priority}")


if __name__ == "__main__":
    main()
