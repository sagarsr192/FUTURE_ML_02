 # Support Ticket Classification & Prioritization

 https://futureml02-fmflj9jbdffabm8ilxbqfq.streamlit.app/

 

This project implements **Machine Learning Task 2 (2026)**: automatically classify support tickets and assign priority levels.

## Objective

Given a support ticket text, the system predicts:

- `category`: Billing, Technical Issue, Account, General Query
- `priority`: High, Medium, Low

## Features Implemented

- Text cleaning (lowercasing, punctuation removal, stopword removal)
- TF-IDF feature extraction (unigrams + bigrams)
- Ticket category classification
- Priority prediction
- Model evaluation with accuracy and class-wise precision/recall/F1
- CLI-based inference for new tickets

## Project Structure

```text
.
 data/
    support_tickets.csv
 models/
 src/
    text_utils.py
    train.py
    predict.py
 requirements.txt
 README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train Models

```bash
python src/train.py --data data/support_tickets.csv --model_dir models --reports_dir reports
```

This command trains two models and saves:

- `models/ticket_category_model.joblib`
- `models/ticket_priority_model.joblib`

It also saves evaluation artifacts:

- `reports/category_confusion_matrix.png`
- `reports/priority_confusion_matrix.png`
- `reports/metrics.json`

## Predict on New Ticket

```bash
python src/predict.py --text "Payment was deducted twice and dashboard is down" --model_dir models
```

## Streamlit Demo App

Run the interactive UI:

```bash
streamlit run app.py
```

The app predicts category and priority, and also shows confidence values.

## Replace with Real Dataset (Recommended)

Use one of the recommended datasets from the task page (Kaggle/Zenodo), then make sure your CSV has these columns:

- `ticket_text`
- `category`
- `priority`

If the dataset has different names (e.g., `text`, `label`), rename them before training.

## Business Value

This system helps support teams:

- route tickets to correct teams faster
- identify urgent cases earlier
- reduce manual triage workload
- improve response SLAs and customer satisfaction

## Suggested Submission Add-ons

- Confusion matrix plots
- Error analysis for misclassified tickets
- Short demo video or screenshots
- Deployed web demo (Streamlit/Flask)
