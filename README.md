# Resume Category Classification

A notebook experiment that predicts resume topic categories using TF-IDF and Random Forest. It does **not** rank candidates or measure suitability for a job; no web application is included.

## Workflow

1. Load `clean_resume_data.csv` and inspect category balance.
2. Drop missing/empty text and duplicate resume text.
3. Reserve a stratified 20% holdout before oversampling.
4. Oversample minority categories in training data only.
5. Fit TF-IDF on training text, train Random Forest, and evaluate on untouched test text.
6. Save the fitted classifier and vectorizer under `models/`.

## Run locally

From the repository root in a separate Python environment:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab resume_category_classification.ipynb
```

## Evaluation correction

The original implementation oversampled before splitting, allowing copies of the same resume into train and test sets. The revised notebook prevents that overlap. Old notebook scores and model artifacts were removed; rerun training before quoting new results or using saved models.

## Limitations

Exact duplicate removal does not detect near-duplicate resumes or shared applicant identities. A grouped split is preferable when those identifiers are available. Validate class-level precision/recall, macro-F1 and a baseline. Dataset source, consent and redistribution terms need verification before distributing or extending the included data.
