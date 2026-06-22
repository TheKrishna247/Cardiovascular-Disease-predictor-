# Cardiovascular Disease Predictor Using Decision Tree

A machine learning project that predicts whether a patient is likely to have
cardiovascular disease, based on basic health and lifestyle data.

## Overview

Cardiovascular disease is one of the leading causes of death worldwide. This
project uses a Decision Tree Classifier to predict the presence of
cardiovascular disease from patient attributes like age, blood pressure,
cholesterol, and lifestyle habits.

The project covers the full ML workflow: loading data, cleaning it,
exploring it visually, training a model, evaluating it, and using it to
predict on a new patient.

## Dataset Information

- File: `cardio_train.csv`
- Source: Cardiovascular Disease dataset (Kaggle)
- Rows: 70,000 patient records
- Columns: 12 (11 features + 1 target)
- Separator: `;`

| Column | Description |
|---|---|
| age | Age in days (converted to years during cleaning) |
| gender | 1 or 2 (coding not specified by dataset source) |
| height | Height in cm |
| weight | Weight in kg |
| ap_hi | Systolic blood pressure |
| ap_lo | Diastolic blood pressure |
| cholesterol | 1 = normal, 2 = above normal, 3 = well above normal |
| gluc | Glucose level, same scale as cholesterol |
| smoke | 1 = smoker, 0 = non-smoker |
| alco | 1 = drinks alcohol, 0 = does not |
| active | 1 = physically active, 0 = not |
| cardio | Target: 1 = disease present, 0 = no disease |

## Features Used

All 11 features are used directly — age (in years), gender, height, weight,
ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active.

## Technologies

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn (Decision Tree Classifier)

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

Make sure `cardio_train.csv` is in the same folder as the script, then run:

```bash
python cardiovascular_disease_predictor.py
```

The script will load and clean the data, generate EDA plots into
`screenshots/`, train the model, print evaluation metrics, and run a sample
prediction.

Alternatively, open `cardiovascular_disease_predictor.ipynb` in Jupyter
Notebook and run all cells in order.

## Results

| Metric | Score |
|---|---|
| Accuracy | ~73% |
| Precision | ~73% |
| Recall | ~73% |
| F1-Score | ~73% |

Results are reproducible with `random_state=42`.

## Screenshots

All visualizations are saved to the `screenshots/` folder when the script
runs, including target distribution, age distribution, cholesterol vs
disease, blood pressure vs disease, correlation heatmap, confusion matrix,
feature importance, and the decision tree.

## Future Scope

- Try other algorithms (Logistic Regression, Random Forest) for comparison
- Use cross-validation for more reliable accuracy estimates
- Engineer a BMI feature from height and weight
- Tune `max_depth` and other hyperparameters

## Author

B.Tech CSE Student
