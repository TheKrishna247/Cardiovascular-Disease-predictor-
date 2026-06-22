# Project Report

## Cardiovascular Disease Predictor Using Decision Tree

| Field | Details |
|---|---|
| Course | Machine Learning |
| Algorithm | Decision Tree Classifier |
| Language | Python 3 |
| Dataset | Cardiovascular Disease Dataset (Kaggle) |

---

## 1. Abstract

Cardiovascular disease is one of the leading causes of death worldwide.
Identifying at-risk patients early can help doctors recommend lifestyle
changes or treatment before the disease becomes serious. This project uses
a Decision Tree Classifier to predict whether a patient has cardiovascular
disease, based on 11 health and lifestyle attributes such as age, blood
pressure, cholesterol, and smoking habits. The dataset contains 70,000
patient records. After cleaning incorrect entries, the model achieves
approximately 73% accuracy on unseen test data, with systolic blood
pressure, age, and cholesterol emerging as the strongest predictors —
consistent with established medical knowledge.

---

## 2. Introduction

Cardiovascular diseases (CVDs) include conditions like heart attacks,
strokes, and hypertension-related illness. Risk factors include age, high
blood pressure, high cholesterol, smoking, and lack of physical activity.

Machine learning can support early screening by identifying patients at
higher risk using easily collected data — age, blood pressure, weight,
and basic lifestyle habits — without needing expensive diagnostic tests.

### Why Decision Trees?

A Decision Tree splits data using a sequence of yes/no questions about
feature values, making its predictions easy to trace and explain. This
interpretability is valuable in medical applications, where a doctor needs
to understand why a prediction was made, not just what it is.

---

## 3. Objective

1. Load and clean the cardiovascular disease dataset
2. Explore the data through visualizations
3. Train a Decision Tree Classifier to predict disease presence
4. Evaluate the model using standard classification metrics
5. Identify which features most influence the prediction
6. Demonstrate prediction on a sample patient

---

## 4. Dataset Description

The dataset contains 70,000 patient records with 11 features and 1 target
column (`cardio`).

| Feature | Type | Description |
|---|---|---|
| age(in days) | Numerical | Age in days, converted to years |
| Gender(1 : woman | 2 : Man) | Categorical | 1 or 2 |
| Height(in cm) | Numerical | Height in cm |
| weight(in kg) | Numerical | Weight in kg |
| sys_pressure | Numerical | Systolic blood pressure |
| dia_pressure | Numerical | Diastolic blood pressure |
| cholesterol | Categorical | 1 = normal, 2 = above normal, 3 = well above normal |
| gluc | Categorical | Glucose level, same scale as cholesterol |
| smoke | Categorical | 1 = smoker, 0 = non-smoker |
| alco | Categorical | 1 = drinks alcohol, 0 = does not |
| active | Categorical | 1 = physically active, 0 = not |
| **cardio** | **Target** | **0 = no disease, 1 = disease present** |

The target classes are well balanced (close to 50/50), which means accuracy
is a meaningful metric here without needing to correct for class imbalance.

---

## 5. Data Preprocessing

The raw dataset required cleaning before it could be used reliably:

1. **Duplicate rows**: 24 fully duplicate rows were removed.
2. **Age conversion**: Age was stored in days. It was converted to years
   (`age / 365.25`) since years are easier to interpret and explain.
3. **Blood pressure errors**: Some rows contained impossible values for
   `sys_pressure` and `dia_pressure`, including negative numbers and values in the
   thousands (likely data entry mistakes, such as a misplaced decimal
   point). Rows were kept only if `sys_pressure` was between 70–250, `dia_pressure` was
   between 40–200, and `sys_pressure` was greater than `dia_pressure` (systolic pressure
   should always be higher than diastolic).
4. **Height/weight outliers**: A small number of rows had implausible
   height (under 100cm) or weight (under 30kg) values and were removed.

After cleaning, the dataset has approximately 68,600 rows — about 2% of
rows were removed, almost entirely due to the blood pressure errors. No
missing values were present in the original dataset.

---

## 6. Methodology

```
cardio_train.csv
      |
      v
Load Dataset (pandas, semicolon-separated)
      |
      v
Data Cleaning
(remove duplicates, convert age, filter invalid BP/height/weight)
      |
      v
Exploratory Data Analysis
      |
      v
Feature Selection (X = 11 features, y = cardio)
      |
      v
Train-Test Split (80% / 20%)
      |
      v
Decision Tree Classifier (gini, max_depth=5)
      |
      v
Evaluation (Accuracy, Confusion Matrix, Classification Report)
      |
      v
Feature Importance + Tree Visualization
      |
      v
Sample Patient Prediction
```

### Decision Tree Configuration

```python
DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)
```

**Gini Impurity** measures how mixed a node is:

```
Gini = 1 - sum(p_i^2)
```

where `p_i` is the proportion of class `i` in that node. A Gini of 0 means
the node contains only one class (perfectly pure).

`max_depth=5` limits the tree to 5 levels, which keeps it simple and
reduces the risk of overfitting — an unrestricted tree would memorize the
training data and perform worse on new patients.

---

## 7. Exploratory Data Analysis — Findings

- **Target distribution**: Close to balanced between disease and no
  disease, which is unusual for real medical data but expected here since
  the dataset was curated for ML practice.
- **Age**: Patients range from about 30 to 65 years old. Disease becomes
  noticeably more common in older age groups.
- **Cholesterol**: Patients with "above normal" or "well above normal"
  cholesterol show a higher proportion of disease.
- **Blood pressure**: Patients with disease tend to have higher systolic
  blood pressure (`sys_pressure`), visible clearly in the box plot comparison.
- **Correlation heatmap**: `sys_pressure` (0.43), `dia_pressure` (0.34), `age` (0.24),
  and `cholesterol` (0.22) show the strongest positive correlation with
  the target. Lifestyle factors (`smoke`, `alco`, `active`) show very weak
  correlation with the target in this dataset.

---

## 8. Results

### 8.1 Accuracy

The Decision Tree Classifier achieved approximately **73% accuracy** on
the held-out test set (20% of the cleaned data).

### 8.2 Classification Report

| Metric | No Disease | Disease |
|---|---|---|
| Precision | ~0.71 | ~0.76 |
| Recall | ~0.80 | ~0.66 |
| F1-Score | ~0.75 | ~0.71 |

The model is somewhat better at correctly identifying healthy patients
(higher recall for "No Disease") than at catching every disease case. In a
medical context, missed disease cases (false negatives) are more
concerning than false alarms, so this is a limitation worth noting — a
more conservative threshold or a different model could improve recall on
the disease class at the cost of more false positives.

### 8.3 Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | sys_pressure (systolic BP) | ~0.78 |
| 2 | age | ~0.12 |
| 3 | cholesterol | ~0.08 |
| 4 | gluc | ~0.01 |
| 5 | weight | ~0.004 |

Systolic blood pressure dominates the model's decisions, which matches
medical understanding — hypertension is one of the strongest known risk
factors for cardiovascular disease. Age and cholesterol are the next most
useful predictors. `gender`, `alco`, and `active` had effectively zero
importance in this tree, meaning the model never found them useful for
splitting at `max_depth=5`.

---

## 9. Conclusion

This project built a complete machine learning pipeline to predict
cardiovascular disease using a Decision Tree Classifier on a real-world
dataset of 70,000 patients. Key takeaways:

1. The dataset required meaningful cleaning (blood pressure errors, age
   format) before it could be used reliably — a reminder that real-world
   data is rarely ready to use as-is.
2. The model achieved a realistic ~73% accuracy, with systolic blood
   pressure as the dominant predictor, followed by age and cholesterol.
3. Decision Trees remain useful for this kind of project because the
   model's reasoning can be visualized and explained step by step.
4. The model is better at identifying healthy patients than catching every
   disease case, which is an important limitation for a real medical
   screening tool.

---

## 10. Limitations

- ~73% accuracy means roughly 1 in 4 predictions is wrong — not accurate
  enough for real medical use without further improvement.
- The dataset does not specify which value of `gender` (1 or 2)
  corresponds to male or female.
- Lifestyle factors (smoking, alcohol, activity) showed almost no
  predictive value in this dataset, which may reflect self-reported data
  being unreliable, or these factors having a smaller direct effect than
  blood pressure and age.
- The model should not be used for actual medical diagnosis.

---

## 11. References

1. Kaggle. Cardiovascular Disease Dataset.
2. Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). Classification
   and Regression Trees. Wadsworth.
3. Scikit-Learn Documentation — Decision Tree Classifier.
   https://scikit-learn.org/stable/modules/tree.html
4. World Health Organization. Cardiovascular diseases (CVDs) Fact Sheet.
