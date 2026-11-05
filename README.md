# Employee Attrition Prediction Models

IBM HR Analytics assignment: predict employee attrition (classification) and, as the optional business task,
predict job satisfaction (regression).

## What is inside
| Path | Purpose |
|---|---|
| `notebooks/employee_attrition_models.ipynb` | **Model Notebook** – EDA, preprocessing, 4 classifiers, metrics, 5-fold CV, hyper-parameter tuning, feature importance, regression |
| `reports/Evaluation_Report.docx` | **Evaluation Report** – metrics tables, ROC/confusion matrices, tuning, limitations |
| `reports/Business_Interpretation_Report.docx` | **Business Interpretation Report** – drivers of attrition and HR recommendations |
| `results/` | metrics CSVs, `summary.json`, figures |
| `src/attrition_models.py` | the notebook as a plain script |
| `src/build_reports.py` | regenerates both reports from `results/` |
| `src/generate_synthetic_data.py` | creates the stand-in dataset (same schema as IBM file) |
| `data/` | dataset (see below) |

## Models and metrics
Logistic Regression, Decision Tree, Random Forest, XGBoost – evaluated with Accuracy, Precision, Recall, F1, ROC-AUC.
Improvements: RandomizedSearchCV hyper-parameter tuning, stratified 5-fold cross-validation, permutation / tree / coefficient feature importance.

## Dataset  (important)
The notebook uses the **real IBM file** if you place it at `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`
([Kaggle: IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)).
If it is missing, it falls back to `data/synthetic_hr_attrition.csv` (synthetic, identical columns).
The numbers committed in `results/` and `reports/` were produced on the synthetic file with a fallback gradient-boosting model
in place of XGBoost, because the build environment had no internet. **Re-run with the real file and xgboost installed to get final numbers.**

## Run
```bash
pip install -r requirements.txt
# (optional) put the real CSV in data/
jupyter nbconvert --to notebook --execute --inplace notebooks/employee_attrition_models.ipynb   # or open it and Run All
python src/build_reports.py      # refresh the .docx reports with the new numbers
```
