"""Build the Evaluation Report and Business Interpretation Report (.docx) from results/ produced by the notebook.
Run:  python src/build_reports.py   (after running the notebook)"""
import json, pathlib, pandas as pd
from docx import Document
from docx.shared import Pt, Inches
ROOT = pathlib.Path(__file__).resolve().parent.parent
R, F, OUT = ROOT / "results", ROOT / "results" / "figures", ROOT / "reports"
S = json.load(open(R / "summary.json"))
base, tuned = pd.read_csv(R/"baseline_metrics.csv", index_col=0), pd.read_csv(R/"tuned_metrics.csv", index_col=0)
cv, reg = pd.read_csv(R/"cv_metrics.csv", index_col=0), pd.read_csv(R/"regression_metrics.csv", index_col=0)
imp = pd.read_csv(R/"permutation_importance.csv", index_col=0)["importance"]
best = S["best_model"]; xgb_note = "" if S["xgboost_real"] else (
    " NOTE: in this run the xgboost package was unavailable, so the XGBoost row is a gradient-boosted-tree fallback "
    "(sklearn HistGradientBoosting). Re-running the notebook with `pip install xgboost` replaces it with true XGBoost.")
synthetic = "synthetic" in S["data_source"]
data_note = ("DATA NOTE: the figures in this report were produced on a synthetic dataset that has the same schema as the IBM HR Analytics file "
             "(the real file could not be downloaded in the build environment). Re-run the notebook with the real file in data/ and then "
             "`python src/build_reports.py` to refresh every number and chart.") if synthetic else ""

def doc(title, sub):
    d = Document(); st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
    d.add_heading(title, 0); d.add_paragraph(sub).italic = True
    if data_note: p = d.add_paragraph(); r = p.add_run(data_note); r.bold = True; r.font.size = Pt(9.5)
    return d
def table(d, df, idx="Model"):
    t = d.add_table(rows=1, cols=len(df.columns) + 1); t.style = "Light Grid Accent 1"
    for i, h in enumerate([idx] + list(df.columns)): t.rows[0].cells[i].text = str(h)
    for n, row in df.iterrows():
        c = t.add_row().cells; c[0].text = str(n).replace(" (fallback: sklearn HistGradientBoosting)", "*")
        for i, v in enumerate(row): c[i + 1].text = f"{v:.3f}" if isinstance(v, float) else str(v)
    d.add_paragraph()
def fig(d, name, w=6.3, cap=""):
    d.add_picture(str(F / name), width=Inches(w)); 
    if cap: d.add_paragraph(cap).runs[0].italic = True
def bullets(d, items):
    for i in items: d.add_paragraph(i, style="List Bullet")

# ---------------- Evaluation report ----------------
d = doc("Evaluation Report – Employee Attrition Prediction Models", "Classification of employee attrition with Logistic Regression, Decision Tree, Random Forest and XGBoost")
d.add_heading("1. Objective", 1)
d.add_paragraph("Predict whether an employee will leave the company (Attrition = Yes) using HR records, compare four classification algorithms, "
                "improve them with cross-validation and hyper-parameter tuning, and identify the features that drive attrition.")
d.add_heading("2. Data", 1)
d.add_paragraph(f"Source: {S['data_source']}. {S['n_rows']} employees, {S['n_features']} predictor columns (after dropping constant/ID columns). "
                f"Attrition rate = {S['attrition_rate']:.1%}, i.e. the classes are imbalanced (about 1 leaver per 5 stayers).")
d.add_paragraph("Preprocessing: stratified 80/20 train/test split (random_state=42); numeric features standardised, categorical features one-hot encoded, "
                "all inside a scikit-learn Pipeline to avoid leakage. Class imbalance handled with class weights (scale_pos_weight for XGBoost).")
fig(d, "01_eda_attrition.png", cap="Figure 1 – Class balance and attrition rate by OverTime / Job Role.")
d.add_heading("3. Methodology", 1)
bullets(d, ["Metrics: Accuracy, Precision, Recall, F1, ROC-AUC (positive class = leaver).",
            "Cross-validation: 5-fold stratified CV on the training set for every model.",
            "Tuning: RandomizedSearchCV (20 candidates, 5-fold CV, scoring = ROC-AUC) per model.",
            "Final scores: tuned models evaluated once on the untouched 20 % test set.",
            "Interpretation: permutation importance, random-forest importance and logistic-regression coefficients."])
d.add_heading("4. Results", 1)
d.add_heading("4.1 Baseline models (test set)", 2); table(d, base)
d.add_heading("4.2 5-fold cross-validation (training set, mean ± std)", 2); table(d, cv)
d.add_heading("4.3 Tuned models (test set)", 2); table(d, tuned)
if xgb_note: d.add_paragraph("* " + xgb_note.strip())
d.add_heading("4.4 Best hyper-parameters", 2)
for n, p in S["best_params"].items(): d.add_paragraph(f"{n.split(' (')[0]}: {p}", style="List Bullet")
fig(d, "03_roc_and_tuning.png", cap="Figure 2 – ROC curves of the tuned models and ROC-AUC before vs after tuning.")
fig(d, "04_confusion_matrices.png", cap="Figure 3 – Confusion matrices (tuned models, test set).")
d.add_heading("5. Feature importance", 1)
fig(d, "05_permutation_importance.png", 5.5, f"Figure 4 – Permutation importance for the best model ({best.split(' (')[0]}).")
fig(d, "06_rf_and_lr_importance.png", cap="Figure 5 – Random Forest importance and Logistic Regression coefficients.")
d.add_heading("6. Optional regression task – predicting Job Satisfaction", 1)
table(d, reg, "Model")
d.add_paragraph("Job satisfaction (1–4) is only weakly predictable from the HR record: both regressors barely improve on predicting the mean "
                "(R² ≈ 0). Conclusion: satisfaction needs survey/engagement data; it should be collected rather than modelled from HR fields.")
d.add_heading("7. Discussion", 1)
t = tuned
bm = t["ROC-AUC"].idxmax(); br = t["Recall"].idxmax()
bullets(d, [f"Best ROC-AUC on the test set: {bm.split(' (')[0]} ({t.loc[bm,'ROC-AUC']:.3f}). Highest recall: {br.split(' (')[0]} ({t.loc[br,'Recall']:.3f}).",
            "Accuracy is misleading here: a model that always predicts 'stay' would score about "
            f"{1-S['attrition_rate']:.0%}. Recall, F1 and ROC-AUC are the relevant measures.",
            "Tuning and class weighting improved recall/F1 for most models, at the cost of some precision – an acceptable trade-off, because missing a leaver is costlier than a retention conversation with someone who would have stayed.",
            "Simple, regularised linear models are competitive with ensembles on this small tabular dataset (≈1.5k rows); ensembles need more data to pull ahead.",
            "Single decision trees are the weakest and most unstable; they are useful only as an easy-to-read rule set."])
d.add_heading("8. Limitations", 1)
bullets(d, ["Small dataset (1,470 rows, ~230 leavers) → wide confidence intervals (see ± in CV table).",
            "Features are a snapshot; no time dimension, so attrition cannot be forecast by month.",
            "Associations are not causal; model outputs should support, not replace, manager judgement.",
            "Probability threshold fixed at 0.5; HR could choose a lower threshold to catch more leavers."])
d.save(OUT / "Evaluation_Report.docx")

# ---------------- Business report ----------------
d = doc("Business Interpretation Report – Employee Attrition", "What the models tell HR leadership, and what to do about it")
d.add_heading("1. Executive summary", 1)
ot = S["overtime_attrition_pct"]
d.add_paragraph(f"About {S['attrition_rate']:.0%} of employees in the dataset left. Our best model ({best.split(' (')[0]}) ranks employees by leaving-risk with a "
                f"ROC-AUC of {tuned.loc[best,'ROC-AUC']:.2f} on unseen data and flags roughly {tuned.loc[best,'Recall']:.0%} of actual leavers. "
                f"The strongest, most actionable drivers are pay level, overtime, job satisfaction and tenure/management stability. "
                f"Employees working overtime left at {ot.get('Yes')}% versus {ot.get('No')}% for those who do not – nearly "
                f"{ot.get('Yes',0)/max(ot.get('No',1),1e-9):.1f}× the rate.")
d.add_heading("2. What drives attrition", 1)
top = list(S["top_features"].items())[:6]
d.add_paragraph("Ranked by permutation importance (how much predictive power is lost when the feature is shuffled):")
for k, v in top: d.add_paragraph(f"{k}  (importance {v:.3f})", style="List Bullet")
fig(d, "05_permutation_importance.png", 5.3, "Figure 1 – Top drivers of attrition risk.")
d.add_paragraph("Direction of effect (logistic regression, standardised coefficients):")
d.add_paragraph("Raises risk: " + ", ".join(k.split("__")[-1] for k in S["top_lr_positive"]), style="List Bullet")
d.add_paragraph("Lowers risk: " + ", ".join(k.split("__")[-1] for k in S["top_lr_negative"]), style="List Bullet")
fig(d, "06_rf_and_lr_importance.png", cap="Figure 2 – Importance and direction of effect.")
d.add_heading("3. Business interpretation", 1)
bullets(d, ["Compensation: lower monthly income is the single strongest warning sign; low-paid staff in early job levels are the most mobile.",
            "Workload: overtime is the clearest controllable lever – overworked employees leave at a much higher rate.",
            "Engagement: low job satisfaction and poor work-life balance raise risk, but satisfaction itself could not be predicted from HR fields (R² ≈ 0), so it must be measured directly via surveys.",
            "Career stability: short tenure with the current manager, many previous employers and no stock options all point to flight risk.",
            "Profile: single employees and frequent business travellers show higher attrition."])
d.add_heading("4. Recommendations", 1)
bullets(d, ["Run a pay-equity / market-benchmark review for low-income, junior roles first; targeted raises are cheaper than replacement.",
            "Cap and monitor overtime; flag teams where a large share of staff work overtime and rebalance headcount or deadlines.",
            "Extend stock-option / retention-bonus eligibility to high-risk, high-performing staff.",
            "Add stay-interviews and a pulse survey (job satisfaction, work-life balance) for employees in the top risk decile.",
            "Strengthen manager continuity: avoid unnecessary manager reassignments; coach managers of high-risk teams.",
            "Review travel-heavy roles for travel reduction or compensation."])
d.add_heading("5. How HR should use the model", 1)
bullets(d, ["Score the workforce monthly and review the top 10–15 % highest-risk employees with their manager.",
            "Use it as a conversation starter, not a decision about an individual; never use it for firing, pay-cuts or promotion decisions.",
            "Check fairness regularly (age, gender, marital status) – these attributes are correlated with outcomes and must not drive unequal treatment.",
            "Retrain every 6–12 months and track whether interventions actually reduce attrition (A/B or matched-group comparison)."])
d.add_heading("6. Caveats", 1)
bullets(d, ["Predictive ≠ causal: reducing overtime is expected to help, but it should be piloted and measured.",
            "Precision is modest (many flagged employees would have stayed), so interventions should be low-cost and supportive.",
            "Small dataset; results should be validated on the company's own, larger and more recent data."])
d.save(OUT / "Business_Interpretation_Report.docx")
print("reports written to", OUT)
