"""Generate a synthetic stand-in for the IBM HR Analytics Employee Attrition dataset.

Same 35 columns / value ranges as the Kaggle file `WA_Fn-UseC_-HR-Employee-Attrition.csv`
(1470 rows, ~16% attrition). Used ONLY when the real file is not present in data/.
"""
import numpy as np, pandas as pd, sys, pathlib

def generate(n=1470, seed=42):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(37, 9, n).round(), 18, 60).astype(int)
    total_years = np.clip(((age - 18) * rng.uniform(0.1, 0.75, n)).round(), 0, 40).astype(int)
    job_level = np.clip(1 + (total_years / 8 + rng.normal(0, .8, n)).round(), 1, 5).astype(int)
    income = np.clip(2000 + job_level * 3300 + total_years * 90 + rng.normal(0, 1500, n), 1009, 19999).round().astype(int)
    years_co = np.clip((total_years * rng.uniform(0.1, 0.9, n)).round(), 0, 40).astype(int)
    yrs_role = np.clip((years_co * rng.uniform(0.2, 1, n)).round(), 0, 18).astype(int)
    yrs_promo = np.clip((years_co * rng.uniform(0, .7, n)).round(), 0, 15).astype(int)
    yrs_mgr = np.clip((years_co * rng.uniform(0.2, 1, n)).round(), 0, 17).astype(int)
    pick = lambda opts, p: rng.choice(opts, n, p=p)
    travel = pick(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], [.71, .19, .10])
    dept = pick(["Research & Development", "Sales", "Human Resources"], [.65, .30, .05])
    role = np.empty(n, dtype=object)
    for i, d in enumerate(dept):
        opts = {"Research & Development": ["Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Research Director"],
                "Sales": ["Sales Executive", "Sales Representative", "Manager"],
                "Human Resources": ["Human Resources", "Manager"]}[d]
        role[i] = rng.choice(opts)
    field = pick(["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"], [.41, .32, .11, .09, .05, .02])
    marital = pick(["Married", "Single", "Divorced"], [.46, .32, .22])
    overtime = pick(["No", "Yes"], [.72, .28])
    stock = np.where(marital == "Single", pick([0, 1, 2, 3], [.6, .25, .1, .05]), pick([0, 1, 2, 3], [.35, .42, .15, .08]))
    dist = np.clip(rng.gamma(1.6, 5, n).round() + 1, 1, 29).astype(int)
    env = pick([1, 2, 3, 4], [.19, .19, .30, .32]); jsat = pick([1, 2, 3, 4], [.20, .19, .30, .31])
    inv = pick([1, 2, 3, 4], [.05, .25, .59, .11]); rel = pick([1, 2, 3, 4], [.19, .21, .31, .29])
    wlb = pick([1, 2, 3, 4], [.05, .23, .61, .11])
    ncomp = np.clip(rng.poisson(2.0, n), 0, 9)
    hike = np.clip(rng.normal(15.2, 3.6, n).round(), 11, 25).astype(int)
    perf = np.where(hike > 18, 4, 3)
    train = np.clip(rng.poisson(2.8, n), 0, 6)
    # --- attrition: logistic model on realistic drivers ---
    z = (-1.9 + 1.5 * (overtime == "Yes") + 0.9 * (travel == "Travel_Frequently") - 0.7 * (travel == "Non-Travel")
         + 0.9 * (marital == "Single") - 0.00020 * (income - 6500) - 0.045 * (age - 37)
         - 0.40 * (jsat - 2.5) - 0.30 * (env - 2.5) - 0.30 * (wlb - 2.5) + 0.035 * dist
         - 0.45 * stock - 0.06 * years_co + 0.07 * yrs_promo + 0.12 * ncomp
         + 0.5 * (role == "Sales Representative") + rng.normal(0, .6, n))
    # calibrate intercept to ~16% attrition
    lo, hi = -6, 6
    for _ in range(40):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > 0.161: hi = mid
        else: lo = mid
    p = 1 / (1 + np.exp(-(z + mid)))
    attr = np.where(rng.uniform(size=n) < p, "Yes", "No")
    df = pd.DataFrame({
        "Age": age, "Attrition": attr, "BusinessTravel": travel,
        "DailyRate": rng.integers(102, 1500, n), "Department": dept, "DistanceFromHome": dist,
        "Education": pick([1, 2, 3, 4, 5], [.11, .19, .39, .27, .04]), "EducationField": field,
        "EmployeeCount": 1, "EmployeeNumber": np.arange(1, n + 1), "EnvironmentSatisfaction": env,
        "Gender": pick(["Male", "Female"], [.6, .4]), "HourlyRate": rng.integers(30, 101, n),
        "JobInvolvement": inv, "JobLevel": job_level, "JobRole": role, "JobSatisfaction": jsat,
        "MaritalStatus": marital, "MonthlyIncome": income, "MonthlyRate": rng.integers(2094, 26999, n),
        "NumCompaniesWorked": ncomp, "Over18": "Y", "OverTime": overtime, "PercentSalaryHike": hike,
        "PerformanceRating": perf, "RelationshipSatisfaction": rel, "StandardHours": 80,
        "StockOptionLevel": stock, "TotalWorkingYears": total_years, "TrainingTimesLastYear": train,
        "WorkLifeBalance": wlb, "YearsAtCompany": years_co, "YearsInCurrentRole": yrs_role,
        "YearsSinceLastPromotion": yrs_promo, "YearsWithCurrManager": yrs_mgr})
    return df

if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_hr_attrition.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    d = generate(); d.to_csv(out, index=False)
    print(out, d.shape, "attrition rate:", (d.Attrition == "Yes").mean().round(3))
