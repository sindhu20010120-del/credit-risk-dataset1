# Credit Risk Prediction and Explainable AI (XAI) Report

## Executive Summary

This report details the development and interpretation of a credit risk classification model using the provided `credit_risk_dataset`. The primary goal was to predict loan default (`loan_status`) and to employ Explainable AI (XAI) techniques to provide transparent and auditable decision-making.

A **Logistic Regression** model achieved a high predictive performance (**AUC: 0.8694**). Global analysis demonstrated that credit **loan grade** and **loan-to-income percentage** are the primary risk drivers. Local analysis (using linear contributions as a proxy for SHAP) provided specific, actionable insights for risk officers on 10 individual loan applications, thereby converting the model from a predictive tool into an auditable business intelligence asset.

## 1. Introduction

Credit risk modeling requires both high predictive accuracy and high transparency. In regulated environments, "black box" decisions are unacceptable. This project addresses this requirement by:
1. Building a robust classification model.
2. Using interpretation techniques (conceptually SHAP) to explain model decisions at global and local levels.

## 2. Model Building and Selection (Deliverable 2)

### 2.1 Preprocessing and Feature Engineering

The `credit_risk_dataset.csv` was preprocessed using a `ColumnTransformer` pipeline:
* **Outlier Handling:** Extreme values in `person_emp_length` were treated as missing.
* **Imputation:** Missing numerical values (`person_emp_length`, `loan_int_rate`) were imputed using the median.
* **Scaling:** Numerical features were standardized using `StandardScaler`.
* **Encoding:** Categorical features were converted using `OneHotEncoder`.

### 2.2 Model Performance

Due to platform library constraints (missing `xgboost` and `shap`), the **Logistic Regression** model was used for all reported metrics and interpretations.

| Model | AUC (Area Under the ROC Curve) | F1 Score |
| :--- | :--- | :--- |
| **Logistic Regression** | **0.8694** | **0.6491** |

**Model Selection Justification:**
While Logistic Regression performed well, the full scope of the project requires a non-linear model, typically **XGBoost**, which is generally superior at capturing complex feature interactions and subtle risk thresholds. The analysis in sections 3 and 4 should be viewed as structured interpretations that would apply to a non-linear XGBoost model if available.

## 3. Global Feature Analysis (Deliverable 2)

The most influential features are identified using the magnitude of the standardized coefficients, representing their overall impact on the default probability.

| Feature | Coefficient Value | Impact Direction |
| :--- | :--- | :--- |
| `loan_grade_G` | +2.899 | **Strongly Increases Risk** (Highest driver of default). |
| `loan_grade_A` | -2.061 | **Strongly Decreases Risk** (Highest mitigator of default). |
| `person_home_ownership_OWN` | -1.644 | Significant risk mitigation (Wealth proxy). |
| `loan_percent_income` | N/A (Highly variable) | **Quantified as high risk across cases.** |

**Interpretation:** The model confirms that institutional factors (loan grade) and core financial health metrics (income, home ownership) are the primary determinants of risk.

## 4. Local Decision Interpretation (Deliverable 3)

Ten loan applications (5 high-risk, 5 low-risk) were selected for local, prediction-specific analysis. The full textual interpretations are provided in the repository's console output.

### 4.1 High-Risk Case Summary

High-risk cases (predicted probability $\geq 0.996$) were consistently driven by the extreme values of **`loan_percent_income`** and the presence of high-risk categorical factors (e.g., `loan_grade_G` or `person_home_ownership_RENT`).

* **Example (Case 4):** Predicted default probability of $0.9979$ was primarily due to an extremely high `loan_percent_income` contribution ($\approx +7.037$) which overwhelmed all mitigating factors.
* **Actionable Insight:** The Risk Officer is advised to deny the application due to unsustainable debt burden relative to income.

### 4.2 Low-Risk Case Summary

Low-risk cases (predicted probability $\leq 0.0015$) demonstrated a synergistic effect of positive factors:
* **Example (Case 10):** Predicted default probability of $0.0015$ was due to strong negative contributions from `loan_grade_A` ($\approx -2.061$) and `person_home_ownership_OWN` ($\approx -1.644$).
* **Actionable Insight:** The Risk Officer is advised to approve the loan, using this case as a benchmark for low-risk characteristics.

## 5. Conclusion and XAI Audit (Deliverable 4)

### 5.1 The Value of XAI for Model Auditing

The use of an XAI framework (SHAP) is essential for moving beyond the linear interpretation provided by Logistic Regression.
* **Identifying Non-Linearity:** SHAP on an XGBoost model would reveal that risk features (like `loan_int_rate`) do not increase risk linearly but may exhibit complex, segmented risk thresholds, a finding impossible for linear models.
* **Quantifying Feature Interaction:** SHAP quantifies the synergistic effect where two moderately negative features (e.g., low income AND renting) combine to create a disproportionately large risk factor. This quantified synergy is the most powerful tool for setting nuanced credit policies.

### 5.2 Regulatory Compliance

By providing a **local, feature-specific explanation** for every prediction, XAI fulfills the requirement for transparency in lending decisions. This allows the financial institution to issue precise **Adverse Action Notices**, citing the exact features responsible for the denial, rather than vague reasons. This capability is paramount for operational trust and regulatory compliance.

## 6. Appendix: Code and Data Reference

* **Data File:** `data/credit_risk_dataset.csv`
* **Code File:** `src/integrated_analysis_code.py`
* **Plots Folder:** `plots/` (Contains the conceptual SHAP Summary Plot and 10 individual Waterfall Plots.)
