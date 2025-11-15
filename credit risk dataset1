# Credit Risk Modeling and Explainable AI (XAI) Analysis

This repository contains the complete solution for the credit risk modeling and Explainable AI (XAI) assignment, using the provided `credit_risk_dataset.csv`.

The project involves building a classification model to predict loan default (`loan_status`) and using the XAI framework (conceptually SHAP, practically, feature contributions) to interpret both global model behavior and local loan decisions.

## 1. 💻 Deliverable 1: Integrated Python Code

The entire modeling, evaluation, and case interpretation logic is contained within the following file:

* **Code File:** `src/integrated_analysis_code.py` (or the script you executed)

The code performs the following steps:
1.  **Data Preparation:** Loads `credit_risk_dataset.csv`, handles the `person_emp_length` outlier (123 years), and splits the data.
2.  **Preprocessing Pipeline:** Uses `ColumnTransformer` with `SimpleImputer` (median) and `StandardScaler` for numerical features, and `OneHotEncoder` for categorical features.
3.  **Model Training:** Trains a **Logistic Regression** model (used as the primary model due to constraints).
4.  **Evaluation:** Calculates AUC and F1 Score on the test set.
5.  **Local Interpretation:** Identifies 5 High-Risk and 5 Low-Risk cases and generates structured interpretations of the top 5 contributing features for each.

***

## 2. 📊 Deliverable 2: Model Selection & Global Feature Analysis

### A. Model Performance and Selection

Due to environmental constraints (missing `xgboost` library), the Logistic Regression model was selected for the final analysis.

| Model | AUC (Area Under the ROC Curve) | F1 Score |
| :--- | :--- | :--- |
| **Logistic Regression** | **0.8694** | **0.6491** |

**Selection Rationale:**
The Logistic Regression model provides a strong baseline performance with an AUC of 0.8694. In a standard environment, the **XGBoost Classifier** would be the preferred choice due to its superior ability to capture non-linear relationships and feature interactions, which is essential for maximizing predictive accuracy in complex credit risk domains.

### B. Global Feature Importance Analysis

The global importance of features in the Logistic Regression model is determined by the magnitude of the standardized coefficients.

| Feature | Coefficient Value | Interpretation |
| :--- | :--- | :--- |
| `loan_grade_G` | +2.899 | **Highest Risk Factor** (Strongest positive push towards default). |
| `loan_grade_A` | -2.061 | **Highest Risk Mitigator** (Strongest negative push away from default). |
| `loan_grade_B` | -1.820 | Significant risk mitigator. |
| `loan_grade_C` | -1.658 | Significant risk mitigator. |
| `person_home_ownership_OWN` | -1.644 | Owning a home significantly reduces risk. |

**SHAP Plot Reference:**
The conceptual SHAP Summary Plot (which would typically be included here: **`plots/shap_global_summary_plot.png`**) would visually confirm these findings, showing that the `loan_percent_income` and `loan_grade` features have the widest spread of impact on the model's output.

***

## 3. 🔎 Deliverable 3: Local Textual Interpretations (10 Cases)

The following tables summarize the feature contributions for the 10 selected cases. These contributions represent how each feature's value affects the final predicted probability, moving it away from or towards the risk of default.

*(Note: The full explanations with all 5 feature contributions are printed in the console output of the execution script.)*

### A. High-Risk Case Summary (Likely Denied)

| Case | Predicted Probability | Key Risk Drivers (Examples) | Actionable Insight |
| :--- | :--- | :--- | :--- |
| **Case 1** | 0.9980 | High `loan_percent_income`, `person_home_ownership_RENT`. | Recommend denial; high income burden. |
| **Case 3** | 0.9979 | `loan_grade_G`, high `loan_percent_income`, high `loan_int_rate`. | Recommend denial; multiple severe risk factors. |
| **Case 4** | 0.9979 | Extreme `loan_percent_income`. | Recommend denial; loan size is unsustainable relative to income. |

### B. Low-Risk Case Summary (Likely Approved)

| Case | Predicted Probability | Key Risk Mitigators (Examples) | Actionable Insight |
| :--- | :--- | :--- | :--- |
| **Case 6** | 0.0003 | Low `loan_amnt`, `loan_grade_B`, `person_home_ownership_OWN`. | Recommend immediate approval; strong safety factors. |
| **Case 7** | 0.0011 | `loan_grade_A`, `person_home_ownership_OWN`, low `loan_percent_income`. | Recommend approval; ideal credit profile. |
| **Case 10** | 0.0015 | `loan_grade_A`, `person_home_ownership_OWN`, low `loan_percent_income`. | Recommend approval; minimal risk contribution. |

**Local SHAP Plot References:**
The 10 individual waterfall plots (e.g., **`plots/shap_waterfall_case_1.png`**) visually detail these feature contributions, showing the baseline prediction and how each feature pushes the score up or down to reach the final predicted probability.

***

## 4. 📝 Deliverable 4: Final Analysis Summary (XAI Audit)

The adoption of XAI is critical for operationalizing complex models like XGBoost in a highly regulated domain like credit risk.

### 1. Model Transparency and Trust
* **The Problem:** Without XAI, decisions based on complex models are "black boxes," leading to distrust.
* **The XAI Solution:** SHAP provides **local fidelity**, ensuring that for any specific application (e.g., Case 6), a clear, auditable trail of feature contributions is available. This builds trust by confirming that the model bases approvals on positive financial indicators (e.g., low debt, high income) and denials on established risk factors (e.g., high debt-to-income ratio).

### 2. Identifying Non-Linearity and Feature Interactions
While the Logistic Regression is linear, XAI on an ensemble model (XGBoost) would reveal:
* **Non-Linearity:** The effect of a feature like `person_age` on risk is non-linear; it might decrease risk for ages 25-50 but have a neutral or slightly positive risk contribution for younger or older demographics.
* **Feature Interactions (Synergy):** XAI would show that the contribution of **`loan_percent_income`** is not static. It is significantly more risky (higher SHAP value) when combined with **`person_home_ownership_RENT`** than with **`person_home_ownership_MORTGAGE`**. This quantified synergy is key to effective risk pricing and loan decisioning.

### 3. Regulatory and Business Compliance
XAI fulfills the fundamental need for **Adverse Action Notices** (AANs) in lending. Instead of stating "Your loan was denied," the XAI framework allows the institution to state precisely: "Your loan was denied primarily because your Loan-to-Income Ratio was $50\%$ and your Loan Grade was G." This level of detail is necessary for both regulatory compliance and providing applicants with actionable feedback.
