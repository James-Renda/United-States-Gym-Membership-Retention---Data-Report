# United States Gym Data Report: Membership Churn Analysis

Using the CRISP-DM methodology, this project analyzes a U.S. gym's membership data to understand churn behavior, identify key predictors of cancellation, and build machine learning models to support customer retention strategies.

## Team
- James Renda
- Zane Davis
- Jackson Wainwright

## Business Understanding

The primary business goal is to reduce customer churn by identifying the key drivers behind gym membership cancellations. Using Python-based machine learning techniques, the objective is to predict which members are most at risk of churn and provide actionable insights for retention efforts.

## Data Understanding

The dataset contains 4,000 rows and 14 columns, including tenure, monthly charges, total charges, usage frequency, and demographic information. These variables are used to predict `Churn` (1 = churned, 0 = retained).

- **Retained:** 2,939 members (~75%)
- **Churned:** 1,061 members (~25%)

## Data Preparation

- Verified there were no null values using `.info()`
- Removed `Avg_class_frequency_current_month` after identifying data leakage — it reflected same-month behavior and artificially inflated correlation with churn
- Binned numeric features into categorical groups for clearer behavioral patterns and better modeling:
  - **Lifetime:** `0_6_months`, `7_12_months`, `13_24_months`, `25_plus_months`
  - **Age:** `18_24`, `25_29`, `30_34`, `35_plus`
  - **Month_to_end_contract → Contract_end_group:** highlights churn risk as members approach contract expiration
- Most cancellations occurred within the **1–3 month** contract-end window (2,244 members)
- Converted categorical columns using `get_dummies` for modeling, dropping original numeric columns

## Modeling

- **Target variable:** `Churn` (1 = churned, 0 = retained)
- **Predictors:** Income, Days_since_last_visit, Avg_class_frequency_total, Avg_PT_total, Total_charges, Lifetime_group, Age_group, Contract_end_group, Contract_period
- **Train/test split:** 70/30
- **Models compared:** Logistic Regression, Decision Tree, Random Forest, XGBoost
- No unsupervised methods were used

### Model Evaluation

| Model | Accuracy | ROC AUC | PR AUC |
|---|---|---|---|
| Logistic Regression | 0.841 | 0.899 | 0.745 |
| Decision Tree | 0.826 | 0.852 | 0.593 |
| Random Forest | 0.841 | 0.883 | 0.692 |
| XGBoost | 0.816 | 0.886 | 0.693 |

**Logistic Regression performed best overall**, with the strongest ROC AUC and PR AUC, suggesting the data is best captured by a linear decision boundary. The Decision Tree performed worst, particularly on PR AUC.

## Key Predictors of Churn

Using permutation importance, the top predictors identified were:
1. **Age group 30–34** — the most important predictor; this age group shows a higher likelihood of churn
2. **Lifetime group 7–12 months** — members in this tenure range show uncertain retention commitment
3. **Average class frequency total** — members who attend class less frequently are much more likely to cancel

### Model Interpretability (SHAP & LIME)

- **SHAP Beeswarm Plot:** Higher feature values (older age groups, longer contract periods) push predictions toward non-churn; lower feature values (low class frequency, shorter remaining contract time) push predictions toward churn.
- **SHAP Waterfall Plot:** Being in the 35+ age group and having a 6-month contract reduces predicted churn likelihood, while being in the 30–34 age group and higher additional charges increase churn risk.
- **LIME Explainer:** For an example customer, the model predicted only an 8% chance of churn — driven by age (35+), a 6-month contract period, and a 4–6 month contract-end group, despite some offsetting risk factors.

## Discussion & Insights

- **Engagement matters most.** Members with low class attendance or overall usage were consistently more likely to churn, signaling declining commitment.
- **Contract structure is a strong signal.** Short-term or monthly plans, and contracts nearing expiration, were closely tied to higher churn. Longer contract periods and later contract stages showed better retention.
- **Demographics matter.** Members aged 30–34 are a higher-risk group, possibly due to lifestyle or financial shifts, while members 35+ are among the most stable segments.

## Recommendations

1. **Personalized engagement** for high-risk members — tailored class reminders, attendance rewards, or wellness check-ins for members with declining usage.
2. **Contract retention incentives** — renewal discounts, value-added programs, or structured incentives for members approaching contract expiration.
3. **Demographic targeting** — direct marketing and retention resources toward age groups most likely to churn (e.g., 30–34).

## Deliverables

- Jupyter Notebook (data cleaning, modeling, and evaluation)
- Written report with findings and retention recommendations

## Tools & Libraries

- Python, pandas
- scikit-learn (Logistic Regression, Decision Tree, Random Forest)
- XGBoost
- SHAP, LIME (model interpretability)
