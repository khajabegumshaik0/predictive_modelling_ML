# 📊 Exploratory Data Analysis (EDA) Report
**Project Name:** Loan Application Approvals Analysis  
**Prepared For:** Risk & Underwriting Division  
**Data Size:** 200 Records  

---

## 1. Executive Summary
This report presents an Exploratory Data Analysis (EDA) on a dataset of 200 synthetic but statistically realistic loan applications. The objective is to identify the key demographic and financial factors that influence loan approvals, examine variable distributions, and assess feature correlation patterns. 

These findings directly guide the feature engineering and modeling steps for our machine learning classifiers (Logistic Regression, Decision Tree, and Random Forest).

### Core Statistics
* **Total Applications:** 200
* **Approved Applications:** 114 (57.0%)
* **Rejected Applications:** 86 (43.0%)
* **Overall Approval Rate:** 57.0%

---

## 2. Descriptive Statistics
Below is the statistical summary of the features across all 200 applicants:

| Variable | Metric Description | Mean | Std Dev (σ) | Min | Max |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Age** | Applicant's Age in Years | 38.3 years | 10.4 years | 18.0 | 66.0 |
| **Income** | Annual Gross Income | $69,219 | $21,170 | $20,000 | $127,104 |
| **Credit Score** | Equifax Credit Score | 694.3 | 82.5 points | 480.0 | 850.0 |
| **Debt-to-Income (DTI)** | Monthly debt / Gross income | 35.8% | 14.6% | 10.0% | 60.0% |
| **Loan Amount** | Requested Loan Principal | $17,046 | $6,451 | $5,000 | $41,202 |

---

## 3. Correlation Analysis
Correlation analysis measures the linear relationship between variables and the target variable (`Approved`). The values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).

### Feature Correlation with Loan Approval
The features are ordered below by their influence (strength of correlation) on approval:

1. **Credit Score (`r = +0.716`)** — *Strong Positive Influence*  
   Credit score is the single most critical factor. Applicants with high credit scores have a substantially higher probability of approval.
   
2. **Debt-to-Income Ratio (`r = -0.347`)** — *Moderate Negative Influence*  
   A higher DTI ratio indicates high outstanding debt obligations relative to income. As DTI increases, the approval rate drops significantly.
   
3. **Annual Income (`r = +0.334`)** — *Moderate Positive Influence*  
   Higher annual income increases the capacity to repay, resulting in a positive correlation with loan approval.
   
4. **Loan Amount (`r = -0.016`)** — *Negligible Influence*  
   Loan amount by itself has a near-zero correlation with approval, suggesting that loan size is evaluated relative to the applicant's income and creditworthiness rather than as a standalone metric.
   
5. **Age (`r = -0.011`)** — *Negligible Influence*  
   Age shows no statistically significant correlation with approval. The underwriting criteria do not discriminate or cluster significantly based on applicant age.

---

## 4. Key Visual Discoveries & Patterns

### A. Credit Score Distribution
- **The Pattern:** A clear decision boundary is visible around a credit score of **600-650**.
- **Insights:** Applicants with a credit score below **580** are almost universally rejected (Approved = 0). Conversely, applicants with a credit score above **720** are almost guaranteed approval (Approved = 1), unless their DTI ratio is exceptionally high. The region between **580 and 700** is a transitional zone where income and DTI decide the outcome.

### B. Income vs. Credit Score Interaction
- **The Pattern:** Looking at the scatter plot of Credit Score (x-axis) vs. Income (y-axis) colored by Approval:
- **Insights:**
  - High-income applicants ($90,000+) can occasionally secure approvals with moderate credit scores (~600).
  - Low-income applicants (<$35,000) are frequently rejected even with decent credit scores (~650).
  - This demonstrates a non-linear interaction effect between Income and Credit Score that the machine learning models (specifically Decision Trees and Random Forests) are well-equipped to capture.

### C. Debt-to-Income (DTI) Boxplot Behavior
- **The Pattern:** Approved loans are heavily concentrated at lower DTI levels.
- **Insights:** The median DTI for approved applicants is **~28%**, with very few approved applicants exceeding a **45%** DTI. In contrast, rejected applicants have a median DTI of **~47%**, showing that high leverage is a key rejection trigger.

---

## 5. Machine Learning Pipeline Implications
1. **Feature Scaling:** Since variables have widely different scales (e.g., Credit Score ranges from 300 to 850, while Income ranges up to $150,000), applying `StandardScaler` is crucial to prevent features with larger scales from dominating algorithms like Logistic Regression.
2. **Model Selection:** Logistic Regression provides a solid, interpretable linear baseline. Decision Trees and Random Forests are able to leverage interaction effects (like high income mitigating a lower credit score) to achieve higher classification precision.
