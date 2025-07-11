# Loan Approval Prediction – Machine Learning FinTech Innovations Loan Approval model

This repository contains a complete data science pipeline for predicting loan approvals based on applicant financial and demographic information. 
The project follows the **CRISP-DM framework** and was completed as part of a final evaluation in a data science bootcamp.

---

## 📊 Problem Statement

Financial institutions face asymmetric costs in loan decisions:
- **$50,000** lost for approving a bad loan (false positive)
- **$8,000** in missed profit when rejecting a good one (false negative)

The goal is to build a **classification model** that minimizes these business costs while maintaining fairness and generalizability.

---

## 💼 Business Understanding

The dataset simulates a real-world loan application scenario. Predicting the `LoanApproved` status helps optimize financial decisions, reduce risk, and improve customer experience.

Key challenges:
- Imbalanced cost of classification errors
- Need for explainable, trustworthy models for stakeholders
- Avoiding overfitting while capturing relevant risk signals

---

## 📂 Dataset

The dataset contains 15,000+ anonymized loan applications and includes features such as:

- `CreditScore`, `RiskScore`, `MonthlyIncome`
- `LoanAmount`, `DebtToIncomeRatio`, `PreviousLoanDefaults`
- `EducationLevel`, `HomeOwnershipStatus`, `EmploymentStatus`, etc.

Target variable: **`LoanApproved`** and RiskScore as secondary target

---

## 🛠️ Methodology (CRISP-DM)

### 1. Business Understanding
- Defined high-stakes cost-sensitive objectives
- Stakeholder needs emphasized model transparency

### 2. Data Understanding
- Performed EDA, null value assessment, and target distribution checks
- Created approval rate charts by categorical segments

### 3. Data Preparation
- Handled missing values with imputation strategies
- Applied scaling to numerical features and one-hot encoding to categorical ones
- Removed data leakage risks (e.g., RiskScore not used as a feature in training)

### 4. Modeling
- Developed and tuned both **Logistic Regression** and **Random Forest** models using `RandomizedSearchCV`
- Primary metric: **F1 Score**
- Secondary metrics: **Precision**, **Recall**, and **Custom Business Cost**

### 5. Evaluation
- Final model: **Logistic Regression** with `liblinear` solver and L1 penalty
- Perfect performance metrics (F1, precision, recall) on the test set — indicating **overfitting**
- Confusion matrix and ROC curve confirm this overperformance
- Segment analysis showed uniform results across categories, further signaling lack of generalization

### 6. Deployment Readiness
- Due to model overfitting, deployment is not recommended without further validation on unseen/external data
- Feature importance analysis performed using model coefficients

---

## 🔍 Feature Importance (Top 5)

| Feature                         | Impact         |
|--------------------------------|----------------|
| RiskScore                      | Most Predictive|
| BankruptcyHistory              | High Impact    |
| DebtToIncomeRatio              | Moderate       |
| EmploymentStatus_SelfEmployed | Moderate       |
| MonthlyIncome                  | Moderate       |

---

This metric was planned to visualize financial impact but was not calculated due to unrealistic model perfection, which invalidates meaningful cost trade-offs.

---

## ⚠️ Limitations & Recommendations

- **Overfitting:** Perfect metrics suggest model memorized training data
- **Fairness testing:** No detected bias, but further validation needed
- **Next Steps:** 
  - Validate on an external holdout set or future data
  - Explore regularization strength and ensemble techniques
  - Incorporate business feedback to simulate real-world rejections/approvals
    
---

## 👨‍💻 Technologies Used

- Python (pandas, scikit-learn, matplotlib, seaborn)
- Jupyter Notebook
- CRISP-DM methodology


