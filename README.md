# Bank Customer Churn Prediction

End-to-end machine learning project to predict bank customer churn and translate model results into business metrics. The analysis focuses on identifying customers with high churn risk, selecting the model that best captures churn cases, and estimating the potential revenue impact of targeted retention actions.

## Project Overview

Customer churn creates direct revenue risk for the bank. In the analyzed dataset, the churn rate is around **16.07%**, with an estimated annual revenue loss of **5,035,607** and an average loss per churned customer of **3,095**.

The main business objective is to build an early warning model that helps the bank prioritize retention campaigns for customers who are most likely to leave.

## Key Business Questions

- Which customer behaviors are most associated with churn?
- Which model is better for detecting churn customers?
- How much potential revenue can be protected if the model is used for retention targeting?
- What retention actions should be prioritized by the business team?

## Notebook

Main analysis: data/bank_churn_data.csv

The notebook covers:

- Business understanding and metric selection
- Exploratory data analysis
- Data preprocessing and feature handling
- Train-test split, encoding, scaling, and SMOTE
- Random Forest and XGBoost modeling
- Model evaluation with recall, precision, F1-score, and ROC-AUC
- SHAP-based model interpretability
- Business impact simulation
- Executive summary and retention recommendations

## Dataset

The notebook expects the dataset at:

```data/bank_churn_data.csv
```

The dataset is not included in this repository. Place the CSV file in the `data/` folder before running the notebook.

## Modeling Approach

The project uses **Recall** as the primary metric because missing a churn customer creates higher business risk than targeting a customer who would not churn.

Business interpretation:

- **False Negative**: the model predicts a customer will stay, but the customer actually churns. This creates missed retention opportunity and potential revenue loss.
- **False Positive**: the model predicts a customer will churn, but the customer actually stays. This may waste retention budget, but the cost is lower than losing a customer.

## Model Performance

The best model is **XGBoost**.

| Model | Churn Recall | Churn Precision | Accuracy | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.82 | 0.86 | 0.95 | 0.981 |
| XGBoost | 0.86 | 0.88 | 0.96 | 0.988 |

XGBoost performs better because it detects more churn customers and produces fewer false negatives.

## Main Insights

- Low transaction activity is the strongest churn signal.
- Customers with low `total_trans_ct` and low `total_trans_amt` are more likely to churn.
- Customers with `total_revolving_bal` close to zero have higher churn risk.
- Demographic features such as gender, education, and marital status are less influential than transaction behavior.
- Customers inactive for around three months show higher churn tendency.

## Business Impact Simulation

Using the XGBoost model:

- Predicted high-risk customers: **320**
- True churn customers detected: **282**
- Assumed retention success rate: **50%**
- Estimated customers saved: **141**
- Gross revenue saved: **436,395**
- Retention cost: **16,000**
- Estimated net revenue protected: **420,395**

Without a model, the bank risks losing **327 churn customers**, equal to an estimated **1,012,065** in revenue.

## Recommendations

- Deploy the XGBoost model as an early warning system for churn prevention.
- Prioritize customers with declining transaction count, low transaction amount, and low revolving balance.
- Allocate retention budget for targeted promotions, cashback, discounts, or point redemption.
- Monitor recall regularly so the model continues to capture as many churn customers as possible.

## How To Run

1. Clone the repository.
2. Add the dataset to `data/bank_churn_data.csv`.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open and run the notebook:

```bash
jupyter notebook notebooks/churn_prediction_business_metrics.ipynb
```

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- SHAP
- Jupyter Notebook
