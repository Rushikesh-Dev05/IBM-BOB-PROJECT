# Rushikesh Online Shopper Purchase Intention Analysis and AI Prediction

## IBM SkillsBuild Data Analytics with AI Academic Internship

### 1. Project overview
This project analyzes online shopping-session behavior and builds a machine-learning model to predict whether a session is likely to end in a purchase.

The project covers both sides of a Data Analytics with AI internship:

- Data loading and validation
- Data cleaning
- Exploratory Data Analysis (EDA)
- Visualization
- Feature engineering
- Classification
- Model comparison
- Model evaluation
- Cross-validation
- Model interpretability using permutation importance
- Reusable prediction function
- Saving a trained model with `joblib`

## 2. Dataset

**Dataset:** Online Shoppers Purchasing Intention Dataset

**Official UCI source:** https://archive.ics.uci.edu/dataset/468/online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset

**Kaggle mirror:** https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset

The UCI record describes 12,330 online-shopping sessions and 18 columns including the binary `Revenue` target. The dataset is designed for classification/clustering and contains numerical and categorical attributes.

The project does **not** use the classroom sales dataset or house-price dataset described in the internship instructions; it uses a separate online-shopping behavior dataset.

## 3. Project question

> Can browsing behavior and session characteristics be used to estimate whether an online shopping session will result in a purchase?

## 4. Main features

Examples include:

- Administrative pages and time
- Informational pages and time
- Product-related pages and time
- Bounce rate
- Exit rate
- Page value
- Special-day closeness
- Month
- Operating system
- Browser
- Region
- Traffic type
- Visitor type
- Weekend

Target:

- `Revenue = True/1`: purchase session
- `Revenue = False/0`: non-purchase session

## 5. AI methodology

The notebook compares:

1. Logistic Regression
2. Random Forest

Both models use preprocessing pipelines. Numeric features are imputed/scaled and categorical features are one-hot encoded. Class weighting is used because the purchase class is smaller than the non-purchase class.

The model is selected automatically using test-set F1-score. Accuracy, precision, recall, F1-score and ROC-AUC are reported.

## 6. Feature engineering

The notebook creates:

- `TotalPages`
- `TotalPageDuration`
- `AvgTimePerPage`
- `EngagementScore`

These features summarize overall browsing activity.

## 7. How to run

### Option A — Jupyter Notebook

```bash
pip install -r requirements.txt
jupyter notebook
```

Open:

`Rushikesh_OnlineShopper_Intent_Analysis.ipynb`

Run all cells from top to bottom.

### Option B — Google Colab

1. Upload the `.ipynb` file to Google Colab.
2. Run the installation cell.
3. Run all cells.
4. The notebook retrieves the dataset through `ucimlrepo`.

## 8. Generated artifacts

After a successful run, the notebook creates:

```text
artifacts/
├── model_comparison.csv
├── cross_validation_f1.csv
├── feature_importance.csv
├── results_summary.txt
└── online_shopper_purchase_intention_model.joblib
```

These files are supporting outputs. The internship's mandatory submission files are still the notebook, `requirements.txt`, project report and `README.md`.

## 9. Reproducibility

A fixed random seed (`42`) is used for train/test splitting, model training and cross-validation so that results are reproducible when the same software versions and dataset are used.

## 10. Limitations

- The data represents historical sessions and should not be treated as a guarantee of future customer behavior.
- The target is purchase completion, not purchase amount or revenue value.
- Model performance depends on the dataset and preprocessing choices.
- A production system would require monitoring, new data, threshold tuning and privacy/security review.

## 11. References

Sakar, C. O., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. DOI: 10.24432/C5F88Q.

UCI dataset page: https://archive.ics.uci.edu/dataset/468/online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset
