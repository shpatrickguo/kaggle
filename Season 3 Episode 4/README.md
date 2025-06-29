# Credit Card Fraud Detection with Ensemble Learning

## Executive Summary

This project presents a credit card fraud detection system built for a highly imbalanced dataset derived from European cardholder transactions in September 2013. The simulated data includes 503,936 transactions, of which only 961 (0.19%) are fraudulent. To tackle this challenge, the approach integrates an ensemble of XGBoost, LightGBM, and a Multi-Layer Perceptron (MLP), resulting in an ROC AUC score of 0.82. This evaluation metric was chosen to emphasize the accurate ranking of rare fraud cases over legitimate ones—an essential consideration given the significant class imbalance.

## Data

The dataset is sourced from Kaggle and contains 31 features, including 28 anonymized variables (V1-V28) derived from PCA transformations, along with 'Time', 'Amount', and 'Class' (indicating fraud). The data is highly imbalanced, with only 0.19% of transactions labeled as fraudulent.

## Methodology

The project employs an ensemble learning approach, combining the strengths of three models:
1. **XGBoost**: A gradient boosting framework that is efficient and effective for classification tasks.
2. **LightGBM**: A gradient boosting framework that uses tree-based learning algorithms, known for its speed and efficiency.
3. **Multi-Layer Perceptron (MLP)**: A type of neural network that can capture complex patterns in the data.

The ensemble is trained on the training set and evaluated on a separate test set. The models are combined using a voting classifier to improve prediction accuracy.

## Results

The ensemble model achieved an ROC AUC score of 0.82, indicating a strong ability to distinguish between fraudulent and legitimate transactions. The model's performance was validated using cross-validation techniques to ensure robustness.

## Conclusion

The credit card fraud detection system demonstrates the effectiveness of ensemble learning in handling highly imbalanced datasets. The combination of XGBoost, LightGBM, and MLP provides a robust solution for identifying fraudulent transactions, achieving a commendable ROC AUC score of 0.82. Future work may explore further enhancements, such as additional feature engineering, to improve model performance.

## Installation

To run the code, ensure you have the following Python packages installed:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow
```

