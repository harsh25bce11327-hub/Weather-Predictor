# Project Statement

## Weather Prediction Using Naive Bayes

**Subject:** Fundamentals of Artificial Intelligence / Machine Learning
**Type:** Academic Mini Project
**Algorithm:** Naive Bayes (Gaussian NB)

---

## Problem Statement

Weather forecasting is a critical real-world challenge that affects agriculture, urban planning, disaster management, and everyday human decisions. Traditional weather prediction relies on complex atmospheric simulations that require significant computational resources and domain expertise.

This project addresses the following core problem:

> **Can a simple probabilistic Machine Learning model accurately predict whether it will rain tomorrow, based on observable weather conditions recorded today?**

The model takes historical weather attributes — such as temperature, humidity, wind, outlook, and atmospheric pressure — as input, and classifies the next day's weather as either **Rain** or **No Rain**.

---

## Objectives

1. **Implement** a Naive Bayes classification model for binary weather prediction.
2. **Preprocess** weather data by handling missing values, encoding categorical features, and removing irrelevant columns.
3. **Train and evaluate** the model using standard ML metrics: accuracy score, confusion matrix, and classification report.
4. **Scale** the solution to support both small sample datasets and large real-world CSV datasets.
5. **Identify** the most influential weather features (top predictors) that drive rainfall classification.

---

## Scope

### In Scope
- Binary classification: predicting `RainTomorrow` as Yes or No
- Support for built-in sample data (14 rows) for quick testing
- Support for external CSV datasets (e.g. the Kaggle "Rain in Australia" dataset with 145,000+ records)
- Automatic preprocessing: null handling, label encoding, column filtering
- Model evaluation using accuracy, confusion matrix, and F1-score
- Command-line interface for flexible usage

### Out of Scope
- Multi-day or hourly weather forecasting
- Numerical weather prediction (NWP) simulations
- Real-time data ingestion or live API integration
- Deep learning or ensemble methods
- Graphical user interface (GUI) or web deployment *(noted as a future improvement)*

---

## Motivation

| Stakeholder | Need |
|---|---|
| Farmers | Plan irrigation, planting, and harvest schedules |
| City planners | Prepare infrastructure for floods or drought |
| Emergency services | Anticipate extreme weather events |
| General public | Make informed daily decisions |
| ML students | Learn probabilistic classification in a real-world context |

---

## Why Naive Bayes?

Naive Bayes was chosen as the algorithm for this project for the following reasons:

- **Simplicity** — Easy to understand and implement, making it ideal for academic exploration.
- **Speed** — Trains and predicts extremely fast even on large datasets.
- **Interpretability** — The probabilistic reasoning is transparent; predictions can be traced back to individual feature contributions.
- **Effectiveness on small data** — Performs well even with limited training samples, unlike deep learning models that need large datasets to generalise.
- **Baseline value** — Establishes a strong, honest baseline before exploring more complex models.

---

## Known Limitations

- **Independence assumption** — Naive Bayes assumes all features are independent, which is not true in weather data (e.g. humidity and pressure are correlated). This reduces precision but does not prevent useful predictions.
- **Class imbalance** — Sunny/dry days typically outnumber rainy days in datasets, causing the model to be biased toward predicting "No Rain." Addressed through evaluation with per-class F1-scores.
- **Gaussian assumption** — Gaussian NB assumes features follow a normal distribution, which may not hold perfectly for all weather variables.

---

## Expected Outcomes

- A working Python-based classification model that predicts rainfall
- Accuracy in the range of **75–85%** on real-world datasets
- Identification of **humidity** and **pressure** as the strongest rainfall predictors
- A clean, documented codebase suitable for academic submission and future extension

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| Pandas | Data loading and preprocessing |
| Scikit-learn | Model training and evaluation |
| GaussianNB | Naive Bayes classifier |
| argparse | Command-line interface |

---

## Future Extensions

- Add data visualisation (correlation heatmaps, class distribution plots)
- Experiment with other classifiers: Random Forest, SVM, XGBoost
- Implement cross-validation for robust performance estimates
- Build a Streamlit-based web interface for interactive predictions
- Deploy as a REST API for integration with weather dashboards
