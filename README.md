# Weather Prediction using Naive Bayes

## Project Overview

This project implements a Machine Learning model to predict whether it will rain tomorrow or not, based on weather factors such as outlook, temperature, humidity, wind, presure, and UV index. It supports both a built-in sample dataset and loading **real-world CSV datasets** (e.g. from Kaggle or BOM) for larger-scale training and evaluation.

---

## Algorithm Used by me: 

* Naive Bayes Algorithm 
* Bayes Theorem
* Assumes all features are conditionally independent

---

## Dataset Description

The model supports two data modes:

### Mode 1 — Built-in Sample Dataset
A small 14-row dataset with categorical features, useful for quick testing:

| Feature | Values |
|---------|--------|
| Outlook | Sunny, Overcast, Rain |
| Temperature | Hot, Mild, Cool |
| Humidity | High, Normal |
| Wind | Weak, Strong |
| RainTomorrow | Yes / No |

### Mode 2 — External CSV Dataset (Recommended)
Load any CSV file with weather data. The script auto-detects and encodes features.

Recommended public dataset: [Rain in Australia – Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

Expected columns (configurable in script):
```
Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am,
WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm,
Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow
```

---

## Technologies Used

* Python 3.8+
* Pandas
* Scikit-learn
* Matplotlib / Seaborn (optional, for visualisation)

---

## Installation

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

## How to Run

### 1. Run with built-in sample data (default)
```bash
python weather_predictor.py
```

### 2. Run with your own CSV dataset
```bash
python weather_predictor.py --csv path/to/weatherAUS.csv
```

### 3. Specify target column and test split
```bash
python weather_predictor.py --csv weatherAUS.csv --target RainTomorrow --test-size 0.2
```

---

## Sample Output

```
=== Weather Prediction - Naive Bayes ===
Data source    : weatherAUS.csv
Total samples  : 145,460
Training set   : 116,368
Test set       : 29,092

Accuracy       : 84.32 %

Confusion Matrix:
[[21543  1847]
 [ 2721  2981]]

Classification Report:
              precision    recall  f1-score   support
          No       0.89      0.92      0.90     23390
         Yes       0.62      0.52      0.57      5702
    accuracy                           0.84     29092
```

---

## Project Structure

```
weather-prediction/
│
├── weather_predictor.py     # Main script
├── README.md                # This file
├── Weather_Prediction_Report.md  # Project report
└── data/
    └── weatherAUS.csv       # (place your dataset here)
```

---

## Features

* Works with tiny sample data **or** large real-world CSVs
* Automatic handling of missing values
* Label encoding for categorical columns
* Detailed evaluation: accuracy, confusion matrix, classification report
* Command-line arguments for flexible usage
* Feature importance display (top predictors)

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy Score | Overall percentage of correct predictions |
| Confusion Matrix | Breakdown of TP, TN, FP, FN |
| Classification Report | Per-class precision, recall, F1-score |
| Feature Importance | Which features most influence predictions |

---

## Learning Outcomes

* Understanding of supervised learning and probabilistic classifiers
* Hands-on experience with real-world data preprocessing
* Model evaluation beyond simple accuracy
* Working with command-line Python tools

---

## Future Improvements

* Add data visualisation (correlation heatmaps, rainfall distribution)
* Experiment with other classifiers (Random Forest, XGBoost)
* Build a Streamlit web UI for interactive predictions
* Deploy the model as a REST API
* Add cross-validation for more robust evaluation

---

## Conclusion

This project demonstrates how Machine Learning can be used to predict weather conditions using both small sample data and large real-world datasets. Scaling up the data significantly improves model reliability and exposes practical challenges like missing values and class imbalance.
