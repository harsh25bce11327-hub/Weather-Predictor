"""
weather_predictor.py
--------------------
Naive Bayes weather prediction model.

My project upports two data modes:
  1. Built-in 14-row sample dataset 
  2. External CSV file with real-world weather data

Usage:
  python weather_predictor.py
  python weather_predictor.py --csv path/to/weatherAUS.csv
  python weather_predictor.py --csv weatherAUS.csv --target RainTomorrow --test-size 0.2
"""

import argparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Built-in sample dataset (used when no CSV is provided)

SAMPLE_DATA = {
    'Outlook':       ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast',
                      'Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature':   ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild',
                      'Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity':      ['High','High','High','High','Normal','Normal','Normal','High',
                      'Normal','Normal','Normal','High','Normal','High'],
    'Wind':          ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak',
                      'Weak','Weak','Strong','Strong','Weak','Strong'],
    'RainTomorrow':  ['No','No','Yes','Yes','Yes','No','Yes','No',
                      'Yes','Yes','Yes','Yes','Yes','No'],
}


# Helpers

def load_sample_data():
    """Return the built-in toy dataset as a DataFrame."""
    return pd.DataFrame(SAMPLE_DATA)


def load_csv_data(filepath, target_col):
    """
    Load a CSV file, drop columns that are unhelpful (dates, IDs, free text),
    handle missing values, and return a cleaned DataFrame.
    """
    print(f"Loading CSV: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Raw shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Drop columns that are almost entirely null (>60% missing)
    threshold = 0.6
    null_frac = df.isnull().mean()
    drop_cols = null_frac[null_frac > threshold].index.tolist()
    if drop_cols:
        print(f"  Dropping cols with >{threshold*100:.0f}% nulls: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Drop common non-feature columns if present
    auto_drop = ['Date', 'date', 'Location', 'location', 'id', 'ID', 'row_id']
    present = [c for c in auto_drop if c in df.columns]
    if present:
        print(f"  Dropping identifier cols: {present}")
        df = df.drop(columns=present)

    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows where the target is null
    before = len(df)
    df = df.dropna(subset=[target_col])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with null target")

    # Fill remaining nulls: mode for categoricals, median for numerics
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == object:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    print(f"  Clean shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def encode_features(df):
    """
    Label-encode all object (string) columns, including the target.
    Returns the encoded DataFrame and the LabelEncoder fitted on the target.
    """
    le_map = {}
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include=['object', 'str']).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_map[col] = le
    return df_enc, le_map


def show_feature_importance(model, feature_names, top_n=10):
    """
    Gaussian NB doesn't have built-in feature importance, so we approximate
    it using the variance of the class-conditional means (a proxy for
    how much each feature shifts across classes).
    """
    # theta_ shape: (n_classes, n_features)
    importance = model.theta_.var(axis=0)
    ranked = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

    print("\nTop feature predictors (by class-mean variance):")
    for i, (feat, score) in enumerate(ranked[:top_n], 1):
        bar = "█" * int(score / max(importance) * 20 + 1)
        print(f"  {i:2}. {feat:<25} {bar}  ({score:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Weather prediction using Gaussian Naive Bayes"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to a CSV dataset file. Omit to use the built-in sample data."
    )
    parser.add_argument(
        "--target", type=str, default="RainTomorrow",
        help="Name of the target column (default: RainTomorrow)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.3,
        help="Fraction of data used for testing (default: 0.3)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n=== Weather Prediction — Naive Bayes ===\n")

    # 1. Load data
    if args.csv:
        df = load_csv_data(args.csv, args.target)
        print(f"\nData source   : {args.csv}")
    else:
        df = load_sample_data()
        print("Data source   : built-in sample dataset (14 rows)")
        print("Tip: Use --csv <file.csv> to train on real-world data\n")

    print(f"Total samples : {len(df):,}")

    # 2. Encode categorical features
    df_enc, le_map = encode_features(df)

    # 3. Split features and target
    X = df_enc.drop(columns=[args.target])
    y = df_enc[args.target]
    feature_names = list(X.columns)

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )
    print(f"Training set  : {len(X_train):,}")
    print(f"Test set      : {len(X_test):,}\n")

    # 5. Train model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # 6. Predict
    y_pred = model.predict(X_test)

    # 7. Evaluate
    acc = accuracy_score(y_test, y_pred) * 100
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred, zero_division=0)

    print(f"Accuracy      : {acc:.2f} %\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    # 8. Feature importance (only meaningful with multiple features)
    if len(feature_names) > 1:
        show_feature_importance(model, feature_names)

    print("\n========================================\n")


if __name__ == "__main__":
    main()
