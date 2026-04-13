import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(df, target_col="churned", test_size=0.2, random_state=42):
    """Split a DataFrame into train and test sets with stratification.

    Args:
        df: DataFrame with features and target column.
        target_col: Name of the target column.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def compute_classification_metrics(y_true, y_pred):
    """Compute classification metrics from true and predicted labels.

    Args:
        y_true: Array of true labels (0 or 1).
        y_pred: Array of predicted labels (0 or 1).

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
        Values are floats.
    """
   
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred))
    }
    return metrics


def run_cross_validation(X_train, y_train, n_folds=5, random_state=42):
    """Run stratified k-fold cross-validation with LogisticRegression.

    Args:
        X_train: Training features (numeric only).
        y_train: Training labels.
        n_folds: Number of CV folds.
        random_state: Random seed.

    Returns:
        Dictionary with keys: 'scores' (array of fold scores),
        'mean' (float), 'std' (float).
    """
    # إنشاء النموذج بنفس المعايير المطلوبة في المهام السابقة
    model = LogisticRegression(
        random_state=random_state, 
        max_iter=1000, 
        class_weight="balanced"
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=n_folds, scoring="accuracy")
    
  
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores))
    }


if __name__ == "__main__":

    try:
        df = pd.read_csv("data/telecom_churn.csv")
        print(f"Loaded {len(df)} rows")

        # Task 1: Split
        numeric_cols = ["tenure", "monthly_charges", "total_charges",
                        "num_support_calls", "senior_citizen", "has_partner",
                        "has_dependents"]
        df_numeric = df[numeric_cols + ["churned"]]

        result = split_data(df_numeric)
        if result is not None:
            X_train, X_test, y_train, y_test = result
            print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

            # Task 2: Metrics
            model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_classification_metrics(y_test, y_pred)
            if metrics:
                print(f"Metrics: {metrics}")

            # Task 3: Cross-validation
            cv_results = run_cross_validation(X_train, y_train)
            if cv_results:
                print(f"CV Mean Accuracy: {cv_results['mean']:.3f} +/- {cv_results['std']:.3f}")
    
    except FileNotFoundError:
        print("Error: 'data/telecom_churn.csv' not found. Please ensure the data directory exists.")