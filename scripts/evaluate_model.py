import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def evaluate_model(X_test_path, y_test_path, model_paths):
    # Load test data
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)  # Loaded as DataFrame
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values.ravel()  # Convert to 1D
    elif isinstance(y_test, np.ndarray):
        y_test = y_test.ravel()
    print("Test data loaded from", X_test_path)

    # Load pre-trained models
    models = {}
    for name, path in model_paths.items():
        models[name] = joblib.load(path)
        print(f"{name} model loaded from {path}")

    # Evaluate each pre-trained model
    results = {}
    for name, model in models.items():
        y_test_pred = model.predict(X_test).ravel()  # Ensure 1D
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        results[name] = {'MSE': mse, 'R²': r2}
        print(f"\n{name} Performance on Test Set:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R² Score: {r2:.2f}")

    # Evaluate with subset of features (RM, LSTAT)
    subset_features = ['RM', 'LSTAT']
    X_test_subset = X_test[subset_features]

    # Load training data to fit a new model on the subset
    X_train = pd.read_csv('data/X_train.csv')  # Need training data for fitting
    y_train = pd.read_csv('data/y_train.csv')
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.values.ravel()
    elif isinstance(y_train, np.ndarray):
        y_train = y_train.ravel()
    X_train_subset = X_train[subset_features]

    # Train a new Linear Regression model on the subset
    lr_subset = LinearRegression()
    lr_subset.fit(X_train_subset, y_train)
    y_test_pred_subset = lr_subset.predict(X_test_subset).ravel()
    mse_subset = mean_squared_error(y_test, y_test_pred_subset)
    r2_subset = r2_score(y_test, y_test_pred_subset)
    print("\nPerformance with Subset Features (RM, LSTAT):")
    print(f"Mean Squared Error (MSE): {mse_subset:.2f}")
    print(f"R² Score: {r2_subset:.2f}")

    return results

if __name__ == "__main__":
    X_test_path = 'data/X_test.csv'
    y_test_path = 'data/y_test.csv'
    model_paths = {
        'Linear Regression': 'models/linear_regression_model.pkl',
        'Ridge Regression': 'models/ridge_model.pkl'
    }
    results = evaluate_model(X_test_path, y_test_path, model_paths)