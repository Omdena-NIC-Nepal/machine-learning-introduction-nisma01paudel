import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model(X_train_path, y_train_path, output_dir):
    # Load preprocessed data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    print("Training data loaded from", X_train_path)

    # Use all features (can be modified based on feature selection)
    selected_features = X_train.columns
    X_train_selected = X_train[selected_features]
    print("Selected Features:", list(selected_features))

    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_selected, y_train)
    y_train_pred = lr_model.predict(X_train_selected)
    
    # Evaluate
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    print("\nLinear Regression - Training Performance:")
    print(f"Mean Squared Error (MSE): {train_mse:.2f}")
    print(f"R^2 Score: {train_r2:.2f}")

    # Cross-validation
    cv_scores = cross_val_score(lr_model, X_train_selected, y_train, cv=5, scoring='r2')
    print(f"Average CV R^2 Score: {cv_scores.mean():.2f}")

    # Train Ridge model with hyperparameter tuning
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_scores = []
    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha)
        cv_score = cross_val_score(ridge_model, X_train_selected, y_train, cv=5, scoring='r2').mean()
        ridge_scores.append(cv_score)
    
    best_alpha = alphas[np.argmax(ridge_scores)]
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X_train_selected, y_train)
    print(f"Best Ridge model (alpha={best_alpha}) trained.")

    # Save models
    joblib.dump(lr_model, f'{output_dir}/linear_regression_model.pkl')
    joblib.dump(ridge_model, f'{output_dir}/ridge_model.pkl')
    print(f"Models saved to {output_dir}")

if __name__ == "__main__":
    X_train_path = 'data/X_train.csv'
    y_train_path = 'data/y_train.csv'
    output_dir = 'models'
    import os
    os.makedirs(output_dir, exist_ok=True)
    train_model(X_train_path, y_train_path, output_dir)