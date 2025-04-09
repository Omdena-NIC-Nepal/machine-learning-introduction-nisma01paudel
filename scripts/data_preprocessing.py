import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

def preprocess_data(input_path, output_dir):
    # Load dataset
    data = pd.read_csv(input_path)
    print("Dataset loaded from", input_path)

    # Handle missing values (none expected, but check)
    if data.isnull().sum().sum() > 0:
        data = data.dropna()
        print("Missing values dropped.")
    else:
        print("No missing values found.")

    # Cap outliers
    key_features = ['CRIM', 'RM', 'LSTAT', 'MEDV']
    for feature in key_features:
        data = cap_outliers(data, feature)
    print("Outliers capped for key features.")

    # Separate features and target
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    print("Features standardized.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # Save preprocessed data
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    print(f"Preprocessed data saved to {output_dir}")

if __name__ == "__main__":
    input_path = 'data/BostonHousing.csv'
    output_dir = 'data'
    preprocess_data(input_path, output_dir)