import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Drop customerID (not useful)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert target
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    return df