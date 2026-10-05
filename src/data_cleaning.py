import os
import pandas as pd
import numpy as np

def main():
    print("=" * 50)
    print("STARTING DATA CLEANING")
    print("=" * 50)
    
    # Original dataset path
    data_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return None
        
    df = pd.read_csv(data_path)
    
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst few rows:")
    print(df.head())
    
    print("\n" + "=" * 50)
    print("CLEANING TotalCharges")
    print("=" * 50)
    # Convert TotalCharges from string to float, replacing empty spaces with NaN
    df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True).astype(float)
    
    # Fill NaN with 0
    num_nans = df['TotalCharges'].isna().sum()
    print(f"Found {num_nans} missing values in TotalCharges. Filling with 0.")
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    print("\n" + "=" * 50)
    print("HANDLING customerID")
    print("=" * 50)
    # Save customerID mapping
    customer_ids = df['customerID'].copy()
    # Drop customerID
    df = df.drop(columns=['customerID'])
    print("Dropped customerID column.")
    
    print("\n" + "=" * 50)
    print("CONVERTING SeniorCitizen")
    print("=" * 50)
    # Convert SeniorCitizen from 0/1 to 'No'/'Yes'
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    print("Converted SeniorCitizen to Yes/No.")
    
    print("\n" + "=" * 50)
    print("CHECKING DUPLICATES")
    print("=" * 50)
    duplicates = df.duplicated().sum()
    print(f"Found {duplicates} duplicate rows.")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed duplicates. New shape: {df.shape}")
        
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(df.describe(include='all'))
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save cleaned data
    output_path = 'data/telco_churn_cleaned.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nSaved cleaned data to {output_path}")
    
    return df

if __name__ == "__main__":
    main()
