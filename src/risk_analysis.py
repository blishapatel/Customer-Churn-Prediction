import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Light pastel color scheme from conventions
PALETTE = ['#A8D8EA', '#AA96DA', '#FCBAD3', '#FFFFD2', '#B5EAD7', '#FFB7B2', '#E2F0CB', '#C7CEEA']
RISK_COLORS = {'Low Risk': '#B5EAD7', 'Medium Risk': '#FFFFD2', 'High Risk': '#FFB7B2'}

def main():
    print("=" * 50)
    print("Starting Risk Analysis Module")
    print("=" * 50)
    
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    cleaned_path = 'data/telco_churn_cleaned.csv'
    model_path = 'models/churn_model.pkl'
    orig_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # 1. Load data
    df = pd.read_csv(cleaned_path)
    
    # 2. Load model
    model_dict = joblib.load(model_path)
        
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_columns = model_dict['feature_columns']
    
    # 3. Prepare dataset
    if 'Churn' in df.columns:
        if df['Churn'].dtype == 'object':
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    else:
        df['Churn'] = 0 # Fallback if missing
        
    X = pd.get_dummies(df.drop('Churn', axis=1, errors='ignore'))
    
    # Align columns with saved feature_columns
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    
    # Scale numeric features (must match training: only tenure, MonthlyCharges, TotalCharges)
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    num_cols_present = [c for c in numeric_cols if c in X.columns]
    if num_cols_present:
        X[num_cols_present] = scaler.transform(X[num_cols_present])
    
    # 4. Predict churn probability
    probs = model.predict_proba(X)[:, 1]
    
    # 5. Categorize risk
    def get_risk_cat(p):
        if p <= 0.3:
            return 'Low Risk'
        elif p <= 0.6:
            return 'Medium Risk'
        else:
            return 'High Risk'
            
    risk_categories = [get_risk_cat(p) for p in probs]
    
    df['Churn_Probability'] = probs
    df['Risk_Category'] = risk_categories
    
    # Reload original CSV to get customerID and merge
    df_orig = pd.read_csv(orig_path)
    if len(df) == len(df_orig):
        df['customerID'] = df_orig['customerID']
    else:
        # Assuming index alignment or similar if data dropped
        pass
        
    # 6. Print risk distribution
    dist = df['Risk_Category'].value_counts()
    dist_pct = df['Risk_Category'].value_counts(normalize=True) * 100
    
    print("\nRisk Distribution:")
    for cat in ['Low Risk', 'Medium Risk', 'High Risk']:
        if cat in dist:
            print(f"{cat}: {dist[cat]} ({dist_pct[cat]:.1f}%)")
            
    # 7. Profile high-risk customers
    high_risk = df[df['Risk_Category'] == 'High Risk']
    print("\nHigh-Risk Customer Profile:")
    if 'tenure' in high_risk.columns:
        print(f"Average Tenure: {high_risk['tenure'].mean():.1f} months")
    if 'Contract' in high_risk.columns:
        print(f"Most Common Contract: {high_risk['Contract'].mode()[0]}")
    if 'InternetService' in high_risk.columns:
        print(f"Most Common Internet Service: {high_risk['InternetService'].mode()[0]}")
    if 'MonthlyCharges' in high_risk.columns:
        print(f"Average Monthly Charges: ${high_risk['MonthlyCharges'].mean():.2f}")
        
    # 8. Generate charts
    sns.set_theme(style='whitegrid', palette=PALETTE)
    
    # a. risk_distribution.png
    plt.figure(figsize=(8, 6))
    colors = [RISK_COLORS[cat] for cat in dist.index]
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Risk Category Distribution', color='#333333')
    plt.savefig('reports/figures/risk_distribution.png', bbox_inches='tight')
    plt.close()
    
    # b. risk_by_tenure.png
    plt.figure(figsize=(8, 6))
    order = ['Low Risk', 'Medium Risk', 'High Risk']
    sns.boxplot(x='Risk_Category', y='tenure', data=df, palette=RISK_COLORS, order=order)
    plt.title('Tenure by Risk Category', color='#333333')
    plt.savefig('reports/figures/risk_by_tenure.png', bbox_inches='tight')
    plt.close()
    
    # c. risk_by_charges.png
    plt.figure(figsize=(8, 6))
    if 'MonthlyCharges' in df.columns:
        sns.boxplot(x='Risk_Category', y='MonthlyCharges', data=df, palette=RISK_COLORS, order=order)
        plt.title('Monthly Charges by Risk Category', color='#333333')
        plt.savefig('reports/figures/risk_by_charges.png', bbox_inches='tight')
    plt.close()
    
    # d. risk_probability_histogram.png
    plt.figure(figsize=(8, 6))
    for cat, color in RISK_COLORS.items():
        subset = df[df['Risk_Category'] == cat]['Churn_Probability']
        if len(subset) > 0:
            sns.histplot(subset, color=color, label=cat, bins=20, alpha=0.7, kde=True)
    plt.title('Churn Probability Distribution', color='#333333')
    plt.xlabel('Probability of Churn')
    plt.legend()
    plt.savefig('reports/figures/risk_probability_histogram.png', bbox_inches='tight')
    plt.close()
    
    # 9. Save the full dataset
    cols = ['customerID'] + [c for c in df.columns if c != 'customerID'] if 'customerID' in df.columns else df.columns.tolist()
    df[cols].to_csv('data/customer_risk_scores.csv', index=False, encoding='utf-8')
    
    # 10. Print sample high-risk customers
    print("\nTop 10 High-Risk Customers:")
    top_10 = df.nlargest(10, 'Churn_Probability')
    display_cols = ['customerID', 'Churn_Probability', 'Risk_Category'] if 'customerID' in df.columns else ['Churn_Probability', 'Risk_Category']
    print(top_10[display_cols].to_string(index=False))
    
    print("=" * 50)
    return df

if __name__ == '__main__':
    main()
